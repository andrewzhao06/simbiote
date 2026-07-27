"""Reading Teammate 1's scene graph — spec §6b.5.

The graph shape here is a PROPOSAL derived from what §4.3 says Step 1's
``build_graph.py`` produces (labelled nodes with pose and bounding volume,
relationships as edges) and what §5.5 says the grasp task reads off objects
(``is_graspable`` / ``grasp_type`` / ``mass_kg``).

Everything that knows the on-disk format is confined to :func:`_parse`. When
Teammate 1's real schema lands, that function changes and nothing downstream
of it does.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from simbiote.robot_iface.actions import Pose

__all__ = [
    "Location",
    "SceneObject",
    "SceneGraph",
    "load_scene",
    "default_scene_path",
    "list_locations",
    "list_objects",
]

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "hospital_scene_graph.json"

#: Below this, ``resolve()`` returns None rather than guessing. A wrong id is
#: worse than a clean "I don't know what you mean" — it sends the robot to the
#: wrong room instead of reporting a parse failure.
_MIN_MATCH_SCORE = 1.0


def default_scene_path() -> Path:
    """The hand-written hospital fixture, used until Step 1's export exists."""
    return _FIXTURE


@dataclass(frozen=True)
class Location:
    """A navigable region — a room, a corridor, a bay."""

    id: str
    label: str
    pose: Pose
    navigable: bool = True
    aliases: tuple[str, ...] = ()
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.label,) + self.aliases


@dataclass(frozen=True)
class SceneObject:
    """A physical object, possibly graspable.

    ``grasp_type`` and ``mass_kg`` are carried through untouched so that
    ``grasp_task.py`` (Step 2) and the agentic skills read the same tags rather
    than each hardcoding assumptions — spec §5.5.
    """

    id: str
    label: str
    pose: Pose
    is_graspable: bool = False
    grasp_type: str | None = None
    mass_kg: float | None = None
    aliases: tuple[str, ...] = ()
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    #: Present on objects with a dedicated grasp point (the wheelchair handle).
    handle_pose: Pose | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.label,) + self.aliases


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str


@dataclass
class SceneGraph:
    """The queryable scene. Built only via :func:`load_scene` / :func:`_parse`."""

    scene_id: str
    locations: dict[str, Location] = field(default_factory=dict)
    objects: dict[str, SceneObject] = field(default_factory=dict)
    edges: tuple[Edge, ...] = ()
    schema_version: str = "0.1"

    # ---- inventory -------------------------------------------------------

    def list_locations(self) -> list[Location]:
        return list(self.locations.values())

    def list_objects(self) -> list[SceneObject]:
        return list(self.objects.values())

    def graspable_objects(self) -> list[SceneObject]:
        return [o for o in self.objects.values() if o.is_graspable]

    # ---- lookup ----------------------------------------------------------

    def has_location(self, node_id: str) -> bool:
        return node_id in self.locations

    def has_object(self, node_id: str) -> bool:
        return node_id in self.objects

    def get_location(self, node_id: str) -> Location | None:
        return self.locations.get(node_id)

    def get_object(self, node_id: str) -> SceneObject | None:
        return self.objects.get(node_id)

    def location_of(self, object_id: str) -> Location | None:
        """The location an object sits in, via an ``in`` / ``on`` edge."""
        for edge in self.edges:
            if edge.src == object_id and edge.relation in ("in", "on", "inside", "on-top-of"):
                loc = self.locations.get(edge.dst)
                if loc is not None:
                    return loc
        return None

    # ---- natural-language resolution ------------------------------------

    def resolve(self, phrase: str, kind: str | None = None) -> str | None:
        """Map a noun phrase onto a node id.

        ``kind`` narrows the search to ``"location"`` or ``"object"``. Returns
        ``None`` when nothing scores above :data:`_MIN_MATCH_SCORE`.
        """
        norm = _normalize(phrase)
        candidates: Iterable[Location | SceneObject]
        if kind == "location":
            candidates = self.locations.values()
        elif kind == "object":
            candidates = self.objects.values()
        else:
            candidates = [*self.objects.values(), *self.locations.values()]

        best_id: str | None = None
        best_score = 0.0
        for node in candidates:
            score = self._score(node, norm)
            if score > best_score:
                best_score, best_id = score, node.id

        return best_id if best_score >= _MIN_MATCH_SCORE else None

    def _score(self, node: Location | SceneObject, norm_phrase: str) -> float:
        phrase_tokens = set(norm_phrase.split())
        best = 0.0
        for name in node.names:
            name_norm = _normalize(name)
            if not name_norm:
                continue
            if name_norm in norm_phrase:
                # Longer exact substring hits win: "supply room" beats "room".
                best = max(best, 2.0 + 0.1 * len(name_norm.split()))
            else:
                name_tokens = set(name_norm.split())
                if name_tokens:
                    overlap = len(name_tokens & phrase_tokens) / len(name_tokens)
                    best = max(best, 1.5 * overlap)

        # Disambiguation nudge: an object whose containing room is also named in
        # the phrase ("the tray in the supply room") beats a same-label object
        # elsewhere in the building.
        if best > 0 and isinstance(node, SceneObject):
            loc = self.location_of(node.id)
            if loc and any(_normalize(n) in norm_phrase for n in loc.names):
                best += 0.5
        return best

    # ---- prompt support --------------------------------------------------

    def inventory_text(self) -> str:
        """A compact scene listing for the LLM prompt.

        Giving the model the real ids is what keeps it from inventing
        ``room_7``; ``tool_schema.validate_calls`` is the backstop if it does
        anyway.
        """
        lines = ["LOCATIONS (valid location_id values):"]
        for loc in self.list_locations():
            alias = f" (aka {', '.join(loc.aliases)})" if loc.aliases else ""
            lines.append(f"  - {loc.id}: {loc.label}{alias}")
        lines.append("OBJECTS (valid object_id values):")
        for obj in self.list_objects():
            alias = f" (aka {', '.join(obj.aliases)})" if obj.aliases else ""
            where = self.location_of(obj.id)
            where_s = f", in {where.id}" if where else ""
            grasp = "graspable" if obj.is_graspable else "NOT graspable"
            lines.append(f"  - {obj.id}: {obj.label}{alias} [{grasp}{where_s}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter seam — the only code that knows the on-disk format.
# ---------------------------------------------------------------------------


def _parse(raw: dict[str, Any]) -> SceneGraph:
    """Turn Teammate 1's JSON into a :class:`SceneGraph`.

    ADAPTER SEAM. If Step 1's real export differs from the proposal, rewrite
    this function only.
    """
    locations: dict[str, Location] = {}
    objects: dict[str, SceneObject] = {}

    for node in raw.get("nodes", []):
        node_id = node["id"]
        kind = node.get("kind")
        pose = Pose.from_dict(node["pose"]) if node.get("pose") else Pose()
        aliases = tuple(node.get("aliases", ()))
        bbox = _parse_bbox(node.get("bbox"))

        if kind == "location":
            locations[node_id] = Location(
                id=node_id,
                label=node.get("label", node_id),
                pose=pose,
                navigable=bool(node.get("navigable", True)),
                aliases=aliases,
                bbox=bbox,
            )
        elif kind == "object":
            handle = node.get("handle_pose")
            objects[node_id] = SceneObject(
                id=node_id,
                label=node.get("label", node_id),
                pose=pose,
                is_graspable=bool(node.get("is_graspable", False)),
                grasp_type=node.get("grasp_type"),
                mass_kg=node.get("mass_kg"),
                aliases=aliases,
                bbox=bbox,
                handle_pose=Pose.from_dict(handle) if handle else None,
            )
        else:
            raise ValueError(
                f"scene graph node {node_id!r} has unknown kind {kind!r} "
                "(expected 'location' or 'object')"
            )

    edges = tuple(
        Edge(src=e["from"], dst=e["to"], relation=e.get("relation", "related_to"))
        for e in raw.get("edges", [])
    )

    return SceneGraph(
        scene_id=raw.get("scene_id", "unknown"),
        locations=locations,
        objects=objects,
        edges=edges,
        schema_version=str(raw.get("schema_version", "0.1")),
    )


def _parse_bbox(
    bbox: dict[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not bbox:
        return None
    lo, hi = bbox["min"], bbox["max"]
    return (
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )


def load_scene(path: str | Path | None = None) -> SceneGraph:
    """Load a scene graph, defaulting to the hospital fixture."""
    p = Path(path) if path is not None else default_scene_path()
    if not p.exists():
        raise FileNotFoundError(f"scene graph not found: {p}")
    return _parse(json.loads(p.read_text(encoding="utf-8")))


_PUNCT = re.compile(r"[^a-z0-9 ]+")
_STOPWORDS = frozenset({"the", "a", "an", "to", "in", "at", "into", "from", "of", "please"})


def _normalize(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(t for t in text.split() if t not in _STOPWORDS)


# Module-level conveniences matching the names in the build spec (§6b.5).


def list_locations(scene: SceneGraph) -> list[Location]:
    return scene.list_locations()


def list_objects(scene: SceneGraph) -> list[SceneObject]:
    return scene.list_objects()
