"""Occupancy grid + A* waypoint planning over `hospital.usd`.

Why this exists
---------------
`NavEnv` (nav_task.py) is a 4 x 4 m arena with three obstacles. The policy
trained in it steers well over short distances but has never seen a 76 x 42 m
building, and `skills.fit_locations_to_arena()` deals with that by *shrinking
the hospital onto the arena* -- the robot never actually drives the building.

To traverse the real hospital the long-range problem (which way around the
walls) has to be solved separately from the short-range one (steer to a point
without clipping anything). This module is the long-range half: it rasterises
the hospital's own geometry into an occupancy grid and A*s through it, handing
`isaac_nav.py` a list of waypoints that are each close enough to be inside the
trained policy's competence.

Everything here is pure `pxr` -- no `SimulationApp`, no PhysX. Booting Isaac
Sim just to read bounding boxes costs minutes; this takes seconds and the
result is cached to an `.npz` beside the USD.

Coordinate frames: hospital.usd is Z-up and metres (`metersPerUnit == 1`), so
grid XY is world XY directly. `z_band` is the slice of height the robot's body
sweeps -- a prim only blocks the robot if it occupies that band, which is why
the floor itself (z ~ 0) and the ceiling lights do not become obstacles.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

DEFAULT_HOSPITAL_USD = (
    "/home/dell/AI/assets/isaac-5.1/Isaac/Environments/Hospital/hospital.usd"
)

# The Ridgeback's body sweeps roughly 0.15..1.6 m off the floor once the Franka
# is folded. Matches the band check_isaac_hospital.py used to find its spawn.
Z_BAND = (0.15, 1.6)

# The Ridgeback's footprint is 0.96 x 0.79 m, so it circumscribes at ~0.62 m
# and the grid has to be inflated by at least that or A* returns paths through
# gaps the base physically cannot fit. Planning at 0.45 m did exactly that:
# traversals clipped corners and jammed against door frames at measured
# clearances of 0.00-0.22 m.
#
# The obvious worry is sealing doorways, but this hospital is open-plan -- the
# whole reachable region stays one connected component even when eroded by
# 0.85 m, and all 30 ordered location pairs still plan at 0.75 m. 0.65 m clears
# the base with a small margin for yaw and the folded arm.
ROBOT_RADIUS = 0.65

GRID_RESOLUTION = 0.10  # metres per cell

# Where the scene graph's location ids actually are inside hospital.usd.
#
# `simbiote/assets/scenes/hospital_scene_graph.json` is a hand-written stand-in for
# Teammate 1's output and its poses span about 14 x 24 m around the origin --
# they are a *layout*, not hospital.usd coordinates. Dropping them onto the
# real building puts every location in the same room, or inside a wall.
#
# These anchors preserve the fixture's relative layout (corridor in the middle;
# supply room north-west, room 1 north-east, room 2 south-east, nurse station
# south-west of it) while being real, reachable, high-clearance points in the
# building. Each was chosen as a local maximum of the clearance field within
# the component reachable from SPAWN, then checked: all 30 ordered pairs plan,
# 10-74 m apart. Clearances in metres are noted per entry.
HOSPITAL_LOCATIONS: dict[str, tuple[float, float]] = {
    "corridor": (-2.74, 10.70),       # 2.12 -- main east-west corridor, centre
    "supply_room": (-26.74, 18.70),   # 2.68 -- west wing, north
    "room_1": (13.26, 31.20),         # 2.42 -- north-east wing
    "room_2": (17.26, 5.20),          # 4.04 -- east wing, south
    "nurse_station": (-29.24, 5.70),  # 3.14 -- west wing, south
}

# Clearest point on the hospital floor (2.75-2.80 m to the nearest prim in the
# robot's height band). Shared with check_isaac_hospital.py / view_isaac_hospital.py.
SPAWN = (7.81, 8.25)


@dataclass(frozen=True)
class GridSpec:
    """Maps world metres <-> grid indices."""

    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int((x - self.origin_x) / self.resolution),
            int((y - self.origin_y) / self.resolution),
        )

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        # Cell centre, not corner -- a path that returns corners drifts half a
        # cell toward the origin at every waypoint.
        return (
            self.origin_x + (ix + 0.5) * self.resolution,
            self.origin_y + (iy + 0.5) * self.resolution,
        )

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height


def _world_bounds_in_band(
    usd_path: str, z_band: tuple[float, float]
) -> tuple[list[tuple[float, float, float, float]], tuple[float, float, float, float]]:
    """Axis-aligned XY footprints of every prim intersecting the height band,
    plus the XY extent of the floor geometry.

    The floor is tracked separately because "not an obstacle" is not the same
    as "walkable": everything outside the building is also free of obstacles,
    and without a floor mask A* happily routes the robot around the *outside*
    of the hospital to reach the next room.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"could not open {usd_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )

    z_lo, z_hi = z_band
    footprints: list[tuple[float, float, float, float]] = []
    floor_min_x = floor_min_y = math.inf
    floor_max_x = floor_max_y = -math.inf

    # Every one of the hospital's 1064 `Geo_` prims is `instanceable = true`,
    # so a plain `stage.Traverse()` stops at the instance and finds 23 meshes
    # in the whole building -- enough to produce a completely empty occupancy
    # grid that looks like a successful build. Descending into instance
    # proxies finds the real 2059.
    prim_range = Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    )

    for prim in prim_range:
        if not prim.IsA(UsdGeom.Gprim):
            continue
        # The mesh itself is a generic `Section0`; the name that says what this
        # is ("Geo_M3_Floor_396") is one or two levels up, and under instancing
        # the useful ancestor can be either. Walk up rather than guess a depth.
        ancestry = []
        walker = prim
        for _ in range(4):
            if not walker or walker.IsPseudoRoot():
                break
            ancestry.append(walker.GetName())
            walker = walker.GetParent()
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if box.IsEmpty():
            continue
        lo, hi = box.GetMin(), box.GetMax()

        # Naming alone is not enough: `S_WetFloorSign` contains "Floor" and is
        # a 0.79 m tall obstacle sitting in the middle of a corridor. Skipping
        # it as floor left a hole in the grid that the planner routed straight
        # through, and the base jammed on it in what looked like open space
        # (2.5 m of apparent clearance). Require the prim to actually be flat
        # and on the ground before believing the name.
        thickness = float(hi[2] - lo[2])
        is_floor = (
            any("Floor" in name for name in ancestry)
            and thickness < 0.3
            and float(lo[2]) < 0.3
        )
        if is_floor:
            floor_min_x = min(floor_min_x, lo[0])
            floor_min_y = min(floor_min_y, lo[1])
            floor_max_x = max(floor_max_x, hi[0])
            floor_max_y = max(floor_max_y, hi[1])
            # A floor slab is thin and sits at z~0, so it falls out of the band
            # test below on its own. Skipping explicitly keeps a slab with a
            # raised lip from walling off the room it belongs to.
            continue

        # Ceiling-mounted lights and the roof are above the band; floor decals
        # are below it. Only what the robot would actually hit counts.
        if hi[2] < z_lo or lo[2] > z_hi:
            continue
        footprints.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))

    if not math.isfinite(floor_min_x):
        raise RuntimeError(
            f"no floor geometry found in {usd_path}; cannot bound the walkable region"
        )
    return footprints, (floor_min_x, floor_min_y, floor_max_x, floor_max_y)


def _inflate(occupied: np.ndarray, cells: int) -> np.ndarray:
    """Grow obstacles by `cells` using a chamfer distance transform.

    scipy is not installed under Isaac Sim's interpreter, so this is a
    hand-rolled two-pass transform rather than `binary_dilation`. At 0.1 m
    cells over a 76 x 42 m building it is ~320k cells -- fast enough vectorised
    per row/column.
    """
    if cells <= 0:
        return occupied.copy()
    # Squared-distance transform via the standard two-pass 1-D decomposition.
    inf = occupied.size + 1
    dist = np.where(occupied, 0, inf).astype(np.int32)

    # Pass along x, then along y, using the exact 1-D EDT (Felzenszwalb) per
    # axis. For the small radii we use, a chamfer approximation would round
    # doorways shut, so do it exactly.
    def edt_1d(f: np.ndarray) -> np.ndarray:
        n = f.shape[0]
        d = np.empty(n, dtype=np.float64)
        v = np.zeros(n, dtype=np.int64)
        z = np.empty(n + 1, dtype=np.float64)
        k = 0
        v[0] = 0
        z[0], z[1] = -np.inf, np.inf
        for q in range(1, n):
            while True:
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                if s <= z[k]:
                    k -= 1
                    if k < 0:
                        k = 0
                        break
                else:
                    break
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            d[q] = (q - v[k]) ** 2 + f[v[k]]
        return d

    work = dist.astype(np.float64)
    for ix in range(work.shape[0]):
        work[ix, :] = edt_1d(work[ix, :])
    for iy in range(work.shape[1]):
        work[:, iy] = edt_1d(work[:, iy])
    return work <= float(cells * cells)


class HospitalMap:
    """Occupancy grid over hospital.usd, with A* between free cells."""

    def __init__(
        self,
        occupied: np.ndarray,
        spec: GridSpec,
        usd_path: str = DEFAULT_HOSPITAL_USD,
        robot_radius: float = ROBOT_RADIUS,
    ):
        self.spec = spec
        self.usd_path = usd_path
        self.robot_radius = robot_radius
        # `raw` is the true geometry; `blocked` is what the planner uses.
        self.raw = occupied
        inflate_cells = round(robot_radius / spec.resolution)
        self.blocked = _inflate(occupied, inflate_cells)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        usd_path: str = DEFAULT_HOSPITAL_USD,
        resolution: float = GRID_RESOLUTION,
        z_band: tuple[float, float] = Z_BAND,
        robot_radius: float = ROBOT_RADIUS,
    ) -> HospitalMap:
        footprints, floor = _world_bounds_in_band(usd_path, z_band)
        min_x, min_y, max_x, max_y = floor
        pad = 1.0
        origin_x, origin_y = min_x - pad, min_y - pad
        width = math.ceil((max_x - min_x + 2 * pad) / resolution)
        height = math.ceil((max_y - min_y + 2 * pad) / resolution)
        spec = GridSpec(origin_x, origin_y, resolution, width, height)

        # Start fully blocked, carve out the floor, then stamp obstacles back
        # in. Anything off the floor slabs stays blocked, which is what keeps
        # A* from routing around the outside of the building.
        occupied = np.ones((width, height), dtype=bool)
        fx0, fy0 = spec.to_cell(min_x, min_y)
        fx1, fy1 = spec.to_cell(max_x, max_y)
        occupied[max(fx0, 0) : fx1 + 1, max(fy0, 0) : fy1 + 1] = False

        for lo_x, lo_y, hi_x, hi_y in footprints:
            x0, y0 = spec.to_cell(lo_x, lo_y)
            x1, y1 = spec.to_cell(hi_x, hi_y)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, width - 1), min(y1, height - 1)
            if x1 < x0 or y1 < y0:
                continue
            occupied[x0 : x1 + 1, y0 : y1 + 1] = True

        return cls(occupied, spec, usd_path=usd_path, robot_radius=robot_radius)

    @classmethod
    def load_or_build(
        cls,
        usd_path: str = DEFAULT_HOSPITAL_USD,
        cache_path: Path | None = None,
        resolution: float = GRID_RESOLUTION,
        z_band: tuple[float, float] = Z_BAND,
        robot_radius: float = ROBOT_RADIUS,
    ) -> HospitalMap:
        cache_path = Path(
            cache_path
            or Path(__file__).resolve().parent.parent.parent
            / "checkpoints"
            / "hospital_occupancy.npz"
        )
        if cache_path.is_file():
            data = np.load(cache_path)
            # A cache built at a different resolution or inflation radius would
            # silently plan paths the robot cannot take.
            if (
                float(data["resolution"]) == resolution
                and float(data["robot_radius"]) == robot_radius
            ):
                spec = GridSpec(
                    float(data["origin_x"]),
                    float(data["origin_y"]),
                    float(data["resolution"]),
                    int(data["width"]),
                    int(data["height"]),
                )
                return cls(
                    data["occupied"], spec, usd_path=usd_path, robot_radius=robot_radius
                )

        hospital_map = cls.build(
            usd_path, resolution=resolution, z_band=z_band, robot_radius=robot_radius
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            occupied=hospital_map.raw,
            origin_x=hospital_map.spec.origin_x,
            origin_y=hospital_map.spec.origin_y,
            resolution=hospital_map.spec.resolution,
            width=hospital_map.spec.width,
            height=hospital_map.spec.height,
            robot_radius=robot_radius,
        )
        return hospital_map

    # -- queries ------------------------------------------------------------

    def is_free(self, x: float, y: float) -> bool:
        ix, iy = self.spec.to_cell(x, y)
        return self.spec.in_bounds(ix, iy) and not self.blocked[ix, iy]

    def clearance(self, x: float, y: float, limit: float = 5.0) -> float:
        """Distance from (x, y) to the nearest *raw* obstacle cell, in metres.

        Uses `raw` rather than `blocked` so the number means what a human
        expects ("2.75 m to the nearest prop") instead of being offset by the
        inflation radius.
        """
        cx, cy = self.spec.to_cell(x, y)
        max_cells = int(limit / self.spec.resolution)
        for radius in range(max_cells + 1):
            x0, x1 = max(cx - radius, 0), min(cx + radius + 1, self.spec.width)
            y0, y1 = max(cy - radius, 0), min(cy + radius + 1, self.spec.height)
            window = self.raw[x0:x1, y0:y1]
            if window.any():
                hits = np.argwhere(window)
                dx = (hits[:, 0] + x0 - cx) * self.spec.resolution
                dy = (hits[:, 1] + y0 - cy) * self.spec.resolution
                return float(np.min(np.hypot(dx, dy)))
        return limit

    def nearest_free(
        self, x: float, y: float, max_radius: float = 6.0
    ) -> tuple[float, float] | None:
        """Closest planner-free point to (x, y).

        Scene-graph anchors are authored by hand and land inside a bed or a
        wall often enough that failing outright would make the demo brittle.
        """
        if self.is_free(x, y):
            return (x, y)
        cx, cy = self.spec.to_cell(x, y)
        for radius in range(1, int(max_radius / self.spec.resolution) + 1):
            x0, x1 = max(cx - radius, 0), min(cx + radius + 1, self.spec.width)
            y0, y1 = max(cy - radius, 0), min(cy + radius + 1, self.spec.height)
            window = self.blocked[x0:x1, y0:y1]
            free = np.argwhere(~window)
            if free.size:
                dx = free[:, 0] + x0 - cx
                dy = free[:, 1] + y0 - cy
                best = free[np.argmin(dx * dx + dy * dy)]
                return self.spec.to_world(int(best[0]) + x0, int(best[1]) + y0)
        return None

    # -- planning -----------------------------------------------------------

    _NEIGHBOURS: ClassVar[list[tuple[int, int, float]]] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        simplify: bool = True,
    ) -> list[tuple[float, float]] | None:
        """A* from `start` to `goal`, returned as world-space waypoints.

        Returns None when the goal is unreachable, rather than a partial path
        -- a nav skill that reports success after stopping at a wall is worse
        than one that reports it could not plan.
        """
        start_free = self.nearest_free(*start)
        goal_free = self.nearest_free(*goal)
        if start_free is None or goal_free is None:
            return None

        start_cell = self.spec.to_cell(*start_free)
        goal_cell = self.spec.to_cell(*goal_free)
        if start_cell == goal_cell:
            return [goal_free]

        blocked = self.blocked
        width, height = self.spec.width, self.spec.height

        def heuristic(cell: tuple[int, int]) -> float:
            dx = abs(cell[0] - goal_cell[0])
            dy = abs(cell[1] - goal_cell[1])
            # Octile: admissible for 8-connected grids where a Euclidean
            # heuristic under-estimates badly enough to expand the whole map.
            return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

        open_heap = [(heuristic(start_cell), 0.0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start_cell: 0.0}
        closed = set()

        while open_heap:
            _, g, cell = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)
            if cell == goal_cell:
                break
            cx, cy = cell
            for dx, dy, step in self._NEIGHBOURS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < width and 0 <= ny < height) or blocked[nx, ny]:
                    continue
                # Refuse to cut a diagonal between two blocked cells; the base
                # is 0.79 m wide and would clip the corner it slipped past.
                if dx and dy and (blocked[cx + dx, cy] or blocked[cx, cy + dy]):
                    continue
                candidate = g + step
                if candidate < cost.get((nx, ny), math.inf):
                    cost[(nx, ny)] = candidate
                    came_from[(nx, ny)] = cell
                    heapq.heappush(
                        open_heap, (candidate + heuristic((nx, ny)), candidate, (nx, ny))
                    )
        else:
            return None

        if goal_cell not in came_from and goal_cell != start_cell:
            return None

        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()

        points = [self.spec.to_world(*cell) for cell in cells]
        return self.simplify(points) if simplify else points

    def _line_is_clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        samples = max(int(distance / (self.spec.resolution * 0.5)), 1)
        for i in range(samples + 1):
            t = i / samples
            if not self.is_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    def simplify(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """String-pull the grid path down to the corners that matter.

        A raw A* path is one waypoint per 0.1 m cell, which would make the nav
        controller re-target 300 times crossing a corridor. Collapsing it to
        line-of-sight segments leaves ~5 waypoints for the same trip.
        """
        if len(points) < 3:
            return list(points)
        out = [points[0]]
        anchor = 0
        for index in range(2, len(points)):
            if not self._line_is_clear(points[anchor], points[index]):
                out.append(points[index - 1])
                anchor = index - 1
        out.append(points[-1])
        return out

    # -- debugging ----------------------------------------------------------

    def ascii_map(
        self,
        marks: dict[str, tuple[float, float]] | None = None,
        path: Iterable[tuple[float, float]] | None = None,
        step: int = 8,
    ) -> str:
        """Coarse text render. `step` is in cells, so 8 at 0.1 m is 0.8 m/char."""
        rows = []
        overlay: dict[tuple[int, int], str] = {}
        if path:
            for x, y in path:
                cell = self.spec.to_cell(x, y)
                overlay[(cell[0] // step, cell[1] // step)] = "*"
        for label, (x, y) in (marks or {}).items():
            cell = self.spec.to_cell(x, y)
            overlay[(cell[0] // step, cell[1] // step)] = label[0].upper()

        for iy in range(self.spec.height // step, -1, -1):
            row = []
            for ix in range(self.spec.width // step):
                if (ix, iy) in overlay:
                    row.append(overlay[(ix, iy)])
                    continue
                window = self.blocked[
                    ix * step : (ix + 1) * step, iy * step : (iy + 1) * step
                ]
                if window.size == 0:
                    row.append(" ")
                elif window.all():
                    row.append("#")
                elif window.any():
                    row.append("+")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)
