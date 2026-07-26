"""OpenUSD export and Step 2 handoff validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from simbiote.mapper.models import SceneGraph, SceneNode


@dataclass(slots=True)
class ValidationReport:
    valid: bool
    checks: dict[str, bool]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"node_{cleaned}"
    return cleaned


def _usd_value(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        return "bool", str(value).lower()
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "double", repr(value)
    return "string", json.dumps(str(value))


def _node_usda(node: SceneNode) -> str:
    center = ", ".join(f"{value:.8g}" for value in node.bounds.center)
    size = ", ".join(f"{value:.8g}" for value in node.bounds.size)
    lines = [
        f'    def Cube "{_identifier(node.node_id)}" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "    )",
        "    {",
        "        double size = 1",
        f"        double3 xformOp:translate = ({center})",
        f"        float3 xformOp:scale = ({size})",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        f"        custom string simbiote:nodeId = {json.dumps(node.node_id)}",
        f"        custom string simbiote:label = {json.dumps(node.label)}",
        f"        custom string simbiote:kind = {json.dumps(node.kind)}",
        f"        custom double simbiote:confidence = {node.confidence:.8g}",
    ]
    for key, value in sorted(node.attributes.items()):
        value_type, serialized = _usd_value(value)
        lines.append(
            f"        custom {value_type} simbiote:{_identifier(key)} = {serialized}"
        )
    lines.append("    }")
    return "\n".join(lines)


def export_usd(graph: SceneGraph, out_path: str | Path) -> Path:
    output = Path(out_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nodes = "\n\n".join(_node_usda(node) for node in graph.nodes)
    proxy = str(bool(graph.metadata.get("proxy"))).lower()
    geometry_path = graph.metadata.get("geometry_path")
    geometry_prim = ""
    if isinstance(geometry_path, str) and geometry_path:
        asset_path = Path(geometry_path).resolve().as_posix()
        geometry_prim = f"""
    def Xform "ReconstructedGeometry" (
        prepend references = @{asset_path}@
    )
    {{
    }}
"""
    content = f"""#usda 1.0
(
    defaultPrim = "SimbioteScene"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "SimbioteScene"
{{
    custom string simbiote:schemaVersion = "{graph.schema_version}"
    custom string simbiote:coordinateSystem = "{graph.coordinate_system}"
    custom bool simbiote:proxy = {proxy}
{geometry_prim}

{nodes}
}}
"""
    output.write_text(content, encoding="utf-8", newline="\n")
    graph_path = output.with_suffix(".scene_graph.json")
    graph_path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
    return output


def validate_map(path: str | Path, *, allow_proxy: bool = False) -> ValidationReport:
    usd_path = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not usd_path.is_file():
        return ValidationReport(False, {"file_exists": False}, [f"Missing {usd_path}"])

    content = usd_path.read_text(encoding="utf-8")
    checks = {
        "file_exists": True,
        "meters": 'metersPerUnit = 1' in content,
        "up_axis": 'upAxis = "Y"' in content,
        "collision": "PhysicsCollisionAPI" in content,
        "navigable_floor": "simbiote:is_navigable = true" in content,
        "graspable_object": "simbiote:is_graspable = true" in content,
        "mass": "simbiote:mass_kg" in content,
        "grasp_type": "simbiote:grasp_type" in content,
        "reconstructed_geometry": (
            'def Xform "ReconstructedGeometry"' in content
            and "prepend references" in content
        ),
    }
    proxy = "simbiote:proxy = true" in content
    checks["non_proxy"] = not proxy
    if proxy:
        message = "USD contains proxy geometry, not reconstructed production geometry"
        if allow_proxy:
            warnings.append(message)
        else:
            errors.append(message)
    for check, passed in checks.items():
        if check not in {"non_proxy", "reconstructed_geometry"} and not passed:
            errors.append(f"Failed contract check: {check}")
    if not proxy and not checks["reconstructed_geometry"]:
        errors.append("Production USD does not compose reconstructed geometry")

    try:
        from pxr import Usd  # type: ignore[import-not-found]

        stage = Usd.Stage.Open(str(usd_path))
        checks["pxr_opens"] = bool(stage)
        if not stage:
            errors.append("USD parser could not open the stage")
    except ImportError:
        warnings.append("usd-core/pxr unavailable; skipped parser-level validation")

    return ValidationReport(not errors, checks, errors, warnings)
