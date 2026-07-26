"""Verification 2 — the scene graph loads and resolves phrases to real ids."""

from __future__ import annotations

import json

import pytest

from factoryflow.agentic.scene_query import load_scene


def test_fixture_loads_with_expected_inventory(scene):
    assert scene.scene_id == "hospital_01"
    assert {loc.id for loc in scene.list_locations()} == {
        "corridor", "supply_room", "room_1", "room_2", "nurse_station",
    }
    assert {obj.id for obj in scene.list_objects()} == {
        "tray_01", "cart_01", "wheelchair_01", "iv_stand_01",
    }


def test_grasp_tags_survive_parsing(scene):
    """Step 2's grasp task reads these tags rather than hardcoding assumptions
    (master doc 5.5), so they must come through untouched."""
    tray = scene.get_object("tray_01")
    assert tray.is_graspable is True
    assert tray.grasp_type == "top"
    assert tray.mass_kg == pytest.approx(0.8)
    assert scene.get_object("iv_stand_01").is_graspable is False


def test_wheelchair_carries_a_handle_pose(scene):
    wheelchair = scene.get_object("wheelchair_01")
    assert wheelchair.handle_pose is not None
    assert wheelchair.handle_pose.z > wheelchair.pose.z


def test_location_of(scene):
    assert scene.location_of("tray_01").id == "supply_room"
    assert scene.location_of("wheelchair_01").id == "room_1"


def test_resolve_object_in_a_named_room(scene):
    assert scene.resolve("the tray in the supply room", kind="object") == "tray_01"


def test_resolve_location(scene):
    assert scene.resolve("Room 2", kind="location") == "room_2"
    assert scene.resolve("the supply room", kind="location") == "supply_room"


def test_resolve_uses_aliases(scene):
    assert scene.resolve("the trolley", kind="object") == "cart_01"
    assert scene.resolve("the hallway", kind="location") == "corridor"


def test_resolve_returns_none_rather_than_guessing(scene):
    """A wrong id sends the robot to the wrong room; None is a clean failure."""
    assert scene.resolve("the flux capacitor", kind="object") is None
    assert scene.resolve("the operating theatre", kind="location") is None


def test_inventory_text_lists_real_ids(scene):
    text = scene.inventory_text()
    assert "supply_room" in text and "tray_01" in text
    assert "NOT graspable" in text  # the IV stand must be marked


def test_unknown_node_kind_is_rejected(tmp_path):
    """A schema drift from Teammate 1 should fail loudly at load, not silently
    drop nodes that the parser later cannot find."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"scene_id": "x", "nodes": [{"id": "n1", "kind": "widget"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown kind"):
        load_scene(bad)
