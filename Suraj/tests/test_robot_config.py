import pytest

from simbiote.robot.robot_config import (
    RIDGEBACK_FRANKA_CONFIG,
    STAND_IN_CONFIG,
    get_config,
)


def test_stand_in_config_resolves_urdf_path():
    path = STAND_IN_CONFIG.resolve_asset_path()
    assert path.endswith("stand_in_robot.urdf")


def test_stand_in_urdf_file_exists():
    from pathlib import Path

    assert Path(STAND_IN_CONFIG.urdf_path).exists()


def test_wheelchair_urdf_file_exists():
    from pathlib import Path

    from simbiote.robot.robot_config import ASSETS_DIR

    assert (ASSETS_DIR / "wheelchair" / "wheelchair.urdf").exists()


def test_ridgeback_franka_config_resolves_usd_path():
    path = RIDGEBACK_FRANKA_CONFIG.resolve_asset_path()
    assert path.endswith("ridgeback_franka.usd")


def test_configs_share_same_shape():
    """Spec §5.4: "same config shape whether it's pointing at a PyBullet URDF
    today or ridgeback_franka.usd tomorrow." """
    assert type(STAND_IN_CONFIG) is type(RIDGEBACK_FRANKA_CONFIG)
    assert set(vars(STAND_IN_CONFIG)) == set(vars(RIDGEBACK_FRANKA_CONFIG))
    assert len(STAND_IN_CONFIG.arm_joint_names) > 0
    assert len(RIDGEBACK_FRANKA_CONFIG.arm_joint_names) > 0


def test_get_config_registry():
    assert get_config("stand_in") is STAND_IN_CONFIG
    assert get_config("ridgeback_franka") is RIDGEBACK_FRANKA_CONFIG
    with pytest.raises(ValueError):
        get_config("nonexistent_robot")


def test_engine_asset_mismatch_raises():
    from dataclasses import replace

    broken = replace(STAND_IN_CONFIG, urdf_path=None)
    with pytest.raises(ValueError):
        broken.resolve_asset_path()
