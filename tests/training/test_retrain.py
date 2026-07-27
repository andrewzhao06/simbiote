import pytest

from simbiote.robot_iface.trajectory import make_toy_trajectory
from simbiote.training import retrain


@pytest.fixture(autouse=True)
def _isolate_demo_store(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "DEMO_STORE_DIR", tmp_path / "demos")
    yield


def test_ingest_demo_writes_a_file():
    traj = make_toy_trajectory("run-1", obs_dim=5, length=8, task="nav")
    path = retrain.ingest_demo(traj)
    assert path.exists()
    assert path.name == "run-1.json"


def test_list_and_load_demos_for_task():
    for i in range(3):
        traj = make_toy_trajectory(f"nav-{i}", obs_dim=5, length=4, task="nav", seed=i)
        retrain.ingest_demo(traj)
    grasp_traj = make_toy_trajectory("grasp-0", obs_dim=5, length=4, task="grasp")
    retrain.ingest_demo(grasp_traj)

    nav_demos = retrain.list_demos("nav")
    assert len(nav_demos) == 3
    grasp_demos = retrain.list_demos("grasp")
    assert len(grasp_demos) == 1

    loaded = retrain.load_all_demos("nav")
    assert {t.session_id for t in loaded} == {"nav-0", "nav-1", "nav-2"}


def test_finetune_policy_without_demos_raises():
    with pytest.raises(RuntimeError):
        retrain.finetune_policy("nav")
