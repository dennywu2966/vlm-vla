"""Smoke tests for VLA project structure — no GPU required."""
import pytest


def test_configs_importable():
    from vlm_vla.configs import SmolVLATrainConfig, OpenVLAInferConfig, EvalConfig

    cfg = SmolVLATrainConfig()
    assert cfg.batch_size == 8
    assert cfg.lora_rank == 32
    assert cfg.env_task == "libero_object"


def test_eval_engine_importable():
    from vlm_vla.eval_engine import EvalReport, TaskResult, compare_reports

    tr = TaskResult("test_task", 0.8, 20, 150.0, {})
    report = EvalReport("test_model", "test_suite", [tr])
    assert report.avg_success_rate == 0.8
    assert "test_task" in report.to_table()


def test_eval_report_save_load(tmp_path):
    from vlm_vla.eval_engine import EvalReport, TaskResult

    tr = TaskResult("pick_block", 0.75, 20, 120.0, {"timeout": 5})
    report = EvalReport("SmolVLA", "libero_object", [tr])

    path = str(tmp_path / "report.json")
    report.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data["avg_success_rate"] == 0.75
    assert data["results"][0]["task_name"] == "pick_block"


def test_compare_reports():
    from vlm_vla.eval_engine import EvalReport, TaskResult, compare_reports

    r1 = EvalReport("ModelA", "suite", [TaskResult("t1", 0.8, 20, 100, {})])
    r2 = EvalReport("ModelB", "suite", [TaskResult("t1", 0.6, 20, 150, {})])
    table = compare_reports(r1, r2)
    assert "ModelA" in table
    assert "ModelB" in table


def test_env_adapter():
    from vlm_vla.env_adapter import ActionSpaceConfig, LIBERO_ACTION_SPACE

    assert LIBERO_ACTION_SPACE.dims == 7
    assert LIBERO_ACTION_SPACE.groups["arm"] == 6
    assert LIBERO_ACTION_SPACE.groups["gripper"] == 1


def test_data_utils_importable():
    pytest.importorskip("lerobot", reason="lerobot not installed; skipping data_utils import test")
    from vlm_vla.data_utils import inspect_dataset, get_task_descriptions
    # Just verify importable — actual dataset tests need network
