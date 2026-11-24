"""Phase 3E: the retraining gate and the order of the pipeline's stages."""

import json

from sardrone import pipeline
from sardrone.pipeline import decide, run_pipeline, stages


def test_gate_opens_only_for_enough_new_hard_frames():
    state = {"hard_cases": 1000}
    assert decide(1199, state).retrain is False  # 199 new: one short
    assert decide(1200, state).retrain is True  # exactly 200 new
    assert decide(1500, state).new_hard_cases == 500
    assert decide(900, state).retrain is False  # fewer than before (frames were removed)
    assert "199 new hard frames" in decide(1199, state).reason


def test_first_run_trains_because_there_is_nothing_to_compare_with():
    decision = decide(50, {})
    assert decision.retrain is True and decision.hard_cases_at_last_training == 0
    assert decision.reason == "no previous training recorded"


def test_stage_order_and_which_stages_are_gated():
    plan = stages("v9", "v9-export", "why", max_train=6000, python="python")
    assert [name for name, _, _ in plan] == ["ingest", "curate", "export", "train", "evaluate"]
    assert [gated for _, _, gated in plan] == [False, False, True, True, True]
    assert plan[2][1][-2:] == ["--max-train", "6000"]
    assert plan[3][1][-4:] == ["--export", "v9-export", "--version", "v9"]


def run(tmp_path, hard_cases, execute, state=None):
    state_path = tmp_path / "state.json"
    if state is not None:
        state_path.write_text(json.dumps(state))
    ran, messages = [], []
    decision = run_pipeline(
        "v9", "v9", "why", execute=execute, state_path=state_path,
        run=lambda command, **kwargs: ran.append(command[2].rsplit(".", 1)[1]),
        count=lambda: hard_cases, log=messages.append,
    )  # fmt: skip
    return decision, ran, messages, state_path


def test_dry_run_runs_nothing_and_writes_nothing(tmp_path):
    decision, ran, messages, state_path = run(tmp_path, 5000, execute=False, state={"hard_cases": 100})
    assert ran == [] and decision.retrain is True
    assert json.loads(state_path.read_text()) == {"hard_cases": 100}
    assert any("(dry run)" in m for m in messages)


def test_gate_closed_runs_the_data_stages_only(tmp_path):
    decision, ran, messages, state_path = run(tmp_path, 1100, execute=True, state={"hard_cases": 1000})
    assert decision.retrain is False
    assert ran == ["frames", "curate"]
    assert sum("skipped by the gate" in m for m in messages) == 3
    assert json.loads(state_path.read_text())["hard_cases"] == 1000, "no training, so the baseline stays"


def test_gate_open_runs_everything_and_records_the_new_baseline(tmp_path):
    decision, ran, _, state_path = run(tmp_path, 1300, execute=True, state={"hard_cases": 1000})
    assert decision.retrain is True
    assert ran == ["frames", "curate", "export", "train", "evaluate"]
    state = json.loads(state_path.read_text())
    assert state["hard_cases"] == 1300 and state["version"] == "v9"
    # Running again straight away: nothing new, so no second training.
    decision, ran, _, _ = run(tmp_path, 1300, execute=True)
    assert decision.retrain is False and ran == ["frames", "curate"]


def test_state_helpers(tmp_path):
    path = tmp_path / "s.json"
    assert pipeline.read_state(path) == {}
    written = pipeline.write_state(42, "v1", path)
    assert pipeline.read_state(path) == written and written["hard_cases"] == 42


def test_export_options_pass_through_to_the_export_stage():
    plan = pipeline.stages("v13", "v13-6k", "rich content", 6000, python="py",
                           export_args=["--train", "train-v13-rich", "--realism", "--add-real", "300"])  # fmt: skip
    export = dict((name, command) for name, command, _ in plan)["export"]
    assert export == ["py", "-m", "sardrone.train.export", "--name", "v13-6k", "--max-train", "6000",
                      "--train", "train-v13-rich", "--realism", "--add-real", "300"]  # fmt: skip


def test_training_options_pass_through_to_the_train_stage():
    plan = pipeline.stages("v13", "v13-6k", "x", None, python="py", train_args=["--epochs", "10"])
    train = dict((name, command) for name, command, _ in plan)["train"]
    assert train[-2:] == ["--epochs", "10"]
