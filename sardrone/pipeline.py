"""The data loop as one command, with a gate that decides whether to retrain.

    make pipeline                  # show what would run (changes nothing)
    make pipeline RUN=1            # run the stages that are due

    ingest -> curate -> GATE -> export -> train -> evaluate

Retraining costs hours, so it should happen when the DATA says so, not on
a timer. The gate compares the number of frames in `sar-hard-cases` (lying,
occluded, small, dusk) with the number recorded at the last training. Only
when at least 200 new hard frames have arrived does the expensive part run.

Why not Airflow? The plan asks for a small Airflow DAG or a reason for
something lighter. This is the lighter thing, on purpose:

* One machine, one user, six stages that run in a fixed order. Airflow's
  scheduler, web server and metadata database would be more moving parts
  than the pipeline itself.
* Airflow pins many dependencies and would not share an environment with
  FiftyOne, PyTorch and Ultralytics without conflicts.
* What matters is in place without it: each stage is a separate command
  that can be rerun alone, every run records its inputs (data hash, commit),
  and the gate's decision is written down.

docs/WORKFLOW.md shows how these same stages map onto an Airflow DAG, for
when there are several machines or people.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "runs" / "pipeline_state.json"
MIN_NEW_HARD_FRAMES = 200


@dataclass(frozen=True)
class GateDecision:
    retrain: bool
    hard_cases_now: int
    hard_cases_at_last_training: int
    new_hard_cases: int
    reason: str


def decide(hard_cases_now: int, state: dict, threshold: int = MIN_NEW_HARD_FRAMES) -> GateDecision:
    """Should the model be retrained? Pure function of two numbers.

    `state` is the record left by the previous training ({} if there was
    none). The first time, with nothing to compare against, training runs.
    """
    if "hard_cases" not in state:
        return GateDecision(True, hard_cases_now, 0, hard_cases_now, "no previous training recorded")
    before = int(state["hard_cases"])
    new = hard_cases_now - before
    if new >= threshold:
        return GateDecision(True, hard_cases_now, before, new, f"{new} new hard frames (threshold {threshold})")
    return GateDecision(False, hard_cases_now, before, new, f"only {new} new hard frames (threshold {threshold})")


def read_state(path: Path = STATE_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_state(hard_cases: int, version: str, path: Path = STATE_FILE) -> dict:
    state = {
        "hard_cases": hard_cases,
        "version": version,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def stages(version: str, export: str, change: str, max_train: int | None, python: str = sys.executable,
           export_args: list[str] | None = None,
           train_args: list[str] | None = None) -> list[tuple[str, list[str], bool]]:  # fmt: skip
    """(name, command, runs only if the gate says retrain). `export_args` are
    passed to the export unchanged (e.g. --train, --realism, --add-real)."""
    export_command = [python, "-m", "sardrone.train.export", "--name", export]
    if max_train:
        export_command += ["--max-train", str(max_train)]
    export_command += list(export_args or [])
    return [
        ("ingest", [python, "-m", "sardrone.fo.frames"], False),
        ("curate", [python, "-m", "sardrone.fo.curate"], False),
        ("export", export_command, True),
        ("train", [python, "-m", "sardrone.train.train", "--export", export, "--version", version,
                   *(train_args or [])], True),
        ("evaluate", [python, "-m", "sardrone.train.evaluate", "--version", version, "--change", change], True),
    ]


def count_hard_cases() -> int:
    from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

    import fiftyone as fo

    from sardrone.fo.curate import VIEW_HARD
    from sardrone.fo.frames import DATASET_NAME

    return len(fo.load_dataset(DATASET_NAME).load_saved_view(VIEW_HARD))


def run_pipeline(version: str, export: str, change: str, *, execute: bool, max_train: int | None = None,
                 state_path: Path = STATE_FILE, run=subprocess.run, count=count_hard_cases, log=print,
                 export_args: list[str] | None = None, train_args: list[str] | None = None) -> GateDecision:  # fmt: skip
    """Run the data stages, ask the gate, then (maybe) the training stages.

    With `execute=False` nothing is run: the plan and the gate's decision
    from the data as it is now are printed.
    """
    plan = stages(version, export, change, max_train, export_args=export_args, train_args=train_args)
    for name, command, gated in plan:
        if gated:
            continue
        log(f"[{name}] {' '.join(command[1:])}" + ("" if execute else "   (dry run)"))
        if execute:
            run(command, cwd=PROJECT_ROOT, check=True)

    decision = decide(count(), read_state(state_path))
    log(f"[gate] retrain={decision.retrain}: {decision.reason}; "
        f"hard cases now {decision.hard_cases_now}, at last training {decision.hard_cases_at_last_training}")  # fmt: skip

    for name, command, gated in plan:
        if not gated:
            continue
        if not decision.retrain:
            log(f"[{name}] skipped by the gate")
            continue
        log(f"[{name}] {' '.join(command[1:])}" + ("" if execute else "   (dry run)"))
        if execute:
            run(command, cwd=PROJECT_ROOT, check=True)
    if execute and decision.retrain:
        write_state(decision.hard_cases_now, version, state_path)
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the data loop with a retraining gate.")
    parser.add_argument("--version", required=True, help="model version to produce if the gate opens")
    parser.add_argument("--export", help="export name (default: same as --version)")
    parser.add_argument("--change", default="retrained by the pipeline")
    parser.add_argument("--max-train", type=int, help="sample this many training frames")
    parser.add_argument("--run", action="store_true", help="really run (default: dry run)")
    parser.add_argument("--export-arg", action="append", default=[], dest="export_args",
                        help="passed to the export unchanged; repeat, e.g. --export-arg=--realism")  # fmt: skip
    parser.add_argument("--train-arg", action="append", default=[], dest="train_args",
                        help="passed to training unchanged; e.g. --train-arg=--epochs --train-arg=10 (the "
                             "scorecard's runs use 10; the code default is 50)")  # fmt: skip
    parser.add_argument("--record-current", action="store_true",
                        help="record today's hard-case count as the state of an existing model, without training")  # fmt: skip
    args = parser.parse_args(argv)
    if args.record_current:
        print(json.dumps(write_state(count_hard_cases(), args.version), indent=2))
        return 0
    run_pipeline(args.version, args.export or args.version, args.change, execute=args.run, max_train=args.max_train,
                 export_args=args.export_args, train_args=args.train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
