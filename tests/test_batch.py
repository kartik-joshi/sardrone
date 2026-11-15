"""Phase 2C: the batch runner's bookkeeping, tested with a stand-in episode
runner (no simulator, no worlds)."""

import csv
import json

import pytest

from sardrone import batch
from sardrone.batch import COMPLETED, FAILED, TIMEOUT, EpisodeResult, run_batch
from sardrone.worlds.matrix import load_matrix, sample_rows


def rows(count=4):
    return sample_rows(
        {"prefix": "t", "count": count, "seed": 3, "world": ["meadow", "park"], "people": [1, 2]}
    )


class Runner:
    """Pretends to fly: writes the episode files and returns a scripted outcome."""

    def __init__(self, episode_dir, script=None):
        self.episode_dir = episode_dir
        self.script = script or {}  # episode id -> list of outcomes, one per attempt
        self.calls = []

    def __call__(self, row):
        self.calls.append(row.id)
        outcomes = self.script.get(row.id, [COMPLETED])
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if outcome == "crash":
            raise RuntimeError("boom")
        if outcome in (COMPLETED, FAILED):
            (self.episode_dir / f"{row.id}.mcap").write_bytes(b"x" * 2_500_000)
            (self.episode_dir / f"{row.id}.json").write_text(
                json.dumps(
                    {
                        "people_placed": row.people,
                        "people_found": row.people if outcome == COMPLETED else 0,
                        "success": outcome == COMPLETED,
                        "detector": "oracle",
                        "duration_s": 123.456,
                    }
                )
            )
        return EpisodeResult(outcome, "" if outcome == COMPLETED else "it broke")


def run(tmp_path, runner, the_rows=None, **kwargs):
    messages = []
    counts = run_batch(
        the_rows or rows(), manifest_path=tmp_path / "m.csv", episode_dir=tmp_path,
        runner=runner, log=messages.append, **kwargs,
    )  # fmt: skip
    return counts, messages, batch.read_manifest(tmp_path / "m.csv")


# ---- matrix sampling ---------------------------------------------------------


def test_sample_rows_are_seeded_and_balanced():
    block = {
        "prefix": "p", "count": 10, "seed": 7, "world": ["meadow", "forest_edge", "park"],
        "people": [1, 2, 3, 4, 5], "altitude_m": [15, 25], "area_m": 80,
    }  # fmt: skip
    first, second = sample_rows(block), sample_rows(block)
    assert first == second
    assert [r.id for r in first][:2] == ["p-001", "p-002"] and first[-1].id == "p-010"
    assert [r.seed for r in first] == list(range(7001, 7011))  # each world gets its own seed
    assert all(r.area_m == 80 for r in first)  # a non-list value is the same for every row
    # Dealt like cards: 10 rows over 3 worlds is 3 or 4 each, never 7 / 2 / 1.
    for world in ("meadow", "forest_edge", "park"):
        assert sum(r.world == world for r in first) in (3, 4)
    assert sorted(r.people for r in first) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert sample_rows({**block, "seed": 8}) != first


def test_pilot10_matrix(tmp_path):
    from pathlib import Path

    the_rows = load_matrix(Path(__file__).resolve().parents[1] / "configs" / "pilot10.yaml")
    assert len(the_rows) == 10 and len({r.id for r in the_rows}) == 10
    assert {r.world for r in the_rows} == {"meadow", "forest_edge", "park"}
    assert {r.time_of_day for r in the_rows} == {"noon", "golden_hour", "dusk"}
    empty = tmp_path / "e.yaml"
    empty.write_text("defaults: {area_m: 80}\n")
    with pytest.raises(ValueError, match="no episodes"):
        load_matrix(empty)


# ---- batch -------------------------------------------------------------------


def test_batch_runs_every_row_and_writes_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "git_commit", lambda: ("abc1234", False))
    runner = Runner(tmp_path)
    counts, _, manifest = run(tmp_path, runner)
    assert runner.calls == ["t-001", "t-002", "t-003", "t-004"]
    assert counts[COMPLETED] == 4 and counts[FAILED] == 0 and counts["skipped"] == 0

    assert [line["episode_id"] for line in manifest] == runner.calls
    line = manifest[0]
    assert line["outcome"] == COMPLETED and line["success"] == "yes" and line["attempt"] == "1"
    assert line["commit"] == "abc1234" and line["dirty"] == "no"
    assert line["mcap"] == "t-001.mcap" and float(line["size_mb"]) == 2.5
    assert float(line["duration_s"]) == 123.5 and line["detector"] == "oracle"
    assert line["people_found"] == line["people_placed"]
    # The matrix row can be rebuilt from the manifest alone.
    stored = json.loads(line["row"])
    assert stored["id"] == "t-001" and stored["seed"] == int(line["seed"]) == 3001
    assert stored["world"] == line["world"]
    with open(tmp_path / "m.csv", newline="") as handle:
        assert next(csv.reader(handle)) == batch.MANIFEST_FIELDS


def test_running_again_skips_finished_episodes(tmp_path):
    run(tmp_path, Runner(tmp_path))
    again = Runner(tmp_path)
    counts, messages, manifest = run(tmp_path, again)
    assert again.calls == [] and counts["skipped"] == 4
    assert len(manifest) == 4, "skipping must not add manifest lines"
    assert "already done" in messages[0]


def test_resume_after_a_crash_part_way(tmp_path):
    """The batch was killed after two episodes: only the rest is flown."""
    run(tmp_path, Runner(tmp_path), the_rows=rows()[:2])
    resumed = Runner(tmp_path)
    counts, _, manifest = run(tmp_path, resumed)
    assert resumed.calls == ["t-003", "t-004"]
    assert counts["skipped"] == 2 and counts[COMPLETED] == 2 and len(manifest) == 4


def test_a_deleted_episode_file_is_recorded_again(tmp_path):
    run(tmp_path, Runner(tmp_path))
    (tmp_path / "t-002.mcap").unlink()
    again = Runner(tmp_path)
    counts, _, _ = run(tmp_path, again, max_attempts=5)
    assert again.calls == ["t-002"]


def test_failed_episode_is_retried_once_then_the_batch_moves_on(tmp_path):
    runner = Runner(tmp_path, {"t-002": [FAILED, COMPLETED], "t-003": [TIMEOUT]})
    counts, _, manifest = run(tmp_path, runner)
    assert runner.calls == ["t-001", "t-002", "t-002", "t-003", "t-003", "t-004"]
    assert counts == {"skipped": 0, "gave_up": 0, COMPLETED: 3, FAILED: 1, TIMEOUT: 2}
    second = [line for line in manifest if line["episode_id"] == "t-002"]
    assert [(l["attempt"], l["outcome"]) for l in second] == [("1", FAILED), ("2", COMPLETED)]
    assert second[0]["error"] == "it broke" and second[0]["success"] == ""
    assert second[0]["people_found"] == "0", "a failed episode keeps its partial numbers"
    third = [line for line in manifest if line["episode_id"] == "t-003"]
    assert [l["outcome"] for l in third] == [TIMEOUT, TIMEOUT] and third[0]["mcap"] == ""


def test_an_episode_that_keeps_failing_is_not_retried_forever(tmp_path):
    run(tmp_path, Runner(tmp_path, {"t-001": [FAILED]}))
    later = Runner(tmp_path)
    counts, messages, manifest = run(tmp_path, later)
    assert later.calls == [] and counts["gave_up"] == 1 and counts["skipped"] == 3
    assert "not retrying" in messages[0]
    assert len([l for l in manifest if l["episode_id"] == "t-001"]) == 2


def test_a_crashing_runner_does_not_end_the_batch(tmp_path):
    runner = Runner(tmp_path, {"t-001": ["crash"]})
    counts, _, manifest = run(tmp_path, runner, max_attempts=1)
    assert runner.calls == ["t-001", "t-002", "t-003", "t-004"]
    assert manifest[0]["outcome"] == FAILED and "runner crashed: boom" in manifest[0]["error"]
    assert counts[COMPLETED] == 3


def test_batch_stops_before_filling_the_disk(tmp_path):
    space = iter([10.0, 10.0, 2.0])
    runner = Runner(tmp_path)
    counts, messages, manifest = run(tmp_path, runner, free_gb=lambda: next(space), min_free_gb=3.0)
    assert runner.calls == ["t-001", "t-002"] and counts["stopped"] == "disk"
    assert len(manifest) == 2 and "disk" in messages[-1]


def test_uncommitted_changes_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "git_commit", lambda: ("abc1234", True))
    _, _, manifest = run(tmp_path, Runner(tmp_path), the_rows=rows(1))
    assert manifest[0]["dirty"] == "yes"


def test_git_commit_reads_this_repository():
    commit, dirty = batch.git_commit()
    assert commit != "unknown" and len(commit) >= 7 and isinstance(dirty, bool)


def test_v1_matrix_keeps_train_and_test_worlds_apart():
    from pathlib import Path

    the_rows = load_matrix(Path(__file__).resolve().parents[1] / "configs" / "v1.yaml")
    train = [r for r in the_rows if r.id.startswith("train-")]
    test = [r for r in the_rows if r.id.startswith("test-")]
    assert len(train) == 60 and len(test) == 15 and len(the_rows) == 75
    # No test world can equal a training world: the seeds do not overlap.
    assert not {r.seed for r in train} & {r.seed for r in test}
    assert len({r.seed for r in the_rows}) == 75
    for group in (train, test):
        assert {r.world for r in group} == {"meadow", "forest_edge", "park"}
        assert {r.time_of_day for r in group} == {"noon", "golden_hour", "dusk"}
        assert {r.altitude_m for r in group} == {15.0, 25.0, 40.0}
        assert {r.people for r in group} == {1, 2, 3, 4, 5}
    # Balanced: 60 rows over 3 worlds is exactly 20 each.
    assert [sum(r.world == w for r in train) for w in ("meadow", "forest_edge", "park")] == [20, 20, 20]


def test_two_blocks_with_the_same_seed_are_refused(tmp_path):
    matrix = tmp_path / "m.yaml"
    matrix.write_text(
        "samples:\n"
        "  - {prefix: a, count: 2, seed: 5, world: [meadow]}\n"
        "  - {prefix: b, count: 2, seed: 5, world: [park]}\n"
    )
    with pytest.raises(ValueError, match="share a world seed"):
        load_matrix(matrix)
