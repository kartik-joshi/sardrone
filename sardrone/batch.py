"""Record many episodes unattended, one after another.

    make batch MATRIX=configs/pilot10.yaml          # real PX4 + Gazebo
    make batch-fake MATRIX=configs/pilot10.yaml     # fake drone, seconds

For every row of the matrix: generate the world, start Gazebo and PX4, fly
the search mission, stop the simulator, and add one line to the MANIFEST
(a CSV). The manifest is what makes a batch trustworthy and restartable:

* It is written after EVERY episode, so a crash loses at most the episode
  in progress. Running the same command again skips what is already done.
* Each line records the matrix row, the seed and the git commit, so any
  episode can be traced back to exactly the code and settings that made it.

Each episode runs in its own process with a time limit. If one hangs or
crashes, it is recorded as failed and the batch moves on.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sardrone.worlds.matrix import VariationRow, load_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_DIR = PROJECT_ROOT / "data" / "episodes"
DEFAULT_WORLD_ROOT = PROJECT_ROOT / "data" / "worlds"

COMPLETED, FAILED, TIMEOUT = "completed", "failed", "timeout"

MANIFEST_FIELDS = [
    "episode_id",
    "attempt",
    "outcome",  # completed | failed | timeout  (did the mission run to the end?)
    "success",  # did it also find everyone? (empty unless completed)
    "people_placed",
    "people_found",
    "detector",
    "world",
    "seed",
    "duration_s",  # flight time inside the episode
    "wall_s",  # real time the episode took, including simulator start
    "size_mb",
    "commit",
    "dirty",  # "yes" if there were uncommitted changes when it ran
    "started_utc",
    "mcap",
    "error",
    "row",  # the full matrix row, as JSON
]


@dataclass
class EpisodeResult:
    outcome: str
    error: str = ""


def git_commit(root: Path = PROJECT_ROOT) -> tuple[str, bool]:
    """Current commit hash and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    return git("rev-parse", "--short", "HEAD") or "unknown", bool(git("status", "--porcelain"))


# ---- manifest ----------------------------------------------------------------


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def append_manifest(path: Path, entry: dict) -> None:
    """Add one line, and make sure it is really on disk before continuing."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({field: entry.get(field, "") for field in MANIFEST_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def attempts_of(manifest: list[dict], episode_id: str) -> list[dict]:
    return [line for line in manifest if line["episode_id"] == episode_id]


def is_done(manifest: list[dict], episode_id: str, episode_dir: Path) -> bool:
    """Done = the manifest says completed AND the files are still there."""
    lines = attempts_of(manifest, episode_id)
    if not lines or lines[-1]["outcome"] != COMPLETED:
        return False
    mcap = episode_dir / f"{episode_id}.mcap"
    return mcap.exists() and mcap.with_suffix(".json").exists()


# ---- running one episode -----------------------------------------------------


def _search_command(row: VariationRow, world_dir: Path, mcap: Path, fake: bool, detector: str = "oracle",
                    second_pass: bool = False) -> list[str]:  # fmt: skip
    command = [
        sys.executable, "-m", "sardrone.mission.search",
        "--world-dir", str(world_dir), "--out", str(mcap), "--overwrite",
    ]  # fmt: skip
    if detector != "oracle":
        command += ["--detector", detector]
    if second_pass:
        command += ["--second-pass"]
    return command + (["--fake"] if fake else [])


def run_episode(
    row: VariationRow,
    *,
    episode_dir: Path = DEFAULT_EPISODE_DIR,
    world_root: Path = DEFAULT_WORLD_ROOT,
    timeout_s: float = 900.0,
    fake: bool = False,
    detector: str = "oracle",
    second_pass: bool = False,
) -> EpisodeResult:
    """Generate the world and fly it. Never raises for an ordinary failure:
    the batch must survive one bad episode."""
    from sardrone.sim.launch import Simulation, SimulationError, gz_python
    from sardrone.worlds.generate import generate

    mcap = episode_dir / f"{row.id}.mcap"
    try:
        paths = generate(row, world_root)
    except Exception as error:  # noqa: BLE001 - reported in the manifest
        return EpisodeResult(FAILED, f"world generation: {error}")

    log_dir = paths.folder / "logs"
    log_dir.mkdir(exist_ok=True)
    # PYTHONUNBUFFERED: the episode's log is written as it happens, so a
    # hanging episode can be watched (the first batch showed an empty log).
    env = dict(os.environ, SARDRONE_GZ_PYTHON=str(gz_python()), PYTHONUNBUFFERED="1")

    def fly() -> EpisodeResult:
        with open(log_dir / "search.log", "w") as log:
            try:
                done = subprocess.run(
                    _search_command(row, paths.folder, mcap, fake, detector, second_pass),
                    cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s,
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                return EpisodeResult(TIMEOUT, f"no result after {timeout_s:g} s")
        if done.returncode != 0:
            tail = (log_dir / "search.log").read_text(errors="replace").strip().splitlines()[-1:]
            return EpisodeResult(FAILED, f"exit code {done.returncode}: {' '.join(tail)}"[:300])
        return EpisodeResult(COMPLETED)

    if fake:
        return fly()
    try:
        with Simulation(paths.folder):
            return fly()
    except SimulationError as error:
        return EpisodeResult(FAILED, f"simulator: {error}")


# ---- the batch ---------------------------------------------------------------


def _entry(row: VariationRow, result: EpisodeResult, episode_dir: Path, attempt: int,
           started: datetime, wall_s: float, commit: str, dirty: bool) -> dict:  # fmt: skip
    mcap = episode_dir / f"{row.id}.mcap"
    entry = {
        "episode_id": row.id,
        "attempt": attempt,
        "outcome": result.outcome,
        "world": row.world,
        "seed": row.seed,
        "wall_s": round(wall_s, 1),
        "commit": commit,
        "dirty": "yes" if dirty else "no",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mcap": mcap.name if mcap.exists() else "",
        "error": result.error,
        "row": json.dumps(asdict(row), sort_keys=True),
    }
    if mcap.exists():
        entry["size_mb"] = round(mcap.stat().st_size / 1e6, 1)
    sidecar_path = mcap.with_suffix(".json")
    if sidecar_path.exists():
        # A failed episode still has a (partial) sidecar; its numbers are kept
        # for debugging, but `success` only counts for completed missions.
        sidecar = json.loads(sidecar_path.read_text())
        entry.update(
            people_placed=sidecar.get("people_placed", ""),
            people_found=sidecar.get("people_found", ""),
            detector=sidecar.get("detector", ""),
            duration_s=round(sidecar.get("duration_s", 0.0), 1),
        )
        if result.outcome == COMPLETED:
            entry["success"] = "yes" if sidecar.get("success") else "no"
    return entry


def run_batch(
    rows: list[VariationRow],
    *,
    manifest_path: Path,
    episode_dir: Path = DEFAULT_EPISODE_DIR,
    runner: Callable[[VariationRow], EpisodeResult],
    max_attempts: int = 2,
    min_free_gb: float = 3.0,
    free_gb: Callable[[], float] | None = None,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict:
    """Run every row that is not done yet. Returns counts per outcome.

    `runner` flies one episode; it is a parameter so that tests can run the
    whole batch logic without a simulator.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    free_gb = free_gb or (lambda: shutil.disk_usage(episode_dir).free / 1e9)
    commit, dirty = git_commit()
    counts = {"skipped": 0, "gave_up": 0, COMPLETED: 0, FAILED: 0, TIMEOUT: 0}

    for number, row in enumerate(rows, start=1):
        label = f"[{number}/{len(rows)}] {row.id}"
        manifest = read_manifest(manifest_path)
        if is_done(manifest, row.id, episode_dir):
            counts["skipped"] += 1
            log(f"{label}: already done, skipping")
            continue

        attempt = len(attempts_of(manifest, row.id))
        result = None
        while attempt < max_attempts:
            if free_gb() < min_free_gb:
                log(f"stopping: less than {min_free_gb:g} GB of disk left")
                counts["stopped"] = "disk"
                return counts
            attempt += 1
            log(f"{label}: attempt {attempt} ({row.world}, {row.people} people, {row.time_of_day})")
            started, clock = datetime.now(timezone.utc), time.monotonic()
            try:
                result = runner(row)
            except Exception as error:  # noqa: BLE001 - a broken runner must not end the batch
                result = EpisodeResult(FAILED, f"runner crashed: {error}")
            entry = _entry(
                row, result, episode_dir, attempt, started, time.monotonic() - clock, commit, dirty
            )
            append_manifest(manifest_path, entry)
            counts[result.outcome] += 1
            found = f", found {entry.get('people_found', '?')}/{entry.get('people_placed', '?')}"
            log(f"{label}: {result.outcome}{found if result.outcome == COMPLETED else ''} "
                f"in {entry['wall_s']} s {result.error}".rstrip())  # fmt: skip
            if result.outcome == COMPLETED:
                break
        if result is None:
            counts["gave_up"] += 1
            log(f"{label}: failed {attempt} times before, not retrying (delete its manifest lines to retry)")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record every episode of a variation matrix.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="default: data/episodes/<matrix name>.manifest.csv")
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--world-root", type=Path, default=DEFAULT_WORLD_ROOT)
    parser.add_argument("--fake", action="store_true", help="fake drone, no simulator")
    parser.add_argument("--limit", type=int, help="only the first N rows")
    parser.add_argument("--only", help="only this episode id")
    parser.add_argument("--prefix", help="only episodes whose id starts with this, e.g. test-")
    parser.add_argument(
        "--detector", default="oracle",
        help="'oracle', or a model version (closed loop). With a model, give --episode-dir too, "
        "so these episodes do not replace the oracle ones.",
    )  # fmt: skip
    parser.add_argument("--second-pass", action="store_true", help="add a lower pass over tree cover")
    parser.add_argument("--ids", help="comma-separated episode ids to run")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per episode")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=3.0, help="stop before free disk drops below this")
    args = parser.parse_args(argv)

    rows = load_matrix(args.matrix)
    if args.only:
        rows = [row for row in rows if row.id == args.only]
    if args.prefix:
        rows = [row for row in rows if row.id.startswith(args.prefix)]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [row for row in rows if row.id in wanted]
    if args.limit:
        rows = rows[: args.limit]
    if args.detector != "oracle" and args.episode_dir == DEFAULT_EPISODE_DIR:
        parser.error("--detector with a model needs its own --episode-dir")
    suffix = ".fake.manifest.csv" if args.fake else ".manifest.csv"
    manifest = args.manifest or args.episode_dir / f"{args.matrix.stem}{suffix}"

    def runner(row: VariationRow) -> EpisodeResult:
        return run_episode(
            row, episode_dir=args.episode_dir, world_root=args.world_root,
            timeout_s=args.timeout, fake=args.fake, detector=args.detector, second_pass=args.second_pass,
        )  # fmt: skip

    started = time.monotonic()
    try:
        counts = run_batch(
            rows, manifest_path=manifest, episode_dir=args.episode_dir, runner=runner,
            max_attempts=args.max_attempts, min_free_gb=args.min_free_gb,
        )  # fmt: skip
    except KeyboardInterrupt:
        print(f"\ninterrupted - run the same command again to continue. Manifest: {manifest}")
        return 130
    print(f"batch finished in {(time.monotonic() - started) / 60:.1f} min: {counts}")
    print(f"manifest: {manifest}")
    not_done = [row.id for row in rows if not is_done(read_manifest(manifest), row.id, args.episode_dir)]
    if not_done:
        print(f"not completed: {', '.join(not_done)}")
    return 1 if not_done else 0


if __name__ == "__main__":
    raise SystemExit(main())
