"""Load recorded episodes into FiftyOne and open the App.

    make view                      # build/refresh the dataset, launch the App
    python -m sardrone.fo.load --no-app

One sample per .mcap file. FiftyOne gives any sample whose filepath ends in
.mcap the "multimodal" media type (fiftyone/core/media.py, checked in the
installed 1.22.1), and the App then replays every stream in the file on one
shared timeline. The JSON sidecar becomes ordinary sample fields, so episodes
can be filtered in the sidebar (weather == "fog", success == False, ...).
"""

from pathlib import Path

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# sardrone.fo.env (imported above) switches on the multimodal viewer and keeps
# the database inside this project. Launching the App still creates two tiny
# files, ~/.fiftyone/var/uid and welcome.json; FiftyOne hard-codes that place.

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import fiftyone as fo  # noqa: E402

DATASET_NAME = "sar-episodes"
DEFAULT_EPISODE_DIR = PROJECT_ROOT / "data" / "episodes"

# Sidecar keys copied onto each sample unchanged.
SIDECAR_FIELDS = (
    "episode_id",
    "world",
    "weather",
    "time_of_day",
    "mission_prompt",
    "people_placed",
    "people_found",
    "success",
    "duration_s",
    # Generated worlds and search missions (Phase 2A onwards).
    "world_id",
    "seed",
    "area_m",
    "altitude_m",
    "occlusion",
    "camera_pitch_deg",
    "sun_elevation_deg",
    "detector",
)


def sidecar_to_fields(sidecar: dict) -> dict:
    """Turn a sidecar dict into flat FiftyOne field values."""
    fields = {key: sidecar[key] for key in SIDECAR_FIELDS if key in sidecar}
    if "duration_s" in fields:
        fields["duration_s"] = float(fields["duration_s"])

    start_ns = sidecar.get("start_time_ns")
    if start_ns:
        # FiftyOne stores datetimes with millisecond precision; the exact
        # nanosecond value stays in the .mcap and the sidecar.
        fields["start_time"] = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)

    counts = sidecar.get("message_counts", {})
    fields["num_images"] = int(counts.get("/camera/front", 0))
    fields["num_messages"] = int(sum(counts.values()))
    fields["num_events"] = int(sum(sidecar.get("event_counts", {}).values()))
    for key in ("area_m", "altitude_m", "camera_pitch_deg", "sun_elevation_deg"):
        if key in fields:
            fields[key] = float(fields[key])
    if "reports" in sidecar:
        fields["num_reports"] = len(sidecar["reports"])
        errors = [r["error_m"] for r in sidecar["reports"] if "error_m" in r]
        if errors:
            fields["mean_report_error_m"] = float(sum(errors) / len(errors))
    if "search" in sidecar and sidecar["search"]:
        fields["lanes"] = int(sidecar["search"]["lanes"])
    return fields


def build_dataset(episode_dir: Path = DEFAULT_EPISODE_DIR, name: str = DATASET_NAME) -> fo.Dataset:
    """Create or refresh the persistent dataset: one sample per .mcap in `episode_dir`.

    Safe to run repeatedly: existing samples are updated in place, new files
    are added, and samples whose file has been deleted are removed.
    """
    episode_dir = Path(episode_dir).resolve()
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    else:
        dataset = fo.Dataset(name, persistent=True)

    existing = {sample.filepath: sample for sample in dataset}
    on_disk = sorted(episode_dir.glob("*.mcap"))

    for mcap_path in on_disk:
        filepath = str(mcap_path)
        sample = existing.pop(filepath, None) or fo.Sample(filepath=filepath)

        sidecar_path = mcap_path.with_suffix(".json")
        if sidecar_path.exists():
            for key, value in sidecar_to_fields(json.loads(sidecar_path.read_text())).items():
                sample[key] = value
            sample.tags = [tag for tag in sample.tags if tag != "no_sidecar"]
        elif "no_sidecar" not in sample.tags:
            sample.tags.append("no_sidecar")

        if sample.in_dataset:
            sample.save()
        else:
            dataset.add_sample(sample)

    # Whatever is left in `existing` belongs to this folder but has no file any more.
    stale = [s.id for path, s in existing.items() if Path(path).parent == episode_dir]
    if stale:
        dataset.delete_samples(stale)

    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load .mcap episodes into FiftyOne.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_EPISODE_DIR, help="episode folder")
    parser.add_argument("--name", default=DATASET_NAME, help="dataset name")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--no-app", action="store_true", help="build the dataset and exit")
    args = parser.parse_args(argv)

    dataset = build_dataset(args.dir, args.name)
    print(f"dataset {dataset.name!r}: {len(dataset)} episode(s), media type {dataset.media_type}")
    for sample in dataset.sort_by("filepath"):
        try:
            summary = f"{sample['duration_s']:6.1f} s  success={sample['success']}"
        except (KeyError, AttributeError, TypeError):
            summary = "(no sidecar)"
        print(f"  {Path(sample.filepath).name:24s} {summary}")

    if len(dataset) == 0:
        print(f"no .mcap files in {args.dir} - run `make synthetic` or `make record` first")
        return 1
    if args.no_app:
        return 0

    session = fo.launch_app(dataset, port=args.port)
    print(f"App running at http://localhost:{args.port} - press Ctrl-C to stop")
    try:
        session.wait(-1)  # block until Ctrl-C, even if no browser tab is connected
    except KeyboardInterrupt:
        print("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
