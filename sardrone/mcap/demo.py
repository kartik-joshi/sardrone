"""Make a short animated GIF from an episode: camera, tracks, mission state.

    python -m sardrone.mcap.demo data/closed_loop/v4/test-001.mcap docs/images/demo.gif --start 18 --end 58

Every frame shows what the drone's camera saw, the tracker's boxes in
yellow (as stored on /detections), the simulator's label boxes in magenta
when asked for, and the mission state from /events. Nothing is re-computed:
the GIF is a view of what is in the file.
"""

import argparse
import io
from pathlib import Path

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image, ImageDraw

from sardrone.mcap.episode_writer import (
    TOPIC_DETECTIONS,
    TOPIC_EVENTS,
    TOPIC_GROUND_TRUTH_BOXES,
    TOPIC_IMAGE,
)
from sardrone.mcap.timeutil import NS_PER_SEC

YELLOW, MAGENTA = (255, 230, 0), (255, 0, 255)


def _rectangles(annotations) -> list[tuple[str, float, float, float, float]]:
    out = []
    for outline, text in zip(annotations.points, annotations.texts):
        xs, ys = [p.x for p in outline.points], [p.y for p in outline.points]
        out.append((text.text, min(xs), min(ys), max(xs), max(ys)))
    return out


def make_gif(mcap_path: Path, out_path: Path, start_s: float, end_s: float, every: int = 4,
             width: int = 480, show_truth: bool = False, frame_ms: int = 120) -> int:  # fmt: skip
    """Write the GIF; returns the number of frames in it."""
    topics = [TOPIC_IMAGE, TOPIC_DETECTIONS, TOPIC_GROUND_TRUTH_BOXES, TOPIC_EVENTS]
    frames, start_ns, state, note, note_until = [], None, "", "", 0
    pending: dict = {}
    count = 0
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, raw, message in reader.iter_decoded_messages(topics=topics, log_time_order=True):
            if start_ns is None:
                start_ns = raw.log_time
            t = (raw.log_time - start_ns) / NS_PER_SEC
            if channel.topic == TOPIC_EVENTS:
                text = message.message
                if text.startswith("state: "):
                    state = text.split("-> ", 1)[1].split(" ", 1)[0]
                elif text.startswith(("person_reported", "false_alarm", "person_confirmed")):
                    note, note_until = text.split(",")[0].replace(":", "").replace("_", " "), t + 3.0
                continue
            if t < start_s or t > end_s:
                continue
            entry = pending.setdefault(raw.log_time, {})
            entry[channel.topic] = message
            if not {TOPIC_IMAGE, TOPIC_DETECTIONS} <= set(entry):
                continue
            del pending[raw.log_time]
            count += 1
            if count % every:
                continue
            image = Image.open(io.BytesIO(entry[TOPIC_IMAGE].data)).convert("RGB")
            scale = width / image.width
            image = image.resize((width, round(image.height * scale)), Image.BILINEAR)
            draw = ImageDraw.Draw(image)
            if show_truth and TOPIC_GROUND_TRUTH_BOXES in entry:
                for _, x0, y0, x1, y1 in _rectangles(entry[TOPIC_GROUND_TRUTH_BOXES]):
                    draw.rectangle([x0 * scale - 4, y0 * scale - 4, x1 * scale + 4, y1 * scale + 4], outline=MAGENTA)
            for label, x0, y0, x1, y1 in _rectangles(entry[TOPIC_DETECTIONS]):
                draw.rectangle([x0 * scale - 2, y0 * scale - 2, x1 * scale + 2, y1 * scale + 2], outline=YELLOW, width=2)
                draw.text((x0 * scale, max(18, y0 * scale - 13)), label.replace("track ", "#"), fill=YELLOW)
            draw.rectangle([0, 0, width, 16], fill=(0, 0, 0))
            caption = f"t={t:5.1f}s  {state}" + (f"  |  {note}" if t <= note_until else "")
            draw.text((4, 3), caption, fill=(255, 255, 255))
            frames.append(image.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    if not frames:
        raise ValueError("no frames in that time range")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)
    return len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Animated GIF of part of an episode.")
    parser.add_argument("mcap", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds from the episode start")
    parser.add_argument("--end", type=float, default=30.0)
    parser.add_argument("--every", type=int, default=4, help="keep every Nth camera frame")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--truth", action="store_true", help="also draw the simulator's label boxes")
    args = parser.parse_args(argv)
    count = make_gif(args.mcap, args.out, args.start, args.end, args.every, args.width, args.truth)
    print(f"wrote {args.out}: {count} frames, {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
