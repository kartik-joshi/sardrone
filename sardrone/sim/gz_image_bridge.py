"""Forward Gazebo camera images (and label boxes) to stdout. Run by GazeboCamera.

This file is a separate program on purpose. The Gazebo Python bindings
(`gz.transport13`, `gz.msgs10`) only exist inside the Python that ships with
Gazebo (PX4's pixi environment, Python 3.13). They cannot be installed into
this project's venv. So GazeboCamera starts this script with Gazebo's Python
and reads its stdout. It therefore uses ONLY the standard library plus `gz`.

Output protocol (binary, on stdout):
    line 1:   one JSON object + "\n" with the lens: width, height, fx, fy, cx, cy
    then, per frame:
        header  struct "<4sIIIIqI" = magic b"SARF", width, height,
                pixel_format, data_length, sim_time_ns, box_count
        data    `data_length` raw bytes (rows of pixels, top to bottom)
        boxes   box_count x struct "<Iffff" = label, min_x, min_y, max_x, max_y
                (box_count is 0xFFFFFFFF when no boxes topic was given)

With `--boxes-topic`, an image is only forwarded together with the boxes
Gazebo rendered AT THE SAME simulation time. The two arrive as separate
messages, so the newest image for which the matching boxes have arrived is
sent. A picture is never paired with labels from another instant.

Only the newest frame is forwarded, at most `--rate` times per second, so a
slow reader never builds up a queue of stale images.
"""

import argparse
import json
import struct
import sys
import threading
import time

from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

MAGIC = b"SARF"
HEADER = struct.Struct("<4sIIIIqI")
BOX = struct.Struct("<Iffff")
NO_BOXES = 0xFFFFFFFF
KEEP_IMAGES = 4  # a few recent images, waiting for their boxes
KEEP_BOXES = 90


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nsec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="gz.msgs.Image topic")
    parser.add_argument("--info-topic", required=True, help="gz.msgs.CameraInfo topic")
    parser.add_argument("--boxes-topic", help="gz.msgs.AnnotatedAxisAligned2DBox_V topic")
    parser.add_argument("--rate", type=float, default=10.0, help="max frames per second")
    args = parser.parse_args()

    lock = threading.Lock()
    images: dict[int, Image] = {}
    boxes: dict[int, list] = {}
    state = {"info": None}

    def on_image(message: Image) -> None:
        with lock:
            images[stamp_ns(message)] = message
            for old in sorted(images)[:-KEEP_IMAGES]:
                del images[old]

    def on_boxes(message: AnnotatedAxisAligned2DBox_V) -> None:
        found = [
            (b.label, b.box.min_corner.x, b.box.min_corner.y, b.box.max_corner.x, b.box.max_corner.y)
            for b in message.annotated_box
        ]
        with lock:
            boxes[stamp_ns(message)] = found
            for old in sorted(boxes)[:-KEEP_BOXES]:
                del boxes[old]

    def on_info(message: CameraInfo) -> None:
        with lock:
            if state["info"] is None:
                state["info"] = message

    node = Node()
    subscriptions = [(Image, args.topic, on_image), (CameraInfo, args.info_topic, on_info)]
    if args.boxes_topic:
        subscriptions.append((AnnotatedAxisAligned2DBox_V, args.boxes_topic, on_boxes))
    for message_type, topic, callback in subscriptions:
        if not node.subscribe(message_type, topic, callback):
            print(f"could not subscribe to {topic}", file=sys.stderr)
            return 2

    out = sys.stdout.buffer
    period = 1.0 / args.rate
    info_sent = False
    last_sent = -1
    try:
        while True:
            time.sleep(period)
            with lock:
                info = state["info"]
                ready = [t for t in images if t > last_sent and (not args.boxes_topic or t in boxes)]
                newest = max(ready) if ready else None
                image = images.get(newest) if newest is not None else None
                found = boxes.get(newest) if args.boxes_topic and newest is not None else None

            if not info_sent:
                if info is None:
                    continue
                k = info.intrinsics.k  # row-major 3x3: fx 0 cx / 0 fy cy / 0 0 1
                lens = {
                    "width": info.width,
                    "height": info.height,
                    "fx": k[0],
                    "fy": k[4],
                    "cx": k[2],
                    "cy": k[5],
                }
                out.write(json.dumps(lens).encode() + b"\n")
                out.flush()
                info_sent = True

            if image is None:
                continue
            last_sent = newest
            out.write(
                HEADER.pack(
                    MAGIC,
                    image.width,
                    image.height,
                    image.pixel_format_type,
                    len(image.data),
                    newest,
                    NO_BOXES if found is None else len(found),
                )
            )
            out.write(image.data)
            for box in found or []:
                out.write(BOX.pack(*box))
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        return 0  # the reader went away: normal shutdown


if __name__ == "__main__":
    raise SystemExit(main())
