"""Forward Gazebo camera images to stdout. Run by GazeboCamera, not by hand.

This file is a separate program on purpose. The Gazebo Python bindings
(`gz.transport13`, `gz.msgs10`) only exist inside the Python that ships with
Gazebo (PX4's pixi environment, Python 3.13). They cannot be installed into
this project's venv. So GazeboCamera starts this script with Gazebo's Python
and reads its stdout. It therefore uses ONLY the standard library plus `gz`.

Output protocol (binary, on stdout):
    line 1:   one JSON object + "\n" with the lens: width, height, fx, fy, cx, cy
    then, per frame:
        header  struct "<4sIIIIq" = magic b"SARF", width, height,
                pixel_format, data_length, sim_time_ns
        data    `data_length` raw bytes (rows of pixels, top to bottom)

Only the newest frame is forwarded, at most `--rate` times per second, so a
slow reader never builds up a queue of stale images.
"""

import argparse
import json
import struct
import sys
import threading
import time

from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

MAGIC = b"SARF"
HEADER = struct.Struct("<4sIIIIq")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="gz.msgs.Image topic")
    parser.add_argument("--info-topic", required=True, help="gz.msgs.CameraInfo topic")
    parser.add_argument("--rate", type=float, default=10.0, help="max frames per second")
    args = parser.parse_args()

    lock = threading.Lock()
    latest: dict = {"image": None, "info": None}

    def on_image(message: Image) -> None:
        with lock:
            latest["image"] = message

    def on_info(message: CameraInfo) -> None:
        with lock:
            if latest["info"] is None:
                latest["info"] = message

    node = Node()
    if not node.subscribe(Image, args.topic, on_image):
        print(f"could not subscribe to {args.topic}", file=sys.stderr)
        return 2
    if not node.subscribe(CameraInfo, args.info_topic, on_info):
        print(f"could not subscribe to {args.info_topic}", file=sys.stderr)
        return 2

    out = sys.stdout.buffer
    period = 1.0 / args.rate
    info_sent = False
    try:
        while True:
            time.sleep(period)
            with lock:
                image, latest["image"] = latest["image"], None
                info = latest["info"]

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
            stamp = image.header.stamp
            sim_time_ns = stamp.sec * 1_000_000_000 + stamp.nsec
            out.write(
                HEADER.pack(
                    MAGIC,
                    image.width,
                    image.height,
                    image.pixel_format_type,
                    len(image.data),
                    sim_time_ns,
                )
            )
            out.write(image.data)
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        return 0  # the reader went away: normal shutdown


if __name__ == "__main__":
    raise SystemExit(main())
