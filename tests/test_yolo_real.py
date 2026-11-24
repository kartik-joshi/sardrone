"""YOLO-style real test sets (the WiSARD sample, issue #9)."""

import pytest

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

fo = pytest.importorskip("fiftyone")

from sardrone.fo import yolo_real  # noqa: E402


def test_yolo_lines_become_top_left_boxes():
    boxes = yolo_real.read_yolo_boxes("0 0.5 0.5 0.2 0.1\n\nbad line\n0 0.1 0.2 0.02 0.04\n")
    assert boxes == [pytest.approx([0.4, 0.45, 0.2, 0.1]), pytest.approx([0.09, 0.18, 0.02, 0.04])]


def test_build_keeps_frame_order_and_empty_frames(tmp_path):
    from PIL import Image

    folder = tmp_path / "flight"
    folder.mkdir()
    for number, text in [(9, "0 0.5 0.5 0.1 0.1\n"), (1, ""), (4, "0 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.1 0.1\n")]:
        Image.new("RGB", (64, 48)).save(folder / f"x_{number:08d}.jpeg")
        (folder / f"x_{number:08d}.txt").write_text(text)
    (folder / "count.txt").write_text("number of humans: 3")
    name = "yolo-real-test"
    try:
        dataset = yolo_real.build(name, folder)
        view = dataset.load_saved_view(f"{name}-test")
        assert view.values("frame") == [1, 4, 9]
        assert view.count("ground_truth.detections") == 3
        assert len(view.match(fo.ViewField("ground_truth.detections").length() == 0)) == 1
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
