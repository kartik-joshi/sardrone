"""The sim2real-gap FiftyOne plugin: core functions and both operators, run
the way FiftyOne runs them (plugin discovered from its folder). Embeddings
are given, so no model is downloaded."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

fo = pytest.importorskip("fiftyone")
pytest.importorskip("sklearn")

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
spec = importlib.util.spec_from_file_location("sim2real_gap_core_test", PLUGIN_DIR / "sim2real_gap" / "gap.py")
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


def test_separability_is_high_for_different_sets_and_near_half_for_the_same():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (80, 16))
    b = rng.normal(3, 1, (80, 16))
    assert gap.separability(a, b) > 0.95
    assert 0.3 < gap.separability(*gap.random_halves(a)) < 0.7
    with pytest.raises(ValueError):
        gap.separability(a[:3], b[:3])


@pytest.fixture
def tiny(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(1)
    name = "s2r-plugin-test"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    data = fo.Dataset(name)
    samples = []
    for index in range(80):
        path = tmp_path / f"img_{index}.jpg"
        Image.new("RGB", (64, 48), (index, 80, 120)).save(path)
        domain = "sim" if index < 40 else "real"
        sample = fo.Sample(filepath=str(path), tags=[domain])
        sample["clip_embedding"] = (rng.normal(0 if domain == "sim" else 2.5, 1, 32)).tolist()
        truth = [fo.Detection(label="person", bounding_box=[0.1, 0.1, 0.2, 0.3])]
        preds = [
            fo.Detection(label="person", bounding_box=[0.1, 0.1, 0.2, 0.3], confidence=0.9),  # right
            fo.Detection(label="person", bounding_box=[0.6, 0.6, 0.2, 0.2], confidence=0.8 if index == 0 else 0.4),
        ]
        sample["ground_truth"] = fo.Detections(detections=truth)
        sample["predictions"] = fo.Detections(detections=preds)
        samples.append(sample)
    data.add_samples(samples, progress=False)
    data.save_view("sim", data.match_tags("sim"))
    data.save_view("real", data.match_tags("real"))
    yield data
    fo.delete_dataset(name)


def test_measure_scores_writes_domains_and_refuses_overlap(tiny):
    summary = gap.measure(tiny, tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key=None)
    assert summary["separability"] > 0.95
    assert 0.25 < summary["sanity_sim_vs_sim"] < 0.75
    assert summary["verdict"].startswith("trivially separable")
    assert tiny.count_values(gap.DOMAIN_FIELD) == {"sim": 40, "real": 40}
    assert tiny.info["sim2real_gap"]["separability"] == summary["separability"]
    with pytest.raises(ValueError):
        gap.measure(tiny, tiny.view(), tiny.load_saved_view("real"), brain_key=None)


def test_false_alarms_are_the_unmatched_boxes_most_confident_first(tiny):
    patches = gap.false_alarms(tiny.view(), "predictions", min_confidence=0.3)
    assert len(patches) == 80  # one wrong box per image
    assert patches.first()["predictions"].confidence == pytest.approx(0.8)
    assert len(gap.false_alarms(tiny.view(), "predictions", min_confidence=0.5)) == 1


def test_both_operators_run_as_a_fiftyone_plugin(tiny, monkeypatch):
    import fiftyone.operators as foo

    monkeypatch.setattr(fo.config, "plugins_dir", str(PLUGIN_DIR))
    measured = foo.execute_operator(
        "@kartik-joshi/sim2real-gap/measure_sim2real_gap",
        ctx={"dataset": tiny, "params": {"sim_view": "sim", "real_view": "real", "brain_key": "s2r_map"}},
    )
    assert measured.error is None, measured.error
    # Shown as percentages; the stored summary keeps fractions.
    assert measured.result["separability"].endswith(" %") and float(measured.result["separability"][:-2]) > 95
    assert tiny.info["sim2real_gap"]["separability"] > 0.95
    assert "s2r_map" in tiny.list_brain_runs()

    reviewed = foo.execute_operator(
        "@kartik-joshi/sim2real-gap/review_false_alarms",
        ctx={"dataset": tiny, "view": tiny.view(),
             "params": {"pred_field": "predictions", "gt_field": "ground_truth", "min_confidence": 0.3}},  # fmt: skip
    )
    assert reviewed.error is None, reviewed.error
    assert reviewed.result == {"false_alarms": 80}
