"""The sim2real-gap FiftyOne plugin (v2): statistics, per-item scores, saved
views, group breakdown, object level, two datasets, and every operator run
the way FiftyOne runs it. Embeddings are given, so no model is downloaded.

The synthetic sets are built so the right answers are known: simulated
"world A" lies close to the real set, "world B" far from it, and a few real
images ("odd") are unlike anything simulated."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

fo = pytest.importorskip("fiftyone")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
spec = importlib.util.spec_from_file_location("sim2real_gap_core_test", PLUGIN_DIR / "sim2real_gap" / "gap.py")
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)

DIM = 32


def cloud(rng, centre, n, spread=1.0):
    return rng.normal(centre, spread, (n, DIM))


# ---- statistics ------------------------------------------------------------------


def test_separability_with_spread_and_the_sanity_check():
    rng = np.random.default_rng(0)
    a, b = cloud(rng, 0, 80), cloud(rng, 3, 80)
    mean, std = gap.separability(a, b)
    assert mean > 0.95 and 0 <= std < 0.1
    assert 0.3 < gap.separability(*gap.random_halves(a))[0] < 0.7
    with pytest.raises(ValueError):
        gap.separability(a[:3], b[:3])


def test_balanced_accuracy_does_not_reward_unequal_set_sizes():
    rng = np.random.default_rng(1)
    same = cloud(rng, 0, 300)
    mean, _ = gap.separability(same[:40], same[40:])  # one distribution, 40 vs 260 items
    assert mean < 0.65  # plain accuracy could reach 0.87 by always answering "b"


def test_frechet_is_near_zero_for_one_distribution_and_grows_with_the_shift():
    rng = np.random.default_rng(2)
    base = cloud(rng, 0, 400)
    same = gap.frechet_distance(base[:200], base[200:])
    near = gap.frechet_distance(base[:200], cloud(rng, 0.5, 200))
    far = gap.frechet_distance(base[:200], cloud(rng, 2.0, 200))
    assert same < near < far


def test_nearest_distance_is_cosine_distance_to_the_closest_row():
    q = gap.normalise([[1, 0], [0, 1]])
    r = gap.normalise([[1, 0], [1, 1]])
    d = gap.nearest_distance(q, r)
    assert d[0] == pytest.approx(0.0) and d[1] == pytest.approx(1 - np.sqrt(0.5))


def test_realism_is_out_of_fold_and_ranks_the_closer_items_higher():
    rng = np.random.default_rng(3)
    real = cloud(rng, 3, 60)
    close, far = cloud(rng, 2.0, 30), cloud(rng, -1.0, 30)
    p_sim, p_real = gap.realism_scores(np.vstack([close, far]), real)
    assert p_sim[:30].mean() > p_sim[30:].mean()
    assert p_real.mean() > p_sim.mean()


# ---- the measurement on a dataset ------------------------------------------------


def _image(path):
    from PIL import Image

    Image.new("RGB", (64, 48), (90, 120, 60)).save(path)
    return str(path)


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(4)
    name = "s2r-plugin-test"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    data = fo.Dataset(name)
    samples = []
    # Embeddings are compared by direction (the plugin normalises them, as is
    # usual for CLIP), so the clouds differ in direction: real around r, world
    # A between r and s (close to real), world B around s (far), and "odd"
    # real images off in a direction of their own, like real content the
    # simulator never renders.
    basis = np.linalg.qr(rng.normal(size=(DIM, 3)))[0].T * 6.0
    r, s_dir, o = basis
    a_dir = 0.8 * r + 0.2 * s_dir
    specs = ([("sim", "A", a_dir)] * 30 + [("sim", "B", s_dir)] * 30 + [("real", None, r)] * 36
             + [("real", "odd", o)] * 4)  # fmt: skip
    for index, (domain, world, centre) in enumerate(specs):
        sample = fo.Sample(filepath=_image(tmp_path / f"img_{index}.jpg"), tags=[domain])
        sample["world"] = world
        sample["clip_embedding"] = rng.normal(centre, 1.0, DIM).tolist()
        sample["ground_truth"] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=[0.1, 0.1, 0.2, 0.3])])
        sample["predictions"] = fo.Detections(detections=[
            fo.Detection(label="person", bounding_box=[0.1, 0.1, 0.2, 0.3], confidence=0.9),
            fo.Detection(label="person", bounding_box=[0.6, 0.6, 0.2, 0.2], confidence=0.8 if index == 0 else 0.4),
        ])  # fmt: skip
        samples.append(sample)
    data.add_samples(samples, progress=False)
    data.save_view("sim", data.match_tags("sim"))
    data.save_view("real", data.match_tags("real"))
    yield data
    fo.delete_dataset(name)


def test_measure_writes_scores_views_and_a_group_breakdown(tiny):
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), group_field="world",
                          brain_key=None)  # fmt: skip
    # Half the simulated set (world A) was built close to real, so the sets are
    # clearly, but not trivially, separable.
    assert 0.6 < summary["separability"] < 0.99 and summary["frechet"] > 0
    assert 0.2 < summary["sanity_sim_vs_sim"] < 0.8
    assert tiny.count_values("s2r_domain") == {"sim": 60, "real": 40}
    # World A was built close to real, B far: A comes first, and its frames look more real.
    assert [g["value"] for g in summary["groups"]] == ["A", "B"]
    F = fo.ViewField
    realism = {w: np.mean(tiny.match(F("world") == w).values("s2r_realism")) for w in ("A", "B")}
    assert realism["A"] > realism["B"]
    # The "uncovered" view leads with the real images unlike anything simulated.
    uncovered = tiny.load_saved_view("s2r-uncovered-real")
    assert set(uncovered.limit(4).values("world")) == {"odd"}
    realistic = tiny.load_saved_view("s2r-realistic-sim")
    assert realistic.first()["world"] == "A"
    assert tiny.info["s2r"]["separability"] == summary["separability"]


def test_measure_refuses_overlapping_views_and_mismatched_embeddings(tiny):
    with pytest.raises(ValueError):
        gap.measure(tiny.view(), tiny.load_saved_view("real"), brain_key=None)


def test_two_datasets_score_both_without_a_map(tiny, tmp_path):
    other = "s2r-plugin-test-other"
    if fo.dataset_exists(other):
        fo.delete_dataset(other)
    real2 = fo.Dataset(other)
    rng = np.random.default_rng(5)
    samples = []
    for index in range(30):
        sample = fo.Sample(filepath=_image(tmp_path / f"other_{index}.jpg"))
        sample["clip_embedding"] = rng.normal(3.0, 1.0, DIM).tolist()
        samples.append(sample)
    real2.add_samples(samples, progress=False)
    try:
        summary = gap.measure(tiny.load_saved_view("sim"), real2.view(), brain_key="never_used", prefix="x")
        assert "views" not in summary and "brain_key" not in summary
        assert real2.count_values("x_domain") == {"real": 30}
        assert "x" in real2.info and "x" in tiny.info
    finally:
        fo.delete_dataset(other)


def test_object_level_scores_each_box(tiny):
    rng = np.random.default_rng(6)
    for sample in tiny:  # object embeddings given, like the image ones
        centre = 3.0 if "real" in sample.tags else -1.0
        for det in sample.ground_truth.detections:
            det["obj_embedding"] = rng.normal(centre, 1.0, DIM).tolist()
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), embeddings_field="obj_embedding",
                          patches_field="ground_truth", prefix="obj", brain_key=None)  # fmt: skip
    assert summary["level"] == "objects" and summary["sim_items"] == 60
    values = [d for boxes in tiny.values("ground_truth.detections.obj_realism") for d in boxes]
    assert len(values) == 100 and all(0 <= v <= 1 for v in values)
    patches = tiny.load_saved_view("obj-uncovered-real")
    assert len(patches) == 40  # one patch per real box


def test_review_finds_false_alarms_and_misses(tiny):
    alarms = gap.review(tiny.view(), "predictions", min_confidence=0.3)
    assert len(alarms) == 100 and alarms.first()["predictions"].confidence == pytest.approx(0.8)
    assert len(gap.review(tiny.view(), "predictions", kind="misses")) == 0  # every person was found


# ---- operators, run the way FiftyOne runs them ----------------------------------


def test_the_operators_run_as_a_fiftyone_plugin(tiny, monkeypatch):
    import fiftyone.operators as foo

    monkeypatch.setattr(fo.config, "plugins_dir", str(PLUGIN_DIR))
    measured = foo.execute_operator(
        "@kartik-joshi/sim2real-gap/measure_sim2real_gap",
        ctx={"dataset": tiny, "params": {"sim_view": "sim", "real_view": "real", "group_field": "world",
                                         "brain_key": "s2r_map"}},  # fmt: skip
    )
    assert measured.error is None, measured.error
    report = measured.result["report"]
    assert "Separability" in report and "%" in report and "Closest to real first" in report
    assert "s2r_map" in tiny.list_brain_runs()

    reviewed = foo.execute_operator(
        "@kartik-joshi/sim2real-gap/review_detector",
        ctx={"dataset": tiny, "view": tiny.view(),
             "params": {"pred_field": "predictions", "gt_field": "ground_truth", "min_confidence": 0.3,
                        "save_as": "worst-alarms"}},  # fmt: skip
    )
    assert reviewed.error is None, reviewed.error
    assert reviewed.result["count"] == 100 and "worst-alarms" in tiny.list_saved_views()

    legacy = foo.execute_operator(  # the 1.0 operator name still works
        "@kartik-joshi/sim2real-gap/review_false_alarms",
        ctx={"dataset": tiny, "view": tiny.view(), "params": {"pred_field": "predictions", "gt_field": "ground_truth"}},
    )
    assert legacy.error is None, legacy.error


def test_groups_are_still_ranked_when_separability_saturates():
    # Two groups, both trivially separable from real: Frechet distance still
    # tells the nearer one apart.
    rng = np.random.default_rng(7)
    real = gap.normalise(cloud(rng, 0, 80) + np.r_[np.full(16, 4.0), np.zeros(16)])
    near = gap.normalise(cloud(rng, 0, 40) + np.r_[np.zeros(16), np.full(16, 4.0)])
    far = gap.normalise(cloud(rng, 0, 40) + np.r_[np.zeros(16), np.full(16, 9.0)] + np.r_[np.full(16, -3.0), np.zeros(16)])
    assert gap.separability(near, real)[0] > 0.95 and gap.separability(far, real)[0] > 0.95
    assert gap.frechet_distance(near, real) < gap.frechet_distance(far, real)
