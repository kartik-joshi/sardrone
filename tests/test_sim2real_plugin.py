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
    specs = ([("sim", "A", a_dir)] * 60 + [("sim", "B", s_dir)] * 60 + [("real", None, r)] * 72
             + [("real", "odd", o)] * 8)  # fmt: skip
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
    assert tiny.count_values("s2r_domain") == {"sim": 120, "real": 80}
    # World A was built close to real, B far: A comes first, and its frames look more real.
    assert [g["value"] for g in summary["groups"]] == ["A", "B"]
    F = fo.ViewField
    realism = {w: np.mean(tiny.match(F("world") == w).values("s2r_realism")) for w in ("A", "B")}
    assert realism["A"] > realism["B"]
    # The "uncovered" view leads with the real images unlike anything simulated.
    uncovered = tiny.load_saved_view("s2r-uncovered-real")
    assert set(uncovered.limit(8).values("world")) == {"odd"}
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
    for index in range(60):
        sample = fo.Sample(filepath=_image(tmp_path / f"other_{index}.jpg"))
        sample["clip_embedding"] = rng.normal(3.0, 1.0, DIM).tolist()
        samples.append(sample)
    real2.add_samples(samples, progress=False)
    try:
        summary = gap.measure(tiny.load_saved_view("sim"), real2.view(), brain_key="never_used", prefix="x")
        assert "views" not in summary and "brain_key" not in summary
        assert real2.count_values("x_domain") == {"real": 60}
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
    assert summary["level"] == "objects" and summary["sim_items"] == 120
    values = [d for boxes in tiny.values("ground_truth.detections.obj_realism") for d in boxes]
    assert len(values) == 200 and all(0 <= v <= 1 for v in values)
    patches = tiny.load_saved_view("obj-uncovered-real")
    assert len(patches) == 80  # one patch per real box


def test_review_finds_false_alarms_and_misses(tiny):
    alarms = gap.review(tiny.view(), "predictions", min_confidence=0.3)
    assert len(alarms) == 200 and alarms.first()["predictions"].confidence == pytest.approx(0.8)
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
    assert reviewed.result["count"] == 200 and "worst-alarms" in tiny.list_saved_views()

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


def test_domain_colours_are_set_once_and_never_override_the_user(tiny):
    assert gap.set_domain_colours(tiny, "s2r_domain") is True
    entry = next(f for f in tiny.app_config.color_scheme.fields if f["path"] == "s2r_domain")
    assert {v["value"] for v in entry["valueColors"]} == {"sim", "real"}
    assert gap.set_domain_colours(tiny, "s2r_domain") is False  # already there: left alone
    tiny.app_config.color_scheme.fields = [{"path": "mine", "fieldColor": "#ff00ff"}]
    tiny.save()
    gap.set_domain_colours(tiny, "s2r_domain")
    paths = [f["path"] for f in tiny.app_config.color_scheme.fields]
    assert paths == ["mine", "s2r_domain"]  # the user's entry is kept


# ---- regression tests for the independent review's findings ----------------------


def test_a_second_run_replaces_the_first_runs_results_and_views(tiny):
    F = fo.ViewField
    real = tiny.load_saved_view("real")
    gap.measure(tiny.match(F("world") == "A"), real, brain_key=None)
    gap.measure(tiny.match(F("world") == "B"), real, brain_key=None)
    realistic = tiny.load_saved_view("s2r-realistic-sim")
    assert set(realistic.values("world")) == {"B"}  # world A's samples are gone
    assert tiny.count_values("s2r_domain") == {"sim": 60, "real": 80, None: 60}  # A's scores cleared too


def test_reviewing_two_prediction_fields_works_and_keeps_one_evaluation_each(tiny):
    for sample in tiny:
        sample["predictions2"] = sample["predictions"].copy()
        sample.save()
    gap.review(tiny.view(), "predictions")
    gap.review(tiny.view(), "predictions2")
    gap.review(tiny.view(), "predictions")  # again: replaced, not an error
    assert sorted(tiny.list_evaluations()) == ["s2r_predictions", "s2r_predictions2"]


def test_a_rare_group_is_listed_but_neither_ranked_nor_shrinking_the_others(tiny):
    F = fo.ViewField
    sim = tiny.load_saved_view("sim")
    sim.match(F("world") == "B").limit(8).set_field("world", "B-rare").save()  # 8 images: too few to rank
    rng = np.random.default_rng(12)
    real = gap.normalise(rng.normal(0, 1, (120, DIM)))  # 120 real: half is 60
    rows = gap.groups(tiny.load_saved_view("sim"), real, "world", "clip_embedding", None)
    by_value = {g["value"]: g for g in rows}
    assert by_value["B-rare"]["ranked"] is False and by_value["B-rare"]["separability"] is None
    assert by_value["A"]["ranked"] and by_value["B"]["ranked"]
    assert {by_value["A"]["n"], by_value["B"]["n"]} == {52}  # the smallest RANKED group (B, 52), not 8
    assert rows[-1]["value"] == "B-rare"  # unranked groups come last


def test_no_group_is_ranked_when_the_real_set_is_too_small_for_a_baseline(tiny):
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), group_field="world",
                          brain_key=None)  # 80 real: 40 per half, under 50  # fmt: skip
    assert summary["groups"] and not any(g["ranked"] for g in summary["groups"])


def test_missing_group_values_form_their_own_group(tiny):
    F = fo.ViewField
    tiny.load_saved_view("sim").match(F("world") == "A").limit(15).set_field("world", None).save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), group_field="world",
                          brain_key=None)  # fmt: skip
    assert "(none)" in [g["value"] for g in summary["groups"]]


def test_p_value_and_equal_size_frechet_are_reported(tiny):
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key=None)
    assert summary["p_value"] < 0.05 and summary["frechet_n"] == 40  # half of the smaller set (80 real)
    assert summary["frechet_excess"] == pytest.approx(
        summary["frechet"] - (summary["frechet_baseline_sim"] + summary["frechet_baseline_real"]) / 2, abs=1e-3)
    rng = np.random.default_rng(9)
    same = rng.normal(0, 1, (60, DIM))
    assert gap.permutation_p_value(same[:30], same[30:]) > 0.05  # one distribution: not significant


def test_mixed_embedding_sizes_give_a_clear_error(tiny):
    first = tiny.first()
    first["clip_embedding"] = list(range(8))
    first.save()
    with pytest.raises(ValueError, match="different sizes"):
        gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key=None)


def test_frechet_baseline_shows_the_small_sample_bias():
    rng = np.random.default_rng(10)
    one = gap.normalise(rng.normal(0, 1, (400, DIM)))
    fd = gap.frechet_with_baseline(one[:200], one[200:], 100)
    assert fd["frechet"] > 0  # biased: not zero even for one source...
    assert abs(fd["excess"]) < 0.5 * fd["frechet"]  # ...but the excess over the baseline is small


def test_a_failed_map_keeps_every_score(tiny, monkeypatch):
    import fiftyone.brain as fob

    def broken(*args, **kwargs):
        raise RuntimeError("no map today")

    monkeypatch.setattr(fob, "compute_visualization", broken)
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key="m")
    assert "no map today" in summary["map_error"]
    assert tiny.count_values("s2r_domain") == {"sim": 120, "real": 80}
    assert tiny.info["s2r"]["separability"] == summary["separability"]


def test_object_mode_builds_its_map_from_undeclared_embedding_attributes(tiny):
    rng = np.random.default_rng(11)
    for sample in tiny:  # dynamic, undeclared attributes, as in the review's crash
        for det in sample.ground_truth.detections:
            det["crop_vec"] = rng.normal(3.0 if "real" in sample.tags else -1.0, 1.0, DIM).tolist()
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), embeddings_field="crop_vec",
                          patches_field="ground_truth", prefix="obj", brain_key="obj_map")  # fmt: skip
    assert summary.get("map_error") is None and summary["brain_key"] == "obj_map"


# ---- fixes from the third review -------------------------------------------------


def test_only_objects_without_an_embedding_are_computed(tiny, monkeypatch):
    import fiftyone.core.collections as foc
    import fiftyone.zoo as foz

    for sample in tiny:
        for det in sample.ground_truth.detections:
            det["vec"] = [1.0] * 16
        sample.save()
    first = tiny.first()
    first.ground_truth.detections[0]["vec"] = None
    first.save()
    seen = {}

    def fake(self, model, patches_field, embeddings_field=None, **kwargs):
        seen["labels"] = self.count(f"{patches_field}.detections")

    monkeypatch.setattr(foz, "load_zoo_model", lambda name: object())
    monkeypatch.setattr(foc.SampleCollection, "compute_patch_embeddings", fake)
    gap.ensure_embeddings(tiny.view(), "vec", patches_field="ground_truth")
    assert seen["labels"] == 1  # only the one box without an embedding, not all 200
    others = [v for boxes in tiny.values("ground_truth.detections.vec") for v in boxes if v is not None]
    assert len(others) == 199 and all(len(v) == 16 for v in others)  # nothing overwritten


def test_an_object_level_rerun_clears_the_previous_runs_box_scores(tiny):
    F = fo.ViewField
    rng = np.random.default_rng(13)
    for sample in tiny:
        for det in sample.ground_truth.detections:
            det["obj_vec"] = rng.normal(3.0 if "real" in sample.tags else -1.0, 1.0, DIM).tolist()
        sample.save()
    kwargs = {"embeddings_field": "obj_vec", "patches_field": "ground_truth", "prefix": "o", "brain_key": None}
    gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), **kwargs)
    world_a = tiny.load_saved_view("sim").match(F("world") == "A")
    gap.measure(world_a, tiny.load_saved_view("real"), **kwargs)
    world_b = tiny.load_saved_view("sim").match(F("world") == "B")
    assert set(v for boxes in world_b.values("ground_truth.detections.o_domain") for v in boxes) == {None}
    assert tiny.has_field("ground_truth.detections.o_realism")  # declared, so the App can filter by it


def test_scores_left_by_an_older_version_are_cleared_too(tiny):
    ids = [i for boxes in tiny.values("ground_truth.detections.id") for i in boxes]
    tiny.set_label_values("ground_truth.detections.old_domain", {ids[0]: "sim"})  # undeclared, as 2.2 wrote it
    assert not tiny.has_field("ground_truth.detections.old_domain")
    gap.clear_previous(tiny, "old", "ground_truth")
    assert tiny.count("ground_truth.detections.old_domain") == 0


def _clustered(rng, images, per_image, offset_scale=2.0):
    """Crops from `images` images: crops of one image share that image's offset."""
    offsets = rng.normal(0, offset_scale, (images, DIM))
    x = np.repeat(offsets, per_image, axis=0) + rng.normal(0, 0.3, (images * per_image, DIM))
    return gap.normalise(x), np.repeat(np.arange(images), per_image)


def test_clustered_items_do_not_make_one_distribution_look_separable():
    rng = np.random.default_rng(14)
    a, ga = _clustered(rng, 60, 5)
    b, gb = _clustered(rng, 60, 5)  # the SAME distribution, 5 crops per image
    leaky = gap.separability(a, b)[0]
    grouped = gap.separability(a, b, groups_a=ga, groups_b=gb)[0]
    assert leaky > 0.6  # crops of one image on both sides of a fold: looks separable
    assert grouped < 0.56  # images kept whole: about chance
    assert gap.permutation_p_value(a, b, permutations=30, groups_a=ga, groups_b=gb) > 0.05
    halves = gap.random_halves(a, 0, ga)
    assert not set(halves[2]) & set(halves[3])  # no image in both halves


def test_clusters_still_find_a_real_gap():
    rng = np.random.default_rng(15)
    a, ga = _clustered(rng, 40, 5)
    b, gb = _clustered(rng, 40, 5)
    b = gap.normalise(b + 0.4)
    assert gap.separability(a, b, groups_a=ga, groups_b=gb)[0] > 0.9
    assert gap.permutation_p_value(a, b, permutations=30, groups_a=ga, groups_b=gb) < 0.05


def test_object_level_keeps_each_images_boxes_together(tiny):
    rng = np.random.default_rng(16)
    for sample in tiny:  # two boxes per image, so images are clusters
        sample.ground_truth.detections.append(fo.Detection(label="person", bounding_box=[0.5, 0.5, 0.2, 0.2]))
        for det in sample.ground_truth.detections:
            det["v2"] = rng.normal(3.0 if "real" in sample.tags else -1.0, 1.0, DIM).tolist()
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), embeddings_field="v2",
                          patches_field="ground_truth", prefix="c", brain_key=None)  # fmt: skip
    assert summary["split_by"] == "image" and summary["clusters"] == {"sim": 120, "real": 80}


def test_a_split_field_keeps_recordings_together(tiny):
    for k, sample in enumerate(tiny):
        sample["recording"] = f"rec{k // 4}"  # 4 frames per recording
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), split_field="recording",
                          brain_key=None)  # fmt: skip
    assert summary["split_by"] == "recording" and summary["clusters"] == {"sim": 30, "real": 20}


def test_patches_views_are_refused_early(tiny):
    patches = tiny.load_saved_view("sim").to_patches("ground_truth")
    with pytest.raises(ValueError, match="whole images"):
        gap.measure(patches, tiny.load_saved_view("real"), brain_key=None)


def test_the_app_offers_the_current_view_and_an_entire_dataset(tiny, tmp_path, monkeypatch):
    import types as pytypes

    import fiftyone.operators as foo

    monkeypatch.setattr(fo.config, "plugins_dir", str(PLUGIN_DIR))
    plugin = importlib.util.spec_from_file_location("s2r_init", PLUGIN_DIR / "sim2real_gap" / "__init__.py")
    module = importlib.util.module_from_spec(plugin)
    plugin.loader.exec_module(module)
    other = "s2r-plugin-test-entire"
    if fo.dataset_exists(other):
        fo.delete_dataset(other)
    real2 = fo.Dataset(other)  # no saved views at all
    rng = np.random.default_rng(17)
    real2.add_samples([fo.Sample(filepath=_image(tmp_path / f"e_{i}.jpg"), clip_embedding=rng.normal(3.0, 1.0, DIM).tolist())
                       for i in range(60)], progress=False)  # fmt: skip
    try:
        ctx = pytypes.SimpleNamespace(dataset=tiny, view=tiny.view(), params={"real_dataset": other})
        form = str(module.MeasureSim2RealGap().resolve_input(ctx).to_json())
        assert module.CURRENT_VIEW in form and module.ENTIRE_DATASET in form
        assert "s2r-uncovered-real" not in form  # the plugin's own results are not offered as inputs
        result = foo.execute_operator(
            "@kartik-joshi/sim2real-gap/measure_sim2real_gap",
            ctx={"dataset": tiny, "view": tiny.load_saved_view("sim"),
                 "params": {"sim_view": module.CURRENT_VIEW, "real_dataset": other,
                            "real_view": module.ENTIRE_DATASET, "prefix": "e"}},
        )  # fmt: skip
        assert result.error is None, result.error
        assert real2.count_values("e_domain") == {"real": 60} and tiny.count_values("e_domain")["sim"] == 120
    finally:
        fo.delete_dataset(other)


# ---- fixes from the fourth review ------------------------------------------------


def _recordings(rng, recordings, frames, shift=0.0):
    """Video-like data: frames of one recording are near-copies of each other."""
    centres = rng.normal(shift, 2.0, (recordings, DIM))
    x = np.repeat(centres, frames, axis=0) + rng.normal(0, 0.05, (recordings * frames, DIM))
    return gap.normalise(x), np.repeat([f"rec{k}" for k in range(recordings)], frames)


def test_long_recordings_keep_every_recording_under_the_cap():
    rng = np.random.default_rng(20)
    _, groups = _recordings(rng, 12, 120)
    kept = gap._take(len(groups), groups, 500, rng)
    assert len(kept) <= 500 and len(set(groups[kept])) == 12  # a share of every recording, not 4 whole ones


def test_the_p_value_works_with_long_recordings():
    rng = np.random.default_rng(21)
    a, ga = _recordings(rng, 12, 120)
    b, gb = _recordings(rng, 12, 120)
    p = gap.permutation_p_value(a, b, permutations=20, groups_a=ga, groups_b=gb)
    assert 0 < p <= 1


def test_sanity_halves_keep_unequal_clusters_whole():
    rng = np.random.default_rng(22)
    x = gap.normalise(rng.normal(0, 1, (600, DIM)))
    groups = np.r_[np.repeat("big", 400), np.repeat([f"r{k}" for k in range(20)], 10)]
    h1, h2, g1, g2 = gap.random_halves(x, 3, groups)
    assert not set(g1) & set(g2) and len(h1) + len(h2) == 600


def test_too_few_recordings_give_notes_not_a_crash(tiny):
    for k, sample in enumerate(tiny.load_saved_view("sim")):
        sample["recording"] = f"rec{k % 6}"  # 6 recordings: halves of 3 cannot make 5 folds
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), split_field="recording",
                          brain_key=None)  # fmt: skip
    assert summary["separability"] is not None and summary["sanity_sim_vs_sim"] is None
    assert any("sanity check" in note for note in summary["notes"])


def test_a_large_p_value_makes_the_verdict_not_significant():
    assert gap.verdict(0.78, 0.4).startswith("not significant")
    assert gap.verdict(0.78, 0.01).startswith("clearly separable")


def test_near_duplicates_are_flagged_when_no_split_field_is_given():
    rng = np.random.default_rng(23)
    frames, _ = _recordings(rng, 10, 30)
    photos = gap.normalise(rng.normal(0, 1, (300, DIM)))
    assert gap.near_duplicate_share(frames) > 0.9 and gap.near_duplicate_share(photos) < 0.05


def test_the_detector_review_runs_again_on_its_own_patches_view(tiny):
    first = gap.review(tiny.view(), "predictions", min_confidence=0.3)
    again = gap.review(first, "predictions", min_confidence=0.3)  # what the App passes on a second run
    assert len(again) == len(first) == 200


def test_false_alarms_without_a_confidence_count_at_zero(tiny):
    sample = tiny.first()
    sample.predictions.detections[1].confidence = None
    sample.save()
    assert len(gap.review(tiny.view(), "predictions", min_confidence=0.0)) == 200
    assert len(gap.review(tiny.view(), "predictions", min_confidence=0.3)) == 199


def test_a_split_field_missing_from_the_second_dataset_is_per_sample_there(tiny, tmp_path):
    other = "s2r-plugin-test-nosplit"
    if fo.dataset_exists(other):
        fo.delete_dataset(other)
    real2 = fo.Dataset(other)
    rng = np.random.default_rng(24)
    real2.add_samples([fo.Sample(filepath=_image(tmp_path / f"n_{i}.jpg"), clip_embedding=rng.normal(3.0, 1.0, DIM).tolist())
                       for i in range(60)], progress=False)  # fmt: skip
    for k, sample in enumerate(tiny.load_saved_view("sim")):
        sample["recording"] = f"rec{k // 4}"
        sample.save()
    try:
        summary = gap.measure(tiny.load_saved_view("sim"), real2.view(), split_field="recording", prefix="n")
        assert summary["clusters"] == {"sim": 30, "real": None}  # no field there: every real item on its own
        assert summary["near_duplicates"]["sim"] is None and summary["near_duplicates"]["real"] is not None
    finally:
        fo.delete_dataset(other)


# ---- fixes from the fifth review -------------------------------------------------


def test_nan_embeddings_are_refused_with_their_ids(tiny):
    bad = tiny.load_saved_view("sim").first()
    vec = list(bad.clip_embedding)
    vec[3] = float("nan")
    bad["clip_embedding"] = vec
    bad.save()
    with pytest.raises(ValueError, match="NaN or infinity") as error:
        gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key=None)
    assert bad.id in str(error.value)
    with pytest.raises(ValueError, match="NaN"):
        gap.normalise([[1.0, float("inf")], [1.0, 0.0]])


def test_sanity_with_few_clusters_is_averaged_and_flagged(tiny):
    for k, sample in enumerate(tiny.load_saved_view("sim")):
        sample["recording"] = f"rec{k % 15}"  # 15 recordings: about 7 per half
        sample.save()
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), split_field="recording",
                          brain_key=None)  # fmt: skip
    low, high = summary["sanity_ranges"]["sim"]
    assert low <= summary["sanity_sim_vs_sim"] <= high
    assert any("noisy" in note for note in summary["notes"])


def test_an_empty_split_field_claims_no_clusters(tiny):
    tiny.add_sample_field("empty_split", fo.StringField)  # declared, but None everywhere
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), split_field="empty_split",
                          brain_key=None)  # fmt: skip
    assert summary["clusters"] is None and summary["split_by"] is None
    assert any("groups no items" in note for note in summary["notes"])
    assert summary["near_duplicates"]["sim"] is not None  # still checked


def test_the_frechet_excess_comes_with_a_spread(tiny):
    summary = gap.measure(tiny.load_saved_view("sim"), tiny.load_saved_view("real"), brain_key=None)
    assert summary["frechet_repeats"] == 3 and summary["frechet_excess_spread"] >= 0
    report = _plugin_module().report(summary)
    assert "mean ± spread of 3 draws" in report


def _plugin_module():
    spec = importlib.util.spec_from_file_location("s2r_init_report", PLUGIN_DIR / "sim2real_gap" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_report_shows_notes_and_near_duplicate_warnings():
    module = _plugin_module()
    summary = {"level": "images", "separability": 1.0, "separability_std": 0.0, "p_value": None,
               "verdict": "x", "sanity_sim_vs_sim": None, "sanity_real_vs_real": 0.5, "sanity_ranges": {},
               "frechet_n": 10, "frechet": 0.5, "frechet_baseline_sim": 0.1, "frechet_baseline_real": 0.1,
               "frechet_excess": 0.4, "frechet_excess_spread": 0.01, "frechet_repeats": 3, "model": "m",
               "sim_items": 20, "real_items": 20, "realism_sim_mean": 0.1, "notes": ["no permutation p-value: few"],
               "near_duplicates": {"sim": 0.5, "real": None},
               "fields": {"domain": "d", "realism": "r", "nearest": "n"}}  # fmt: skip
    text = module.report(summary)
    assert "permutation p = -" in text and "Note: no permutation p-value" in text
    assert "50 % of simulated items have a near-copy" in text and "of real" not in text


def test_long_recordings_keep_a_real_share_of_every_recording():
    rng = np.random.default_rng(25)
    groups = np.repeat([f"r{k}" for k in range(12)], 120)
    kept = gap._take(len(groups), groups, 500, rng)
    assert len(kept) >= 12 * (500 // 12)  # 41 frames from each, not one
