"""Hugging Face export: only the chosen fields, tagged by split, no episode in two splits."""

import pytest

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

fo = pytest.importorskip("fiftyone")

from sardrone.fo import hub  # noqa: E402


@pytest.fixture
def frames(tmp_path):
    from PIL import Image

    name, out = "hub-test-src", "hub-test-out"
    for n in (name, out):
        if fo.dataset_exists(n):
            fo.delete_dataset(n)
    data = fo.Dataset(name)
    samples = []
    for index in range(9):
        path = tmp_path / f"f{index}.jpg"
        Image.new("RGB", (32, 24)).save(path)
        sample = fo.Sample(filepath=str(path))
        sample["episode_id"] = f"ep-{index // 3}"  # three frames per flight
        sample["ground_truth"] = fo.Detections(
            detections=[] if index == 0 else [fo.Detection(label="person", bounding_box=[0.1, 0.1, 0.1, 0.2])]
        )
        sample["pred_v4"] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=[0, 0, 0.1, 0.1])])
        sample["altitude_m"] = 25.0
        samples.append(sample)
    data.add_samples(samples, progress=False)
    F = fo.ViewField
    data.save_view("tr", data.match(F("episode_id") == "ep-0"))
    data.save_view("va", data.match(F("episode_id") == "ep-1"))
    data.save_view("te", data.match(F("episode_id") == "ep-2"))
    yield data, out
    for n in (name, out):
        if fo.dataset_exists(n):
            fo.delete_dataset(n)


def test_build_keeps_labels_and_metadata_but_drops_predictions(frames):
    source, out = frames
    built = hub.build(source, {"train": "tr", "val": "va", "test": "te"}, name=out)
    fields = set(built.get_field_schema())
    assert {"ground_truth", "episode_id", "altitude_m", "split"} <= fields
    assert "pred_v4" not in fields
    assert built.count_sample_tags() == {"train": 3, "val": 3, "test": 3}
    assert hub.check(built)["splits"] == {"train": 3, "val": 3, "test": 3}
    summary = hub.check(built)
    assert summary["samples"] == 9 and summary["people_boxes"] == 8
    assert summary["frames_without_people"] == 1
    assert summary["episodes_in_more_than_one_split"] == 0
    assert set(source.values("id")).isdisjoint(built.values("id"))  # a copy, the source is untouched
    assert "pred_v4" in source.get_field_schema()


def test_the_dataset_card_states_licence_and_attribution():
    card = hub.CARD.read_text()
    assert "license: cc-by-4.0" in card
    assert "Rescue Randy" in card and "Open Robotics" in card
    assert "visible" in card  # what a label means
