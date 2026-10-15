"""A rebuild keeps the ids of the release it replaces."""
from datasets import Dataset
from scripts.build_jev_dataset import carry_published_example_ids, carry_published_ids


PUBLISHED = {
    "rows": {("t", "train:7"): "t-1:train:0", ("t", "train:9"): "t-1:train:1"},
    "owners": {"t-1:train:0": ("t", "train:7"), "t-1:train:1": ("t", "train:9")},
    "examples": {"t-1:train:0": "e7", "t-1:train:1": "e9"},
}


def test_resampled_rows_keep_their_published_ids():
    rows = Dataset.from_dict({
        # a cap moved train:9 to position 0, and its variant with it; train:5 is new at position 1
        "id": ["t-1:train:0", "t-1:train:0:check", "t-1:train:1"],
        "source": ["t", "t", "t"],
        "source_index": ["train:9", "train:9", "train:5"],
        "example_id": ["new9", "new9", "new5"],
    })
    rows = carry_published_ids(rows, PUBLISHED)
    assert rows["id"] == ["t-1:train:1", "t-1:train:1:check", "t-1:train:src-train-5"]
    rows = carry_published_example_ids(rows, PUBLISHED)
    assert rows["example_id"] == ["e9", "e9", "new5"]


def test_no_release_changes_nothing():
    rows = Dataset.from_dict({"id": ["t-1:train:0"], "source": ["t"], "source_index": ["train:9"],
                              "example_id": ["x"]})
    assert carry_published_ids(rows, None)["id"] == ["t-1:train:0"]
    assert carry_published_example_ids(rows, {"rows": {}, "owners": {}, "examples": {}})["example_id"] == ["x"]
