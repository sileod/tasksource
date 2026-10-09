#!/usr/bin/env python3
"""Audit a materialized Jev vision config.

The checks are deliberately source-agnostic: every image must decode, every
decision must have a non-empty state/question, unique options, a valid target,
and parseable provenance. A small textual sample is emitted for manual image
review; the script does not pretend that an automated check can establish
visual answerability.
"""

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Image, Sequence, load_from_disk
from PIL import Image as PILImage
from tasksource.vision_tasks import clean_cauldron_question


def image_bytes(image):
    # ``datasets`` decodes Image features to PIL objects on normal indexing.
    # The audit needs the stored bytes so that the provenance hashes can be
    # checked without re-encoding the image.
    if isinstance(image, PILImage.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "PNG")
        return buffer.getvalue()
    if image.get("bytes") is not None:
        return image["bytes"]
    return Path(image["path"]).read_bytes()


def audit(root, output, sample_per_source=3):
    dataset = load_from_disk(str(Path(root) / "dataset"))
    report = {
        "scope": "Automated structural and image-decoding checks over the materialized Jev vision config.",
        "manual_review": "The emitted sample rows must be opened with their images to judge visual answerability.",
        "splits": {},
        "sources": {},
        "sample_rows": [],
    }
    samples = defaultdict(int)
    for split, rows in dataset.items():
        assert rows.features["images"] == Sequence(Image()), rows.features["images"]
        raw_rows = rows.cast_column("images", Sequence(Image(decode=False)))
        split_report = Counter()
        for row, raw_row in zip(rows, raw_rows):
            split_report["rows"] += 1
            source = row["source"]
            report["sources"].setdefault(source, Counter())
            report["sources"][source][split] += 1
            assert row["state"].strip() and row["question"].strip()
            options = row["options"]
            assert len(options) >= 2 and len({option.casefold() for option in options}) == len(options)
            assert len(row["target"]) == len(options)
            assert all(value >= 0 and value <= 1 for value in row["target"])
            assert abs(sum(row["target"]) - 1) < 1e-6
            assert max(range(len(options)), key=row["target"].__getitem__) < len(options)
            images = row["images"]
            assert images
            hashes = []
            for image, raw_image in zip(images, raw_row["images"]):
                data = image_bytes(raw_image)
                with PILImage.open(io.BytesIO(data)) as decoded:
                    decoded.verify()
                    split_report["decoded_images"] += 1
                hashes.append(hashlib.sha256(data).hexdigest())
            metadata = json.loads(row["metadata"])
            if metadata.get('source_question'):
                assert clean_cauldron_question(row['state']) == row['state'], 'Leftover source answer-format instruction'
            assert metadata.get("image_group_id") and metadata.get("provenance")
            assert metadata.get("image_ids") == hashes
            if samples[source] < sample_per_source:
                samples[source] += 1
                report["sample_rows"].append({
                    "split": split,
                    "source": source,
                    "state": row["state"],
                    "question": row["question"],
                    "options": options,
                    "target": row["target"],
                    "image_sha256": hashes,
                    "image_group_id": metadata["image_group_id"],
                    "manual_review": "pending",
                })
        report["splits"][split] = dict(split_report)
    report["sources"] = {key: dict(value) for key, value in report["sources"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-source", type=int, default=3)
    args = parser.parse_args()
    audit(args.root, args.output, args.sample_per_source)


if __name__ == "__main__":
    main()
