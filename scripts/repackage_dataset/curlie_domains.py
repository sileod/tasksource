"""Locally convert the Curlie domain-level benchmark archive into a data-only mirror; never upload.

Downloads the pinned Zenodo archive (or reads it from --input), checks its MD5, and writes the native
train/validation/test JSONL as parquet: ``domain`` and ``labels`` (the list of Curlie top-level categories).
"""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Sequence, Value

from tasksource.citation import card_section

RECORD = '19943663'
URL = f'https://zenodo.org/api/records/{RECORD}/files/curlie.tar.gz/content'
MD5 = 'dcf2531fd68d4702f4cc7b6eebb01e31'
SPLITS = {'train': 'train.jsonl', 'validation': 'val.jsonl', 'test': 'test.jsonl'}
EXPECTED = {'train': 566826, 'validation': 161950, 'test': 80976}


def convert(archive, output):
    data = archive.read_bytes()
    if hashlib.md5(data).hexdigest() != MD5:
        raise ValueError('Archive MD5 mismatch')
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        files = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        metadata = json.load(tar.extractfile(files['metadata.json']))
        classes = metadata['classes']
        features = Features({'domain': Value('string'), 'labels': Sequence(ClassLabel(names=classes))})
        output.mkdir(parents=True, exist_ok=True)
        sizes, multi = {}, {}
        for split, name in SPLITS.items():
            rows = []
            for line in tar.extractfile(files[name]):
                x = json.loads(line)
                labels = [i for i, v in enumerate(x['class_vector']) if v]
                if len(x['class_vector']) != len(classes) or not labels:
                    raise ValueError(f'Bad class vector: {x}')
                rows.append({'domain': x['text'], 'labels': labels})
            Dataset.from_list(rows, features=features).to_parquet(str(output / f'{split}.parquet'))
            sizes[split], multi[split] = len(rows), sum(len(r['labels']) > 1 for r in rows)
    assert sizes == EXPECTED, sizes
    provenance = dict(mirror='tasksource/curlie-domains', source=f'https://zenodo.org/records/{RECORD}',
        archive={'url': URL, 'md5': MD5}, classes=classes, rows=sizes, multi_label_rows=multi,
        license='cc-by-4.0', license_evidence=f'https://zenodo.org/records/{RECORD}',
        settings={'split': 'native train/val/test of the benchmark (val renamed validation)',
                  'labels': 'indices of the set class_vector entries'})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'README.md').write_text(f'''---
license: cc-by-4.0
tags:
- tasksource
- text-classification
- web
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---
# Curlie domain categories

Apex domain names (`0-100.it`, `000webhostapp.com`) with their top-level categories in the human-edited
[Curlie](https://curlie.org) web directory: {", ".join(c.replace("_", " ") for c in classes)}. About 2% of domains
have several categories. Domains come from many countries and languages; only the name is given, not the page.

Data-only mirror of the [Curlie Domain-Level Multi-Label Content Classification Benchmark](https://zenodo.org/records/{RECORD})
(Weiß and Baumann, 2026), built from the Curlie release of Homepage2Vec (Lugeon et al., 2022). Reproduce with
[scripts/repackage_dataset/curlie_domains.py](https://github.com/sileod/tasksource/blob/main/scripts/repackage_dataset/curlie_domains.py);
[provenance.json](provenance.json) records the archive MD5 and settings. Native splits: {sizes["train"]:,} train,
{sizes["validation"]:,} validation, {sizes["test"]:,} test. License: CC BY 4.0; attribute Curlie and the
Homepage2Vec authors.

Schema: `domain`, `labels` (list of category ids; names in the features).

## Citation

Please cite the original benchmark and data:

```bibtex
@dataset{{weiss_baumann_2026_curlie,
  author    = {{Weiß, Thomas and Baumann, Timo}},
  title     = {{Curlie Domain-Level Multi-Label Content Classification Benchmark}},
  year      = {{2026}},
  publisher = {{Zenodo}},
  doi       = {{10.5281/zenodo.{RECORD}}},
  url       = {{https://zenodo.org/records/{RECORD}}},
}}

@inproceedings{{lugeon2022homepage2vec,
  title     = {{Homepage2Vec: Language-Agnostic Website Embedding and Classification}},
  author    = {{Lugeon, Sylvain and Piccardi, Tiziano and West, Robert}},
  booktitle = {{Proceedings of the International AAAI Conference on Web and Social Media}},
  volume    = {{16}},
  pages     = {{1285--1291}},
  year      = {{2022}},
}}
```

{card_section(title='Citing tasksource')}''')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='curlie.tar.gz (downloaded from Zenodo when omitted)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = args.input
    if archive is None:
        archive = args.output.parent / 'curlie.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, archive)
    convert(archive, args.output)
