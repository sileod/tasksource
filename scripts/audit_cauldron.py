"""Audit pinned Cauldron QA filtering without downloading its image payloads.

python scripts/audit_cauldron.py --output docs/jev/cauldron-audit.json
Add --samples build/cauldron-samples to save tiny native-image smoke fixtures.
Then use --smoke-only --samples build/cauldron-samples --output smoke.json.
Counts audit every question; smoke fixtures contain only the first native groups.
"""
import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from tasksource import list_tasks
from tasksource.vision_tasks import normalize_cauldron_answer, parse_cauldron_mc


def audit(config, tasks, files, revision, samples=None):
    fs = HfFileSystem()
    counts = {task.id: Counter() for task in tasks}
    image_groups = 0
    for path in files:
        if not path.startswith(config + '/') or not path.endswith('.parquet'):
            continue
        with fs.open(f'datasets/HuggingFaceM4/the_cauldron@{revision}/{path}', 'rb', block_size=65536) as handle:
            parquet = pq.ParquetFile(handle)
            image_groups += parquet.metadata.num_rows
            if samples and not (samples / (config + '.parquet')).exists():
                samples.mkdir(parents=True, exist_ok=True)
                batch = next(parquet.iter_batches(batch_size=32))
                import pyarrow as pa
                pq.write_table(pa.Table.from_batches([batch]), samples / (config + '.parquet'))
            for batch in parquet.iter_batches(batch_size=500, columns=['texts']):
                for row in batch.to_pylist():
                    for qa in row['texts']:
                        for task in tasks:
                            if task.task_type == 'VisualMultipleChoice':
                                _, reason = parse_cauldron_mc(qa)
                            else:
                                reason = None if normalize_cauldron_answer(qa['assistant']) in task.mapping.label_values else 'outside_vocabulary'
                            counts[task.id]['total'] += 1
                            counts[task.id][reason or 'retained'] += 1
        print(config, path, flush=True)
    return {task: {'image_groups': image_groups, 'total_qas': c['total'], 'retained_qas': c['retained'],
                   'dropped_by_reason': {key: value for key, value in sorted(c.items()) if key not in ('total', 'retained')}}
            for task, c in counts.items()}


def smoke(samples, tasks):
    """Run the public loader/recast on bounded, unmodified native-image fixtures."""
    import hashlib
    from unittest.mock import patch
    from datasets import load_dataset, Sequence, Image, ClassLabel, disable_progress_bars
    disable_progress_bars()
    from tasksource import load_task
    from tasksource.preprocess import disable_image_decoding
    report = {}
    for task in tasks.itertuples():
        path = samples / (task.config_name + '.parquet')
        native = load_dataset('parquet', data_files={'train': str(path)}, streaming=True)
        import io
        from PIL import Image as PILImage
        for row in disable_image_decoding(native)['train']:
            for image in row['images']:
                with PILImage.open(io.BytesIO(image['bytes'])) as decoded:
                    decoded.verify()
        groups = {tuple(hashlib.sha256(image['bytes']).hexdigest() for image in row['images'])
                  for row in disable_image_decoding(native)['train']}
        with patch('tasksource.access.load_dataset', return_value=native):
            ds = load_task(task.id, vision=True, max_rows=10)
            jev = load_task(task.id, vision=True, max_rows=10, recast='jev')
        assert set(ds) == {'train'} and len(ds['train'])
        assert ds['train'].features['images'] == Sequence(Image())
        assert isinstance(ds['train'].features['labels'], ClassLabel)
        for row in disable_image_decoding(jev)['train']:
            assert row['answer'] == row['criteria'][row['label']]
            assert tuple(hashlib.sha256(image['bytes']).hexdigest() for image in row['images']) in groups
            metadata = json.loads(row['metadata'])
            assert metadata['image_group_id']
            if task.task_type == 'VisualMultipleChoice':
                qa, reason = parse_cauldron_mc({'user': metadata['source_question'], 'assistant': metadata['source_answer']})
                assert reason is None and row['answer'] == qa['choices_list'][qa['labels']]
            else:
                expected = task.mapping.label_values[normalize_cauldron_answer(metadata['source_answer'])]
                assert row['answer'] == expected
        report[task.id] = {'native_groups': len(groups), 'rows': len(ds['train']), 'jev_rows': len(jev['train'])}
        print(task.id, report[task.id], flush=True)
    return {'scope': 'Loader injection of the first 32 native image groups per config, without modifying source records.', 'tasks': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=Path)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    tasks = list_tasks(vision=True)
    tasks = tasks[tasks.dataset_name == 'HuggingFaceM4/the_cauldron']
    if args.smoke_only:
        if not args.samples:
            parser.error('--smoke-only requires --samples')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(smoke(args.samples, tasks), indent=2) + '\n')
        return
    revision = tasks.iloc[0].mapping.load_dataset_kwargs['revision']
    files = HfApi().list_repo_files('HuggingFaceM4/the_cauldron', repo_type='dataset', revision=revision)
    groups = [list(group.itertuples()) for _, group in tasks.groupby('config_name')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        reports = list(pool.map(lambda group: audit(group[0].config_name, group, files, revision, args.samples), groups))
    report = {'dataset': 'HuggingFaceM4/the_cauldron', 'revision': revision,
              'scope': 'Full native QA text audit; image integrity is checked separately on native smoke fixtures.',
              'tasks': {key: value for result in reports for key, value in result.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
