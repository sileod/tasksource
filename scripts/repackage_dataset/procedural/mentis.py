"""Fresh MentisOculi paper-fold MC instances; never ingest benchmark examples."""
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile

from .common import checkout, seed_for, write_static

SOURCE = 'https://github.com/Jana-Z/mentis-oculi'
REVISION = 'ac6d97acf231767f408a9e5d75515872b90190d6'


def mentis_paper_fold(max_rows=1000, max_rows_eval=100, seed=42):
    """Fresh procedural MentisOculi paper-fold questions with five native labeled panels.

    Source: https://github.com/Jana-Z/mentis-oculi. Uses new split-specific seeds,
    not the released benchmark puzzles. Preserves the generator's complete question
    and option-panel image. Final unfolded hole sets must remain mutually distinguishable;
    this rechecks the geometry after the generator's final unfolding step. Answers and
    generation traces stay in metadata. Source generator license: Apache-2.0.
    """
    root,archive_sha = checkout('Jana-Z/mentis-oculi',REVISION)
    path = root/'datasets/paper-fold/main.py'
    spec = importlib.util.spec_from_file_location('tasksource_mentis_paper_fold',path)
    native = importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    # Observe original final render coordinates without changing generation or images.
    captured = {}
    draw = native._plot_state
    def capture(*args,**kwargs):
        captured[Path(args[4]).name] = list(args[3])
        return draw(*args,**kwargs)
    native._plot_state = capture
    known_images = {hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('datasets/**/combined.png')}
    splits,excluded = {},[]
    for split,limit in (('validation',max_rows_eval),('train',max_rows)):
        rows = [];splits[split] = rows
        for index in range(limit):
            instance_seed = seed_for('mentis-paper-fold',seed,split,index)
            folds = (2,3,4)[index%3]
            try:
                captured.clear()
                with tempfile.TemporaryDirectory(dir=root.parent) as directory:
                    metadata = native.generate_sequence(n=3,min_steps=folds,max_steps=folds,
                        seed=instance_seed,save_dir=directory,puzzle_id=index,min_distinguishable_distance=0.3)
                    gold = captured[metadata['cot_images'][-1]]
                    points = [captured[f'wrong_{i}.png'] for i in range(4)]
                    label = metadata['correct_choice_index'];points.insert(label,gold)
                    if not native._are_answer_sets_distinguishable(points,0.3):raise ValueError('Indistinguishable final unfolded choices')
                    scores = [int(not native._is_sufficiently_different(p,gold,0.3)) for p in points]
                    if scores != [int(i==label) for i in range(5)]:raise ValueError('Multiple correct final patterns')
                    image = (Path(directory)/'combined.png').read_bytes()
                    if hashlib.sha256(image).hexdigest() in known_images:raise ValueError('Image matches a published source example')
                    rows.append(dict(id=f'mentis-paper-fold:{instance_seed}',image=image,
                        inputs=metadata['question']+' The top row shows successive folds from left to right, ending with the hole punch. Choose the unfolded pattern in panels A–E below.',
                        choices_list=[f'Panel {letter}' for letter in 'ABCDE'],labels=label,
                        metadata=dict(seed=instance_seed,source_revision=REVISION,num_folds=folds,grid_size=3,
                            correct_choice=metadata['correct_choice'],verifier_scores=scores,
                            final_choice_holes=[[[p.x,p.y] for p in choice] for choice in points],
                            choice_method='native_geometrically_verified_panels')))
            except (ValueError,KeyError,RuntimeError) as error:
                excluded.append(dict(split=split,seed=instance_seed,reason=str(error)))
            if (index+1)%10==0:print('Mentis paper-fold',split,index+1,'kept',len(rows),flush=True)
    report = dict(source=SOURCE,revision=REVISION,source_archive_sha256=archive_sha,
        license='apache-2.0',license_url=SOURCE+'/blob/'+REVISION+'/LICENSE',annotation_origin='procedural',
        configuration=dict(seed=seed,max_rows=max_rows,max_rows_eval=max_rows_eval,folds=[2,3,4],grid_size=3,min_distance=0.3),
        split_policy='Fresh split-specific generator seeds, held-out image copies excluded; no published puzzle ingestion.',
        known_source_example_images=len(known_images),generation_exclusions=excluded,
        exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
        rendering_versions={p:importlib.metadata.version(p) for p in ('Pillow','matplotlib','numpy','shapely')})
    return write_static('mentis-oculi-paper-fold',splits,report,mentis_paper_fold)
