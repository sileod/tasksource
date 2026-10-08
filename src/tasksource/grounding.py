"""Box-based projections and deterministic SoM; no source-specific loading."""
import hashlib
import io
import json
import math
import random

from datasets import Image, Sequence
from PIL import Image as PILImage, ImageDraw, ImageFont


def normalized_box(box):
    """All shared geometry is normalized xyxy in the displayed image."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Expected four finite normalized xyxy coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Box outside displayed image or empty')
    return list(box)


def grid_labels(point, bins=(7, 7), depth=1):
    """Row/column labels from a normalized point, including nested cell refinements."""
    if (len(bins) != 2 or any(type(n) is not int or n <= 0 for n in bins)
            or type(depth) is not int or depth <= 0):
        raise ValueError('Grid dimensions and depth must be positive integers')
    x, y = point
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
        raise ValueError('Point coordinates must be finite and in [0, 1]')
    nx, ny = bins
    cells = []
    for _ in range(depth):
        c, r = min(int(x * nx), nx - 1), min(int(y * ny), ny - 1)
        cells.append((r, c))
        x, y = x * nx - c, y * ny - r
    return cells


def grid_center(cells, bins=(7, 7)):
    """Decode a cell path to its center in the original normalized image."""
    grid_labels((0, 0), bins, len(cells))  # validate grid and depth
    x = y = 0.5
    nx, ny = bins
    for r, c in reversed(cells):
        if not (type(r) is int and type(c) is int and 0 <= r < ny and 0 <= c < nx):
            raise ValueError('Cell outside grid')
        x, y = (c + x) / nx, (r + y) / ny
    return x, y


def grounding_row(images, instruction, target_bbox, *, metadata, candidates=None, bins=(7, 7)):
    """Project native geometry to a visual grid or MC row, without adding distractors.

    Candidates are {bbox: normalized xyxy, text: native description} in stable source
    order. The target must match exactly one supplied box; it is never injected.
    Source identity, split, revision and screenshot/trajectory IDs belong in metadata.
    """
    if not images or not instruction.strip():
        raise ValueError('Grounding requires images and a nonempty instruction')
    target = normalized_box(target_bbox)
    point = [(target[0] + target[2]) / 2, (target[1] + target[3]) / 2]
    info = {**metadata, 'target_bbox': target, 'bbox_units': 'normalized_xyxy', 'point': point}
    row = {'images': images, 'inputs': instruction}
    if candidates is None:
        r, c = grid_labels(point, bins)[0]
        row['labels'] = r * bins[0] + c
        info.update(bins=list(bins), stage=info.get('stage', 1))
    else:
        boxes = [normalized_box(candidate['bbox']) for candidate in candidates]
        if not 2 <= len(boxes) <= 26 or len({tuple(box) for box in boxes}) != len(boxes):
            raise ValueError('MC grounding needs 2–26 unique native candidate boxes')
        matches = [i for i, box in enumerate(boxes) if max(abs(a-b) for a, b in zip(box, target)) <= 1e-6]
        if len(matches) != 1:
            raise ValueError('Target must match one native candidate, without insertion')
        row.update(choices_list=[f'Element {i+1}: {candidate.get("text", "")}; box={json.dumps(boxes[i])}' for i, candidate in enumerate(candidates)],
                   labels=matches[0])
        info['candidate_boxes'] = boxes
    row['metadata'] = json.dumps(info, sort_keys=True)
    return row


def check_grounding_splits(dataset):
    """Reject shared screenshots or trajectories across preserved native splits."""
    seen = {}
    for split, rows in dataset.items():
        for row in rows.select_columns(['images', 'metadata']):
            info = json.loads(row['metadata'] or '{}')
            source = info.get('source_dataset', '')
            keys = [('image', hashlib.sha256(image['bytes']).hexdigest()) for image in row['images']]
            keys += [(key, source, str(info[key])) for key in ('image_group_id', 'trajectory_id') if key in info]
            for key in keys:
                if key in seen and seen[key] != split:
                    raise ValueError(f'Grounding {key[0]} crosses splits {seen[key]} and {split}')
                seen[key] = split


def augment_grounding(dataset, probabilities=None, seed=0, excluded_sources=(), mark_size=0.03, line_width=2):
    """Apply SoM after sampling and before recasting; mark IDs survive option shuffling.

    Probabilities map plain/som+text/som-only to weights summing to one. Exclusions
    are source-dataset identities, checked before any image pixels are read.
    Metadata and target geometry are never rendered as model inputs. Only the first
    displayed image is annotated. Other images keep their encoded bytes/order.
    """
    weights = {'plain': 0, 'som+text': 1, 'som-only': 0} if probabilities is None else dict(sorted(probabilities.items()))
    if (not weights or set(weights) - {'plain', 'som+text', 'som-only'}
            or any(not math.isfinite(v) or v < 0 for v in weights.values())
            or not math.isclose(sum(weights.values()), 1)):
        raise ValueError('SoM variant probabilities must be nonnegative and sum to one')
    if not math.isfinite(mark_size) or not 0 < mark_size <= .2 or type(line_width) is not int or line_width < 1:
        raise ValueError('Invalid mark size or line width')
    excluded = set(excluded_sources)
    features = dataset['train'].features['images']
    dataset = dataset.cast_column('images', Sequence(Image(decode=False)))
    check_grounding_splits(dataset)

    def augment(row):
        info = json.loads(row.get('metadata') or '{}')
        source = info.get('source_dataset')
        if source in excluded:
            raise ValueError(f'Excluded grounding source: {source}')
        boxes = [normalized_box(box) for box in info.get('candidate_boxes', [])]
        if len({tuple(box) for box in boxes}) != len(boxes):
            raise ValueError('SoM requires distinct candidate boxes')
        if not boxes:
            raise ValueError('SoM requires native candidate_boxes, never target-only boxes')
        identity = info.get('source_row', info.get('id'))
        if identity is None:
            raise ValueError('Grounding augmentation requires stable source_row or id')
        digest = hashlib.sha256(json.dumps([seed, source, identity], sort_keys=True).encode()).hexdigest()
        variant = random.Random(int(digest, 16)).choices(list(weights), weights=list(weights.values()))[0]
        columns = sorted((key for key in row if key.startswith('choice') and key[6:].isdigit()),
                         key=lambda key: int(key[6:]))
        present = [key for key in columns if row[key] is not None]
        if columns and len(present) != len(boxes):
            raise ValueError('Candidate boxes and present choices disagree')
        if columns:
            gold = int(row['labels'])
            target = normalized_box(info['target_bbox'])
            matches = [i for i, box in enumerate(boxes) if max(abs(a-b) for a, b in zip(box, target)) <= 1e-6]
            if matches != [gold]:
                raise ValueError('Gold label and native target box disagree')
        updates = {}
        pixel_boxes = []
        image_size = None
        if variant != 'plain':
            image = row['images'][0]
            with PILImage.open(io.BytesIO(image['bytes']) if image.get('bytes') is not None else image['path']) as decoded:
                canvas = decoded.convert('RGB')
            draw = ImageDraw.Draw(canvas)
            width, height = canvas.size
            image_size = [width, height]
            font = ImageFont.load_default()
            for number, box in enumerate(boxes, 1):
                rect = [min(width-1, int(box[0]*width)), min(height-1, int(box[1]*height)),
                        min(width-1, max(0, math.ceil(box[2]*width)-1)), min(height-1, max(0, math.ceil(box[3]*height)-1))]
                pixel_boxes.append(rect)
                draw.rectangle(rect, outline='red', width=line_width)
                text = str(number)
                bounds = draw.textbbox((0, 0), text, font=font)
                label = PILImage.new('RGB', (bounds[2]+4, bounds[3]+4), 'white')
                ImageDraw.Draw(label).text((2, 2), text, fill='red', font=font)
                size = max(label.height, round(min(width, height)*mark_size))
                label = label.resize((round(label.width*size/label.height), size))
                x, y = rect[:2]
                canvas.paste(label, (min(x, max(0, width-label.width)), min(y, max(0, height-label.height))))
            buffer = io.BytesIO()
            canvas.save(buffer, format='PNG')
            updates['images'] = [{'bytes': buffer.getvalue(), 'path': None}] + row['images'][1:]
            for number, key in enumerate(present, 1):
                # Stable mark IDs are content, distinct from Jev's permuted option indices.
                updates[key] = f'Mark {number}' + (f': {row[key]}' if variant == 'som+text' else '')
        info['augmentation'] = {'version': 1, 'variant': variant, 'probabilities': weights,
            'seed': seed, 'identity_hash': digest, 'image_index': 0, 'encoding': 'unchanged' if variant == 'plain' else 'PNG',
            'mark_size': mark_size, 'line_width': line_width, 'pixel_boxes': pixel_boxes, 'image_size': image_size, 'pillow_version': PILImage.__version__,
            'mark_ids': list(range(1, len(boxes)+1)), 'candidate_boxes': boxes,
            'original_image_sha256': hashlib.sha256(row['images'][0]['bytes']).hexdigest()}
        updates['metadata'] = json.dumps(info, sort_keys=True)
        return updates

    return dataset.map(augment).cast_column('images', features)
