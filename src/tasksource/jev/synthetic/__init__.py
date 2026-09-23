"""Synthetic Jev dataset-construction package.

Canonical unit is a *state bundle*: one generated state plus 1-N Jev
questions (choice / noul / score) over that same state. Both bundled
(one row per state) and flattened (one row per decision) views are
exported; the flat view follows the tasksource-jev training schema
(id, kind, options, target, state, question, source, variant, split).
"""

from .config import AppConfig, load_config
from .schemas import bundle_to_flat_rows, flat_to_training_row

__all__ = ["AppConfig", "load_config", "bundle_to_flat_rows", "flat_to_training_row"]
