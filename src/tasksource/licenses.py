"""Task licenses from Hub dataset cards and Data Provenance Initiative annotations.

A task's licenses are the ``license`` of the card of each repo it loads (and of
the original behind a tasksource copy), plus the licenses DPI records for it.
``license_use`` takes the most restrictive classified license: ``non-commercial``
if any is non-commercial or academic-only, else ``commercial`` if one allows
commercial use (share-alike and copyleft included), else ``unspecified``
(missing, ``other``, bare ``cc``, no-derivatives). Both snapshots are checked in
(``scripts/update_licenses.py``); nothing is fetched at import.

This is a best-effort filter, not legal advice: cards can be wrong or incomplete.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor

from .metadata.card_licenses import CARD_LICENSES
from .metadata.dpi_licenses import LICENSE_USE as DPI_USE, REPO_LICENSES as DPI_REPOS, TASK_LICENSES as DPI_TASKS

LICENSE_USES = ("commercial", "non-commercial", "unspecified")

# Hub card license ids (lowercase) whose terms allow commercial use; share-alike and
# copyleft included. Anything else (cc, other, no-derivatives, custom) stays unclassified.
PERMISSIVE_LICENSE = re.compile(
    r"^(apache|mit|bsd|cc0|cc-by-\d|cc-by-sa-\d|odc-by|odbl|pddl|afl|gpl|lgpl|agpl|artistic|cdla|ecl|epl|mpl"
    r"|isc|unlicense|wtfpl|bigscience|openrail|creativeml-openrail|llama2|llama3)")
NON_COMMERCIAL_LICENSE = re.compile(r"(^|-)nc(-|$)|academic|research")
DPI_LICENSE_USE = {"All": "commercial", "NC": "non-commercial", "Acad": "non-commercial"}


def card_license_use(license):
    if NON_COMMERCIAL_LICENSE.search(license):
        return "non-commercial"
    return "commercial" if PERMISSIVE_LICENSE.match(license) else None


def dpi_license_use(license):
    # "No License" means all rights reserved, whatever DPI's class says
    return None if license == "No License" else DPI_LICENSE_USE.get(DPI_USE.get(license))


def provenance_repos(info):
    """The Hub repos of a ``task_provenance`` record: loaded, read through hf://, originals."""
    return [r for r in [info.get("dataset"), *info.get("data_files_from", []), *info.get("originals", [])] if r]


def source_license(task_id, repos, cards=None):
    """``license``, ``license_use`` and the card and DPI licenses they come from.

    ``cards`` maps repo -> card licenses and defaults to the checked-in snapshot."""
    cards = CARD_LICENSES if cards is None else cards
    card = {repo: cards[repo] for repo in repos if cards.get(repo)}
    task = task_id.removeprefix("multilingual/")
    # DPI annotated older tasksource ids; a whole-dataset id also covers its configs
    dpi = sorted(set(DPI_TASKS.get(task, DPI_TASKS.get(task.split("/")[0], []))).union(
        *[DPI_REPOS.get(repo.lower(), []) for repo in repos]))
    uses = {card_license_use(license) for licenses in card.values() for license in licenses}
    uses |= {dpi_license_use(license) for license in dpi}
    use = "non-commercial" if "non-commercial" in uses else "commercial" if "commercial" in uses else "unspecified"
    names = sorted({license for licenses in card.values() for license in licenses}) + [f"{name} (DPI)" for name in dpi]
    return {"license": ", ".join(names) or "unspecified", "license_use": use,
            **({"card_licenses": card} if card else {}), **({"dpi_licenses": dpi} if dpi else {})}


def fetch_card_licenses(repos):
    """The current ``license`` of each Hub dataset card, as a list (``unknown`` dropped)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi()

    def read(repo):
        for attempt in range(3):
            try:
                card = api.dataset_info(repo).card_data
                break
            except (GatedRepoError, RepositoryNotFoundError):  # no readable card
                return repo, []
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        value = card.get("license") if card else None
        values = value if isinstance(value, list) else [value]
        return repo, [str(v).lower() for v in values if v and str(v).lower() != "unknown"]

    with ThreadPoolExecutor(16) as pool:
        return dict(pool.map(read, sorted(repos)))
