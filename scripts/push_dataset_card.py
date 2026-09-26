"""Upload a dataset card, keeping the metadata that push_to_hub writes on the Hub.

``push_to_hub`` adds ``configs`` and ``dataset_info`` to the Hub README; a card
uploaded from ``dataset_cards/`` does not have them and would drop them. This
takes them from the current Hub card, or from the latest earlier revision that
has them.

    python scripts/push_dataset_card.py dataset_cards/tasksource-jev.md tasksource/tasksource-jev-typed-decisions
"""

import argparse

from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

KEPT = ("configs", "dataset_info")


def hub_metadata(api, repo_id):
    for commit in api.list_repo_commits(repo_id, repo_type="dataset"):
        try:
            path = hf_hub_download(repo_id, "README.md", repo_type="dataset", revision=commit.commit_id)
        except EntryNotFoundError:  # a revision without a README
            continue
        data = DatasetCard.load(path).data.to_dict()
        kept = {key: data[key] for key in KEPT if key in data}
        if len(kept) == len(KEPT):
            return kept
    return {}


def push_card(card_path, repo_id, message="Update dataset card"):
    api = HfApi()
    card = DatasetCard.load(card_path)
    for key, value in hub_metadata(api, repo_id).items():
        card.data[key] = value
    card.push_to_hub(repo_id, repo_type="dataset", commit_message=message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("card")
    parser.add_argument("repo_id")
    parser.add_argument("--message", default="Update dataset card")
    args = parser.parse_args()
    push_card(args.card, args.repo_id, args.message)
