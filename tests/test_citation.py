"""Cards carry one current tasksource citation and keep citations of the original data."""
from tasksource.citation import BIBTEX, with_citation


def test_adds_refreshes_and_keeps_original_citations():
    plain = '# data\n\nSome card.\n'
    added = with_citation(plain)
    assert added.count('sileo-2024-tasksource') == 1 and with_citation(added) == added

    stale = '# data\n\n## Citation\n\n```bibtex\n@inproceedings{sileo-2024-tasksource,\n year = 2024}\n```\n\n## License\n\nMIT\n'
    refreshed = with_citation(stale)
    assert BIBTEX in refreshed and refreshed.endswith('## License\n\nMIT\n') and refreshed.count('## Citation') == 1

    original = '# data\n\n## Citation\n\n```bibtex\n@dataset{orig2022, title = {Original}}\n```\n'
    both = with_citation(original)
    assert '@dataset{orig2022' in both and '## Citing tasksource' in both and with_citation(both) == both
