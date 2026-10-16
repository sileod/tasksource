"""The tasksource citation, shared by the README, dataset cards and card-writing scripts."""
import re

SENTENCE = ('If you use tasksource, its standardized task annotations, or datasets derived from them in published '
            'research, please cite the tasksource paper (Sileo, LREC-COLING 2024).')

BIBTEX = '''@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361",
    doi = "10.63317/2h4pkr685mkq",
    pages = "15655--15684",
}'''


def card_section(level=2, title='Citation'):
    return f"{'#' * level} {title}\n\n{SENTENCE}\n\n```bibtex\n{BIBTEX}\n```\n"


def with_citation(text):
    """Card body with one current tasksource citation. A section that already cites tasksource is
    replaced; a citation of the original data stays, and the tasksource one goes in its own section."""
    pattern = r'^(#{2,3}) (Citation|Citing tasksource)[^\n]*\n.*?(?=^#{1,3} |\Z)'
    for section in re.finditer(pattern, text, re.M | re.S):
        if 'sileo-2024-tasksource' in section.group(0):
            rest = text[section.end():]
            return (text[:section.start()] + card_section(len(section.group(1)), section.group(2))
                    + ('\n' if rest else '') + rest)
    title = 'Citing tasksource' if re.search(r'^#{2,3} Citation', text, re.M) else 'Citation'
    return text.rstrip() + '\n\n' + card_section(title=title)
