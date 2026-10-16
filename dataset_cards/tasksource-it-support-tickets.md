---
license: cc-by-4.0
language:
- en
- de
- pt
- es
task_categories:
- text-classification
size_categories:
- 1K<n<10K
pretty_name: Classification of IT Support Tickets
---

# Classification of IT Support Tickets

This dataset contains **real support tickets** collected from an IT support
company in the Florianópolis region of Brazil. It contains 2,229 ticket texts,
manually classified into seven categories by three IT support professionals.
The source authors' train/test split is preserved exactly; Tasksource joins
each text and label by the original ID without reshuffling or editing text.

The tickets are mainly in English, German, Portuguese, and Spanish, with other
languages also present. Personally identifiable and sensitive information
was removed by the authors using AWS Comprehend PII Removal, custom regular
expressions, and manual review. Placeholder tags such as `[NAME]`, `[TICKET
ID]`, and `[LOCATION]` indicate masked content.

## Schema

- `text`: original ticket text (`string`)
- `label`: one of the seven source categories (`ClassLabel`)

The source IDs are used to validate and join the paired X/y files, then omitted
from this two-column release. Only accidental CSV index columns are removed.

## Source and license

- Original title: *Classification of IT Support Tickets*
- Author: Leonardo Santiago Benitez Pereira, Federal Institute of Santa Catarina
- DOI: [10.5281/zenodo.7648117](https://doi.org/10.5281/zenodo.7648117)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Citation

```bibtex
@dataset{benitez_pereira_2022_it_support_tickets,
  author       = {Leonardo Santiago Benitez Pereira},
  title        = {Classification of IT Support Tickets},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7648117},
  url          = {https://doi.org/10.5281/zenodo.7648117}
}
```
