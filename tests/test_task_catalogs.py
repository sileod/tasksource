from types import SimpleNamespace

from scripts.write_task_catalogs import dataset_link


def test_json_catalog_links_hub_data_files_without_changing_loader():
    mapping=SimpleNamespace(load_dataset_kwargs={'data_files':{
        'train':'hf://datasets/relbert/lexical_relation_classification/dataset/{config_name}/train.jsonl'}})
    link=dataset_link('json',mapping,'lexical_relation_classification/BLESS')
    assert '[relbert/lexical_relation_classification](https://hf.co/datasets/relbert/lexical_relation_classification)' in link
    assert '(`json` loader)' in link
    assert mapping.load_dataset_kwargs['data_files']['train'].startswith('hf://')


def test_catalog_preserves_direct_sources_and_does_not_invent_missing_repos():
    assert dataset_link('tasksource/discosense')=='[tasksource/discosense](https://hf.co/datasets/tasksource/discosense)'
    assert dataset_link('json',SimpleNamespace(load_dataset_kwargs={}))=='json'
