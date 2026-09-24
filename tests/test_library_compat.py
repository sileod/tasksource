import unittest

from tasksource import list_tasks

BUILDERS = {"csv", "json", "parquet", "text"}


class LibraryCompatTest(unittest.TestCase):
    def test_missing_names_stay_none(self):
        # pandas 3 turns None into NaN in string columns and on explode
        for multilingual in (False, True):
            df = list_tasks(multilingual=multilingual)
            self.assertFalse(df.id.str.contains(r"(^|/)nan(/|$)").any())
            self.assertFalse(df.config_name.map(lambda x: isinstance(x, float)).any())

    def test_repos_have_a_namespace(self):
        # huggingface_hub>=1 rejects repo ids without a namespace
        for multilingual in (False, True):
            names = set(list_tasks(multilingual=multilingual).dataset_name) - BUILDERS
            self.assertEqual({n for n in names if "/" not in n}, set())


if __name__ == "__main__":
    unittest.main()
