"""Task licenses from Hub cards and Data Provenance Initiative annotations."""

import unittest

from tasksource import list_tasks, task_licenses
from tasksource.licenses import source_license


class SourceLicenseTest(unittest.TestCase):
    def test_most_restrictive_license_wins(self):
        repos = ["a/copy", "b/original"]
        self.assertEqual(source_license("x", repos, {"a/copy": ["mit"], "b/original": ["cc-by-nc-4.0"]})["license_use"],
                         "non-commercial")
        self.assertEqual(source_license("x", repos, {"a/copy": ["mit"]})["license_use"], "commercial")
        for unclassified in (["other"], ["cc"], ["cc-by-nd-4.0"], []):
            self.assertEqual(source_license("x", repos, {"a/copy": unclassified})["license_use"], "unspecified")

    def test_dpi_annotations(self):
        # DPI records the SILICONE corpora as non-commercial although the Hub card says CC BY-SA
        meld = source_license("silicone/meld_e", ["eusip/silicone"], {"eusip/silicone": ["cc-by-sa-4.0"]})
        self.assertEqual((meld["license_use"], meld["license"]), ("non-commercial", "cc-by-sa-4.0, CC BY-NC-SA 4.0 (DPI)"))
        # an annotation of the whole dataset covers its configs
        self.assertEqual(source_license("hh-rlhf/helpful-base", ["tasksource/hh-rlhf"], {})["license_use"], "commercial")
        self.assertEqual(source_license("x", [], {}), {"license": "unspecified", "license_use": "unspecified"})


class CatalogTest(unittest.TestCase):
    def test_on_demand(self):
        self.assertNotIn("license_use", list_tasks().columns)  # computed only when asked for
        commercial = list_tasks(license_use="commercial")
        self.assertEqual(set(commercial.license_use), {"commercial"})
        self.assertIn("glue/rte", set(list_tasks(license_use=["commercial", "unspecified"]).id))  # GLUE: "other"
        with self.assertRaises(ValueError):
            list_tasks(license_use="free")

    def test_task_licenses(self):
        table = task_licenses(["glue/rte", "hh-rlhf/helpful-base"]).set_index("id")
        self.assertEqual(table.loc["glue/rte", "card_licenses"], {"nyu-mll/glue": ["other"]})
        self.assertEqual(table.loc["hh-rlhf/helpful-base", "license_use"], "commercial")
        self.assertEqual(set(task_licenses(multilingual=True).license_use) - {"commercial", "non-commercial", "unspecified"},
                         set())


if __name__ == "__main__":
    unittest.main()
