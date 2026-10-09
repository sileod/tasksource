"""Task licenses from Hub cards and Data Provenance Initiative annotations."""

import unittest

from tasksource import list_tasks, task_licenses
from tasksource.licenses import source_license


class SourceLicenseTest(unittest.TestCase):
    def test_original_visual_data_terms_fill_missing_cards(self):
        for task in ('vision/clevr/count', 'clevr/color', 'vision/mapqa/yesno', 'vision/tqa'):
            info = source_license(task, ['HuggingFaceM4/the_cauldron'], {})
            self.assertEqual(info['license_use'], 'commercial')
            self.assertIn('https://', info['source_license_evidence']['url'])
        restricted = source_license('vision/clevr/count', ['test/restricted'], {'test/restricted': ['cc-by-nc-4.0']})
        self.assertEqual(restricted['license_use'], 'non-commercial')
        # Software terms do not fill missing photo rights.
        self.assertEqual(source_license('vision/nlvr2', ['pingzhili/nlvr2'], {})['license_use'], 'unspecified')

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


def test_task_weight_multiplies_hand_and_audit_weights():
    from tasksource.metadata.weights import task_weight
    from tasksource.metadata.audit_weights import AUDIT_WEIGHTS

    assert task_weight("UNLI") == 8 * AUDIT_WEIGHTS.get("UNLI", 1)
    assert task_weight("english-grading/syntax") == 0.5
    assert task_weight("tomi-nli") == 1.5
    assert task_weight("glue/rte") == 1


def test_annotation_license_does_not_clear_image_terms():
    for task in ('aokvqa', 'nlvr2', 'tallyqa/count', 'vsr/yesno', 'intergps', 'spair71k/grid7'):
        info = source_license('vision/' + task, ['mirror/data'], {'mirror/data': ['mit']})
        assert info['license_use'] == 'unspecified'
        assert info['license_review']['unresolved']
        assert info['license_review']['annotations']['license'] != 'unspecified'
    assert source_license('vision/vsr/yesno', [], {})['license_review']['annotations']['license'] == 'cc-by-4.0'


def test_research_image_restrictions_and_no_redistribution_are_recorded():
    for task in ('figureqa', 'bapps/preference'):
        info = source_license('vision/' + task, ['mirror/data'], {'mirror/data': ['mit']})
        assert info['license_use'] == 'non-commercial'
        assert info['license_review']['images']['license']
    figure = source_license('vision/figureqa', [], {})['license_review']
    assert figure['redistribution'] == 'prohibited'
    assert figure['archive_member'].endswith('.pdf')


def test_review_does_not_weaken_known_noncommercial_terms():
    info = source_license('vision/aokvqa', ['upstream'], {'upstream': ['cc-by-nc-4.0']})
    assert info['license_use'] == 'non-commercial'
