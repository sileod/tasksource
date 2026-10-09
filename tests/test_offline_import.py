"""Importing tasksource and listing tasks must not query the Hub or the network."""

import subprocess
import sys
import unittest
from pathlib import Path

CODE = """
import datasets, socket
def refuse(*args, **kwargs):
    raise RuntimeError("network access during import")
datasets.get_dataset_config_names = refuse
socket.socket.connect = refuse
socket.create_connection = refuse
import tasksource
from tasksource import list_tasks, parked, eval_only
print(len(list_tasks()), len(list_tasks(multilingual=True)))
"""


class OfflineImportTest(unittest.TestCase):
    def test_text_catalog_does_not_import_vision_or_pillow(self):
        code = '''
import datasets, builtins, sys
original = builtins.__import__
def without_pillow(name, *args, **kwargs):
    if name == 'PIL' or name.startswith('PIL.'):
        raise ModuleNotFoundError("No module named 'PIL'")
    return original(name, *args, **kwargs)
builtins.__import__ = without_pillow
import tasksource
assert len(tasksource.list_tasks())
assert len(tasksource.list_tasks(multilingual=True))
assert 'tasksource.vision_tasks' not in sys.modules
assert 'tasksource.grounding' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_import_and_catalog_need_no_network(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                                env={"PYTHONPATH": str(root / "src"), "HF_HUB_OFFLINE": "1", "PATH": ""})
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
