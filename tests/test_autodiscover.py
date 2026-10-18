import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from uuid import uuid4

from core.register.autodiscover import autodiscover_components


class AutodiscoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package_name = "discovery_test_" + uuid4().hex
        self.root = Path(self.temp.name) / self.package_name
        self.root.mkdir()
        sys.path.insert(0, self.temp.name)

    def tearDown(self):
        sys.path.remove(self.temp.name)
        for name in list(sys.modules):
            if name == self.package_name or name.startswith(self.package_name + "."):
                del sys.modules[name]
        self.temp.cleanup()

    def write_module(self, name, source=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def test_recurses_into_namespace_and_regular_packages(self):
        self.write_module("models/nested/model.py", "class Model: pass\n")
        self.write_module("tasks/__init__.py", "initialized = True\n")
        self.write_module("tasks/run.py", "class Task: pass\n")

        names = autodiscover_components((self.package_name,))

        self.assertEqual(names, [
            self.package_name,
            self.package_name + ".models.nested.model",
            self.package_name + ".tasks",
            self.package_name + ".tasks.run",
        ])
        task_package = importlib.import_module(self.package_name + ".tasks")
        self.assertTrue(task_package.initialized)

    def test_skips_private_files_and_directories(self):
        fail = "raise RuntimeError('private module must not be imported')\n"
        self.write_module("_private.py", fail)
        self.write_module("__main__.py", fail)
        self.write_module("_private/model.py", fail)
        self.write_module(".hidden/model.py", fail)
        self.write_module("public.py")

        self.assertEqual(autodiscover_components((self.package_name,)), [
            self.package_name,
            self.package_name + ".public",
        ])

    def test_repeated_discovery_does_not_repeat_registration(self):
        self.write_module("model.py", "registrations = []\nregistrations.append(object())\n")
        first = autodiscover_components((self.package_name,))
        module = importlib.import_module(self.package_name + ".model")
        registration = module.registrations[0]

        self.assertEqual(autodiscover_components((self.package_name,)), first)
        self.assertEqual(module.registrations, [registration])

    def test_import_failure_names_module_and_preserves_cause(self):
        self.write_module("broken.py", "raise RuntimeError('broken component')\n")

        with self.assertRaises(ImportError) as error:
            autodiscover_components((self.package_name,))

        self.assertIn(self.package_name + ".broken", str(error.exception))
        self.assertIsInstance(error.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
