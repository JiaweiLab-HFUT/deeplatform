from contextlib import chdir
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import components.datafactory.BatteryLifeDataset as split_module
from components.datafactory.BatteryLifeDataset import BatteryLifeDataSplit


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = {'train': ['b.pkl', 'a.pkl'], 'val': [], 'test': ['c.pkl']}

    def write(self, filename, data):
        path = self.root / filename
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def test_loads_multiple_json_files_and_preserves_lists_and_empty_groups(self):
        self.write('z.json', {'dataset': 'ONE', **self.split})
        self.write('a.json', {'dataset': 'TWO', 'train': [], 'val': ['v.pkl'], 'test': []})
        self.write('m.json', {'dataset': 'THREE', **self.split})
        records = BatteryLifeDataSplit(self.root).split_record
        self.assertEqual(list(records), ['TWO', 'THREE', 'ONE'])
        self.assertEqual(records['ONE'], self.split)
        self.assertEqual(records['TWO']['train'], [])

    def test_instances_have_separate_records_and_reload_new_json_files(self):
        self.write('one.json', {'dataset': 'ONE', **self.split})
        first = BatteryLifeDataSplit(self.root)
        first.split_record['ONE']['train'].append('memory_only.pkl')
        self.write('two.json', {'dataset': 'TWO', **self.split})
        second = BatteryLifeDataSplit(self.root)
        self.assertNotIn('TWO', first.split_record)
        self.assertEqual(second.split_record['ONE']['train'], ['b.pkl', 'a.pkl'])
        self.assertIn('TWO', second.split_record)

    def test_default_directory_is_relative_to_project_not_working_directory(self):
        project = self.root / 'project'
        directory = project / 'datasets' / 'batteryLife' / 'data_split'
        directory.mkdir(parents=True)
        (directory / 'one.json').write_text(json.dumps({'dataset': 'ONE', **self.split}), encoding='utf-8')
        source = project / 'components' / 'datafactory' / 'BatteryLifeDataset.py'
        with patch.object(split_module, '__file__', str(source)), chdir(self.root):
            loaded = BatteryLifeDataSplit()
        self.assertEqual(loaded.split_dir, directory)
        self.assertEqual(loaded.split_record, {'ONE': self.split})

    def test_missing_or_empty_directory_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, '数据划分目录'):
            BatteryLifeDataSplit(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, '没有 JSON 文件'):
            BatteryLifeDataSplit(self.root)

    def test_duplicate_datasets_across_files_name_both_sources(self):
        self.write('one.json', {'dataset': 'ONE', **self.split})
        self.write('two.json', {'dataset': 'ONE', **self.split})
        with self.assertRaises(ValueError) as error:
            BatteryLifeDataSplit(self.root)
        for text in ['ONE', 'one.json', 'two.json', '重复']:
            self.assertIn(text, str(error.exception))

    def test_invalid_or_duplicate_json_keys_are_reported_with_filename(self):
        for text in ['{', '{"dataset": "ONE", "dataset": "TWO", "train": [], "val": [], "test": []}',
                     '{"dataset": "ONE", "train": [], "train": [], "val": [], "test": []}']:
            with self.subTest(text=text):
                self.write('bad.json', {})
                (self.root / 'bad.json').write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'bad.json.*无法解析'):
                    BatteryLifeDataSplit(self.root)

    def test_invalid_descriptions_are_rejected(self):
        missing = deepcopy(self.split)
        del missing['test']
        unknown = dict(self.split, validation=[])
        cases = [[], {}, {'ONE': self.split}, self.split,
                 {'dataset': 'ONE', **missing}, {'dataset': 'ONE', **unknown}]
        for name in [None, 1, '', '  ']:
            cases.append({'dataset': name, **self.split})
        for files in ['a.pkl', [1], [''], ['  ']]:
            cases.append({'dataset': 'ONE', **dict(self.split, train=files)})
        for data in cases:
            with self.subTest(data=data):
                self.write('bad.json', data)
                with self.assertRaisesRegex(ValueError, 'bad.json'):
                    BatteryLifeDataSplit(self.root)

    def test_duplicate_files_and_cross_partition_overlap_are_rejected(self):
        for split in [dict(self.split, train=['a.pkl', 'a.pkl']),
                      dict(self.split, val=['a.pkl']),
                      dict(self.split, val=['c.pkl'])]:
            with self.subTest(split=split):
                self.write('bad.json', {'dataset': 'ONE', **split})
                with self.assertRaisesRegex(ValueError, 'bad.json.*(重复|同时出现在)'):
                    BatteryLifeDataSplit(self.root)

    def test_data_factory_uses_loaded_calce_and_mix_large_records(self):
        from components.datafactory.BatteryLifeDataset import BatteryLifeDatasetFactory

        records = BatteryLifeDataSplit().split_record
        factory = BatteryLifeDatasetFactory.__new__(BatteryLifeDatasetFactory)
        for name in ['CALCE', 'MIX_large']:
            with self.subTest(dataset=name):
                self.assertEqual(factory.get_split_files(name),
                                 tuple(records[name][flag] for flag in ('train', 'val', 'test')))


if __name__ == '__main__':
    unittest.main()
