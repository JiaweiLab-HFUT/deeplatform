import copy
import json
import os
import pickle
from pathlib import Path

from tqdm import tqdm

from denseweight import DenseWeight
import torch
from core.register.registryhub import R
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.augmentation import BatchAugmentation_battery_revised
import numpy as np
import shutil
import pandas as pd
from utils.sync_flat_symlinks import sync_flat_symlinks


def _unique_split_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f'重复 JSON 字段：{key!r}')
        values[key] = value
    return values


class BatteryLifeDataSplit:
    """加载每个 JSON 中的 dataset、train、val、test 字段。"""

    def __init__(self, split_dir=None):
        self.split_dir = (
            Path(__file__).resolve().parents[2] / 'datasets' / 'batteryLife' / 'data_split'
            if split_dir is None else Path(split_dir).resolve()
        )
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f'数据划分目录不存在或不是目录：{self.split_dir}')
        paths = sorted(self.split_dir.glob('*.json'))
        if not paths:
            raise ValueError(f'数据划分目录中没有 JSON 文件：{self.split_dir}')

        records, owners = {}, {}
        for path in paths:
            try:
                with path.open(encoding='utf-8') as stream:
                    data = json.load(stream, object_pairs_hook=_unique_split_keys)
            except ValueError as exc:
                raise ValueError(f'{path}: 无法解析数据划分 JSON：{exc}') from exc
            dataset, split = self._validate_split(path, data)
            if dataset in records:
                raise ValueError(f'{path}: 数据集 {dataset!r} 与 {owners[dataset]} 重复')
            records[dataset] = split
            owners[dataset] = path
        self.split_record = records

    @staticmethod
    def _validate_split(path, data):
        if not isinstance(data, dict):
            raise ValueError(f'{path}: 顶层必须是包含 dataset、train、val、test 的字典')
        required = {'dataset', 'train', 'val', 'test'}
        if set(data) != required:
            missing = sorted(required - set(data))
            unknown = sorted(set(data) - required)
            raise ValueError(f'{path}: 必须包含 dataset、train、val、test；缺少 {missing}，未知字段 {unknown}')
        dataset = data['dataset']
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f'{path}: dataset 必须是非空字符串')
        location = f'{path}: {dataset}'
        owners, split = {}, {}
        for flag in ('train', 'val', 'test'):
            files = data[flag]
            if not isinstance(files, list):
                raise ValueError(f'{location}.{flag}: 必须是文件名列表')
            for index, filename in enumerate(files):
                if not isinstance(filename, str) or not filename.strip():
                    raise ValueError(f'{location}.{flag}[{index}]: 文件名必须是非空字符串')
                if filename in owners:
                    previous = owners[filename]
                    if previous == flag:
                        raise ValueError(f'{location}.{flag}: 文件 {filename!r} 重复')
                    raise ValueError(f'{location}: 文件 {filename!r} 同时出现在 {previous} 和 {flag}')
                owners[filename] = flag
            split[flag] = files
        return dataset, split


@R.dataFactory.register
class BatteryLifeDatasetFactory:
    def __init__(self, dataset, eval_cycle_max, eval_cycle_min, root_path, seq_len, charge_discharge_len, early_cycle_threshold,
                 weighted_loss, batch_size, num_workers = 8):
        self.label_scaler = None
        self.raw_labels = None
        self.weights = None
        self.life_class_scaler = None
        self.scaled_life_classes = None
        self.total_filenames = None
        self.total_seen_unseen_IDs = None
        self.total_cj_aug_charge_discharge_curves = None
        self.total_dataset_ids = None
        self.class_labels = None
        self.unique_labels = None
        self.total_labels = None
        self.total_curve_attn_masks = None
        self.total_charge_discharge_curves = None
        self.unseen_seen_record = None
        self.files = None
        self.flag = None
        self.weighted_loss = weighted_loss
        self.dataset = dataset
        self.eval_cycle_min = eval_cycle_min
        self.eval_cycle_max = eval_cycle_max
        self.root_path = root_path
        self.seq_len = seq_len
        self.charge_discharge_len = charge_discharge_len
        self.early_cycle_threshold = early_cycle_threshold
        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah',
                          'discharge_capacity_in_Ah', 'time_in_s']
        with Path(__file__).with_name('life_classes.json').open(encoding='utf-8') as fp:
            self.life_classes = json.load(fp)
        self.KDE_samples = []
        self.aug_helper = BatchAugmentation_battery_revised()
        self.train_files, self.val_files, self.test_files = self.get_split_files(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        #将seen_unseen_labelsseen_unseen_labels的文件合并在一起
        seen_unseen_folder = Path(root_path + '/seen_unseen_labels')
        merged = {}
        for f in sorted(seen_unseen_folder.glob("*.json")):
            if f.name == "all_seen_unseen_labels.json":  # 跳过输出文件本身
                continue
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            merged.update(data)
        with open(seen_unseen_folder / "all_seen_unseen_labels.json", "w", encoding="utf-8") as fp:
            json.dump(merged, fp, ensure_ascii=False, indent=2)
        #软链接合并所有的电池文件
        src = '/data/BatteryLife_Processed'
        dst = self.root_path + '/data'
        exclude = ['Life labels', '__pycache__', 'life_bank_cache', 'seen_unseen_labels', 'READMEs']
        sync_flat_symlinks(src, dst, exclude)
        #合并所有标签
        self.all_file_life_labels = {}
        labels_folder = Path(root_path + '/Life labels')
        for f in sorted(labels_folder.glob("*.json")):
            with open(f, encoding="utf-8") as fp:
                life_label = json.load(fp)
            self.all_file_life_labels.update(life_label)


    def get_split_files(self, dataset):
        split = BatteryLifeDataSplit()
        train = split.split_record.get(dataset)['train']
        val = split.split_record.get(dataset)['val']
        test = split.split_record.get(dataset)['test']
        return train, val, test


    def get_dataset_loader(self, flag, label_scaler=None, life_class_scaler=None):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        if flag == 'train':
            self.files = [i for i in self.train_files]
        elif flag == 'val':
            self.files = [i for i in self.val_files]
        elif flag == 'test':
            self.files = [i for i in self.test_files]
            self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/all_seen_unseen_labels.json'))
        (self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.unique_labels, self.class_labels,
         self.total_dataset_ids, self.total_cj_aug_charge_discharge_curves, self.total_seen_unseen_IDs, self.total_filenames) = self.read_data()
        self.KDE_samples = copy.deepcopy(self.total_labels) if flag == 'train' else []
        self.weights = self.get_loss_weight()
        if flag == 'train' and label_scaler is None:
            self.label_scaler = StandardScaler()
            self.life_class_scaler = StandardScaler()
            self.label_scaler.fit(np.array(self.unique_labels).reshape(-1, 1))
            self.life_class_scaler.fit(np.array(self.class_labels).reshape(-1, 1))
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1, 1))
            self.scaled_life_classes = np.array(self.class_labels) - 1
        else:
            # validation set or testing set
            assert label_scaler is not None
            self.label_scaler = label_scaler
            self.life_class_scaler = life_class_scaler
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1,1))
            self.scaled_life_classes = np.array(self.class_labels) - 1

        dataset = BatteryLifeDataset(self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.class_labels,
                 self.total_dataset_ids, self.total_cj_aug_charge_discharge_curves, self.total_seen_unseen_IDs,self.scaled_life_classes,self.weights,
                 self.label_scaler, self.life_class_scaler, self.total_filenames)

        if flag == 'test' or flag == 'val':
            shuffle_flag = False
            drop_last = False

        else:
            shuffle_flag = True
            drop_last = True

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last,
            collate_fn=batteryLife_collate_fn_baseline
        )

        return dataset, data_loader

    def get_loss_weight(self, method='KDE'):
        '''
        Get the weight for weighted loss
        method can be ['1/n', '1/log(x+1)', 'KDE']
        '''
        if self.weighted_loss and self.flag == 'train':
            if method == '1/n':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                weights = 1.0 / label_to_count[df["label"]].values
            elif method == '1/log(x+1)':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x) + 1)
                weights = 1 / normalized_x
            elif method == 'KDE':
                # Define DenseWeight
                dw = DenseWeight(alpha=1.0)
                # Fit DenseWeight and get the weights for the 1000 samples
                dw.fit(self.KDE_samples)
                # Calculate the weight for an arbitrary target value
                weights = []
                for label in self.KDE_samples:
                    single_sample_weight = dw(label)[0]
                    weights.append(single_sample_weight)
            else:
                raise Exception('Not implemented')
            return weights
        else:
            return np.ones(len(self.total_charge_discharge_curves))

    def return_label_scaler(self):
        return self.label_scaler

    def return_life_class_scaler(self):
        return self.life_class_scaler

    def __len__(self):
        return len(self.total_labels)

    def read_data(self):
        '''
        read all data from files
        :return: x_enc, x_cycle_numbers, prompts, charge_data, discharge_data, RPT_masks, labels
        '''

        total_charge_discharge_curves = []
        total_curve_attn_masks = []
        total_labels = []  # RUL
        unique_labels = []
        class_labels = []  # the pseudo class for samples
        total_dataset_ids = []
        total_cj_aug_charge_discharge_curves = []
        total_seen_unseen_IDs = []
        total_filenames = []

        for file_name in tqdm(self.files):
            # if file_name not in BatteryLifeDataSplit.MICH_EXP_test_files and file_name not in BatteryLifeDataSplit.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
            #     dataset_id = datasetName2ids[file_name.split('_')[0]]
            # else:
            #     dataset_id = datasetName2ids['MICH_EXP']
            dataset_id = datasetName2ids[file_name.split('_')[0]]
            charge_discharge_curves, attn_masks, labels, eol, cj_aug_charge_discharge_curves = self.read_samples_from_one_cell(
                file_name)
            if eol is None:
                # This battery has not reached end of life
                continue

            for class_label, life_range in self.life_classes.items():
                if eol >= life_range[0] and eol < life_range[1]:
                    class_label = int(class_label)
                    class_labels += [class_label for _ in range(len(charge_discharge_curves))]
                    break

            total_charge_discharge_curves += charge_discharge_curves
            total_cj_aug_charge_discharge_curves += cj_aug_charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            total_filenames += [file_name] * len(labels)
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)
            if self.flag == 'test':
                seen_unseen_id = self.unseen_seen_record[file_name]
                if seen_unseen_id == 'unseen':
                    total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                elif seen_unseen_id == 'seen':
                    total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                else:
                    raise Exception('Check the bug!')
            else:
                total_seen_unseen_IDs += [1 for _ in range(
                    len(labels))]  # 1 indicates seen. This is not used on training or evaluation set
        return total_charge_discharge_curves, total_curve_attn_masks, np.array(
            total_labels), unique_labels, class_labels, total_dataset_ids, total_cj_aug_charge_discharge_curves, total_seen_unseen_IDs, total_filenames

    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        global data
        with open(f'{self.root_path}/data/{file_name}', 'rb') as fp:
            data = pickle.load(fp)
        if file_name.startswith('Tongji'):
            file_name = file_name.replace('--', '-#')
        life_labels = self.all_file_life_labels
        if file_name in life_labels:
            eol = life_labels[file_name]
            if eol <= 0:
                eol = None
        else:
            eol = None
        return data, eol

    def read_cell_df(self, file_name):
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, basic_prompt, eol
        '''
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]

        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
        # print(data.keys())
        cycle_data = data['cycle_data']  # list of cycle data dict

        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan  # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah'] < 0] = np.nan
            cycle_df.bfill(inplace=True)  # deal with NaN
            total_cycle_dfs.append(cycle_df)

            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break

        df = pd.concat(total_cycle_dfs)
        # obtain the charge and discahrge curves
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold,nominal_capacity)
        cj_aug_charge_discharge_curves, fm_aug_charge_discharge_curves = self.aug_helper.batch_aug(
            charge_discharge_curves)

        return df, charge_discharge_curves, eol, nominal_capacity, cj_aug_charge_discharge_curves

    def read_samples_from_one_cell(self, file_name):
        '''
        read all samples using this function
        :param file_name: which file needs to be read
        :return: history_sohs, future_sohs, masks, cycles, prompts, charge_data, discharge_data and RPT_masks in each sample
        '''

        df, charge_discharge_curves_data, eol, nominal_capacity, cj_aug_charge_discharge_curves = self.read_cell_df(
            file_name)
        if df is None or eol <= self.early_cycle_threshold:
            print(file_name, df is None)
            return None, None, None, None, None

        # the charge and discharge data
        charge_discharge_curves = []  # [N, seq_len, fix_charge_resample_len]
        total_cj_aug_charge_discharge_curves = []
        attn_masks = []
        labels = []
        # get the early-life data
        early_charge_discharge_curves_data = charge_discharge_curves_data[:self.early_cycle_threshold]
        early_cj_aug_charge_discharge_curves = cj_aug_charge_discharge_curves[:self.early_cycle_threshold]
        if np.any(np.isnan(early_charge_discharge_curves_data)):
            raise Exception(f'Failure in {file_name} | Early data contains NaN! Cycle life is {eol}!')
        for i in range(self.seq_len, self.early_cycle_threshold + 1):
            if i >= eol:
                # If we encounter a battery whose cycle life is even smaller than early_cycle_threhold
                # We should not include the eol cycle data
                break

            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1  # set 1 not to mask

            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if i <= self.eval_cycle_max and i >= self.eval_cycle_min:
                    # Only keep the val and test samples that satisfy the eval_cycle
                    pass
                else:
                    continue

            # tmp_prompt = basic_prompt
            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            total_cj_aug_charge_discharge_curves.append(early_cj_aug_charge_discharge_curves)
            attn_masks.append(tmp_attn_mask)

        return charge_discharge_curves, attn_masks, labels, eol, total_cj_aug_charge_discharge_curves

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []
        unique_cycles = df['cycle_number'].unique()
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle in range(1, early_cycle_threshold + 1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]

                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records / nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values
                time_in_s_records = cycle_df['time_in_s'].values

                cutoff_voltage_indices = np.nonzero(
                    current_records_in_C >= 0.01)  # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][
                    -1]  # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C <= -0.01)
                discharge_end_index = cutoff_voltage_indices[0][-1]

                # tmp_discharge_capacity_records = max(charge_capacity_records) - discharge_capacity_records
                if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_35', 'CALB_45']:
                    # Every cycle first discharge and then charge
                    # capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]

                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity

                    charge_voltages = charge_voltages[np.abs(charge_current_in_C) > 0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C) > 0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C) > 0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C) > 0.01]
                else:
                    # Every cycle first charge and then discharge
                    # capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity

                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C) > 0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C) > 0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C) > 0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C) > 0.01]

                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]

                # try:
                #     discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curvesv2(discharge_voltages, discharge_currents, discharge_capacities)
                #     charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curvesv2(charge_voltages, charge_currents, charge_capacities)
                # except:
                #     print('file_name', file_name, cycle)

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(
                    discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(
                    charge_voltages, charge_currents, charge_capacities)

                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)

                voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(
                    voltage_records)  # normalize using the cutoff voltage
                current_records = current_records.reshape(1,
                                                          self.charge_discharge_len) / nominal_capacity  # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1,
                                                                  self.charge_discharge_len) / nominal_capacity  # normalize the capacity

                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))

        curves = np.concatenate(curves, axis=0)  # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or dicharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interploted records
        '''
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages) + 1)
        interp_bases = np.linspace(1, len(voltages) + 1, num=charge_discharge_len,
                                   endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery



class BatteryLifeDataset(Dataset):
    def __init__(self,total_charge_discharge_curves, total_curve_attn_masks, total_labels, class_labels,
                 total_dataset_ids, total_cj_aug_charge_discharge_curves, total_seen_unseen_IDs,scaled_life_classes,weights,
                 label_scaler, life_class_scaler, total_filenames):
        self.total_charge_discharge_curves = total_charge_discharge_curves
        self.total_curve_attn_masks = total_curve_attn_masks
        self.total_labels = total_labels
        self.class_labels = class_labels
        self.total_dataset_ids = total_dataset_ids
        self.total_cj_aug_charge_discharge_curves = total_cj_aug_charge_discharge_curves
        self.total_seen_unseen_IDs = total_seen_unseen_IDs
        self.scaled_life_classes = scaled_life_classes
        self.weights = weights
        self.total_filenames = total_filenames
        self.label_scaler = label_scaler
        self.life_class_scaler = life_class_scaler

    def __getitem__(self, index):
        sample = {
            'cycle_curve_data': torch.Tensor(self.total_charge_discharge_curves[index]),
            'curve_attn_mask': torch.Tensor(self.total_curve_attn_masks[index]),
            'labels': self.total_labels[index],
            'life_class': self.class_labels[index],
            'scaled_life_class': self.scaled_life_classes[index],
            'weight': self.weights[index],
            'dataset_id': self.total_dataset_ids[index],
            'cj_cycle_curve_data': self.total_cj_aug_charge_discharge_curves[index],
            'seen_unseen_id': self.total_seen_unseen_IDs[index],
            'file_name':self.total_filenames[index]
        }
        return sample

    def return_label_scaler(self):
        return self.label_scaler

    def return_life_class_scaler(self):
        return self.life_class_scaler

    def __len__(self):
        return len(self.total_labels)

datasetName2ids = {
    'CALCE':0,
    'HNEI':1,
    'HUST':2,
    'MATR':3,
    'RWTH':4,
    'SNL':5,
    'MICH':6,
    'MICH_EXP':7,
    'Tongji1':8,
    'Stanford':9,
    'ISU-ILCC':11,
    'XJTU':12,
    'ZN-coin':13,
    'UL-PUR':14,
    'Tongji2':15,
    'Tongji3':16,
    'CALB':17,
    'ZN42':22,
    'ZN2024':23,
    'CALB42':24,
    'CALB2024':25,
    'NA-ion':27,
    'NA-ion42':28,
    'NA-ion2024':29,
    #新增
    'SDU':30,
    'Stanford_2':31,
    'SUNV2':32
}
def batteryLife_collate_fn_baseline(samples):
    cycle_curve_data = torch.stack([i['cycle_curve_data'] for i in samples])
    curve_attn_mask = torch.stack([i['curve_attn_mask'] for i in samples])

    def stack_values(key):
        return torch.as_tensor(np.asarray([i[key] for i in samples], dtype=np.float32))

    life_class = stack_values('life_class')
    labels = stack_values('labels')
    scaled_life_class = stack_values('scaled_life_class')
    # 每条样本一个权重，与逐样本回归 loss 的 [B, 1] 形状对齐。
    weights = stack_values('weight').reshape(-1, 1)
    seen_unseen_ids = stack_values('seen_unseen_id')
    file_names = [sample['file_name'] for sample in samples]

    cycle_curve_data.masked_fill_(curve_attn_mask.eq(0).unsqueeze(-1).unsqueeze(-1), 0)
    return cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, seen_unseen_ids, file_names
