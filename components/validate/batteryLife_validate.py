import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

from core.register.registryhub import R


@R.validate.register
class BatteryLifeValidate:
    def __init__(self, used_metric):
        self.used_metric_dict = used_metric

    def validate(self, accelerator, model, vali_data, vali_loader, compute_seen_unseen=False):
        total_preds, total_references = [], []
        total_seen_unseen_ids = []
        model.eval()
        with torch.no_grad():
            for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights,
                    seen_unseen_ids) in tqdm(enumerate(vali_loader)):
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)  # [B, S, N]
                curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
                labels = labels.float().to(accelerator.device)
                # encoder - decoder
                outputs = model(cycle_curve_data, curve_attn_mask)
                accelerator.wait_for_everyone()
                std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
                transformed_preds = outputs * std + mean_value
                transformed_labels = labels * std + mean_value

                all_predictions, all_targets, seen_unseen_ids = accelerator.gather_for_metrics(
                    (transformed_preds, transformed_labels, seen_unseen_ids))

                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
                if compute_seen_unseen:
                    total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()

        total_preds = np.array(total_preds)
        total_references = np.array(total_references)
        total_seen_unseen_ids = np.array(total_seen_unseen_ids)

        result = {}
        for name, metric in self.used_metric_dict.items():
            result[name] = metric(total_references, total_preds)

        if compute_seen_unseen:
            # calculate the model performance on the samples from the seen and unseen aging conditions
            seen_references = total_references[total_seen_unseen_ids == 1] if np.any(
                total_seen_unseen_ids == 1) else np.array([0])
            unseen_references = total_references[total_seen_unseen_ids == 0] if np.any(
                total_seen_unseen_ids == 0) else np.array([0])
            seen_preds = total_preds[total_seen_unseen_ids == 1] if np.any(total_seen_unseen_ids == 1) else np.array(
                [1])
            unseen_preds = total_preds[total_seen_unseen_ids == 0] if np.any(total_seen_unseen_ids == 0) else np.array(
                [1])
            for name, metric in self.used_metric_dict.items():
                seen_metric_name = f'seen_{name}'
                result[seen_metric_name] = metric(seen_references, seen_preds)

            for name, metric in self.used_metric_dict.items():
                unseen_metric_name = f'unseen_{name}'
                result[unseen_metric_name] = metric(unseen_references, unseen_preds)
        model.train()
        return result
