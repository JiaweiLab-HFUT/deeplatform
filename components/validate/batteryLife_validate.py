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
            result[name] = float(metric(total_references, total_preds))

        if compute_seen_unseen:
            for prefix, group_id in [('seen', 1), ('unseen', 0)]:
                mask = total_seen_unseen_ids == group_id
                for name, metric in self.used_metric_dict.items():
                    result[f'{prefix}_{name}'] = (
                        float(metric(total_references[mask], total_preds[mask]))
                        if np.any(mask) else None
                    )
        model.train()
        return result
