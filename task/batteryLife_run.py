from core.register.registryhub import R
from core.task.Task import Task
import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import evaluate
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import time
import json
import random
import numpy as np
import os
import json
import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from core.hook.hook_registry import hook_emit


@R.task.register
class BatteryLifeRun(Task):
    def __init__(self, context):
        super().__init__(context)


    def run(self):
        context = self.context
        accelerator = context.accelerator
        device = torch.device(context.device)
        model = context.model
        model.to(device)
        datafactory = context.datafactory
        print("Loading training samples......")
        train_data, train_loader = datafactory.get_dataset_loader('train')
        label_scaler = train_data.return_label_scaler()
        life_class_scaler = train_data.return_life_class_scaler()
        print("Loading vali samples......")
        vali_data, vali_loader = datafactory.get_dataset_loader('val', label_scaler = label_scaler, life_class_scaler = life_class_scaler)
        print("Loading test samples......")
        test_data, test_loader = datafactory.get_dataset_loader('test', label_scaler = label_scaler, life_class_scaler = life_class_scaler)
        joblib.dump(label_scaler, f'{self.checkpoint_path}/label_scaler')
        joblib.dump(life_class_scaler, f'{self.checkpoint_path}/life_class_scaler')
        model_optim = context.optimizer.build()
        scheduler = context.scheduler.build(model_optim, len(train_loader))
        compute_loss = context.loss
        early_stop = context.early_stop
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        accelerator.print(f'Training started on {accelerator.device}: '
                          f'{len(train_data)} train, {len(vali_data)} validation, {len(test_data)} test samples')
        hook_emit('before_train', context)
        for epoch in range(context.epochs):
            hook_emit('before_epoch', context)
            total_loss = 0
            model.train()
            for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights,
                    seen_unseen_ids) in enumerate(train_loader):
                hook_emit('before_iter', context, i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights,
                    seen_unseen_ids))
                with accelerator.accumulate(model):
                    model_optim.zero_grad()
                    cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                    curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                    labels = labels.float().to(accelerator.device)
                    outputs = model(cycle_curve_data, curve_attn_mask)
                    loss = compute_loss(outputs, labels, weights)
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    model_optim.step()
                    scheduler.step()
                hook_emit('after_iter', context, i, total_loss)
            hook_emit('after_epoch_train', context)
            train_result = {'loss': float(total_loss / len(train_loader))}
            vali_result = context.validate.validate(accelerator, model, vali_data, vali_loader, compute_seen_unseen=False)
            test_result = context.validate.validate(accelerator, model, test_data, test_loader, compute_seen_unseen=True)
            early_stop.check(vali_result, epoch + 1)
            if early_stop.is_best:
                self.save_checkpoint(model, self.checkpoint_path, accelerator)
            hook_emit('epoch_over', context, train_result, vali_result, test_result)
            self.log_epoch_over(accelerator, epoch + 1, train_result, vali_result, test_result)
            if early_stop.should_stop:
                accelerator.print("Early stopping")
                accelerator.set_trigger()
            if accelerator.check_trigger():
                break
        hook_emit('after_train', context)

    def save_checkpoint(self, model, checkpoint_path, accelerator = None):
        if accelerator is not None:
            accelerator.save_model(model, checkpoint_path)
            accelerator.print(f'The checkpoint is saved in {checkpoint_path}!')
        else:
            torch.save(model.state_dict(), checkpoint_path + '/model.pth')

    def log_epoch_over(self, accelerator, epoch, train_result, vali_result, test_result):
        accelerator.print(f"epoch:{epoch}\ntrain_result:{train_result}\nvali_result: {vali_result}\ntest_result: {test_result}")
        if accelerator.is_main_process:
            record = {'epoch': epoch, 'train': train_result, 'validation': vali_result, 'test': test_result}
            with open(f'{self.path}/metrics.jsonl', 'a', encoding='utf-8') as fp:
                fp.write(json.dumps(record, default=float, ensure_ascii=False) + '\n')



