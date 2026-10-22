"""可通过 upload_component 安装、供 BatteryLifeRun 使用的线性回归组件。"""

from torch import nn

from core.register.registryhub import R


@R.model.register('ExternalLinear')
class ExternalLinear(nn.Module):
    def __init__(self, early_cycle_threshold=100, charge_discharge_length=300):
        super().__init__()
        self.linear = nn.Linear(early_cycle_threshold * 3 * charge_discharge_length, 1)

    def forward(self, cycle_curve_data, curve_attn_mask):
        masked = cycle_curve_data * curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
        return self.linear(masked.flatten(start_dim=1))
