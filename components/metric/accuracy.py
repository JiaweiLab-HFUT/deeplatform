from core.register.registryhub import R
from sklearn.metrics import root_mean_squared_error
@R.metric.register
class Accuracy:
    def __init__(self, k:float):
        self.k = k
    def __call__(self, y_true, y_pred):
        relative_error = abs(y_pred - y_true) / y_true
        hit_num = sum(relative_error <= self.k)
        return hit_num / len(y_true) * 100