from core.register.registryhub import R
import numpy as np
@R.metric.register
class Accuracy:
    def __init__(self, k:float):
        self.k = k
    def __call__(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        hits = np.abs(y_pred - y_true) <= self.k * np.abs(y_true)
        return float(np.mean(hits) * 100)
