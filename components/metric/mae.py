from core.register.registryhub import R
from sklearn.metrics import mean_absolute_error
@R.metric.register
class Mae:
    def __call__(self, y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))
