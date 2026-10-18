from core.register.registryhub import R
from sklearn.metrics import mean_absolute_percentage_error
@R.metric.register
class Mape:
    def __call__(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)