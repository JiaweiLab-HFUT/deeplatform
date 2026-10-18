from core.register.registryhub import R
from sklearn.metrics import root_mean_squared_error
@R.metric.register
class Rmse:
    def __call__(self, y_true, y_pred):
        return float(root_mean_squared_error(y_true, y_pred))
