from core.register.registryhub import R
from sklearn.metrics import mean_squared_error
@R.metric.register
class Mse:
    def __call__(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)