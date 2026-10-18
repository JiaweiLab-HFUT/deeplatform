from core.register.registryhub import R


@R.others.register
class EarlyStop:
    def __init__(self, enable, patience, least_epoch, monitor,  which_is_better, verbose=True, delta=0.0):
        self.should_stop = False
        self.is_best = False
        self.enable = enable
        self.monitor = monitor
        self.delta = delta
        self.which_is_better = which_is_better
        self.patience = patience
        self.least_epoch = least_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_vali_result = None
        if which_is_better == 'less':
            self.best = float("inf")
        elif which_is_better == 'higher':
            self.best = float("-inf")
        else:
            raise ValueError('which_is_better must be either "less" or "higher"')
    def check(self, vali_result, epoch):
        self.is_best = False
        if self.enable is False:
            return False
        if epoch <= self.least_epoch:
            return False
        monitor_result = vali_result[self.monitor]
        if self.which_is_better == 'less':
            if monitor_result < self.best - self.delta:
                self.best = monitor_result
                self.is_best = True
                self.counter = 0
                self.best_vali_result = vali_result
            else:
                self.counter += 1
        elif self.which_is_better == 'higher':
            if monitor_result > self.best + self.delta:
                self.best = monitor_result
                self.is_best = True
                self.counter = 0
                self.best_vali_result = vali_result
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        if self.verbose:
            if self.is_best:
                print(f'epoch{epoch} is best, {self.monitor} is {self.best}')
            if self.should_stop:
                print(f'early stop at {epoch} epoch, the best {self.monitor} is {self.best}, all metric on {epoch} epoch is {self.best_vali_result}')
            else:
                print(f'early stop {self.counter}/{self.patience}')
        return self.should_stop
