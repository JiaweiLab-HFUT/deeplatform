from core.register.registryhub import R


@R.datatype.register
class Dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
