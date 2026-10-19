from core.hook.hook import Hook
from core.register.registryhub import R


@R.hook.register
class PrintEnd(Hook):
    event = 'end'

    def run(self, data):
        print('结束', data)
