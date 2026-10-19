from core.hook.hook import Hook
from core.register.registryhub import R


@R.hook.register
class PrintStart(Hook):
    event = 'start'

    def run(self):
        print('开始')
