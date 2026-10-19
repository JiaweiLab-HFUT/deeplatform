from core.register.registryhub import R
from core.task.Task import Task

@R.task.register
class MyTask(Task):
    def run(self):
        hooks = self.hooks
        hooks.emit("start")
        data = 'self.load()'
        hooks.emit("end", data)

