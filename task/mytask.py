from core.hook.hook_registry import hook_emit
from core.register.registryhub import R
from core.task.Task import Task

@R.task.register
class MyTask(Task):
    def run(self):
        hook_emit("start")
        data = 'self.load()'
        hook_emit("end", data)

