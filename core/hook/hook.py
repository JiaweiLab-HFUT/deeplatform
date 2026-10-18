from core.hook.hook_registry import hook_reg, hook_registry


class Hook:
    def __init__(self, hook, priority = 0,):
        '''
        hook是钩子位置
        priority是优先级
        :param hook:
        :param priority:
        '''
        hook_registry.register_hook(hook, priority, self.run)


    def run(self, *args, **kwargs):
        raise NotImplementedError