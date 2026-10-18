from core.hook.hook_registry import hook_reg


@hook_reg("start", 1)
def on_start():
    print("开始")