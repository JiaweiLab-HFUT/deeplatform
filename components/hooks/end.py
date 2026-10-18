from core.hook.hook_registry import hook_reg


@hook_reg("end")
def end(data):
    print("结束", data)