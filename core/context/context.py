'''
context管理整个训练过程中的资源
外部调用context必须通过bind，并声明自己的scope
'''

from collections import defaultdict

class Context:
    def __init__(self):
        self.__memory = defaultdict(dict)

    def bind(self, scope):
        return _BoundContext(self.__memory, scope)


class _BoundContext:
    __slots__ = ('_memory', '_scope')

    def __init__(self, memory, scope):
        self._memory = memory
        self._scope = scope


    def __setitem__(self, key, value):
        self._memory[self._scope][key] = value

    def __getitem__(self, key):
        return self._memory[self._scope][key]

    def __delitem__(self, key):
        del self._memory[self._scope][key]

    def __contains__(self, key):
        return key in self._memory[self._scope]

    def __getattr__(self, key):
        try:
            return self._memory[self._scope][key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        # slots 里的内部属性走正常路径
        if key in self.__slots__:
            object.__setattr__(self, key, value)
        else:
            self._memory[self._scope][key] = value

    def __delattr__(self, key):
        if key in self.__slots__:
            object.__delattr__(self, key)
        else:
            try:
                del self._memory[self._scope][key]
            except KeyError:
                raise AttributeError(key) from None