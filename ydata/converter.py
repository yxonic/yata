from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .util import foreach
import numpy as np
from six import get_function_code


class BaseConverter:
    def __init__(self, func):
        self._processor = func

    def __call__(self, other):
        return BaseConverter(lambda k, p:
                             self._processor(k, other._processor(k, p)))

    def apply(self, key, item):
        return self._processor(key, item)


class Converter(BaseConverter):
    def __init__(self, func):
        if get_function_code(func).co_argcount == 1:
            self._processor = foreach(lambda _, x: func(x))
        else:
            self._processor = foreach(func)


class Numeral(BaseConverter):
    def __init__(self, dtype=None):
        if dtype is None:
            self.dtype = 'float32'
        else:
            self.dtype = dtype
        self._processor = foreach(lambda _, x: np.asarray(x, dtype=self.dtype))


class Categorical(BaseConverter):
    def __init__(self, null='<NULL>'):
        self._items = [null]
        self._map = {null: 0}
        self._processor = foreach(lambda _, x: self.to_categorical(x))

    @foreach
    def to_categorical(self, item):
        try:
            return self._map[item]
        except KeyError:
            i = len(self._items)
            self._items.append(item)
            self._map[item] = i
            return i

    @foreach
    def get_original(self, cat):
        try:
            return self._items[cat]
        except IndexError:
            return ''

    @property
    def items(self):
        return self._items


class Words(BaseConverter):
    def __init__(self, sep, length=None, null='<NULL>'):
        if length is None:
            self._processor = foreach(lambda _, x: str(x).split(sep))
        else:
            def check(_, x):
                rv = str(x).split(sep)
                if len(rv) > length:
                    return rv[:length]
                else:
                    return rv + [null] * (length - len(rv))
            self._processor = foreach(check)


class Chars(BaseConverter):
    def __init__(self, length=None, null='<NULL>'):
        if length is None:
            self._processor = foreach(lambda _, x: list(str(x)))
        else:
            def check(_, x):
                rv = list(str(x))
                if len(rv) > length:
                    return rv[:length]
                else:
                    return rv + [null] * (length - len(rv))
            self._processor = foreach(check)
