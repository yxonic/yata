from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from collections import Iterable
import numpy as np


def foreach(func):
    @functools.wraps(func)
    def wrapper(*args):
        item = args[-1]
        if isinstance(item, Iterable) and \
                not isinstance(item, str) and \
                not (isinstance(item, np.ndarray) and item.ndim == 0):
            rv = []
            for i in item:
                rv.append(wrapper(*(args[:-1] + (i,))))
            return np.asarray(rv)
        else:
            return func(*args)
    return wrapper
