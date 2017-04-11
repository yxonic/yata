"""Contains utilities for data manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from collections import Iterable
from six.moves import filterfalse
from six import string_types
import numpy as np


def foreach(func):
    """
    Decorator that enables an one-to-one function to apply on arrays with arbitrary depth  
    """
    @functools.wraps(func)
    def wrapper(*args):
        item = args[-1]
        if isinstance(item, Iterable) and \
                not isinstance(item, string_types) and \
                not (isinstance(item, np.ndarray) and item.ndim == 0):
            rv = []
            for i in item:
                rv.append(wrapper(*(args[:-1] + (i,))))
            return np.asarray(rv)
        else:
            return func(*args)
    return wrapper


def unique(iterable, key=None):
    """List unique elements, preserving order."""
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
