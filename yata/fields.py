"""Contains common field converters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .util import foreach
import numpy as np
from six import get_function_code


class Field:
    _processor = None

    def __call__(self, other):
        f = Field()
        f._processor = lambda k, p: \
            self._processor(k, other._processor(k, p))
        return f

    def apply(self, key, item):
        return self._processor(key, item)


class Converter(Field):
    def __init__(self, func=None):
        """
        Wraps a function so that it can get chained with other converters
        :param func: Convert function. If None, the converter does nothing
        """
        if func is None:
            self._processor = lambda _, x: x
        elif get_function_code(func).co_argcount == 1:
            self._processor = lambda _, x: func(x)
        else:
            self._processor = func


class Numeral(Field):
    def __init__(self, dtype=None):
        """
        Convert to numeral type
        :param dtype: Data type to convert to, can be type or str
        """
        if dtype is None:
            self.dtype = 'float32'
        else:
            self.dtype = dtype
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype).type
        self._processor = foreach(lambda _, x: self.dtype(x))


class Categorical(Field):
    def __init__(self, null='<NULL>'):
        """
        Maps sequences to ints
        :param null: Special token that maps to 0
        """
        self._items = [null]
        self._map = {null: 0}
        self._fixed = False
        self._processor = foreach(lambda _, x: self.to_categorical(x))

    @foreach
    def to_categorical(self, item):
        """
        Maps an item to int
        :rtype: int
        :param item: Any hashable item
        :return: Category index of this item
        """
        try:
            return self._map[item]
        except KeyError:
            if self._fixed:
                return 0  # for null
            else:
                i = len(self._items)
                self._items.append(item)
                self._map[item] = i
                return i

    @foreach
    def get_original(self, cat):
        """
        Get original item that maps to cat
        :param cat: Category index
        :return: Original item
        """
        try:
            return self._items[cat]
        except IndexError:
            return ''

    @property
    def items(self):
        """
        Items in order of their category index
        """
        return self._items

    def load_dict(self, items):
        """
        Use a list to specify mapping dict. After load_dict, items not in dict would be mapped to 0 
        :param items: Item list
        """
        self._items = items
        for i, w in enumerate(self._items):
            self._map[w] = i
        self._fixed = True


class Words(Field):
    def __init__(self, sep, length=None, null='<NULL>'):
        """
        Convert string to list of words
        :param sep: Separator of words
        :param length: If None, length will be varied, else converted list will all be in same size
        :param null: Special token for null word
        """
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


class Chars(Field):
    def __init__(self, length=None, null='<NULL>'):
        """
        Convert string to char array
        :param length: If None, length will be varied, else converted list will all be in same size
        :param null: Special token for null character
        """
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


class File(Field):
    def __init__(self, mode='r'):
        """
        Convert file name to file object
        :param mode: Specifies the mode in which the file is opened
        """
        self._processor = foreach(lambda _, filename: open(filename, mode))


class Image(Field):
    def __init__(self, shape=None):
        """
        Open file as PIL.Image
        :param shape: If not None, resize every image to this shape 
        """
        from PIL import Image

        def open_image(file):
            im = Image.open(file)
            if shape is not None:
                im = im.resize(shape)
            return im

        self._processor = foreach(lambda _, file: open_image(file))
