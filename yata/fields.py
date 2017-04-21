"""Contains common field converters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .util import foreach
import numpy as np
from six import get_function_code, text_type


class Field:
    def __init__(self):
        pass

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
        Field.__init__(self)
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
        Field.__init__(self)
        if dtype is None:
            self.dtype = 'float32'
        else:
            self.dtype = dtype
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype).type
        self._processor = foreach(lambda _, x: self.dtype(x))


class Categorical(Field):
    def __init__(self, null='<NULL>', one_hot=False):
        """
        Maps sequences to ints
        :param null: Special token that maps to 0
        """
        Field.__init__(self)
        self._one_hot = one_hot
        null = text_type(null)
        self._items = [null]
        self._map = {null: 0}
        self._fixed = False
        self._processor = foreach(lambda _, x: self.to_categorical(x))

    @foreach
    def to_categorical(self, item, one_hot=None):
        """
        Maps an item to int
        :rtype: int
        :param item: Any hashable item
        :param one_hot: Whether returns index or one-hotted array
        :return: Category index of this item
        """
        item = text_type(item)
        try:
            cat = self._map[item]
        except KeyError:
            if self._fixed:
                cat = 0  # for null
            else:
                cat = len(self._items)
                self._items.append(item)
                self._map[item] = cat
        if one_hot is None:
            one_hot = self._one_hot
        if one_hot:
            if not self._fixed:
                raise ValueError('can\'t return one hot result with varying size')
            a = np.zeros((self.count,), dtype='int32')
            a[cat] = 1
            return a
        else:
            return cat

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

    @property
    def count(self):
        """
        Number of different categories
        """
        return len(self._items)

    def load_dict(self, items):
        """
        Use a list to specify mapping dict. After load_dict, items not in dict would be mapped to 0 
        :param items: Item list
        """
        self._items = list(items)
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
        Field.__init__(self)
        if length is None:
            self._processor = foreach(lambda _, x: text_type(x).split(sep))
        else:
            def check(_, x):
                rv = text_type(x).split(sep)
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
        Field.__init__(self)
        if length is None:
            self._processor = foreach(lambda _, x: list(text_type(x)))
        else:
            def check(_, x):
                rv = list(text_type(x))
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
        Field.__init__(self)
        self._processor = foreach(lambda _, filename: open(filename, mode))


class Image(Field):
    def __init__(self, shape=None, gray_scale=False):
        """
        Open file as PIL.Image
        :param shape: If not None, resize every image to this shape 
        """
        Field.__init__(self)
        from PIL import Image

        def _alpha_to_color(image, color=(255, 255, 255)):
            image.load()
            background = Image.new('RGB', image.size, color)
            background.paste(image, mask=image.split()[3])
            return background

        def open_image(filename):
            im = _alpha_to_color(Image.open(filename))
            if shape is not None:
                im = im.resize(shape)
            if gray_scale:
                im = im.convert('L')
            return 1 - np.asarray(im) / 255.0

        self._processor = foreach(lambda _, f: open_image(f))
