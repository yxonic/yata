"""Contains data loader classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import math
import copy
import random
import os
import re
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from collections import Iterable, defaultdict, namedtuple, OrderedDict

from .fields import Converter
from .util import unique


def _make_item_class(converters):
    """
    Make item type according to field mapping rules
    :rtype: type
    :param converters: Field mapping rules
    :return: namedtuple type Item
    """
    fields = []
    for f in converters.keys():
        fields.extend(f.split('->')[-1].split(','))
    return namedtuple('Item', fields)


def _parse(doc, key, converters, subset=None):
    """
    Parse item from a tuple or dict
    :rtype: list
    :param doc: Original document (tuple or dict)
    :param key: Key of this document
    :param converters: Field mapping rules
    :param subset: Only parse a subset of fields. If subset is None, then parse the whole set
    :return: Parsed item, fields not in subset set to None
    """
    v = []

    for field_map, converter in converters.items():
        if '->' in field_map:
            before, after = field_map.split('->')
        else:
            before = after = field_map

        if subset is not None and \
                len(set(after.split(',')) & set(subset)) == 0:
            v.extend([None] * len(after.split(',')))
            continue

        if before == '':
            i = doc
        elif isinstance(doc, dict):
            i = [doc[x] for x in before.split(',')]
        else:
            i = [doc[int(x)] for x in before.split(',')]

        o = converter.apply(key, *i)

        if ',' in after:
            v.extend(o)
        else:
            v.append(o)

    return v


class BaseLoader(object):
    __metaclass__ = ABCMeta

    def get(self, key):
        """
        Get an data item. Returns None if not found
        :rtype: namedtuple or NoneType
        :param key: Data key
        :return: Item or None
        """
        if key not in self.keys:
            return None
        return self._get(key)

    @abstractmethod
    def _get(self, key): pass

    def epoch(self, batch_size):
        """
        Returns a generator of an epoch
        :param batch_size: Size of each batch. The actual size may be smaller on conversion error or on last batch
        :return: A generator, each time yields a batch
        """
        n = int(math.ceil(self.size / batch_size))

        def loader():
            for i in range(n):
                keys = self.keys[i*batch_size:(i+1)*batch_size]
                rv = defaultdict(list)
                for key in keys:
                    data = self.get(key)
                    for k, v in data._asdict().items():
                        rv[k].append(np.asarray(v))
                yield keys, {k: np.asarray(rv[k]) for k in rv}

        return loader()

    def sample(self, frac):
        """
        Returns a random sample of items 
        :param frac: Fraction of items to return
        :return: A new loader with sampled keys
        """
        rv = copy.copy(self)
        n = int(math.ceil(len(self.keys) * frac))
        rv._keys = random.sample(self.keys, n)
        rv._indices = OrderedDict()
        for k in rv.keys:
            rv._indices[k] = self._indices[k]
        return rv

    def shuffle(self):
        """
        Returns a loader with item shuffled
        :return: A new loader with item shuffled
        """
        return self.sample(1.0)

    def split(self, frac, on=None):
        """
        Split items into two parts with ratio frac:(1-frac)
        :param frac: Fraction of the first part
        :param on: Field(s) to split on. When specified, same values in these fields won't be at both sides 
        :return: Two loaders.
        """
        left = copy.copy(self)
        right = copy.copy(self)

        if on is None:
            n = int(len(self.keys) * frac)
            left._keys = self._keys[:n]
            right._keys = self._keys[n:]
        else:
            if not isinstance(on, Iterable) or isinstance(on, str):
                on = (on, )

            if set(on) > set(self.Index._fields):
                raise KeyError('can only split on indexed fields')

            indices = self._indices.values()
            value_list = self.Index(*(list(unique(x)) for x in zip(*indices)))
            left_set = dict()
            right_set = dict()
            for on_ in on:
                l = value_list.__getattribute__(on_)
                n = int(math.ceil(len(l) * frac))

                left_set[on_] = set(l[:n])
                right_set[on_] = set(l[n:])

            left._keys = []
            right._keys = []

            for key, index in self._indices.items():
                in_left = True
                in_right = True
                for on_ in on:
                    if index.__getattribute__(on_) not in left_set[on_]:
                        in_left = False
                    if index.__getattribute__(on_) not in right_set[on_]:
                        in_right = False

                if in_left:
                    left._keys.append(key)
                if in_right:
                    right._keys.append(key)

        return left, right

    @property
    def keys(self):
        """
        Returns valid keys in this loader
        :rtype: list
        :return: A list of keys
        """
        return self._keys

    @property
    def size(self):
        """
        Returns item count
        :rtype: int
        :return: Size of valid keys
        """
        return len(self.keys)

    # noinspection PyPep8Naming
    @property
    def Item(self):
        """
        Returns item class of this loader
        :rtype: namedtuple
        :return: namedtuple for item
        """
        if self._Item is None:
            return namedtuple('Item', [])
        else:
            return self._Item

    @property
    def fields(self):
        """
        Returns list of fields provided by this loader
        :rtype: list
        :return: List of available fields
        """
        return self.Item._fields

    _keys = None
    _indices = None
    _Item = None
    _Index = None

    # noinspection PyPep8Naming
    @property
    def Index(self):
        """
        Return index type of this loader
        :return: namedtuple for index
        """
        if self._Index is None:
            return namedtuple('Index', [])
        else:
            return self._Index


class DataLoader(BaseLoader):
    def __init__(self, *args):
        """
        Merge loaders with same keys
        :param args: data loaders that share the same key field
        """
        super(DataLoader, self).__init__()
        self._indices = OrderedDict()
        self._sources = args

        # merge metadata
        keys = None
        item_fields = []
        index_fields = []
        for source in self._sources:
            if keys is None:
                keys = set(source.keys)
            else:
                keys &= set(source.keys)
            item_fields.extend(source.Item._fields)
            index_fields.extend(source.Index._fields)

        self._keys = sorted(keys)
        self._Item = namedtuple('Item', item_fields, rename=True)
        self._Index = namedtuple('Index', index_fields, rename=True)

        # merge index
        for key in self.keys:
            index = []
            for source in self._sources:
                index.extend(source._indices[key])
            self._indices[key] = self.Index(*index)

    def _get(self, key):
        item = []
        for source in self._sources:
            item.extend(source.get(key))
        return self.Item(*item)


class TableLoader(BaseLoader):
    def __init__(self, file, with_header=True, key=None, fields=None, index=None):
        """
        Loads text table separated by '\t'
        :param file: path to table
        :param with_header: Whether table has an extra header line. Without header, each line in data would be treated
               as tuple, else as dict
        :param key: Single object or callable. If key is a single object, then line[key] will be treated as key. If
               key is callable, then key(line) will be the key. There can be multiple keys in a single line, if your
               callable returns an iterable.
        :param fields: A dict that describes field mapping rules. See examples to find out how to describe rules
        :param index: Fields to build indices at. These fields should be small and hashable. Indices are used when
               splitting the loader
        """
        super(TableLoader, self).__init__()
        self._keys = []
        self._indices = OrderedDict()
        if fields is None:
            fields = {}
        else:
            fields = OrderedDict(fields)
        if index is None:
            index = []

        self._field_map = fields

        self._with_header = with_header

        df = pd.read_table(file, dtype=str,
                           header='infer' if with_header else None)
        self._table = df

        self._Item = _make_item_class(fields)
        self._Index = namedtuple('Index', index)
        self._loc = {}
        for i, doc in self._table.iterrows():
            if key is None:
                self.__add_doc(i, i, doc)
            elif callable(key):
                k_ = key(doc)
                if isinstance(k_, Iterable):
                    for k in k_:
                        self.__add_doc(i, k, doc)
            else:
                self.__add_doc(i, doc[key], doc)

    def __add_doc(self, i, key, doc):
        fields = self._field_map
        index = self.Index._fields
        if self._with_header:
            doc = dict(doc)
        else:
            doc = tuple(doc)
        item = self.Item(*_parse(doc, key, fields, index))
        item_dict = item._asdict()
        ind = self.Index(**{f: item_dict[f] for f in index})
        self._keys.append(key)
        self._indices[key] = ind
        self._loc[key] = i

    def _get(self, key):
        data = self._table.loc[self._loc[key]]
        if self._with_header:
            data = dict(data)
        else:
            data = tuple(data)
        return self.Item(*_parse(data, key, self._field_map))


class DirectoryLoader(BaseLoader):
    def __init__(self, dirname, converters=None, pattern=None):
        """
        Loads files under directory
        :param dirname: path to directory
        :param converters: To set converter of files, pass converter or {'file': converter, ...} with other fields.
        :param pattern: Regex to match file name. Use named groups in parallel with converters. Default pattern
               is like: {key}.{ext}
        """
        super(DirectoryLoader, self).__init__()
        self._keys = []
        self._indices = OrderedDict()
        if not isinstance(converters, dict):
            converters = {'file': converters, 'ext': Converter}
        else:
            converters = OrderedDict(converters)
        if pattern is None:
            pattern = r'^(?P<key>.+)\.(?P<ext>.+)$'
        self._converters = converters['file']
        converters['filename'] = Converter()
        self._Item = _make_item_class(converters)
        del converters['file']
        self._Index = namedtuple('Index', converters.keys())

        files = os.listdir(dirname)
        pattern = re.compile(pattern)
        for filename in files:
            for match in pattern.finditer(filename):
                doc = match.groupdict()
                key = doc['key']
                self._keys.append(key)
                del doc['key']
                doc['filename'] = os.path.join(dirname, filename)
                self._indices[key] = self.Index(**doc)

    def _get(self, key):
        ind = self._indices[key]
        f = self._converters.apply(key, ind.filename)
        item_dict = dict(**ind._asdict())
        item_dict['file'] = f
        return self.Item(**item_dict)
