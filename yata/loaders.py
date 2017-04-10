'''Contains data loader classes.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import math
import copy
import random
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from collections import Iterable, defaultdict, namedtuple, OrderedDict

from .util import unique


def _make_item_class(converters):
    fields = []
    for f in converters.keys():
        fields.extend(f.split('->')[-1].split(','))
    return namedtuple('Item', fields)


def _load(doc, key, converters, subset=None):
    v = []

    for field_map, converter in converters.items():
        if '->' in field_map:
            before, after = field_map.split('->')
        else:
            before = after = field_map

        if subset is not None and \
                len(set(before.split(',')) & set(subset)) == 0:
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


class BaseLoader(object, metaclass=ABCMeta):
    def __init__(self):
        self._keys = []
        self._indices = OrderedDict()

    def get(self, key):
        if key not in self.keys:
            raise KeyError(key)
        return self._get(key)

    @abstractmethod
    def _get(self, key): pass

    def epoch(self, batch_size):
        N = int(math.ceil(self.size / batch_size))

        def loader():
            for i in range(N):
                keys = self.keys[i*batch_size:(i+1)*batch_size]
                rv = defaultdict(list)
                for k in keys:
                    data = self.get(k)
                    for k, v in data._asdict().items():
                        rv[k].append(v)
                yield keys, {k: np.asarray(rv[k]) for k in rv}

        return loader()

    def sample(self, frac):
        rv = copy.copy(self)
        N = int(math.ceil(len(self.keys) * frac))
        rv._keys = random.sample(self.keys, N)
        rv._indices = OrderedDict()
        for k in rv.keys:
            rv._indices[k] = self._indices[k]
        return rv

    def shuffle(self):
        return self.sample(1.0)

    def split(self, frac, on=None):
        left = copy.copy(self)
        right = copy.copy(self)

        if on is None:
            N = int(len(self.keys) * frac)
            left._keys = self._keys[:N]
            right._keys = self._keys[N:]
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
                vlist = value_list.__getattribute__(on_)
                N = int(math.ceil(len(vlist) * frac))

                left_set[on_] = set(vlist[:N])
                right_set[on_] = set(vlist[N:])

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
        return self._keys

    @property
    def size(self):
        return len(self.keys)

    @property
    def Item(self):
        if self._Item is None:
            return namedtuple('Item', [])
        else:
            return self._Item

    @property
    def fields(self):
        return self.Item._fields

    _keys = None
    _indices = None
    _Item = None
    _Index = None

    @property
    def Index(self):
        if self._Index is None:
            return namedtuple('Index', [])
        else:
            return self._Index


class DataLoader(BaseLoader):
    def __init__(self, *args):
        super().__init__()
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
    def __init__(self, file, with_header=True, ignore_error=False,
                 key=None, fields=None, index=None):
        super().__init__()
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
            if callable(key):
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
        item = self.Item(*_load(doc, key, fields, index))
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
        return self.Item(*_load(data, key, self._field_map))
