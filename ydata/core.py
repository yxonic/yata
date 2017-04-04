from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import copy
import random
from collections import Iterable, defaultdict
from six.moves import range
import numpy as np
import pandas as pd


class DataSource:
    def __init__(self, file, with_header=True, key=None, fields=None):
        if key is None:
            key = 0
        if fields is None:
            fields = {}

        self._with_header = with_header

        df = pd.read_table(file, dtype=str,
                           header='infer' if with_header else None)
        self._table = df

        index = {}
        new_fields = []
        for f in fields:
            if '->' in f:
                f = f.split('->')[1]
                if ',' in f:
                    for nf in f.split(','):
                        new_fields.append(nf)
                else:
                    new_fields.append(f)
            else:
                new_fields.append(f)

        for i, data in df.iterrows():
            if callable(key):
                k = key(data)
                if isinstance(k, Iterable):
                    for k_ in k:
                        v = self._get_fields(data, k_, fields)
                        if v is not None:
                            index[k_] = i
                else:
                    v = self._get_fields(data, k, fields)
                    if v is not None:
                        index[k] = i
            else:
                v = self._get_fields(data, data[key], fields)
                if v is not None:
                    index[data[key]] = i

        self._index = index
        self._keys = sorted(index.keys())
        self._fields = fields
        self._new_fields = sorted(new_fields)

    def get(self, key):
        data = self._table.loc[self._index[key]]
        return self._get_fields(data, key, self._fields)

    @property
    def keys(self):
        return self._keys

    @property
    def fields(self):
        return self._new_fields

    def _get_fields(self, data, key, fields):
        v = {}
        try:
            for field in fields:
                converter = fields[field]
                if '->' in field:
                    field, new_field = field.split('->')
                else:
                    new_field = field
                if not self._with_header:
                    field = int(field)
                result = converter.apply(key, data[field])
                if ',' in new_field:
                    for i, f in enumerate(new_field.split(',')):
                        v[f] = result[i]
                else:
                    v[new_field] = result
        except Exception:
            return None
        return v


class DataLoader:
    def __init__(self, *args):
        keys = None
        fields = set()
        for source in args:
            if keys is None:
                keys = set(source.keys)
            else:
                keys &= set(source.keys)
            # TODO: warn if fields overlap
            fields.update(source.fields)

        self._sources = args
        self._keys = sorted(keys)
        self._fields = fields

    def get(self, key):
        result = {}
        for s in self._sources:
            result.update(s.get(key))
        return result

    def sample(self, frac=1.0):
        rv = copy.copy(self)
        N = int(len(self.keys) * frac)
        rv._keys = random.sample(self.keys, N)
        return rv

    def epoch(self, batch_size):
        total = len(self.keys)
        N = math.ceil(total / batch_size)

        def loader():
            for i in range(N):
                keys = self.keys[i*batch_size:(i+1)*batch_size]
                rv = defaultdict(list)
                for k in keys:
                    data = self.get(k)
                    for f in data:
                        rv[f].append(data[f])
                yield keys, {f: np.asarray(rv[f]) for f in rv}

        return loader()

    @property
    def keys(self):
        return self._keys

    @property
    def fields(self):
        return self._fields
