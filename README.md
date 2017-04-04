# ydata

```python
import re
from ydata import *

chars = Chars()
words = Words(':')
text_by_char = Categorical()(chars)
text_by_word = Categorical()(words)
int32 = Numeral('int32')
label = Categorical()

source1 = DataSource('file1.table', with_header=True,
                     key='id',
                     fields={'content': text_by_char,
                             'content->words': text_by_word,
                             'label': label})


def get_key(line):
    return re.finditer(r'<[a-z0-9\-]+>', line['content'])


def get_context(key, line):
    line = line.replace('<{}>'.format(key), '###')\
        .replace('<[a-z0-9\\-]*>', '')
    left = re.findall('(.{0,10})###', line)[0]
    right = re.findall('###(.{0,10})', line)[0]
    return (left, right)


source2 = DataSource('file2.table', with_header=True,
                     key=get_key,
                     fields={'content->lcontext,rcontext':
                             text(Converter(get_context))})

loader = DataLoader(sources=[source1, source2])

data = loader.get(id)  # (key, content, label, lcontext, rcontext)

loader.shuffle()

iterator = loader.epoch(batch_size=32, sample_frac=0.1)
```