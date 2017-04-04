import re
from ydata import DataSource, DataLoader, \
    Numeral, Words, Categorical, Chars, Converter


def get_key(line):
    for m in re.finditer(r'<([a-z0-9\-]+)>', line[0]):
        yield m.group(1)


def get_context(key, line):
    line = line.replace('<{}>'.format(key), '###')
    line = re.sub('<[a-z0-9\\-]+>', '', line)
    left = re.findall('(.{0,10})###', line)[0]
    right = re.findall('###(.{0,10})', line)[0]
    return (left, right)


if __name__ == '__main__':
    word_cat = Categorical()
    text = word_cat(Words(':', 5))

    char_cat = Categorical()
    char = char_cat(Chars(10, '#'))
    label = Numeral('int32')

    data1 = DataSource('test.table', key='id',
                       fields={'label': label,
                               'content->words': text})

    data2 = DataSource('headless.table', with_header=False,
                       key=get_key,
                       fields={'0->lc,rc':
                               char(Converter(get_context))})

    loader = DataLoader(data1, data2)

    print('loader:')
    print('keys:', loader.keys)
    print('fields:', loader.fields)
    print()

    for batch in loader.sample().epoch(4):
        lc, rc = batch[1]['lc'], batch[1]['rc']

        a = batch[0]
        b = [''.join(char_cat.get_original(line)) for line in lc]
        c = [''.join(char_cat.get_original(line)) for line in rc]

        for line in zip(a, b, c):
            print(line)
