import re
from yata.loaders import TableLoader, DataLoader
from yata.fields import Numeral, Words, Categorical, Chars, Converter


def get_key(line):
    for m in re.finditer(r'<([a-z0-9\-]+)>', line[0]):
        yield m.group(1)


def get_context(key, line):
    line = line.replace('<{}>'.format(key), '###')
    line = re.sub('<[a-z0-9\\-]+>', '', line)
    left = re.findall('(.{0,10})###', line)[0]
    right = re.findall('###(.{0,10})', line)[0]
    return (left, right)


def main():
    word_cat = Categorical()
    text = word_cat(Words(':', 5))

    char_cat = Categorical()
    char = char_cat(Chars(10, '#'))
    label = Numeral('int32')

    data1 = TableLoader('test.table', key='id',
                        fields={'label': label,
                                'content->words': text,
                                'author': Converter(lambda x: x)},
                        index=['label', 'author'])

    data2 = TableLoader('headless.table', with_header=False,
                        key=get_key,
                        fields={'0->lc,rc':
                                char(Converter(get_context))})

    loader = DataLoader(data1, data2)

    print('loader:')
    print('keys:', loader.keys)
    print('fields:', loader.fields)
    print()

    for batch in loader.shuffle().epoch(3):
        lc, rc = batch[1]['lc'], batch[1]['rc']

        a = batch[0]
        b = [''.join(char_cat.get_original(line)) for line in lc]
        c = [''.join(char_cat.get_original(line)) for line in rc]

        print('new batch')
        for line in zip(a, b, c):
            print(line)

    print()
    left, right = loader.shuffle().split(0.5, on=['label', 'author'])
    print(left.keys)
    print(right.keys)


if __name__ == '__main__':
    main()
