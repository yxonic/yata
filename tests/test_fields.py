from yata.fields import Converter
import unittest


class TestFields(unittest.TestCase):
    def test_converter(self):
        id_converter = Converter()
        self.assertEqual(id_converter.apply('', 'hello'), 'hello')
        self.assertEqual(id_converter.apply('', ['hello']), ['hello'])

        concat_converter = Converter(lambda x: ','.join(x))
        self.assertEqual(concat_converter.apply('', ['1', '2']), '1,2')

        concat_converter = Converter(lambda k, x: k + ':' + ','.join(x))
        self.assertEqual(concat_converter.apply('key', ['1', '2']), 'key:1,2')


if __name__ == '__main__':
    unittest.main()
