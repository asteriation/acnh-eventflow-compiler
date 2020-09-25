from __future__ import annotations

import unittest

from bitstring import Bits

from .str_ import String, StringPool

class TestStr(unittest.TestCase):
    def test_empty_string(self):
        bs = String('').prepare_bitstream()
        self.assertEqual(bs, Bits(b'\0\0\0'))

    def test_odd_string(self):
        bs = String('abc').prepare_bitstream()
        self.assertEqual(bs, Bits(b'\3\0abc\0'))

    def test_even_string(self):
        bs = String('abcdef').prepare_bitstream()
        self.assertEqual(bs, Bits(b'\6\0abcdef\0'))

class TestStringPool(unittest.TestCase):
    def test_empty_pool(self):
        sp = StringPool([])
        bs = sp.prepare_bitstream()
        self.assertEqual(bs, Bits(b'STR \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0'))
        self.assertEqual(sp.empty.offset, 20)
        self.assertEqual(sp.strings, {})

    def test_multiple_strings(self):
        sp = StringPool(['moo', 'iamacow', 'foobar', 'zz'])
        sp.set_offset(10000);
        bs = sp.prepare_bitstream()
        self.assertEqual(bs, Bits(b'STR \0\0\0\0\0\0\0\0\0\0\0\0\4\0\0\0\0\0\0\0\3\0moo\0\7\0iamacow\0\6\0foobar\0\0\2\0zz\0'))
        self.assertEqual(sp.empty.offset, 10020)
        self.assertEqual(list(sp.strings.keys()), ['moo', 'iamacow', 'foobar', 'zz'])
        self.assertEqual(sp['moo'].offset, 10024)
        self.assertEqual(sp['iamacow'].offset, 10030)
        self.assertEqual(sp['foobar'].offset, 10040)
        self.assertEqual(sp['zz'].offset, 10050)


