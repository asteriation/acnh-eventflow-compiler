from __future__ import annotations

import unittest

from bitstring import Bits, pack

from .str_ import StringPool
from .dic_ import Dictionary

class TestDictioanry(unittest.TestCase):
    def setUp(self):
        self.sp = StringPool([
            'Root',
            'MessageID',
            'MessageWindow',
        ])
        self.sp.set_offset(0xbadc0ffee0ddf00d)

    def test_empty_dic(self):
        dc = Dictionary([], self.sp)
        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'DIC \0\0\0\0\xff\xff\xff\xff\0\0\0\0') +
                pack('uintle:64', self.sp.empty.offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10])

    def test_dic_1(self):
        dc = Dictionary(['Root'], self.sp)
        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'DIC \1\0\0\0\xff\xff\xff\xff\1\0\0\0') +
                pack('uintle:64', self.sp.empty.offset) +
                Bits(b'\2\0\0\0\0\0\1\0') +
                pack('uintle:64', self.sp.strings['Root'].offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10, 0x20])

    def test_dic_2(self):
        dc = Dictionary(['MessageID', 'MessageWindow'], self.sp)
        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'DIC \2\0\0\0\xff\xff\xff\xff\2\0\0\0') +
                pack('uintle:64', self.sp.empty.offset) +
                Bits(b'\2\0\0\0\0\0\1\0') +
                pack('uintle:64', self.sp.strings['MessageID'].offset) +
                Bits(b'\0\0\0\0\1\0\2\0') +
                pack('uintle:64', self.sp.strings['MessageWindow'].offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10, 0x20, 0x30])

    # todo: larger tests

