from __future__ import annotations

import unittest

from bitstring import Bits, pack

from .str_ import StringPool
from .dic_ import Dictionary

class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.sp = StringPool([
            'Root',
            'MessageID',
            'MessageWindow',
            'SubEntryName',
            'TalkEntryPointName',
            'IsChangeSameState',
            'TalkEntryFlowName',
            'StateName',
            'IsChangePreState',
            'ArgFlag0',
            'ArgInt0',
            'ArgFloat0',
            'ArgStr1'
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
                pack('uintle:64', self.sp['Root'].offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10, 0x20])

    def test_dic_2(self):
        dc = Dictionary(['MessageID', 'MessageWindow'], self.sp)
        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'DIC \2\0\0\0\xff\xff\xff\xff\2\0\0\0') +
                pack('uintle:64', self.sp.empty.offset) +
                Bits(b'\2\0\0\0\0\0\1\0') +
                pack('uintle:64', self.sp['MessageID'].offset) +
                Bits(b'\0\0\0\0\1\0\2\0') +
                pack('uintle:64', self.sp['MessageWindow'].offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10, 0x20, 0x30])

    def test_dic_3(self):
        dc = Dictionary([
            'SubEntryName',
            'TalkEntryPointName',
            'IsChangeSameState',
            'TalkEntryFlowName',
            'StateName',
            'IsChangePreState',
            'ArgFlag0',
            'ArgInt0',
            'ArgFloat0',
            'ArgStr1'
        ], self.sp)
        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'DIC \x0a\0\0\0\xff\xff\xff\xff\1\0\0\0') +
                pack('uintle:64', self.sp.empty.offset) +
                Bits(b'\0\0\0\0\7\0\2\0') +
                pack('uintle:64', self.sp['SubEntryName'].offset) +
                Bits(b'\x20\0\0\0\x02\0\x03\0') +
                pack('uintle:64', self.sp['TalkEntryPointName'].offset) +
                Bits(b'\x21\0\0\0\x05\0\x04\0') +
                pack('uintle:64', self.sp['IsChangeSameState'].offset) +
                Bits(b'\x22\0\0\0\x06\0\x0a\0') +
                pack('uintle:64', self.sp['TalkEntryFlowName'].offset) +
                Bits(b'\x22\0\0\0\x01\0\x05\0') +
                pack('uintle:64', self.sp['StateName'].offset) +
                Bits(b'\x30\0\0\0\x06\0\x03\0') +
                pack('uintle:64', self.sp['IsChangePreState'].offset) +
                Bits(b'\x04\0\0\0\x00\0\x08\0') +
                pack('uintle:64', self.sp['ArgFlag0'].offset) +
                Bits(b'\x08\0\0\0\x09\0\x07\0') +
                pack('uintle:64', self.sp['ArgInt0'].offset) +
                Bits(b'\x10\0\0\0\x08\0\x09\0') +
                pack('uintle:64', self.sp['ArgFloat0'].offset) +
                Bits(b'\x24\0\0\0\x0a\0\x04\0') +
                pack('uintle:64', self.sp['ArgStr1'].offset)
        )
        self.assertEqual(dc.get_all_pointers(), [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0])

