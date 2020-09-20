from __future__ import annotations

import unittest

from bitstring import Bits, pack

from .relt import RelocationTable

class TestRelocationTable(unittest.TestCase):
    def test_empty_relt(self):
        dc = RelocationTable([])
        dc.set_offset(0xbaadf00d)

        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'RELT\x15\xf0\xad\xba\1\0\0\0\0\0\0\0') +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\x15\xf0\xad\xba\0\0\0\0\0\0\0\0')
        )
        self.assertEqual(dc.get_all_pointers(), [])

    def test_relt_one_entry(self):
        dc = RelocationTable([0x4f8, 0x400, 0x408, 0x418])
        dc.set_offset(0xbaadf00d)

        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'RELT\x15\xf0\xad\xba\1\0\0\0\0\0\0\0') +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\x15\xf0\xad\xba\0\0\0\0\1\0\0\0') +
                Bits(b'\0\4\0\0\x0b\x00\x00\x80')
        )
        self.assertEqual(dc.get_all_pointers(), [])

    def test_relt_multiple_entres(self):
        dc = RelocationTable([0x980, 0x4f8, 0x400, 0x408, 0x418, 0x500])
        dc.set_offset(0xbaadf00d)

        bs = dc.prepare_bitstream()
        self.assertEqual(bs,
                Bits(b'RELT\x15\xf0\xad\xba\1\0\0\0\0\0\0\0') +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\x15\xf0\xad\xba\0\0\0\0\3\0\0\0') +
                Bits(b'\0\4\0\0\x0b\x00\x00\x80') +
                Bits(b'\0\5\0\0\x01\x00\x00\x00') +
                Bits(b'\x80\x09\0\0\x01\x00\x00\x00')
        )
        self.assertEqual(dc.get_all_pointers(), [])

