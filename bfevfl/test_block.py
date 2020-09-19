from __future__ import annotations

import unittest

from bitstring import Bits

from .block import Block, DataBlock, ContainerBlock

Block.__abstractmethods__ = frozenset()
DataBlock.__abstractmethods__ = frozenset()
ContainerBlock.__abstractmethods__ = frozenset()

class TestDataBlock(unittest.TestCase):
    def test_no_pointers(self):
        db = DataBlock(8)
        db.buffer.overwrite(Bits(b'12345678'))

        self.assertEqual(len(db), 8)
        self.assertEqual(db.prepare_bitstream(), Bits(b'12345678'))

    def test_pointers(self):
        b1, b2 = Block(), Block()
        b1.set_offset(0x123456789abcdef0)
        b2.set_offset(0x56789abcdef01234)

        db = DataBlock(16)
        db._add_pointer(8, b2)
        db._add_pointer(0, b1)

        self.assertEqual(db.prepare_bitstream(), Bits(b'\xf0\xde\xbc\x9a\x78\x56\x34\x12\x34\x12\xf0\xde\xbc\x9a\x78\x56'))
        self.assertEqual(db.get_all_pointers(), [0, 8])

class TestContainerBlock(unittest.TestCase):
    def test_empty_block(self):
        cb = ContainerBlock([])

        # should be no-op
        cb.set_offset(-5)

        self.assertEqual(len(cb), 0)
        self.assertEqual(cb.alignment(), 8)
        self.assertEqual(cb.prepare_bitstream(), Bits(b''))

    def test_multiple_subblocks(self):
        b1, b2, b3 = DataBlock(8), DataBlock(3), DataBlock(6)
        b1.alignment = b2.alignment = b3.alignment = lambda: 1
        cb = ContainerBlock([b1, b2, b3])

        b1.buffer.overwrite(Bits(b'abcdefgh'))
        b2.buffer.overwrite(Bits(b'!@#'))
        b3.buffer.overwrite(Bits(b'123456'))

        self.assertEqual(len(cb), 8 + 3 + 6)
        self.assertEqual(cb.prepare_bitstream(), Bits(b'abcdefgh!@#123456'))

    def test_offset(self):
        b1, b2, b3 = DataBlock(8), DataBlock(3), DataBlock(6)
        b1.alignment = b2.alignment = b3.alignment = lambda: 1
        cb = ContainerBlock([b1, b2, b3])

        self.assertEqual(cb.offset, 0)
        self.assertEqual(b1.offset, 0)
        self.assertEqual(b2.offset, 0 + 8)
        self.assertEqual(b3.offset, 0 + 8 + 3)

        cb.set_offset(135)
        self.assertEqual(cb.offset, 135)
        self.assertEqual(b1.offset, 135)
        self.assertEqual(b2.offset, 135 + 8)
        self.assertEqual(b3.offset, 135 + 8 + 3)

    def test_alignment(self):
        b1, b2, b3 = DataBlock(8), DataBlock(3), DataBlock(6)
        b1.alignment = lambda: 5
        b2.alignment = lambda: 7
        b3.alignment = lambda: 4
        cb = ContainerBlock([b1, b2, b3])

        b1.buffer.overwrite(Bits(b'abcdefgh'))
        b2.buffer.overwrite(Bits(b'!@#'))
        b3.buffer.overwrite(Bits(b'123456'))

        # expected layout:
        # [b1 - 8B] [6B pad] [b2 - 3B] [3B pad] [b3 - 6B]
        self.assertEqual(len(cb), 8 + 6 + 3 + 3 + 6)
        self.assertEqual(cb.prepare_bitstream(), Bits(b'abcdefgh\0\0\0\0\0\0!@#\0\0\x00123456'))
        self.assertEqual(cb.alignment(), 5)

    def test_pointers(self):
        b = Block()
        b.set_offset(0x123456789abcdef0)

        db = DataBlock(16)
        db.alignment = lambda: 8
        db._add_pointer(4, b)

        cb = ContainerBlock([db])
        cb.set_offset(58)

        self.assertEqual(db.prepare_bitstream(), Bits(b'\0\0\0\0\xf0\xde\xbc\x9a\x78\x56\x34\x12\0\0\0\0'))
        self.assertEqual(db.get_all_pointers(), [62])
