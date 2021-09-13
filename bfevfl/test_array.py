from __future__ import annotations

import unittest

from bitstring import Bits, pack

from .block import Block, DataBlock
from .array import BlockPtrArray, BlockArray, Uint16Array, IntArray, BoolArray, FloatArray, StringArray

Block.__abstractmethods__ = frozenset() # type: ignore
DataBlock.__abstractmethods__ = frozenset() # type: ignore

class TestArray(unittest.TestCase):
    def test_blockptr_empty(self):
        arr = BlockPtrArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_blockptr(self):
        b = Block()
        b.set_offset(0xbaadf00d)

        arr = BlockPtrArray([b, None])
        self.assertEqual(arr.prepare_bitstream(),
                pack('uintle:64', b.offset) + pack('uintle:64', 0)
        )

    def test_block_empty(self):
        arr = BlockArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_block(self):
        b1, b2, b3 = DataBlock(8), DataBlock(3), DataBlock(6)
        b1.alignment = b2.alignment = b3.alignment = lambda: 1
        arr = BlockArray([b1, b2, b3])

        b1.buffer.overwrite(Bits(b'abcdefgh'))
        b2.buffer.overwrite(Bits(b'!@#'))
        b3.buffer.overwrite(Bits(b'123456'))

        self.assertEqual(len(arr), 8 + 3 + 6)
        self.assertEqual(arr.prepare_bitstream(), Bits(b'abcdefgh!@#123456'))

    def test_uint16_empty(self):
        arr = Uint16Array([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_uint16(self):
        arr = Uint16Array([0x1234, 0x5678, -1])
        self.assertEqual(arr.prepare_bitstream(),
                Bits(b'\x34\x12\x78\x56\xff\xff')
        )

    def test_int_empty(self):
        arr = IntArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_int(self):
        arr = IntArray([0x12345678, 0x9abcdef0, -1])
        self.assertEqual(arr.prepare_bitstream(),
                Bits(b'\x78\x56\x34\x12\xf0\xde\xbc\x9a\xff\xff\xff\xff')
        )

    def test_bool_empty(self):
        arr = BoolArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_bool(self):
        arr = BoolArray([False, True])
        self.assertEqual(arr.prepare_bitstream(),
                Bits(b'\0\0\0\0\1\0\0\x80')
        )

    def test_float_empty(self):
        arr = FloatArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_float(self):
        arr = FloatArray([500.0, -200.1])
        self.assertEqual(arr.prepare_bitstream(),
                pack('floatle:32', 500.0) + pack('floatle:32', -200.1)
        )

    def test_string_empty(self):
        arr = StringArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(''))

    def test_string(self):
        arr = StringArray(['ab', 'zxcvzxcvzxcv', 'vv'])
        self.assertEqual(arr.prepare_bitstream(),
                Bits(b'\2\0ab\0\0\0\0') +
                Bits(b'\x0c\0zxcvzxcvzxcv\0\0') +
                Bits(b'\2\0vv\0')
        )

