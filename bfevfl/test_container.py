from __future__ import annotations

from collections import OrderedDict
import math
import unittest

from bitstring import Bits, pack

from .datatype import Argument, ArgumentType, BoolType, FloatType, IntType, StrType, TypedValue
from .str_ import StringPool
from .dic_ import Dictionary
from .container import ArgumentContainerItem, IntContainerItem, BoolContainerItem, FloatContainerItem, StringContainerItem, Container

class TestContainer(unittest.TestCase):
    def test_argument_container(self):
        ci = ArgumentContainerItem('arg0')
        self.assertEqual(ci.prepare_bitstream(), Bits(b'\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\4\0arg0\0'))
        self.assertEqual(ci.get_all_pointers(), [8])

    def test_int_container(self):
        ci = IntContainerItem(0xbaadf00d)
        self.assertEqual(ci.prepare_bitstream(), Bits(b'\2\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\x0d\xf0\xad\xba'))
        self.assertEqual(ci.get_all_pointers(), [8])

    def test_bool_container(self):
        ci = BoolContainerItem(True)
        self.assertEqual(ci.prepare_bitstream(), Bits(b'\3\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\x80'))
        self.assertEqual(ci.get_all_pointers(), [8])

    def test_float_container(self):
        ci = FloatContainerItem(3.1415)
        self.assertEqual(ci.prepare_bitstream(),
                Bits(b'\4\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0') + pack('floatle:32', 3.1415)
        )
        self.assertEqual(ci.get_all_pointers(), [8])

    def test_string_container(self):
        ci = StringContainerItem('FooBar')
        self.assertEqual(ci.prepare_bitstream(),
                Bits(b'\5\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\6\0FooBar\0')
        )
        self.assertEqual(ci.get_all_pointers(), [8])

    def test_container_small_1(self):
        sp = StringPool(['MessageID', 'MessageWindow'])
        sp.offset = 0xbadc0ffee0ddf00d

        ci = Container(OrderedDict((
            ('MessageID', TypedValue(StrType, 'FooBarBaz')),
            ('MessageWindow', TypedValue(BoolType, True))
        )), sp)
        dict_bs = Dictionary(['MessageID', 'MessageWindow'], sp).prepare_bitstream()
        str_bs = StringContainerItem('FooBarBaz').prepare_bitstream()
        bool_bs = BoolContainerItem(True).prepare_bitstream()
        str_pad = len(dict_bs) // 8 % 8
        str_offset = math.ceil(len(dict_bs) // 8 / 8) * 8
        bool_pad = len(str_bs) // 8 % 8
        bool_offset = str_offset + math.ceil(len(str_bs) // 8 / 8) * 8
        self.assertEqual(ci.prepare_bitstream(),
                Bits(b'\1\0\2\0\0\0\0\0') +
                pack('uintle:64', 0x20) +
                pack('uintle:64', 0x20 + str_offset) +
                pack('uintle:64', 0x20 + bool_offset) +
                dict_bs + Bits(str_pad * 8) + str_bs + Bits(bool_pad * 8) + bool_bs
        )

    def test_container_small_2(self):
        sp = StringPool(['MessageID', 'MessageWindow', 'Test'])
        sp.offset = 0xbadc0ffee0ddf00d

        ci = Container(OrderedDict((
            ('MessageID', TypedValue(IntType, 12345678)),
            ('MessageWindow', TypedValue(FloatType, 3.1415)),
            ('Test', TypedValue(ArgumentType, Argument('arg0')))
        )), sp)
        dict_bs = Dictionary(['MessageID', 'MessageWindow', 'Test'], sp).prepare_bitstream()
        int_bs = IntContainerItem(12345678).prepare_bitstream()
        float_bs = FloatContainerItem(3.1415).prepare_bitstream()
        arg_bs = ArgumentContainerItem('arg0').prepare_bitstream()

        int_pad = len(dict_bs) // 8 % 8
        int_offset = math.ceil(len(dict_bs) // 8 / 8) * 8
        float_pad = len(int_bs) // 8 % 8
        float_offset = int_offset + math.ceil(len(int_bs) // 8 / 8) * 8
        arg_pad = len(float_bs) // 8 % 8
        arg_offset = float_offset + math.ceil(len(float_bs) // 8 / 8) * 8
        self.assertEqual(ci.prepare_bitstream(),
                Bits(b'\1\0\3\0\0\0\0\0') +
                pack('uintle:64', 0x28) +
                pack('uintle:64', 0x28 + int_offset) +
                pack('uintle:64', 0x28 + float_offset) +
                pack('uintle:64', 0x28 + arg_offset) +
                dict_bs +
                Bits(int_pad * 8) + int_bs +
                Bits(float_pad * 8) + float_bs +
                Bits(arg_pad * 8) + arg_bs
        )

