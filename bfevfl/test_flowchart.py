from __future__ import annotations

import unittest

from bitstring import Bits, pack

from .datatype import TypedValue, IntType, FloatType, BoolType
from .actors import Actor, Action, Query
from .str_ import String, StringPool
from .dic_ import Dictionary
from .array import BlockArray, BlockPtrArray
from .container import Container
from .flowchart import (_Actor, _Event, _ActionEvent,
        _SubflowIndexArray, _VarDef, _Pad24, _Entrypoint, _FlowchartHeader, Flowchart)

class Test_Actor(unittest.TestCase):
    def setUp(self):
        self.sp = StringPool(['EventFlowSystemActor', 'sec_name'] +
            [f'action{i}' for i in range(10)] +
            [f'query{i}' for i in range(10)]
        )

    def test_empty(self):
        actor = _Actor('EventFlowSystemActor', '', None, None, self.sp)

        self.assertEqual(actor.get_all_pointers(), [0, 8, 16])
        self.assertEqual(actor.prepare_bitstream(),
            pack('uintle:64', self.sp['EventFlowSystemActor'].offset) +
            pack('uintle:64', self.sp.empty.offset) +
            pack('uintle:64', self.sp.empty.offset) +
            Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0') +
            Bits(b'\0\0\0\0\xff\xff\1')
        )

    def test_nonempty(self):
        actions = BlockPtrArray[String]([self.sp[f'action{i}'] for i in range(10)])
        queries = BlockPtrArray[String]([self.sp[f'query{i}'] for i in range(10)])
        actions.offset = 218093421830921
        queries.offset = 895719471249182
        actor = _Actor('EventFlowSystemActor', 'sec_name', actions, queries, self.sp)

        self.assertEqual(actor.get_all_pointers(), [0, 8, 16, 24, 32])
        self.assertEqual(actor.prepare_bitstream(),
            pack('uintle:64', self.sp['EventFlowSystemActor'].offset) +
            pack('uintle:64', self.sp['sec_name'].offset) +
            pack('uintle:64', self.sp.empty.offset) +
            pack('uintle:64', actions.offset) +
            pack('uintle:64', queries.offset) +
            Bits(b'\0\0\0\0\0\0\0\0') +
            Bits(b'\x0a\0\x0a\0\xff\xff\1')
        )

class Test_ActionEvent(unittest.TestCase):
    def setUp(self):
        self.sp = StringPool(['action'])

    def test_no_params(self):
        action = _ActionEvent('action', 0xFFFF, 0, 4, None, self.sp)

        self.assertEqual(action.get_all_pointers(), [0])
        self.assertEqual(action.prepare_bitstream(),
                pack('uintle:64', self.sp['action'].offset) +
                Bits(b'\0\0\xff\xff\0\0\4\0') +
                pack('uintle:64', 0) +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')
        )

    def test_params(self):
        params = Container({}, self.sp)
        params.offset = 21934812940
        action = _ActionEvent('action', 1, 1, 3, params, self.sp)

        self.assertEqual(action.get_all_pointers(), [0, 16])
        self.assertEqual(action.prepare_bitstream(),
                pack('uintle:64', self.sp['action'].offset) +
                Bits(b'\0\0\1\0\1\0\3\0') +
                pack('uintle:64', params.offset) +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')
        )

class Test_SubflowIndexArray(unittest.TestCase):
    def test_empty(self):
        arr = _SubflowIndexArray([])
        self.assertEqual(arr.prepare_bitstream(), Bits(b''))

    def test_nonempty(self):
        arr = _SubflowIndexArray([0, 5, 4, 0x3321, 7])
        self.assertEqual(arr.prepare_bitstream(), Bits(b'\0\0\5\0\4\0\x21\x33\7\0'))

class Test_VarDef(unittest.TestCase):
    def test_int(self):
        v = _VarDef(TypedValue(IntType, 4))
        self.assertEqual(v.prepare_bitstream(), Bits(b'\4\0\0\0\0\0\0\0\1\0\2\0'))

    def test_bool_true(self):
        v = _VarDef(TypedValue(BoolType, True))
        self.assertEqual(v.prepare_bitstream(), Bits(b'\1\0\0\x80\0\0\0\0\1\0\3\0'))

    def test_bool_false(self):
        v = _VarDef(TypedValue(BoolType, False))
        self.assertEqual(v.prepare_bitstream(), Bits(b'\0\0\0\0\0\0\0\0\1\0\3\0'))

    def test_float(self):
        v = _VarDef(TypedValue(FloatType, -500.32))
        self.assertEqual(v.prepare_bitstream(), pack('floatle:32', -500.32) + Bits(b'\0\0\0\0\1\0\4\0'))

class Test_Entrypoint(unittest.TestCase):
    def test_simple(self):
        ep = _Entrypoint(None, _Pad24(), None, 0x2345)

        self.assertEqual(ep.get_all_pointers(), [])
        self.assertEqual(ep.prepare_bitstream(),
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x45\x23')
        )

    def test_subflow_index(self):
        si = _SubflowIndexArray(list(range(10)))
        si.offset = 1289471295
        ep = _Entrypoint(None, _Pad24(), si, 0x2345)

        self.assertEqual(ep.get_all_pointers(), [0])
        self.assertEqual(ep.prepare_bitstream(),
                pack('uintle:64', si.offset) +
                Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0a\0\0\0\x45\x23')
        )

    def test_vardef(self):
        sp = StringPool(['hello', 'world'])
        names = Dictionary(['hello', 'world'], sp)
        vardefs = BlockArray[_VarDef]([
            _VarDef(TypedValue(IntType, 5123)),
            _VarDef(TypedValue(FloatType, 1e10)),
        ])
        vardefs.offset = 124513
        ep = _Entrypoint(names, vardefs, None, 0x2345)

        self.assertEqual(ep.get_all_pointers(), [8, 16])
        self.assertEqual(ep.prepare_bitstream(),
                Bits(b'\0\0\0\0\0\0\0\0') +
                pack('uintle:64', names.offset) +
                pack('uintle:64', vardefs.offset) +
                Bits(b'\0\0\2\0\x45\x23')
        )

class Test_FlowchartHeader(unittest.TestCase):
    def test(self):
        pool = StringPool(['Flowchart'])
        actors = BlockArray[_Actor]([])
        actors.n = 5
        events = BlockArray[_Event]([])
        events.n = 1
        entrypoint_names = Dictionary([], pool)
        entrypoints = BlockArray[_Entrypoint]([])
        entrypoints.n = 4

        header = _FlowchartHeader('Flowchart', 3, 7, actors, events, entrypoint_names, entrypoints, pool)

        header.offset = 0x80000
        pool.offset = 0x100000
        actors.offset = 0xabcdef01
        events.offset = 0x01234567
        entrypoint_names.offset = 0x456789ab
        entrypoints.offset = 0x32324512

        self.assertEqual(header.get_all_pointers(), [0x80020, 0x80028, 0x80030, 0x80038, 0x80040])
        self.assertEqual(header.prepare_bitstream(),
                Bits(b'EVFL\0\0\x08\0\0\0\0\0\0\0\0\0') +
                Bits(b'\5\0\3\0\7\0\1\0\4\0\0\0\0\0\0\0') +
                pack('uintle:64', pool['Flowchart'].offset) +
                pack('uintle:64', actors.offset) +
                pack('uintle:64', events.offset) +
                pack('uintle:64', entrypoint_names.offset) +
                pack('uintle:64', entrypoints.offset)
        )

class TestFlowchart(unittest.TestCase):
    def setUp(self):
        self.sp = StringPool([
            'Flowchart'
        ] + [f'{s}{i}' for i in range(100) for s in ('Actor', 'Action', 'Query', 'Event')])
        self.sp.offset = 0x1234

    def test_empty(self):
        fc = Flowchart('Flowchart', [], [], [], self.sp)

        self.assertEqual(fc.prepare_bitstream(),
            Bits(b'EVFL\x34\x12\0\0\0\0\0\0\0\0\0\0') +
            Bits(b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0') +
            pack('uintle:64', self.sp['Flowchart'].offset) +
            Bits(b'\x48\0\0\0\0\0\0\0\x48\0\0\0\0\0\0\0') +
            Bits(b'\x48\0\0\0\0\0\0\0\x60\0\0\0\0\0\0\0') +
            Bits(b'DIC \0\0\0\0\xff\xff\xff\xff\0\0\0\0') +
            pack('uintle:64', self.sp.empty.offset)
        )

    def test_actor_filtering(self):
        actors = [Actor(f'Actor{i}') for i in range(100)]
        for actor in actors:
            for i in range(50):
                actor.register_action(Action(actor.name, f'Action{i}', []))
                actor.register_query(Query(actor.name, f'Query{i}', [], IntType))
        actors[0].actions['Action0'].mark_used()

        fc = Flowchart('Flowchart', actors, [], [], self.sp)
        self.assertLess(len(fc), 200)

    # TODO more tests

