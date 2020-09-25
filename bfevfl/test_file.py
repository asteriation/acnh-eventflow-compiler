from __future__ import annotations

import unittest

from bitstring import Bits

from .datatype import IntType, StrType, BoolType, TypedValue
from .actors import Actor, Action, Param
from .nodes import Node, RootNode, ActionNode
from .file import File

class TestFile(unittest.TestCase):
    def test_toilet(self):
        name = 'Ftr_Toilet'

        sys_actor = Actor('EventFlowSystemActor')
        sys_actor.register_action(Action(
            'EventFlowSystemActor',
            'EventFlowActionWaitFrame',
            [Param('WaitFrame', IntType)],
        ))
        sys_actor.actions['EventFlowActionWaitFrame'].mark_used()

        player_actor = Actor('Player')
        player_actor.register_action(Action(
            'Player',
            'EventFlowActionOpenMessageWindow',
            [Param('MessageID', StrType), Param('IsCloseMessageWindow', BoolType)],
        ))
        player_actor.register_action(Action(
            'Player',
            'EventFlowActionPlayerClearFoodPowerup',
            [],
        ))
        player_actor.actions['EventFlowActionOpenMessageWindow'].mark_used()
        player_actor.actions['EventFlowActionPlayerClearFoodPowerup'].mark_used()

        nodes: List[Node] = [
            ActionNode('Event0', player_actor.actions['EventFlowActionOpenMessageWindow'],
                {
                    'MessageID': TypedValue(StrType, 'TalkFtr/FTR_Toilet:001'),
                    'IsCloseMessageWindow': TypedValue(BoolType, False),
                }),
            ActionNode('Event2', player_actor.actions['EventFlowActionPlayerClearFoodPowerup'], {}),
            ActionNode('Root', sys_actor.actions['EventFlowActionWaitFrame'],
                {
                    'WaitFrame': TypedValue(IntType, 30),
                }),
            ActionNode('Event4', sys_actor.actions['EventFlowActionWaitFrame'],
                {
                    'WaitFrame': TypedValue(IntType, 33),
                }),
            RootNode('Root', []),
        ]
        nodes[1].add_out_edge(nodes[3])
        nodes[2].add_out_edge(nodes[1])
        nodes[3].add_out_edge(nodes[0])
        nodes[4].add_out_edge(nodes[2])

        with open('tests/bfevfl/Ftr_Toilet.bfevfl', 'rb') as f:
            expected = f.read()

        file = File('Ftr_Toilet', [sys_actor, player_actor], nodes)
        self.assertEqual(file.prepare_bitstream(), Bits(expected))

