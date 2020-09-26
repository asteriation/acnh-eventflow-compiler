from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Set

from bfevfl.datatype import IntType, StrType, BoolType
from bfevfl.actors import Actor, Action, Query, Param
from bfevfl.nodes import Node, ActionNode
from bfevfl.file import File

from parse import tokenize, parse
from util import find_postorder

def actor_gen(name: str) -> Actor:
    actor = Actor(name)
    actor.register_action(Action(name, 'EventFlowActionWaitFrame', [Param('WaitFrame', IntType)]))
    actor.register_action(Action(name, 'EventFlowActionPlayerClearFoodPowerup', []))
    actor.register_action(Action(name, 'EventFlowActionOpenMessageWindow', [Param('MessageID', StrType), Param('IsCloseMessageWindow', BoolType)]))
    actor.register_action(Action(name, 'EventFlowActionUIMoneyAppear', []))
    actor.register_action(Action(name, 'EventFlowActionUIMoneyDisappear', []))
    actor.register_action(Action(name, 'EventFlowActionUISonkatsuPointAppear', []))
    actor.register_action(Action(name, 'EventFlowActionUISonkatsuPointDisappear', []))
    actor.register_action(Action(name, 'EventFlowActionBellCountDown', [Param('Money', IntType), Param('Reverse', BoolType)]))
    actor.register_action(Action(name, 'EventFlowActionLifeSupportPointCountDown', [Param('Point', IntType), Param('Reverse', BoolType)]))
    return actor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='functions.csv', default='functions.csv',
            help='functions.csv for EventFlow function type information (default: ./functions.csv)')
    parser.add_argument('-d', metavar='output_directory', help='output directory')
    parser.add_argument('-o', metavar='file', help='file to output to, overrides -d, ' +
            'cannot be used for multiple input files')
    parser.add_argument('files', metavar='evfl_file', nargs='+', help='.evfl files to compile')
    args = parser.parse_args()

    if len(args.files) > 1 and args.o:
        print('-o cannot be used with multiple input files', file=sys.stderr)
        sys.exit(1)

    # fcsv = Path(args.f)
    # if not fcsv.exists() or not fcsv.is_file():
        # print(f'cannot open {args.f}', file=sys.stderr)
        # sys.exit(1)
    # actor_gen = lambda n: Actor(n)

    output_dir = Path('.')
    output_name = None
    if args.d and not args.o:
        output_dir = Path(args.d)
        if not output_dir.exists():
            output_dir.mkdir()
        if output_dir.is_file():
            print('output directory is a file', file=sys.stderr)
            sys.exit(1)
    if args.o:
        output_name = args.o

    for filename in args.files:
        if_ = Path(filename)
        of = output_dir / output_name if output_name else output_dir / if_.with_suffix('.bfevfl').name
        name = if_.with_suffix('').name
        if not if_.exists():
            print(f'file {filename} not found, skipping', file=sys.stderr)
            continue

        with if_.open('rt') as f:
            evfl = f.read()

        tokens = list(tokenize(evfl))
        roots, actors = parse(tokens, actor_gen)

        nodes: Set[Node] = set()
        for root in roots:
            for node in find_postorder(root):
                if isinstance(node, ActionNode):
                    node.action.mark_used()
                nodes.add(node)

        # print(nodes)

        bfevfl = File(name, actors, list(nodes))
        with of.open('wb') as f:
            f.write(bfevfl.prepare_bitstream().bytes)

        print(if_, name, 'outfile:', of)
        print(nodes)
