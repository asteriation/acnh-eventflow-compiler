from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Set, Callable, List, Tuple

from bfevfl.datatype import IntType, StrType, BoolType, Type
from bfevfl.actors import Actor, Action, Query, Param
from bfevfl.nodes import Node, ActionNode, SwitchNode, SubflowNode
from bfevfl.file import File

from parse import tokenize, parse
from util import find_postorder

def param_str_to_param(pstr: str) -> Param:
    name, type_ = pstr.split(':')
    name, type_ = name.strip(), type_.strip()
    return Param(name, Type(type_))

def compare_version(v1: str, v2: str) -> int:
    t1 = tuple(int(p) for p in v1.split('.'))
    t2 = tuple(int(p) for p in v2.split('.'))
    if t1 < t2:
        return -1
    elif t1 > t2:
        return 1
    return 0

def actor_gen_prepare(csvr, version: str) -> Callable[[str, str], Actor]:
    actions: List[Tuple[str, List[Param]]] = []
    queries: List[Tuple[str, List[Param], Type]] = []

    # MaxVersion, Type, Name, Parameters[, Return]
    header = next(csvr)
    maxver_i, type_i, name_i, param_i, return_i = (header.index(s) for s in ('MaxVersion', 'Type', 'Name', 'Parameters', 'Return'))
    for row in csvr:
        maxver = row[maxver_i]
        if maxver == 'pseudo' or compare_version(maxver, version) < 0:
            continue
        params = []
        if row[param_i].strip():
            params = [param_str_to_param(p) for p in row[param_i].split(';')]
        if row[type_i] == 'Action':
            actions.append(('EventFlowAction' + row[name_i], params))
        else:
            queries.append(('EventFlowQuery' + row[name_i], params, Type(row[type_i])))

    def inner(name: str, secondary_name: str) -> Actor:
        actor = Actor(name, secondary_name)
        for aname, params in actions:
            actor.register_action(Action(name, aname, params))
        for qname, params, rtype in queries:
            actor.register_query(Query(name, qname, params, rtype, True))
        return actor

    return inner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='functions.csv', default='functions.csv',
            help='functions.csv for EventFlow function type information (default: ./functions.csv)')
    parser.add_argument('-v', metavar='version', default='9.9.9', help='target game version (default: 9.9.9)')
    parser.add_argument('-d', metavar='output_directory', help='output directory')
    parser.add_argument('-o', metavar='file', help='file to output to, overrides -d, ' +
            'cannot be used for multiple input files')
    parser.add_argument('files', metavar='evfl_file', nargs='+', help='.evfl files to compile')
    args = parser.parse_args()

    if len(args.files) > 1 and args.o:
        print('-o cannot be used with multiple input files', file=sys.stderr)
        sys.exit(1)

    fcsv = Path(args.f)
    if not fcsv.exists() or not fcsv.is_file():
        print(f'cannot open {args.f}', file=sys.stderr)
        sys.exit(1)
    with fcsv.open('rt') as f:
        actor_gen = actor_gen_prepare(csv.reader(f), args.v)

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

        tokens = tokenize(evfl)
        roots, actors = parse(tokens, actor_gen)

        nodes: Set[Node] = set()
        entrypoints = set(r.name for r in roots)
        for root in roots:
            for node in find_postorder(root):
                if node in nodes:
                    continue
                if isinstance(node, ActionNode):
                    node.action.mark_used()
                elif isinstance(node, SwitchNode):
                    node.query.mark_used()
                elif isinstance(node, SubflowNode):
                    if node.ns == '' and node.called_root_name not in entrypoints:
                        print(f'subflow call for {node.called_root_name} but matching flow/entrypoint not found')
                        sys.exit(2)

                nodes.add(node)

        bfevfl = File(name, actors, list(nodes))
        with of.open('wb') as f:
            f.write(bfevfl.prepare_bitstream().bytes)

