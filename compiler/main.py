from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Set, Callable, List, Tuple

from bfevfl.datatype import IntType, StrType, BoolType, Type
from bfevfl.actors import Actor, Action, Query, Param
from bfevfl.nodes import Node, ActionNode, SwitchNode, SubflowNode, RootNode
from bfevfl.file import File
from bfevfl.util import find_postorder

from funcparserlib.lexer import LexerError
from funcparserlib.parser import NoParseError

from compiler.parse import tokenize, parse, parse_custom_rules
from compiler.optimize import optimize_names, optimize_merge_identical, make_counter_renamer, make_compact_renamer
from compiler.logger import init_logger, setup_logger, emit_error, emit_fatal, LogException, LogError, LogFatal

def param_str_to_param(pstr: str) -> Param:
    name, type_ = pstr.split(':')
    name, type_ = name.strip(), type_.strip()
    return Param(name, Type(type_))

def actor_gen_prepare(csvr) -> Tuple[Callable[[str, str], Actor], List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    actions: List[Tuple[str, List[Param]]] = []
    queries: List[Tuple[str, List[Param], Type, bool]] = []
    action_rules: List[Tuple[str, str]] = []
    query_rules: List[Tuple[str, str]] = []
    query_op_rules: List[Tuple[str, str]] = []

    header = next(csvr)
    type_i, name_i, param_i, return_i, rtype_i, rule_i = (header.index(s) for s in ('Type', 'Name', 'Parameters', 'Return', 'ParseType', 'ParseRule'))
    for row in csvr:
        row = [s.strip() for s in row]
        params = []
        if row[param_i]:
            params = [param_str_to_param(p) for p in row[param_i].split(';') if p.strip() != '...']
        if row[type_i] == 'Action':
            actions.append(('EventFlowAction' + row[name_i], params))
            action_rules.append((row[name_i], row[rule_i]))
        else:
            type_ = row[return_i]
            inverted = False
            if type_ == 'inverted_bool':
                type_, inverted = 'bool', True
            queries.append(('EventFlowQuery' + row[name_i], params, Type(type_), inverted))
            if row[rtype_i] == 'function':
                query_rules.append((row[name_i], row[rule_i]))
            elif row[rtype_i] == 'predicate':
                query_op_rules.append((row[name_i], row[rule_i]))
            elif row[rtype_i]:
                emit_fatal(f'ParseType for "{row[name_i]}" not recognized: "{row[rtype_i]}"')
                raise LogFatal()

    def inner(name: str, secondary_name: str) -> Actor:
        actor = Actor(name, secondary_name)
        for aname, params in actions:
            actor.register_action(Action((name, secondary_name), aname, params))
        for qname, params, rtype, inverted in queries:
            actor.register_query(Query((name, secondary_name), qname, params, rtype, inverted))
        return actor

    return (inner,
            [rule for rule in action_rules if rule],
            [rule for rule in query_rules if rule],
            [rule for rule in query_op_rules if rule])

def process_file(filename, output_dir, output_name, actor_gen, optimizer_flags, **kwargs):
    init_logger(filename)

    if_ = Path(filename)
    of = output_dir / output_name if output_name else output_dir / if_.with_suffix('.bfevfl').name
    name = if_.with_suffix('').name
    if not if_.exists():
        emit_error('file not found, skipping')
        raise LogError()

    with if_.open('rt') as f:
        evfl = f.read()
        setup_logger(evfl)

    tokens = tokenize(evfl)
    roots, actors = parse(
        tokens,
        actor_gen,
        exported_tco = optimizer_flags['exported_tco'],
        **kwargs
    )
    if optimizer_flags['merge_duplicate']:
        optimize_merge_identical(roots)
    if optimizer_flags['short_event_names']:
        optimize_names(roots, make_compact_renamer)
    else:
        optimize_names(roots, make_counter_renamer)
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
                if node.ns == '':
                    if node.called_root_name not in entrypoints:
                        emit_error(f'subflow call for {node.called_root_name} but matching flow/entrypoint not found')
                        raise LogError()

            nodes.add(node)

    bfevfl = File(name, actors, list(nodes))
    with of.open('wb') as f:
        f.write(bfevfl.prepare_bitstream().bytes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions', metavar='functions.csv', default='functions.csv',
            help='functions.csv for EventFlow function type information (default: ./functions.csv)')
    parser.add_argument('-d', metavar='output_directory', help='output directory')
    parser.add_argument('-o', metavar='file', help='file to output to, overrides -d, ' +
            'cannot be used for multiple input files')
    parser.add_argument('files', metavar='evfl_file', nargs='+', help='.evfl files to compile')
    parser.add_argument('--optimize', action='store_true', help='Enable all optimizer flags')

    optimizer = parser.add_argument_group('Optimizer Flags', description='Optimizer flags for smaller output')
    optimizer.add_argument('--fexported-tco', action='store_true', help='Perform tail-call-optimization for exported flows')
    optimizer.add_argument('--fmerge-duplicate', action='store_true', help='Merge duplicate segments of code')
    optimizer.add_argument('--fshort-event-names', action='store_true', help='Rename all events to very short names (instead of Event0, Event1, etc.)')

    args = parser.parse_args()
    if args.optimize:
        args.fexported_tco = True
        args.fmerge_duplicate = True
        args.fshort_event_names = True

    optimizer_flags = dict(vars(args))
    for k in list(optimizer_flags.keys()):
        if k.startswith('f'):
            optimizer_flags[k[1:]] = optimizer_flags[k]
        del optimizer_flags[k]

    try:
        if len(args.files) > 1 and args.o:
            emit_fatal('-o cannot be used with multiple input files')
            raise LogFatal()

        setattr(args, "f", args.functions)
        fcsv = Path(args.functions)
        if not fcsv.exists() or not fcsv.is_file():
            emit_fatal(f'cannot open {args.f}')
            raise LogFatal()
        with fcsv.open('rt') as f:
            actor_gen, action_rules, query_rules, query_op_rules = actor_gen_prepare(csv.reader(f))

        function_prefix = [
            ('ID', None),
            ('ACTOR', 'placeholder'),
            ('DOT', None),
            ('ID', None),
            ('LPAREN', None),
        ]
        custom_action_parsers = parse_custom_rules(action_rules, function_prefix)
        custom_query_parsers = parse_custom_rules(query_rules, function_prefix)
        custom_query_op_parser = parse_custom_rules(query_op_rules, [])[1]

        output_dir = Path('.')
        output_name = None
        if args.d and not args.o:
            output_dir = Path(args.d)
            if not output_dir.exists():
                output_dir.mkdir()
            if output_dir.is_file():
                emit_fatal('output directory is a file')
                raise LogFatal()
        if args.o:
            output_name = args.o

        success = True
        for filename in args.files:
            try:
                process_file(
                    filename,
                    output_dir,
                    output_name,
                    actor_gen,
                    optimizer_flags,
                    custom_action_parser_pfx = custom_action_parsers[0],
                    custom_action_parser_reg = custom_action_parsers[1],
                    custom_query_parser_pfx = custom_query_parsers[0],
                    custom_query_parser_reg = custom_query_parsers[1],
                    custom_query_parser_op = custom_query_op_parser,
                )
            except LogError:
                success = False
            except NoParseError as e:
                success = False
                pos, msg = e.msg.split(':', 1)
                start, end = pos.split('-', 1)
                start = tuple(int(x) for x in start.split(',', 1))
                end = tuple(int(x) for x in end.split(',', 1))
                emit_error(e.msg, start, end)

        if not success:
            sys.exit(1)
    except LogFatal:
        sys.exit(2)

if __name__ == '__main__':
    main()
