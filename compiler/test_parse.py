from __future__ import annotations

import logging
import unittest
from typing import Dict, List, Set

from funcparserlib.lexer import Token, LexerError
from funcparserlib.parser import NoParseError

from bfevfl.datatype import BoolType, FloatType, IntType, StrType, Type
from bfevfl.actors import Action, Actor, Query, Param
from bfevfl.nodes import RootNode, TerminalNode, TerminalNode_, Node

from compiler.parse import tokenize, parse
from compiler.util import find_postorder
from compiler.logger import set_log_level

class TestTokenize(unittest.TestCase):
    def test_indent_dedent_simple(self):
        tokens = tokenize('\n \n')
        token_types = [t.type for t in tokens]
        self.assertEqual(token_types, ['NL', 'INDENT', 'NL', 'DEDENT'])

    def test_multiple_dedent(self):
        tokens = tokenize('\n \n  \n   \n    \n  \n')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'NL', 'INDENT', 'NL', 'INDENT',
                'NL', 'INDENT', 'NL', 'DEDENT', 'DEDENT', 'NL',
                'DEDENT', 'DEDENT'
        ]
        self.assertEqual(token_types, expected)

    def test_intermediate_dedent_to_zero(self):
        tokens = tokenize('\n a\n  b\n   c\n    d\nabc\n  a')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'NL', 'INDENT', 'ID', 'NL',
                'INDENT', 'ID', 'NL', 'INDENT', 'ID', 'NL',
                'DEDENT', 'DEDENT', 'DEDENT', 'DEDENT',
                'ID', 'NL', 'INDENT', 'ID', 'NL', 'DEDENT',
        ]
        self.assertEqual(token_types, expected)

    def test_comment_indent_1(self):
        tokens = tokenize('\n  # \na # b\n # c')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'ID', 'NL'
        ]
        self.assertEqual(token_types, expected)

    def test_comment_indent_2(self):
        tokens = tokenize('\n  a\n  #b\nc')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'NL', 'DEDENT', 'ID', 'NL'
        ]
        self.assertEqual(token_types, expected)

    def test_comment_misindent(self):
        tokens = tokenize('\n  a\n #b\n  c\n   # d\n  e')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'NL', 'ID', 'NL', 'ID', 'NL', 'DEDENT'
        ]
        self.assertEqual(token_types, expected)

    def test_bad_indent(self):
        with self.assertRaises(LexerError):
            tokenize('\n\t\n \t')

    def test_bad_dedent(self):
        with self.assertRaises(LexerError):
            tokenize('\n\t\n\t \n ')

    def test_paren_mismatch1(self):
        with self.assertRaises(LexerError):
            tokenize('(]')

    def test_paren_mismatch2(self):
        with self.assertRaises(LexerError):
            tokenize('[)')

    def test_paren_mismatch_nested1(self):
        with self.assertRaises(LexerError):
            tokenize('[(([])])')

    def test_paren_mismatch_nested2(self):
        with self.assertRaises(LexerError):
            tokenize('([[()])]')

    def test_paren_unopened1(self):
        with self.assertRaises(LexerError):
            tokenize(')')

    def test_paren_unopened2(self):
        with self.assertRaises(LexerError):
            tokenize(']')

    def test_paren_unclosed1(self):
        with self.assertRaises(LexerError):
            tokenize('(')

    def test_paren_unclosed2(self):
        with self.assertRaises(LexerError):
            tokenize('[')

    def test_nonstart_annotation(self):
        with self.assertRaises(LexerError):
            tokenize(' @')

    def test_id(self):
        with self.assertRaises(LexerError):
            tokenize('-abc')
        with self.assertRaises(LexerError):
            tokenize('_abc')
        with self.assertRaises(LexerError):
            tokenize('abc-')
        with self.assertRaises(LexerError):
            tokenize('abc_')
        self.assertEqual(tokenize('abc_d0-3'), [
            Token('ID', 'abc_d0-3'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize('0ab34'), [
            Token('ID', '0ab34'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize('013-34'), [
            Token('ID', '013-34'),
            Token('NL', '')
        ])
        self.assertNotEqual(tokenize('01334'), [
            Token('ID', '01334'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize('01-a'), [
            Token('ID', '01-a'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize('01A_B_C'), [
            Token('ID', '01A_B_C'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize('`1.233 444\\` ee`'), [
            Token('ID', '`1.233 444\\` ee`'),
            Token('NL', '')
        ])
        with self.assertRaises(LexerError):
            tokenize('01-')

    def test_str(self):
        self.assertEqual(tokenize('"a\\"b\\_c"'), [
            Token('STRING', '"a\\"b\\_c"'),
            Token('NL', '')
        ])
        self.assertEqual(tokenize("'a\\'b\\_c'"), [
            Token('STRING', "'a\\'b\\_c'"),
            Token('NL', '')
        ])

    def test_entrypoints1(self):
        tokens = tokenize('\nentrypoint a:\n\ta\nentrypoint b:\n\tc')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'ID', 'COLON', 'NL', 'ID', 'NL',
                'ID', 'ID', 'COLON', 'NL', 'ID', 'NL', 'DEDENT',
        ]
        self.assertEqual(token_types, expected)

    def test_entrypoints2(self):
        tokens = tokenize('\na\nentrypoint b:\n\tc')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'ID', 'NL', 'INDENT', 'ID', 'ID', 'COLON', 'NL', 'ID', 'NL', 'DEDENT',
        ]
        self.assertEqual(token_types, expected)

    def test_entrypoints3(self):
        tokens = tokenize('\n\ta\n\t\tb\nentrypoint c:\nd')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'NL', 'INDENT', 'ID', 'NL', 'DEDENT', 'DEDENT',
                'ID', 'ID', 'COLON', 'NL', 'ID', 'NL',
        ]
        self.assertEqual(token_types, expected)

    def test_entrypoints4(self):
        with self.assertRaises(LexerError):
            tokenize('\nentrypoint e:\n')

    def test_entrypoints5(self):
        tokens = tokenize('\nentrypoint c:\n # f\nentrypoint d: \n\t\t# f\n\td')
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'ID', 'COLON', 'NL', 'ID', 'ID', 'COLON',
                'NL', 'ID', 'NL', 'DEDENT'
        ]
        self.assertEqual(token_types, expected)

    def test_keyword_start_id(self):
        tokens = tokenize('\nentrypoint_e:\n')
        token_types = [t.type for t in tokens]
        expected = ['NL', 'ID', 'COLON', 'NL']
        self.assertEqual(token_types, expected)

    def test_example(self):
        src = ''' \
            flow Test(a: int = 5, b: float = 3.0): # comment
                if SubflowResults[+5] in (1, 2):
                    System.EventFlags['a'] = 3
                else:
                    fork:
                        branch:
                            System.EventFlags["b"] = 7
                        branch:
                            System.EventFlags["c"] = 7
                        branch:
                            pass'''
        expected = [
            'ID', 'ID', 'LPAREN', 'ID', 'COLON', 'ID', 'ASSIGN',
                'INT', 'COMMA', 'ID', 'COLON', 'ID', 'ASSIGN', 'FLOAT',
                'RPAREN', 'COLON', 'NL',
            'INDENT', 'ID', 'ID', 'LSQUARE', 'INT', 'RSQUARE', 'ID',
                'LPAREN', 'INT', 'COMMA', 'INT', 'RPAREN', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'ID', 'COLON', 'NL',
            'INDENT', 'ID', 'COLON', 'NL',
            'INDENT', 'ID', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'ID', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'ID', 'COLON', 'NL',
            'INDENT', 'ID', 'NL',
            'DEDENT', 'DEDENT', 'DEDENT', 'DEDENT',
        ]

        tokens = tokenize(src)
        token_types = [t.type for t in tokens]
        self.assertEqual(token_types, expected)

class TestParser(unittest.TestCase):
    CASES = [
        ('empty_flow1', None),
        ('empty_flow2', None),
        ('single_action_flow', None),
        ('multi_action_flow', None),
        ('multiple_flows', None),
        ('local_unused', None),
        ('empty', None),
        ('newlines', None),
        ('comments_only', None),
        ('flow_decl_second_line', None),
        ('simple_fork', None),
        ('fork_join_action', None),
        ('annotation', None),
        ('leading_comments', None),
        ('subflow', None),
        ('subflow_tail_call', None),
        ('local_subflow', None),
        ('local_subflow_empty', None),
        ('local_subflow_nontco_silent', None),
        ('local_subflow_nontco_export', None),
        ('switch_nonfull', None),
        ('switch_full', None),
        ('switch_nonfull_nonreturn', None),
        ('switch_full_nonreturn', None),
        ('switch_nested', None),
        ('simple_entrypoint', None),
        ('entrypoint', None),
        ('start_entrypoint', None),
        ('switch_entrypoint', None),
        ('fork_entrypoint', None),
        ('err_out_of_flow', NoParseError),
        ('err_fork_no_branch', NoParseError),
        ('err_fork_pass', NoParseError),
        ('err_fork_action', NoParseError),
        ('err_switch_empty', NoParseError),
        ('err_switch_case_empty', NoParseError),
        ('err_switch_pass', NoParseError),
        ('err_consecutive_entrypoint', NoParseError),
    ]

    TEST_DIR = 'tests/parser'

    def generate_actor(self, name: str, secondary_name: str) -> Actor:
        actor = Actor(name, secondary_name)

        actor.register_action(Action(name, 'EventFlowActionAction0', []))
        actor.register_action(Action(name, 'EventFlowActionAction1', [
            Param('param0', IntType),
        ]))
        actor.register_action(Action(name, 'EventFlowActionAction2', [
            Param('param0', StrType),
        ]))
        actor.register_action(Action(name, 'EventFlowActionAction3', [
            Param('param0', IntType),
            Param('param1', StrType),
            Param('param2', FloatType),
            Param('param3', BoolType),
        ]))
        actor.register_query(Query(name, 'EventFlowQueryQuery0', [], Type('int3'), False))
        actor.register_query(Query(name, 'EventFlowQueryQuery1', [
            Param('param0', IntType),
        ], Type('bool'), False))
        actor.register_query(Query(name, 'EventFlowQueryQuery2', [
            Param('param0', IntType),
        ], Type('bool'), True))

        return actor

    def test_files(self):
        self.maxDiff = None

        set_log_level(level=logging.ERROR)
        for c, err in TestParser.CASES:
            with self.subTest(msg=f'test_{c}'):
                with open(f'{TestParser.TEST_DIR}/{c}.evfl', 'rt') as ef:
                    evfl = ef.read()

                tokens = tokenize(evfl)

                if err is not None:
                    with self.assertRaises(err):
                        parse(tokens, self.generate_actor)
                else:
                    rn, actors = parse(tokens, self.generate_actor)
                    actual = [str(x) for x in extract_and_sort_nodes(rn)]
                    expected = []
                    with open(f'{TestParser.TEST_DIR}/{c}.out', 'rt') as ef:
                        for line in ef:
                            line = line.strip()
                            if line:
                                expected.append(line)
                    self.assertEqual(actual, expected)

def extract_and_sort_nodes(roots: List[RootNode]) -> List[Node]:
    # order: RootNodes (sort by name), non-Root (sort by name)
    int_nodes: Set[Node] = set()
    for root in roots:
        int_nodes.update([x for x in find_postorder(root)
                if not isinstance(x, RootNode)])

    def split(name):
        if not name[-1].isnumeric() or name.isnumeric():
            return (name, 0)
        j = 0
        while name[j - 1].isnumeric():
            j -= 1
        return (name[:j], int(name[j:]))
    roots = sorted(roots, key=lambda x: split(x.name))
    int_nodes_l = sorted(int_nodes, key=lambda x: split(x.name))
    return roots + int_nodes_l # type: ignore

