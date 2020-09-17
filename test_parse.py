import unittest
from typing import Dict, List, Set

from funcparserlib.lexer import Token, LexerError
from funcparserlib.parser import NoParseError

from datatype import BoolType, FloatType, IntType, StrType
from actors import Action, Actor, Param
from nodes import RootNode, TerminalNode, TerminalNode_, Node

from parse import tokenize, parse

class TestTokenize(unittest.TestCase):
    def test_indent_dedent_simple(self):
        tokens = list(tokenize('\n \n'))
        token_types = [t.type for t in tokens]
        self.assertEqual(token_types, ['NL', 'INDENT', 'NL', 'DEDENT'])

    def test_multiple_dedent(self):
        tokens = list(tokenize('\n \n  \n   \n    \n  \n'))
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'NL', 'INDENT', 'NL', 'INDENT',
                'NL', 'INDENT', 'NL', 'DEDENT', 'DEDENT', 'NL',
                'DEDENT', 'DEDENT'
        ]
        self.assertEqual(token_types, expected)

    def test_intermediate_dedent_to_zero(self):
        tokens = list(tokenize('\n a\n  b\n   c\n    d\nabc\n  a'))
        token_types = [t.type for t in tokens]
        expected = [
                'NL', 'INDENT', 'ID', 'NL', 'INDENT', 'ID', 'NL',
                'INDENT', 'ID', 'NL', 'INDENT', 'ID', 'NL',
                'DEDENT', 'DEDENT', 'DEDENT', 'DEDENT',
                'ID', 'NL', 'INDENT', 'ID', 'NL', 'DEDENT',
        ]
        self.assertEqual(token_types, expected)

    def test_bad_indent(self):
        with self.assertRaises(LexerError):
            list(tokenize('\n\t\n \t'))

    def test_bad_dedent(self):
        with self.assertRaises(LexerError):
            list(tokenize('\n\t\n\t \n '))

    def test_paren_mismatch1(self):
        with self.assertRaises(LexerError):
            list(tokenize('(]'))

    def test_paren_mismatch2(self):
        with self.assertRaises(LexerError):
            list(tokenize('[)'))

    def test_paren_mismatch_nested1(self):
        with self.assertRaises(LexerError):
            list(tokenize('[(([])])'))

    def test_paren_mismatch_nested2(self):
        with self.assertRaises(LexerError):
            list(tokenize('([[()])]'))

    def test_paren_unopened1(self):
        with self.assertRaises(LexerError):
            list(tokenize(')'))

    def test_paren_unopened2(self):
        with self.assertRaises(LexerError):
            list(tokenize(']'))

    def test_paren_unclosed1(self):
        with self.assertRaises(LexerError):
            list(tokenize('('))

    def test_paren_unclosed2(self):
        with self.assertRaises(LexerError):
            list(tokenize('['))

    def test_id(self):
        with self.assertRaises(LexerError):
            next(tokenize('-abc'))
        with self.assertRaises(LexerError):
            next(tokenize('_abc'))
        with self.assertRaises(LexerError):
            list(tokenize('abc-'))
        with self.assertRaises(LexerError):
            list(tokenize('abc_'))
        self.assertEqual(list(tokenize('abc_d0-3')), [
            Token('ID', 'abc_d0-3'),
            Token('NL', '')
        ])
        self.assertEqual(list(tokenize('0ab34')), [
            Token('ID', '0ab34'),
            Token('NL', '')
        ])
        self.assertEqual(list(tokenize('013-34')), [
            Token('ID', '013-34'),
            Token('NL', '')
        ])
        self.assertNotEqual(list(tokenize('01334')), [
            Token('ID', '01334'),
            Token('NL', '')
        ])
        self.assertEqual(list(tokenize('01-a')), [
            Token('ID', '01-a'),
            Token('NL', '')
        ])
        self.assertEqual(list(tokenize('01A_B_C')), [
            Token('ID', '01A_B_C'),
            Token('NL', '')
        ])
        with self.assertRaises(LexerError):
            list(tokenize('01-'))

    def test_example(self):
        src = ''' \
            flow Test(a: int = 5, b: float = 3.0): # comment
                if SubflowResults[+5] in (1, 2):
                    System.EventFlags['a'] = 3
                else:
                    fork:
                        branch0:
                            System.EventFlags["b"] = 7
                        branch1:
                            System.EventFlags["c"] = 7
                        branch2:
                            pass'''
        expected = [
            'KW', 'ID', 'LPAREN', 'ID', 'COLON', 'TYPE', 'ASSIGN',
                'INT', 'COMMA', 'ID', 'COLON', 'TYPE', 'ASSIGN', 'FLOAT',
                'RPAREN', 'COLON', 'NL',
            'INDENT', 'KW', 'ID', 'LSQUARE', 'INT', 'RSQUARE', 'KW',
                'LPAREN', 'INT', 'COMMA', 'INT', 'RPAREN', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'KW', 'COLON', 'NL',
            'INDENT', 'KW', 'COLON', 'NL',
            'INDENT', 'KW', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'KW', 'COLON', 'NL',
            'INDENT', 'ID', 'DOT', 'ID', 'LSQUARE', 'STRING', 'RSQUARE',
                'ASSIGN', 'INT', 'NL',
            'DEDENT', 'KW', 'COLON', 'NL',
            'INDENT', 'KW', 'NL',
            'DEDENT', 'DEDENT', 'DEDENT', 'DEDENT',
        ]

        tokens = list(tokenize(src))
        token_types = [t.type for t in tokens]
        self.assertEqual(token_types, expected)

class TestParser(unittest.TestCase):
    CASES = [
        ('empty_flow1', None),
        ('empty_flow2', None),
        ('single_action_flow', None),
        ('multi_action_flow', None),
        ('multiple_flows', None),
        ('empty', None),
        ('newlines', None),
        ('comments_only', None),
        ('flow_decl_second_line', None),
        ('err_out_of_flow', NoParseError),
    ]

    ACTORS: Dict[str, Actor] = {}
    TEST_DIR = 'tests/parser/'

    def setUp(self):
        TestParser.ACTORS = {
            'TestActor1': Actor('TestActor1'),
            'TestActor2': Actor('TestActor2'),
        }
        for name, actor in TestParser.ACTORS.items():
            actor.register_action(Action(name, 'Action0', []))
            actor.register_action(Action(name, 'Action1', [
                Param('param0', IntType),
            ]))
            actor.register_action(Action(name, 'Action2', [
                Param('param0', StrType),
            ]))
            actor.register_action(Action(name, 'Action3', [
                Param('param0', IntType),
                Param('param1', StrType),
                Param('param2', FloatType),
                Param('param3', BoolType),
            ]))

    def test_files(self):
        for c, err in TestParser.CASES:
            with self.subTest(msg=f'test_{c}'):
                with open(f'{TestParser.TEST_DIR}/{c}.evfl', 'rt') as ef:
                    evfl = ef.read()

                tokens = list(tokenize(evfl))

                if err is not None:
                    with self.assertRaises(err):
                        parse(tokens, TestParser.ACTORS)
                else:
                    rn = parse(tokens, TestParser.ACTORS)
                    nodes = iter(extract_and_sort_nodes(rn))
                    with open(f'{TestParser.TEST_DIR}/{c}.out', 'rt') as ef:
                        for line in ef:
                            line = line.strip()
                            if line:
                                self.assertEqual(str(next(nodes)), line)
                    with self.assertRaises(StopIteration):
                        next(nodes)

def __find_postorder_helper(root: Node, visited: Set[str]) -> List[Node]:
    po: List[Node] = []
    for node in root.out_edges:
        if node.name not in visited:
            visited.add(node.name)
            po.extend(__find_postorder_helper(node, visited))
    po.append(root)
    return po

def __find_postorder(root: Node) -> List[Node]:
    return __find_postorder_helper(root, set())

def extract_and_sort_nodes(roots: List[RootNode]) -> List[Node]:
    # order: RootNodes (sort by name), non-Root/TerminalNodes (sort by name), TerminalNode
    int_nodes: List[Node] = []
    for root in roots:
        int_nodes.extend([x for x in __find_postorder(root)
                if not isinstance(x, (RootNode, TerminalNode_))])

    roots = sorted(roots, key=lambda x: x.name)
    int_nodes = sorted(int_nodes, key=lambda x: x.name)
    return roots + int_nodes + [TerminalNode] # type: ignore

