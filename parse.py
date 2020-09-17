from __future__ import annotations

import re
from typing import Dict, Generator, List, Tuple

from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import a, some, maybe, many, finished, skip

from datatype import BoolType, FloatType, IntType, StrType, Type, TypedValue
from actors import Param, Action, Actor
from nodes import Node, RootNode, ActionNode, TerminalNode

def compare_indent(base: str, new: str, pos: Tuple[int, int]) -> int:
    if base.startswith(new):
        return -1 if len(base) > len(new) else 0
    elif new.startswith(base):
        return 1
    raise LexerError(pos, f'mixed tab/space indent')

def tokenize(string: str) -> Generator[Token, None, None]:
    specs = [
        ('COMMENT', (r'#[^\r\n]*',)),
        ('SP', (r'[ \t]+',)),
        ('NL', (r'[\r\n]+',)),
        ('COLON', (r':',)),
        ('LPAREN', (r'\(',)),
        ('RPAREN', (r'\)',)),
        ('LSQUARE', (r'\[',)),
        ('RSQUARE', (r'\]',)),
        ('CMP', (r'[!=]=|[<>]=?',)),
        ('ASSIGN', (r'=',)),
        ('PASSIGN', (r'\+=',)),
        ('DOT', (r'\.',)),
        ('COMMA', (r',',)),
        ('TYPE', (r'int|float|str',)),
        ('KW', (r'flow|internal|entrypoint|if|elif|else|do|while|fork|branch\d+|return|pass|not|and|or|in',)),
        ('BOOL', (r'true|false',)),
        ('ID', (r'''
            [A-Za-z]
                (?:
                    [A-Za-z0-9_\-]*[A-Za-z0-9]
                  | [A-Za-z0-9]*)
          | \d+
                (?:
                    [A-Za-z_\-][A-Za-z0-9_\-]*[A-Za-z0-9]
                  | [A-Za-z][A-Za-z0-9]*)
            ''', re.VERBOSE)),
        ('FLOAT', (r'[+-]?[ \t]*(?:\d+\.\d*|\d*\.\d+)',)), # todo: fix this
        ('INT', (r'[+-]?[ \t]*\d+',)), # todo: hex
        ('STRING', (r'"(?:\\\.|[^"\\])*"|\'(?:\\\.|[^\'\\])*\'',)),
    ]

    t = make_tokenizer(specs)
    pstack: List[Token] = []
    indent = ['']

    if string and string[-1] not in ('\r', '\n'):
        string = string + '\n'

    num_lines = len(re.findall(r'\r\n|\r|\n', string))
    last_token = None
    space_since_nl = False
    for x in t(string):
        if x.type == 'COMMENT':
            continue
        elif x.type == 'LPAREN':
            pstack.append(x)
        elif x.type == 'RPAREN':
            if not pstack:
                raise LexerError(x.start, 'no parentheses to close')
            if pstack.pop().type != 'LPAREN':
                raise LexerError(x.start, "expecting ']' but got ')'")
        elif x.type == 'LSQUARE':
            pstack.append(x)
        elif x.type == 'RSQUARE':
            if not pstack:
                raise LexerError(x.start, 'no bracket to close')
            if pstack.pop().type != 'LSQUARE':
                raise LexerError(x.start, "expecting ')' but got ']'")
        elif x.type == 'NL':
            if pstack:
                continue
            if last_token and last_token.type == 'NL':
                continue
            x = Token('NL', '', start=x.start, end=x.end)
            space_since_nl = False
        elif x.type == 'SP':
            space_since_nl = True
            if last_token and last_token.type == 'NL':
                indent_diff = compare_indent(indent[-1], x.name, x.start)
                if indent_diff < 0:
                    found = False
                    while indent:
                        s = indent.pop()
                        if s == x.name:
                            indent.append(s)
                            break
                        last_token = Token('DEDENT', '', start=x.start, end=x.end)
                        yield last_token
                    if not indent:
                        raise LexerError(x.end, 'dedent to unknown level')
                    continue
                elif indent_diff > 0:
                    indent.append(x.name)
                    x = Token('INDENT', '', start=x.start, end=x.end)
                else:
                    continue
            else:
                continue

        if x.type != 'INDENT' and last_token and last_token.type == 'NL' and not space_since_nl:
            while len(indent) > 1:
                s = indent.pop()
                last_token = Token('DEDENT', '', start=x.start, end=x.end)
                yield last_token

        last_token = x
        yield last_token

    if pstack:
        raise LexerError((num_lines + 1, 0), 'unclosed parentheses/brackets')

    while indent[-1]:
        indent.pop()
        yield Token('DEDENT', '', start=(num_lines + 1, 0), end=(num_lines + 1, 0))

def parse(seq: List[Token], actors: Dict[str, Actor]) -> List[RootNode]:
    tokval = lambda x: x.value
    toktype = lambda t: some(lambda x: x.type == t) >> tokval
    tok = lambda typ, name: skip(a(Token(typ, name)) >> tokval)

    def make_array(n):
        if n is None:
            return []
        else:
            return [x for x in [n[0]] + n[1] if x is not None]

    int_ = lambda n: TypedValue(type=IntType, value=int(n))
    float_ = lambda n: TypedValue(type=FloatType, value=float(n))
    bool_ = lambda n: TypedValue(type=BoolType, value=(n == 'true'))
    string = lambda n: TypedValue(type=StrType, value=n)
    type_ = lambda n: Type(type=n)

    id_ = toktype('ID') >> str

    nid = 0
    def next_id() -> int:
        nonlocal nid
        rv, nid = nid, nid + 1
        return rv

    def make_action(n):
        actor_name, action_name, params = n
        assert actor_name in actors, f'no actor with name "{actor_name}"'
        assert action_name in actors[actor_name].actions, f'actor "{actor_name}" does not have action with name "{action_name}"'

        action = actors[actor_name].actions[action_name]
        try:
            pdict = action.prepare_param_dict(params)
        except AssertionError as e:
            raise e # todo: better messages

        return ActionNode(f'Event{next_id()}', action, pdict)

    def make_none(_):
        return None

    def make_return(_):
        return TerminalNode

    def make_flow(n):
        name, params, body_root = n
        assert not params, 'vardefs todo'
        node = RootNode(name, [])
        node.add_out_edge(body_root)
        return node

    def link_block(n):
        if n is None:
            return TerminalNode
        n = [x for x in [n[0]] + n[1] if x is not None]
        if not n:
            return TerminalNode

        for n1, n2 in zip(n[:-1], n[1:]):
            n1.add_out_edge(n2)
        n[-1].add_out_edge(TerminalNode)
        return n[0]

    def collect_flows(n):
        if n is None:
            return []
        else:
            return [x for x in n if x is not None]

    # value: INT | STRING | FLOAT | BOOL | ID (todo)
    value = (
        toktype('INT') >> int_
        | toktype('FLOAT') >> float_
        | toktype('BOOL') >> bool_
        | toktype('STRING') >> string
    )

    # function_params: | function_params COMMA value
    function_params = maybe(value + many(tok('COMMA', ',') + value)) >> make_array

    # actor_name: id
    actor_name = id_
    # action_name: id
    action_name = id_
    # simple_action: actor_name DOT action_name LPAREN function_params RPAREN NL
    simple_action = (
        actor_name + tok('DOT', '.') + action_name +
        tok('LPAREN', '(') + function_params + tok('RPAREN', ')') +
        tok('NL', '')
    ) >> make_action

    # action: simple_action (todo: converted actions)
    action = simple_action

    # pass: PASS NL
    pass_ = (tok('KW', 'pass') + tok('NL', '')) >> make_none

    # return: RETURN NL
    return_ = (tok('KW', 'return') + tok('NL', '')) >> make_return

    # stmt: action | pass_ | return (todo: queries, blocks)
    stmt = action | pass_ | return_

    # block_body: stmt | block_body stmt
    block_body = stmt + many(stmt) >> link_block

    # block: COLON NL INDENT block_body DEDENT
    block = (
        tok('COLON', ':') + tok('NL', '') + tok('INDENT', '') +
        block_body + tok('DEDENT', '')
    )

    # flow_param: ID COLON TYPE
    flow_param = id_ + tok('COLON', ':') + (toktype('TYPE') >> type_)

    # flow_params:  | flow_params COMMA flow_param
    flow_params = maybe(flow_param + many(tok('COMMA', ',') + flow_param)) >> make_array

    # flow: FLOW ID LPAREN flow_params RPAREN block
    flow = (
        tok('KW', 'flow') + id_ + tok('LPAREN', '(') +
        flow_params + tok('RPAREN', ')') + block
    ) >> make_flow

    # file: | file flow | file NL
    evfl_file = many(flow | (tok('NL', '') >> make_none)) >> collect_flows

    parser = evfl_file + skip(finished)
    return parser.parse(seq)

