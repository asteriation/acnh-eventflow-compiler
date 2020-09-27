from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple

from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import a, some, maybe, many, finished, skip, forward_decl

from bfevfl.datatype import BoolType, FloatType, IntType, StrType, Type, TypedValue
from bfevfl.actors import Param, Action, Actor
from bfevfl.nodes import Node, RootNode, ActionNode, JoinNode, ForkNode, SubflowNode, TerminalNode

def compare_indent(base: str, new: str, pos: Tuple[int, int]) -> int:
    if base.startswith(new):
        return -1 if len(base) > len(new) else 0
    elif new.startswith(base):
        return 1
    raise LexerError(pos, f'mixed tab/space indent')

def tokenize(string: str) -> List[Token]:
    specs = [
        ('ANNOTATION', (r'@[^\r\n]*\r?\n',)),
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
        ('KW', (r'flow|internal|entrypoint|if|elif|else|do|while|fork|branch|return|pass|not|and|or|in|run',)),
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
    tokens: List[Token] = []
    space_since_nl = False
    first_non_annotation = False
    for x in t(string):
        if x.type != 'ANNOTATION':
            first_non_annotation = True
        if first_non_annotation and x.type == 'ANNOTATION':
            raise LexerError(x.start, "unexpected '@' - annotations must be at the top of the file")

        if x.type == 'COMMENT':
            if tokens and tokens[-1].type == 'INDENT':
                indent.pop()
                tokens.pop()
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
            space_since_nl = False
            if tokens and tokens[-1].type == 'NL':
                continue
            x = Token('NL', '', start=x.start, end=x.end)
        elif x.type == 'SP':
            space_since_nl = True
            if tokens and tokens[-1].type == 'NL':
                indent_diff = compare_indent(indent[-1], x.name, x.start)
                if indent_diff < 0:
                    found = False
                    while indent:
                        s = indent.pop()
                        if s == x.name:
                            indent.append(s)
                            break
                        tokens.append(Token('DEDENT', '', start=x.start, end=x.end))
                    if not indent:
                        raise LexerError(x.end, 'dedent to unknown level')
                elif indent_diff > 0:
                    indent.append(x.name)
                    tokens.append(Token('INDENT', '', start=x.start, end=x.end))
            continue

        if x.type != 'INDENT' and tokens and tokens[-1].type == 'NL' and not space_since_nl:
            while len(indent) > 1:
                s = indent.pop()
                tokens.append(Token('DEDENT', '', start=x.start, end=x.end))

        tokens.append(x)

    if pstack:
        raise LexerError((num_lines + 1, 0), 'unclosed parentheses/brackets')

    while indent[-1]:
        indent.pop()
        tokens.append(Token('DEDENT', '', start=(num_lines + 1, 0), end=(num_lines + 1, 0)))

    return tokens

def process_actor_annotations(seq: List[Token]) -> Tuple[Dict[str, str], List[Token]]:
    i = 0
    rv: Dict[str, str] = {}
    while i < len(seq) and seq[i].type == 'ANNOTATION':
        actor_name, sec_name = seq[i].value[1:].strip().split(':', 1)
        rv[actor_name] = sec_name
        i += 1
    return rv, seq[i:]

def parse(seq: List[Token], gen_actor: Callable[[str, str], Actor]) -> Tuple[List[RootNode], List[Actor]]:
    actors: Dict[str, Actor] = {}
    actor_secondary_names, seq = process_actor_annotations(seq)

    tokval = lambda x: x.value
    toktype = lambda t: some(lambda x: x.type == t) >> tokval
    tokop = lambda typ: skip(some(lambda x: x.type == typ))
    tokkw = lambda name: skip(a(Token('KW', name)))

    def make_array(n):
        if n is None:
            return []
        else:
            return [x for x in [n[0]] + n[1] if x is not None]

    int_ = lambda n: TypedValue(type=IntType, value=int(n))
    float_ = lambda n: TypedValue(type=FloatType, value=float(n))
    bool_ = lambda n: TypedValue(type=BoolType, value=(n == 'true'))
    string = lambda n: TypedValue(type=StrType, value=n[1:-1])
    type_ = lambda n: Type(type=n)

    id_ = toktype('ID') >> str

    nid = 0
    def next_id() -> int:
        nonlocal nid
        rv, nid = nid, nid + 1
        return rv

    def make_action(n):
        actor_name, action_name, params = n
        action_name = f'EventFlowAction{action_name}'
        if actor_name not in actors:
            actors[actor_name] = gen_actor(actor_name, actor_secondary_names.get(actor_name, ''))
        assert action_name in actors[actor_name].actions, f'no action with name "{action_name}" found'

        action = actors[actor_name].actions[action_name]
        try:
            pdict = action.prepare_param_dict(params)
        except AssertionError as e:
            raise e # todo: better messages

        return ActionNode(f'Event{next_id()}', action, pdict)

    def make_fork(n):
        for node in n:
            assert node is not None, 'empty branch in fork not allowed'
            __replace_terminal(node, None)

        fork_id, join_id = next_id(), next_id()
        join = JoinNode(f'Event{join_id}')
        fork = ForkNode(f'Event{fork_id}', join, n)

        for node in n:
            fork.add_out_edge(node)

        return fork, join

    def make_subflow_param(n):
        return (n,)

    def make_subflow(n):
        ns, name, params = n
        param_dict = {k[0][0]: k[0][1] for k in params}
        return SubflowNode(f'Event{next_id()}', ns or '', name, param_dict)

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

        n = __flatten_nodes(n)
        n = [x for x in n if x is not None]

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

    block = forward_decl()

    # value = INT | STRING | FLOAT | BOOL | ID (todo)
    value = (
        toktype('INT') >> int_
        | toktype('FLOAT') >> float_
        | toktype('BOOL') >> bool_
        | toktype('STRING') >> string
    )

    # function_params =  [value { COMMA value }]
    function_params = maybe(value + many(tokop('COMMA') + value)) >> make_array

    # actor_name = id
    actor_name = id_
    # action_name = id
    action_name = id_
    # simple_action = actor_name DOT action_name LPAREN function_params RPAREN NL
    simple_action = (
        actor_name + tokop('DOT') + action_name +
        tokop('LPAREN') + function_params + tokop('RPAREN') +
        tokop('NL')
    ) >> make_action

    # action = simple_action (todo: converted actions)
    action = simple_action

    # branches = { BRANCH block }
    # branchless case handled implicitly by lack of INDENT
    branches = many(tokkw('branch') + block)

    # fork = FORK COLON NL INDENT branches DEDENT
    fork = tokkw('fork') + tokop('COLON') + tokop('NL') + \
            tokop('INDENT') + branches + tokop('DEDENT') >> make_fork

    # pass = PASS NL
    pass_ = (tokkw('pass') + tokop('NL')) >> make_none

    # return = RETURN NL
    return_ = (tokkw('return') + tokop('NL')) >> make_return

    # flow_name = [id COLON COLON] id
    flow_name = maybe(id_ + tokop('COLON') + tokop('COLON')) + id_

    # subflow_param = id ASSIGN value
    subflow_param = id_ + tokop('ASSIGN') + value >> make_subflow_param

    # subflow_params = [subflow_param { COMMA subflow_param }]
    subflow_params = maybe(subflow_param + many(tokop('COMMA') + subflow_param)) >> make_array

    # run = RUN flow_name LPAREN subflow_params RPAREN NL
    run = (
        tokkw('run') + flow_name + tokop('LPAREN') + subflow_params + tokop('RPAREN') + tokop('NL')
    ) >> make_subflow

    # stmt = action | fork | run | pass_ | return | NL (todo: queries, blocks)
    stmt = action | fork | run | pass_ | return_ | tokop('NL')

    # block_body = stmt { stmt }
    block_body = stmt + many(stmt) >> link_block

    # block = COLON NL INDENT block_body DEDENT
    block.define(tokop('COLON') + tokop('NL') + tokop('INDENT') + block_body + tokop('DEDENT'))

    # flow_param = ID COLON TYPE
    flow_param = id_ + tokop('COLON') + (toktype('TYPE') >> type_)

    # flow_params = [flow_param { COMMA flow_param }]
    flow_params = maybe(flow_param + many(tokop('COMMA') + flow_param)) >> make_array

    # flow = FLOW ID LPAREN flow_params RPAREN block
    flow = (
        tokkw('flow') + id_ + tokop('LPAREN') + flow_params + tokop('RPAREN') + block
    ) >> make_flow

    # file = { flow | NL }
    evfl_file = many(flow | (tokop('NL') >> make_none)) >> collect_flows

    parser = evfl_file + skip(finished)
    roots: List[RootNode] = parser.parse(seq)
    for n in roots:
        __replace_terminal(n, None)

    return roots, list(actors.values())

def __replace_terminal_helper(root: Node, replacement: Optional[Node], visited: Set[str]) -> None:
    if TerminalNode in root.out_edges:
        root.del_out_edge(TerminalNode)
        if replacement is not None:
            root.add_out_edge(replacement)

    for node in root.out_edges:
        if node.name not in visited:
            visited.add(node.name)
            __replace_terminal_helper(node, replacement, visited)

def __replace_terminal(root: Node, replacement: Optional[Node]) -> None:
    __replace_terminal_helper(root, replacement, set())

def __flatten_nodes_helper(lst: Iterable[Any]) -> Generator[Node, None, None]:
    for x in lst:
        if isinstance(x, Iterable):
            yield from __flatten_nodes_helper(x)
        else:
            yield x

def __flatten_nodes(lst: Sequence[Any]) -> List[Node]:
    return list(__flatten_nodes_helper(lst))
