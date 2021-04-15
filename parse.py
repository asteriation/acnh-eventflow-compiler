from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple

from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import a, some, maybe, many, finished, skip, forward_decl

from logger import LOG
from util import find_postorder

from bfevfl.datatype import BoolType, FloatType, IntType, StrType, Type, TypedValue
from bfevfl.actors import Param, Action, Actor
from bfevfl.nodes import (Node, RootNode, ActionNode, SwitchNode, JoinNode, ForkNode,
        SubflowNode, TerminalNode, ConnectorNode)

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
    buffered: List[Token] = []
    space_since_nl = False
    first_non_annotation = False
    buffering = False

    emit_token = lambda tok: (buffered if buffering else tokens).append(tok)
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
            if tokens and tokens[-1].type == 'NL' and not buffering:
                continue
            x = Token('NL', '', start=x.start, end=x.end)
        elif x.type == 'ID' and x.name == 'entrypoint':
            if space_since_nl:
                raise LexerError(x.start, 'entrypoint must be unindented')
            buffering = True
        elif x.type == 'SP':
            space_since_nl = True
            if tokens and tokens[-1].type == 'NL' and not buffering:
                indent_diff = compare_indent(indent[-1], x.name, x.start)
                if indent_diff < 0:
                    found = False
                    while indent:
                        s = indent.pop()
                        if s == x.name:
                            indent.append(s)
                            break
                        emit_token(Token('DEDENT', '', start=x.start, end=x.end))
                    if not indent:
                        raise LexerError(x.end, 'dedent to unknown level')
                elif indent_diff > 0:
                    indent.append(x.name)
                    emit_token(Token('INDENT', '', start=x.start, end=x.end))

            if not buffering and buffered:
                tokens.extend(buffered)
                buffered = []

            continue

        if x.type != 'INDENT' and tokens and tokens[-1].type == 'NL' and not space_since_nl \
                and not buffering:
            while len(indent) > 1:
                s = indent.pop()
                emit_token(Token('DEDENT', '', start=x.start, end=x.end))
            if not buffering and buffered:
                tokens.extend(buffered)
                buffered = []

        emit_token(x)

        if x.type == 'NL':
            buffering = False

    if pstack:
        raise LexerError((num_lines + 1, 0), 'unclosed parentheses/brackets')

    if buffering or buffered:
        raise LexerError((num_lines + 1, 0), 'unexpected end of file')

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
    tokkw = lambda name: skip(a(Token('ID', name)))

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

        return (), (ActionNode(f'Event{next_id()}', action, pdict),)

    def make_case(n):
        if isinstance(n, tuple):
            return ([x.value for x in n[0]], n[1])
        return n

    def make_switch(n):
        actor_name, query_name, params, branches = n
        cases = branches[0] + branches[2]
        default = branches[1]

        query_name = f'EventFlowQuery{query_name}'
        if actor_name not in actors:
            actors[actor_name] = gen_actor(actor_name, actor_secondary_names.get(actor_name, ''))
        assert query_name in actors[actor_name].queries, f'no query with name "{query_name}" found'

        query = actors[actor_name].queries[query_name]
        try:
            pdict = query.prepare_param_dict(params)
        except AssertionError as e:
            raise e # todo: better error messages

        sw = SwitchNode(f'Event{next_id()}', query, pdict)
        entrypoints = []
        for values, block in cases:
            eps, node, connector = block
            entrypoints.extend(eps)

            sw.add_out_edge(node)
            connector.add_out_edge(sw.connector)

            for value in values:
                sw.add_case(node, value)

        num_values = query.rv.num_values()
        if num_values == 999999999:
            LOG.warning(f'maximum value for {query_name} unknown; assuming 50')
            LOG.warning(f'setting a maximum value in functions.csv may reduce generated bfevfl size')
            num_values = 50

        default_values = set(range(num_values)) - set(sum((v for v, n in cases), []))
        if default_values:
            if default is not None:
                default.add_out_edge(sw.connector)

            default_branch = default or sw.connector
            sw.add_out_edge(default_branch)
            for value in default_values:
                sw.add_case(default_branch, value)
        elif default:
            LOG.warning(f'default branch for {query_name} call is dead code, ignoring')

        return entrypoints, (sw,)

    def make_fork(n_):
        for entrypoints, node, connector in n_:
            for ep in entrypoints:
                __replace_node(ep, connector, None)
            __replace_node(node, connector, None)
        eps = __flatten([ep for ep, _, _ in n_])
        n = [x for _, x, _ in n_]

        fork_id, join_id = next_id(), next_id()
        join = JoinNode(f'Event{join_id}')
        fork = ForkNode(f'Event{fork_id}', join, n)

        for node in n:
            fork.add_out_edge(node)

        return eps, (fork, join)

    def make_subflow_param(n):
        return (n,)

    def make_subflow(n):
        ns, name, params = n
        param_dict = {k[0][0]: k[0][1] for k in params}
        return (), (SubflowNode(f'Event{next_id()}', ns or '', name, param_dict),)

    def make_none(_):
        return None

    def make_return(_):
        return (), (TerminalNode,)

    def make_flow(n):
        local, name, params, body = n
        entrypoints, body_root, body_connector = body
        assert not params, 'vardefs todo'
        node = RootNode(name, local is not None, [])
        node.add_out_edge(body_root)
        body_connector.add_out_edge(TerminalNode)
        return list(entrypoints) + [node]

    def link_ep_block(n):
        connector = ConnectorNode(f'Connector{next_id()}')
        ep, block_info = n
        block_info = [x for x in block_info if x is not None]
        if block_info:
            eps, block = (__flatten(p) for p in zip(*(x for x in block_info if x is not None)))
        else:
            eps, block = [], ()

        if not block:
            if ep is not None:
                ep_node = RootNode(ep, [])
                ep_node.add_out_edge(connector)
                eps.append(ep_node)
            return (eps, connector, connector)

        for n1, n2 in zip(block, block[1:] + [connector]):
            if isinstance(n1, SwitchNode):
                n1.connector.add_out_edge(n2)
            else:
                n1.add_out_edge(n2)

        if ep is not None:
            ep_node = RootNode(ep, [])
            ep_node.add_out_edge(block[0])
            eps.append(ep_node)

        return (eps, block[0], connector)

    def link_block(n):
        connector = ConnectorNode(f'Connector{next_id()}')
        n = make_array(n)
        eps, blocks, connectors = zip(*n)
        eps = __flatten(eps)

        for connector, block in zip(connectors[:-1], blocks[1:]):
            connector.add_out_edge(block)

        return (eps, blocks[0], connectors[-1])

    def collect_flows(n):
        if n is None:
            return []
        else:
            return __flatten([x for x in n if x is not None])

    block = forward_decl()

    # value = INT | STRING | FLOAT | BOOL | ID (todo)
    value = (
        toktype('INT') >> int_
        | toktype('FLOAT') >> float_
        | tokkw('true') >> bool_
        | tokkw('false') >> bool_
        | toktype('STRING') >> string
    )

    # pass = PASS NL
    pass_ = (tokkw('pass') + tokop('NL')) >> make_none

    # return = RETURN NL
    return_ = (tokkw('return') + tokop('NL')) >> make_return

    # function_params =  [value { COMMA value }]
    function_params = maybe(value + many(tokop('COMMA') + value)) >> make_array

    # actor_name = id
    actor_name = id_
    # function_name = id
    function_name = id_
    # function = actor_name DOT action_name LPAREN function_params RPAREN
    function = (
        actor_name + tokop('DOT') + function_name +
        tokop('LPAREN') + function_params + tokop('RPAREN')
    )
    # simple_action = function NL
    simple_action = function + tokop('NL') >> make_action

    # action = simple_action (todo: converted actions)
    action = simple_action

    # int_list = INT {COMMA INT} | LPAREN int_list RPAREN
    int_list = forward_decl()
    int_list.define(((toktype('INT') >> int_) + many(tokop('COMMA') + (toktype('INT') >> int_)) | \
            toktype('LPAREN') + int_list + toktype('RPAREN')) >> make_array)

    # case = CASE int_list block
    case = tokkw('case') + int_list + block >> make_case

    # default = DEFAULT block
    default = tokkw('default') + block >> make_case

    # cases = { case } [ default ] { case } | pass
    cases = many(case) + maybe(default) + many(case) | pass_

    # switch = SWITCH function COLON NL INDENT cases DEDENT
    switch = tokkw('switch') + function + tokop('COLON') + tokop('NL') + \
            tokop('INDENT') + cases + tokop('DEDENT') >> make_switch

    # branches = { BRANCH block }
    # branchless case handled implicitly by lack of INDENT
    branches = many(tokkw('branch') + block)

    # fork = FORK COLON NL INDENT branches DEDENT
    fork = tokkw('fork') + tokop('COLON') + tokop('NL') + \
            tokop('INDENT') + branches + tokop('DEDENT') >> make_fork

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

    # stmt = action | switch | fork | run | pass_ | return | NL
    stmt = action | switch | fork | run | pass_ | return_ | (tokop('NL') >> make_none)

    # entrypoint = ENTRYPOINT id COLON NL
    entrypoint = tokkw('entrypoint') + id_ + tokop('COLON') + tokop('NL')

    # stmts = stmt { stmt }
    stmts = stmt + many(stmt) >> make_array

    # ep_block_body = [entrypoint] stmts
    ep_block_body = maybe(entrypoint) + stmts >> link_ep_block

    # block_body = ep_block_body { ep_block_body }
    block_body = ep_block_body + many(ep_block_body) >> link_block

    # block = COLON NL INDENT block_body DEDENT
    block.define(tokop('COLON') + tokop('NL') + tokop('INDENT') + block_body + tokop('DEDENT'))

    # type = INT | FLOAT | STR | BOOL
    type_atom = tokkw('int') | tokkw('float') | tokkw('str') | tokkw('bool')

    # flow_param = ID COLON TYPE
    flow_param = id_ + tokop('COLON') + type_atom >> type_

    # flow_params = [flow_param { COMMA flow_param }]
    flow_params = maybe(flow_param + many(tokop('COMMA') + flow_param)) >> make_array

    # flow = [LOCAL] FLOW ID LPAREN flow_params RPAREN block
    flow = (
        maybe(a(Token('ID', 'local'))) + tokkw('flow') + id_ + tokop('LPAREN') + flow_params + tokop('RPAREN') + block
    ) >> make_flow

    # file = { flow | NL }
    evfl_file = many(flow | (tokop('NL') >> make_none)) >> collect_flows

    parser = evfl_file + skip(finished)
    roots: List[RootNode] = parser.parse(seq)
    for n in roots:
        __collapse_connectors(n)
        __replace_node(n, TerminalNode, None)

    exported_roots = [r for r in roots if not r.local]
    return exported_roots, list(actors.values())

def __collapse_connectors(root: RootNode) -> None:
    remap: Dict[Node, Node] = {}

    for node in find_postorder(root):
        for onode in node.out_edges:
            if onode in remap:
                node.reroute_out_edge(onode, remap[onode])
        if isinstance(node, ConnectorNode):
            assert len(node.out_edges) == 1
            remap[node] = node.out_edges[0]

def __replace_node(root: Node, replace: Node, replacement: Optional[Node]) -> None:
    for node in find_postorder(root):
        if replace in node.out_edges:
            if replacement is None:
                node.del_out_edge(replace)
            else:
                node.reroute_out_edge(replace, replacement)

def __flatten_helper(lst: Iterable[Any]) -> Generator[Any, None, None]:
    for x in lst:
        if isinstance(x, Iterable):
            yield from __flatten_helper(x)
        else:
            yield x

def __flatten(lst: Sequence[Any]) -> List[Any]:
    return list(__flatten_helper(lst))
