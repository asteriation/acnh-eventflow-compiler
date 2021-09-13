from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, NoReturn

from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import a, some, maybe, many, finished, skip, forward_decl, NoParseError, _Ignored, Parser
from more_itertools import peekable

from compiler.logger import emit_info, emit_warning, emit_fatal, LogFatal
from compiler.util import find_postorder

from bfevfl.datatype import BoolType, FloatType, IntType, StrType, Type, TypedValue
from bfevfl.actors import Param, Action, Actor
from bfevfl.nodes import (Node, RootNode, ActionNode, SwitchNode, JoinNode, ForkNode,
        SubflowNode, TerminalNode, ConnectorNode)

def __compare_indent(base: str, new: str, pos: Tuple[int, int]) -> int:
    if base.startswith(new):
        return -1 if len(base) > len(new) else 0
    elif new.startswith(base):
        return 1
    raise LexerError(pos, f'mixed tab/space indent')

__tokens = [
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

def tokenize(string: str) -> List[Token]:
    t = make_tokenizer(__tokens)
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
    gen = peekable(t(string))
    for x in gen:
        if x.type != 'ANNOTATION':
            first_non_annotation = True
        if first_non_annotation and x.type == 'ANNOTATION':
            raise LexerError(x.start, "unexpected '@' - annotations must be at the top of the file")

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
            nxt = gen.peek(None)
            next_comment = (nxt is not None and nxt.type == 'COMMENT')
            if tokens and tokens[-1].type == 'NL' and not buffering and not next_comment:
                indent_diff = __compare_indent(indent[-1], x.name, x.start)
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

def __process_actor_annotations(seq: List[Token]) -> Tuple[Dict[str, str], List[Token]]:
    i = 0
    rv: Dict[str, str] = {}
    while i < len(seq) and seq[i].type == 'ANNOTATION':
        actor_name, sec_name = seq[i].value[1:].strip().split(':', 1)
        rv[actor_name] = sec_name
        i += 1
    return rv, seq[i:]

__Result = namedtuple('__Result', ('value', 'start', 'end'))

def __collapse_results(n, depth=9999999999):
    if n is None:
        return __Result(None, (0, 0), (0, 0))
    if isinstance(n, __Result):
        return n
    if isinstance(n, Token):
        return __Result(n.value, n.start, n.end)
    if isinstance(n, _Ignored):
        return __Result(n, n.value.start, n.value.end)
    if isinstance(n, (tuple, list)):
        cls = type(n)
        if not n:
            return __Result(cls(), (0, 0), (0, 0))
        if depth == 0:
            _, starts, ends = zip(*(__collapse_results(x) for x in n))
            return __Result(n, min(i for i in starts if i > (0, 0)), max(i for i in ends if i > (0, 0)))
        values, starts, ends = zip(*(__collapse_results(x, depth=depth-1) for x in n))
        return __Result(cls(values), min(i for i in starts if i > (0, 0)), max(i for i in ends if i > (0, 0)))
    raise ValueError

def __wrap_result(f):
    def inner(n):
        n = __collapse_results(n)
        r = f(n.value, n.start, n.end)
        if isinstance(r, __Result):
            return r
        return __Result(r, n.start, n.end)
    return inner

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

def __verify_calls(root: Node, local_roots: Dict[str, RootNode], exported_roots: Dict[str, RootNode]) -> None:
    reroutes = {}
    for node in find_postorder(root):
        if isinstance(node, SubflowNode) and node.ns == '':
            tail_call = (len(node.out_edges) == 1 and node.out_edges[0] is TerminalNode)
            if node.called_root_name not in exported_roots and node.called_root_name not in local_roots:
                emit_warning(f'{node.called_root_name} called but not defined')
            if node.called_root_name in local_roots:
                assert tail_call, 'non-tail-call local subflows not implemented'
                reroutes[node] = local_roots[node.called_root_name].out_edges[0]
            elif tail_call:
                reroutes[node] = exported_roots[node.called_root_name].out_edges[0]

    for old, new in reroutes.items():
        __replace_node(root, old, new)

def __flatten_helper(lst: Iterable[Any]) -> Generator[Any, None, None]:
    for x in lst:
        if isinstance(x, Iterable):
            yield from __flatten_helper(x)
        else:
            yield x

def __flatten(lst: Sequence[Any]) -> List[Any]:
    return list(__flatten_helper(lst))
    
__toktype = lambda t: some(lambda x: x.type == t)
__tokop = lambda typ: skip(some(lambda x: x.type == typ))
__tokkw = lambda name: skip(a(Token('ID', name)))
__identity = __wrap_result(lambda n, s, e: n)

def __make_array(n):
    r = __collapse_results(n, depth=1)
    re
    if n is None:
        return __Result([], (0, 0), (0, 0))
    else:
        return __Result(
            [x.value for x in [n[0]] + n[1] if x.value is not None],
            min(x.start for x in [n[0]] + n[1] if x.start > (0, 0)),
            max(x.end for x in [n[0]] + n[1] if x.end > (0, 0)),
        )

__int = __wrap_result(lambda n, s, e: TypedValue(type=IntType, value=int(n)))
__float = __wrap_result(lambda n, s, e: TypedValue(type=FloatType, value=float(n)))
__bool = __wrap_result(lambda n, s, e: TypedValue(type=BoolType, value=(n == 'true')))
__string = __wrap_result(lambda n, s, e: TypedValue(type=StrType, value=n[1:-1]))
__type = __wrap_result(lambda n, s, e: Type(type=n))

id_ = __toktype('ID')

def __parse_custom_rule(name: str, s: str) -> Tuple[int, Parser]:
    rules = [x.strip() for x in s.split('\n') if x.strip()]
    out = []
    r = re.compile(r'^([A-Z]+)\s*\(\s*(?:(".*")|([^=]+)\s*(?:\=\s*(.+))?|)\s*\)$')
    valid_token_types = set(t for t, m in __tokens)

    def exit_bad_rule(message, mark_rule=True) -> NoReturn:
        emit_fatal(message)
        for r in rules:
            emit_info(r)
            if mark_rule and r == rule:
                emit_info('^' * len(r))
        raise LogFatal()

    actor_set = False
    params = {'.name': name, '.negated': False}

    def eval_(s):
        __int = lambda n: TypedValue(type=IntType, value=int(n))
        __float = lambda n: TypedValue(type=FloatType, value=float(n))
        __bool = lambda n: TypedValue(type=BoolType, value=(n == 'true'))
        __string = lambda n: TypedValue(type=StrType, value=n[1:-1])
        return eval(s, locals())

    def make_parser(type_ ,value, param, parse):
        nonlocal actor_set
        if type_ not in valid_token_types and type_ != 'NULL':
            exit_bad_rule(f'"{type_}" is not a valid token type')
        if param == '.actor':
            actor_set = True
        if type_ == 'NULL':
            if value:
                exit_bad_rule("NULL token cannot be used with value check")
        if value:
            try:
                value = eval_(value)
            except:
                exit_bad_rule("Failed to parse custom rule value check")
            out.append(skip(a(Token(type_, value))))
        elif not param:
            out.append(skip(some(lambda x: x.type == type_)))
        else:
            parse = parse or '__value'
            try:
                f = eval_(f'lambda __type, __value: {parse}')
            except:
                exit_bad_rule("Failed to parse custom rule")

            if type_ == 'NULL':
                params[param] = f(None, None)
                return

            @__wrap_result
            def inner(t, start, end):
                return {param: f(type_, t)}
            
            value_wrapper = {
                'INT': __int,
                'FLOAT': __float,
                'BOOL': __bool,
                'STRING': __string,
                'ID': __string,
            }.get(type_, __identity)
            if param[0] == '.':
                value_wrapper = __identity

            out.append(some(lambda x: x.type == type_) >> value_wrapper >> inner)

    for rule in rules:
        m = r.match(rule)
        if m is None:
            exit_bad_rule("Misformatted custom rule")
        make_parser(*m.groups())

    if not actor_set:
        exit_bad_rule(".actor never set in custom rule", mark_rule=False)

    @__wrap_result
    def final(t, start, end):
        p = {**params}
        if not isinstance(t, dict):
            for x in t:
                p.update(x)
        else:
            p.update(t)
        return p

    return len(out), (sum(out[1:], out[0]) >> final)

def parse_custom_rules(ss: List[Tuple[str, str]]) -> Optional[Parser]:
    # longest rules first
    _, parsers = zip(*sorted((__parse_custom_rule(name, s) for name, s in ss if s.strip()), key=lambda x: x[0]))
    parsers = parsers[::-1]

    if parsers:
        final = parsers[0]
        for parser in parsers[1:]:
            final = final | parser
        return final
    return None

def parse(
    seq: List[Token],
    gen_actor: Callable[[str, str], Actor],
    custom_action_parser: Optional[Parser]=None,
    custom_query_parser: Optional[Parser]=None
) -> Tuple[List[RootNode], List[Actor]]:
    actors: Dict[str, Actor] = {}
    actor_secondary_names, seq = __process_actor_annotations(seq)

    nid = 0
    def next_id() -> int:
        nonlocal nid
        rv, nid = nid, nid + 1
        return rv

    def check_function(actor, name, params, type_, prepare_params=True):
        function_name = f'EventFlow{type_.capitalize()}{name}'
        if actor not in actors:
            actors[actor] = gen_actor(actor, actor_secondary_names.get(actor, ''))
        mp = getattr(actors[actor], ['actions', 'queries'][type_ == 'query'])
        assert function_name in mp, f'no {type_} with name "{function_name}" found'
        function = mp[function_name]
        if prepare_params:
            try:
                pdict = function.prepare_param_dict(params)
            except AssertionError as e:
                raise e # todo: better error messages
        else:
            pdict = None
        return function_name, function, pdict

    @__wrap_result
    def make_action(n, start, end):
        actor_name, action_name, params = n
        action_name, action, pdict = check_function(actor_name, action_name, params, 'action')
        return (), (ActionNode(f'Event{next_id()}', action, pdict),)

    @__wrap_result
    def make_custom_action(n, start, end):
        actor_name = n.pop('.actor')
        action_name = n.pop('.name')
        n.pop('.negated')
        params = n

        action_name, action, _ = check_function(actor_name, action_name, params, 'action', prepare_params=False)
        return (), (ActionNode(f'Event{next_id()}', action, params),)

    @__wrap_result
    def make_case(n, start, end):
        if isinstance(n, tuple):
            return ([x.value for x in n[0]], n[1])
        return n

    def _get_query_num_values(query, start, end):
        num_values = query.num_values
        if num_values == 999999999:
            emit_warning(f'maximum value for {query.name} unknown; assuming 50', start, end, printsource=False)
            emit_warning(f'setting a maximum value in functions.csv may reduce generated bfevfl size', start, end)
            num_values = 50
        return num_values

    @__wrap_result
    def make_switch(n, start, end):
        actor_name, query_name, params, branches = n
        cases = branches[0] + branches[2]
        default = branches[1]

        query_name, query, pdict = check_function(actor_name, query_name, params, 'query')
        sw = SwitchNode(f'Event{next_id()}', query, pdict)
        entrypoints = []
        for values, block in cases:
            eps, node, connector = block
            entrypoints.extend(eps)

            sw.add_out_edge(node)
            connector.add_out_edge(sw.connector)

            for value in values:
                sw.add_case(node, value)

        num_values = _get_query_num_values(query, start, end)
        default_values = set(range(num_values)) - set(sum((v for v, n in cases), []))
        if default_values:
            if default is not None:
                default.add_out_edge(sw.connector)

            default_branch = default or sw.connector
            sw.add_out_edge(default_branch)
            for value in default_values:
                sw.add_case(default_branch, value)
        elif default:
            emit_warning(f'default branch for {query_name} call is dead code, ignoring', start, end)

        return entrypoints, (sw,)

    @__wrap_result
    def make_bool_function(p, start, end):
        if isinstance(p, dict):
            actor_name = p.pop('.actor')
            query_name = p.pop('.name')
            negated = p.pop('.negated')
            params = pdict = p
            query_name, query, _ = check_function(actor_name, query_name, params, 'query', prepare_params=False)
        else:
            actor_name, query_name, params = p
            negated = False
            query_name, query, pdict = check_function(actor_name, query_name, params, 'query')
        num_values = _get_query_num_values(query, start, end)
        if num_values > 2:
            emit_warning(f'call to {query_name} treated as boolean function but may not be', start, end)
        return ((query, pdict), [({0}, query.inverted != negated), (set(range(1, num_values)), (not query.inverted) != negated)])

    @__wrap_result
    def make_in(p, start, end):
        actor_name, query_name, params, values = p
        query_name, query, pdict = check_function(actor_name, query_name, params, 'query')
        num_values = _get_query_num_values(query, start, end)
        matched = set()
        unmatched = set(range(num_values))
        for value in values:
            if 0 > value.value or num_values <= value.value:
                emit_warning('{value.value} never returned by {query_name}, ignored', start, end)
                continue
            matched.add(value.value)
            unmatched.remove(value.value)
        if not matched or not unmatched:
            emit_warning(f'always true or always false check', start, end)
        return ((query, pdict), [(matched, True), (unmatched, False)])

    @__wrap_result
    def make_cmp(p, start, end):
        actor_name, query_name, params, op, value = p
        query_name, query, pdict = check_function(actor_name, query_name, params, 'query')
        num_values = _get_query_num_values(query, start, end)
        if op == '==' or op == '!=':
            matched = {value.value} if 0 <= value.value < num_values else set()
            unmatched = set(i for i in range(num_values) if i != value.value)
        elif op == '<' or op == '>=':
            matched = set(range(min(num_values, value.value)))
            unmatched = set(range(value.value, num_values))
        else:
            matched = set(range(min(num_values, value.value + 1)))
            unmatched = set(range(value.value + 1, num_values))
        if op in ('!=', '>=', '>'):
            matched, unmatched = unmatched, matched
        if not matched or not unmatched:
            emit_warning(f'always true or always false check', start, end)
        return ((query, pdict), [(matched, True), (unmatched, False)])

    def _predicate_replace(values, old, new):
        for i in range(len(values)):
            if values[i][1] == old:
                values[i] = (values[i][0], new)

    @__wrap_result
    def make_or(p, start, end):
        left, right = p
        # todo: can probably optimize for smaller output
        _predicate_replace(left[1], False, right)
        return left

    @__wrap_result
    def make_and(p, start, end):
        left, right = p
        # todo: can probably optimize for smaller output
        _predicate_replace(left[1], True, right)
        return left

    @__wrap_result
    def make_not(p, start, end):
        _predicate_replace(p[1], True, None)
        _predicate_replace(p[1], False, True)
        _predicate_replace(p[1], None, False)
        return p

    def _expand_table(table, current, next_):
        ((query, pdict), values) = table
        sw = SwitchNode(f'Event{next_id()}', query, pdict)
        for match, action in values:
            if isinstance(action, tuple):
                to = _expand_table(action, current, next_)
            elif action:
                to = current
            else:
                to = next_
            for value in match:
                sw.add_case(to, value)
            sw.add_out_edge(to)
        return sw

    @__wrap_result
    def make_ifelse(n, start, end):
        if_, block, elifs, else_ = n
        cond_branches = [(if_, block)] + elifs + ([(None, else_)] if else_ else [])

        entrypoints = []
        next_ = next_connector = ConnectorNode(f'Connector{next_id()}')
        for table, body in cond_branches[::-1]:
            eps, node, branch_connector = body
            entrypoints.extend(eps)

            if table is None:
                next_ = node
                next_connector = branch_connector
            else:
                next_ = _expand_table(table, node, next_)
                branch_connector.add_out_edge(next_.connector)
                next_connector.add_out_edge(next_.connector)
                next_connector = next_.connector
        return entrypoints, (next_,)

    @__wrap_result
    def make_fork(n_, start, end):
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

    @__wrap_result
    def make_subflow_param(n, start, end):
        return (n,)

    @__wrap_result
    def make_subflow(n, start, end):
        ns, name, params = n
        param_dict = {k[0][0]: k[0][1] for k in params}
        return (), (SubflowNode(f'Event{next_id()}', ns or '', name, param_dict),)

    @__wrap_result
    def make_none(_, start, end):
        return None

    @__wrap_result
    def make_return(_, start, end):
        return (), (TerminalNode,)

    @__wrap_result
    def make_flow(n, start, end):
        local, name, params, body = n
        entrypoints, body_root, body_connector = body
        assert not params, 'vardefs todo'
        node = RootNode(name, local is not None, [])
        node.add_out_edge(body_root)
        body_connector.add_out_edge(TerminalNode)
        return list(entrypoints) + [node]

    @__wrap_result
    def link_ep_block(n, start, end):
        connector = ConnectorNode(f'Connector{next_id()}')
        ep, block_info = n
        block_info = [x for x in block_info if x is not None]
        if block_info:
            eps, block = (__flatten(p) for p in zip(*(x for x in block_info if x is not None)))
        else:
            eps, block = [], ()

        if not block:
            if ep is not None:
                ep_node = RootNode(ep, True, [])
                ep_node.add_out_edge(connector)
                eps.append(ep_node)
            return (eps, connector, connector)

        for n1, n2 in zip(block, block[1:] + [connector]):
            if isinstance(n1, SwitchNode):
                n1.connector.add_out_edge(n2)
            else:
                n1.add_out_edge(n2)

        if ep is not None:
            ep_node = RootNode(ep, True, [])
            ep_node.add_out_edge(block[0])
            eps.append(ep_node)

        return (eps, block[0], connector)

    @__wrap_result
    def link_block(n, start, end):
        connector = ConnectorNode(f'Connector{next_id()}')
        n = [n[0]] + n[1]
        eps, blocks, connectors = zip(*n)
        eps = __flatten(eps)

        for connector, block in zip(connectors[:-1], blocks[1:]):
            connector.add_out_edge(block)

        return (eps, blocks[0], connectors[-1])

    @__wrap_result
    def collect_flows(n, start, end):
        if n is None:
            return []
        else:
            return __flatten([x for x in n if x is not None])

    block = forward_decl()

    # value = INT | STRING | FLOAT | BOOL | ID (todo)
    value = (
        __toktype('INT') >> __int
        | __toktype('FLOAT') >> __float
        | __tokkw('true') >> __bool
        | __tokkw('false') >> __bool
        | __toktype('STRING') >> __string
    )

    # pass = PASS NL
    pass_ = (__tokkw('pass') + __tokop('NL')) >> make_none

    # return = RETURN NL
    return_ = (__tokkw('return') + __tokop('NL')) >> make_return

    # function_params =  [value { COMMA value }]
    function_params = maybe(value + many(__tokop('COMMA') + value)) >> __make_array

    # actor_name = id
    actor_name = id_
    # function_name = id
    function_name = id_
    # base_function = actor_name DOT action_name LPAREN function_params RPAREN
    base_function = (
        actor_name + __tokop('DOT') + function_name +
        __tokop('LPAREN') + function_params + __tokop('RPAREN')
    )
    # function = custom_query_parser | base_function | LPAREN function RPAREN
    function = forward_decl()
    if custom_query_parser is not None:
        function_ = base_function | custom_query_parser
    else:
        function_ = base_function
    function_ = function_ | (__tokop('LPAREN') + function + __tokop('RPAREN'))
    function.define(function_)

    # simple_action = base_function NL
    simple_action = base_function + __tokop('NL') >> make_action

    # action = custom_action_parser | simple_action
    if custom_action_parser is not None:
        custom_action_parser = custom_action_parser + __tokop('NL') >> make_custom_action
        action = custom_action_parser | simple_action
    else:
        action = simple_action

    # __intlist = INT {COMMA INT} [COMMA] | LPAREN __intlist RPAREN
    __intlist = forward_decl()
    __intlist.define(((__toktype('INT') >> __int) + many(__tokop('COMMA') + (__toktype('INT') >> __int)) + maybe(__tokop('COMMA')) >> __make_array) | \
            __tokop('LPAREN') + __intlist + __tokop('RPAREN'))

    # case = CASE __intlist block
    case = __tokkw('case') + __intlist + block >> make_case

    # default = DEFAULT block
    default = __tokkw('default') + block >> make_case

    # cases = { case } [ default ] { case } | pass
    cases = many(case) + maybe(default) + many(case) | pass_

    # switch = SWITCH function COLON NL INDENT cases DEDENT
    switch = __tokkw('switch') + function + __tokop('COLON') + __tokop('NL') + \
            __tokop('INDENT') + cases + __tokop('DEDENT') >> make_switch

    predicate = forward_decl()
    predicate0 = forward_decl()
    predicate1 = forward_decl()
    predicate2 = forward_decl()

    # bool_function = function
    bool_function = function >> make_bool_function

    # in_predicate = function IN __intlist
    in_predicate = function + __tokkw('in') + __intlist >> make_in

    # cmp_predicate = function CMP INT
    cmp_predicate = function + __toktype('CMP') + (__toktype('INT') >> __int) >> make_cmp

    # not_predicate = NOT predicate0
    not_predicate = __tokkw('not') + predicate0 >> make_not

    # paren_predicate = LPAREN predicate RPAREN
    paren_predicate = __tokop('LPAREN') + predicate + __tokop('RPAREN')

    # predicate0 = in_predicate | cmp_predicate | not_predicate | bool_function | paren_predicate
    predicate0.define(in_predicate | cmp_predicate | not_predicate | bool_function | paren_predicate)

    # and_predicate = predicate0 AND predicate1
    and_predicate = predicate0 + __tokkw('and') + predicate1 >> make_and

    # predicate1 = and_predicate | predicate0
    predicate1.define(and_predicate | predicate0)

    # or_predicate = predicate1 OR predicate2
    or_predicate = predicate1 + __tokkw('or') + predicate2 >> make_or

    # predicate2 = or_predicate | predicate1
    predicate2.define(or_predicate | predicate1)

    # predicate = predicate2
    predicate.define(predicate2)

    # if = IF predicate block
    if_ = __tokkw('if') + predicate + block

    # elif = ELIF predicate block
    elif_ = __tokkw('elif') + predicate + block

    # else = ELSE block
    else_ = __tokkw('else') + block

    # ifelse = if { elif } [ else ]
    ifelse = if_ + many(elif_) + maybe(else_) >> make_ifelse

    # branches = { BRANCH block }
    # branchless case handled implicitly by lack of INDENT
    branches = many(__tokkw('branch') + block)

    # fork = FORK COLON NL INDENT branches DEDENT
    fork = __tokkw('fork') + __tokop('COLON') + __tokop('NL') + \
            __tokop('INDENT') + branches + __tokop('DEDENT') >> make_fork

    # flow_name = [id COLON COLON] id
    flow_name = maybe(id_ + __tokop('COLON') + __tokop('COLON')) + id_

    # subflow_param = id ASSIGN value
    subflow_param = id_ + __tokop('ASSIGN') + value >> make_subflow_param

    # subflow_params = [subflow_param { COMMA subflow_param }]
    subflow_params = maybe(subflow_param + many(__tokop('COMMA') + subflow_param)) >> __make_array

    # run = RUN flow_name LPAREN subflow_params RPAREN NL
    run = (
        __tokkw('run') + flow_name + __tokop('LPAREN') + subflow_params + __tokop('RPAREN') + __tokop('NL')
    ) >> make_subflow

    # stmt = action | switch | ifelse | fork | run | pass_ | return | NL
    stmt = action | switch | ifelse | fork | run | pass_ | return_ | (__tokop('NL') >> make_none)

    # entrypoint = ENTRYPOINT id COLON NL
    entrypoint = __tokkw('entrypoint') + id_ + __tokop('COLON') + __tokop('NL')

    # stmts = stmt { stmt }
    stmts = stmt + many(stmt) >> __make_array

    # ep_block_body = [entrypoint] stmts
    ep_block_body = maybe(entrypoint) + stmts >> link_ep_block

    # block_body = ep_block_body { ep_block_body }
    block_body = ep_block_body + many(ep_block_body) >> link_block

    # block = COLON NL INDENT block_body DEDENT
    block.define(__tokop('COLON') + __tokop('NL') + __tokop('INDENT') + block_body + __tokop('DEDENT'))

    # type = INT | FLOAT | STR | BOOL
    type_atom = __tokkw('int') | __tokkw('float') | __tokkw('str') | __tokkw('bool')

    # flow_param = ID COLON TYPE
    flow_param = id_ + __tokop('COLON') + type_atom >> __type

    # flow_params = [flow_param { COMMA flow_param }]
    flow_params = maybe(flow_param + many(__tokop('COMMA') + flow_param)) >> __make_array

    # flow = [LOCAL] FLOW ID LPAREN flow_params RPAREN block
    flow = (
        maybe(a(Token('ID', 'local'))) + __tokkw('flow') + id_ + __tokop('LPAREN') + flow_params + __tokop('RPAREN') + block
    ) >> make_flow

    # file = { flow | NL }
    evfl_file = many(flow | (__tokop('NL') >> make_none)) >> collect_flows

    parser = evfl_file + skip(finished)
    roots: List[RootNode] = parser.parse(seq).value
    local_roots = {r.name: r for r in roots if r.local}
    exported_roots = {r.name: r for r in roots if not r.local}
    for n in roots:
        __collapse_connectors(n)
    for n in roots:
        __verify_calls(n, local_roots, exported_roots)
    for n in roots:
        __replace_node(n, TerminalNode, None)

    return list(exported_roots.values()), list(actors.values())
