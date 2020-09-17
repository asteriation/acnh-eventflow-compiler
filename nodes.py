from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union

from datatype import Type, TypedValue
from actors import Action, Query

class Node(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.out_edges: List[Node] = []

    def add_out_edge(self, dest: Node) -> None:
        if dest not in self.out_edges:
            self.out_edges.append(dest)

    def del_out_edge(self, dest: Node) -> None:
        self.out_edges.remove(dest)

    def __str__(self) -> str:
        return f'Node[name={self.name}' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

    def __repr__(self) -> str:
        return str(self)

    def get_dot(self) -> str:
        return f'n{id(self)} [label=<<b>{self.name} ({type(self).__name__})</b><br/>{type(self).__name__}>];' + \
                ''.join(f'n{id(self)} -> n{id(nx)};' for nx in self.out_edges)

class RootNode(Node):
    @dataclass
    class VarDef:
        name: str
        type_: Type
        initial_value: Union[int, bool, float]

    def __init__(self, name: str, vardefs: List[VarDef] = []) -> None:
        Node.__init__(self, name)
        self.vardefs = vardefs[:]

    def __str__(self) -> str:
        return f'RootNode[name={self.name}' + \
            f', vardefs=[{", ".join(str(v) for v in self.vardefs)}]' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class ActionNode(Node):
    def __init__(self, name: str, action: Action, params: Dict[str, TypedValue]) -> None:
        Node.__init__(self, name)
        self.action = action
        self.params = params

    def __str__(self) -> str:
        return f'ActionNode[name={self.name}' + \
            f', action={self.action}' + \
            f', params={self.params}' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

# class SwitchNode(Node):
    # def __init__(self, name: str, query: Query, params: Dict[str, Any]) -> None:
        # Node.__init__(self, name)
        # self.query = query
        # self.params = params
        # self.cases: Dict[str, List[Any]] = {}
        # self.terminal_node: Optional[Node] = None

        # assert sum(len(x) for x in self.cases.values()) <= self.query.rv.num_values()

    # def del_out_edge(self, dest: Node) -> None:
        # Node.del_out_edge(self, dest)
        # if dest.name in self.cases:
            # del self.cases[dest.name]
        # if self.terminal_node is dest:
            # self.terminal_node = None

    # def reroute_out_edge(self, old_dest: Node, new_dest: Node) -> None:
        # Node.reroute_out_edge(self, old_dest, new_dest)
        # if old_dest.name in self.cases:
            # c = self.cases[old_dest.name]
            # del self.cases[old_dest.name]
            # self.cases[new_dest.name] = c
        # if self.terminal_node is old_dest:
            # self.terminal_node = new_dest

    # def add_case(self, node_name: str, value: Any) -> None:
        # if node_name not in self.cases:
            # self.cases[node_name] = []
        # self.cases[node_name].append(value)

    # def register_terminal_node(self, terminal_node: Node) -> None:
        # # todo: improve when switch node doesn't need a terminal node contact
        # if sum(len(x) for x in self.cases.values()) == self.query.rv.num_values():
            # self.terminal_node = None
            # return

        # self.add_out_edge(terminal_node)
        # terminal_node.add_in_edge(self)

        # self.terminal_node = terminal_node

    # def __str__(self) -> str:
        # return f'SwitchNode[name={self.name}' + \
            # f', query={self.query}' + \
            # f', params={self.params}' + \
            # f', cases={self.cases}' + \
            # f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            # f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            # ']'

class ForkNode(Node):
    def __init__(self, name: str, join_node: JoinNode, forks: List[Node]) -> None:
        Node.__init__(self, name)
        self.join_node = join_node
        self.forks = forks

    def __str__(self) -> str:
        return f'ForkNode[name={self.name}' + \
            f', join_node={self.join_node.name}' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

class JoinNode(Node):
    def __init__(self, name: str) -> None:
        Node.__init__(self, name)

    def __str__(self) -> str:
        return f'JoinNode[name={self.name}' + \
            f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            ']'

# class SubflowNode(Node):
    # def __init__(self, name: str, ns: str, called_root_name: str, nxt: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> None:
        # Node.__init__(self, name)
        # self.ns = ns
        # self.called_root_name = called_root_name
        # self.nxt = nxt
        # self.params = params.copy() if params else {}

    # def __str__(self) -> str:
        # return f'SubflowNode[name={self.name}' + \
            # f', ns={self.ns}' + \
            # f', called_root_name={self.called_root_name}' + \
            # f', params={self.params}' + \
            # f', in_edges=[{", ".join(n.name for n in self.in_edges)}]' + \
            # f', out_edges=[{", ".join(n.name for n in self.out_edges)}]' + \
            # ']'

class TerminalNode_(Node):
    def __init__(self) -> None:
        Node.__init__(self, 'TerminalNode')

    def add_out_edge(self, dest: Node) -> None:
        pass

    def del_out_edge(self, dest: Node) -> None:
        pass

    def __str__(self) -> str:
        return f'TerminalNode[]'

TerminalNode = TerminalNode_()

