from typing import Callable, List, Set, Dict, Tuple

from bfevfl.nodes import Node, RootNode, SubflowNode, ForkNode, JoinNode
from compiler.util import find_postorder

def __merge_identical_nodes(a: Node, b: Node) -> None:
    # merge a into b; a is effectively deleted after this
    # this assumes a/b's out edges are the same.
    for node in a.in_edges:
        node.reroute_out_edge(a, b)
    for node in a.out_edges:
        a.del_out_edge(node)

def optimize_merge_identical(roots: List[RootNode]) -> None:
    changed = True
    while changed:
        changed = False
        d: Dict[Tuple[Any], Node] = {}
        visited: Set[Node] = set()
        for root in roots:
            for node in find_postorder(root):
                if node in visited:
                    continue
                visited.add(node)
                if isinstance(node, JoinNode): # JoinNodes are paired with ForkNodes
                    continue
                if not isinstance(node, ForkNode):
                    data = node.get_data()
                    children = tuple(sorted([n.name for n in node.out_edges]))
                    key = (data, children)
                else:
                    data = node.get_data()
                    join_children = tuple(sorted([n.name for n in node.join_node.out_edges]))
                    children = tuple(sorted([n.name for n in node.out_edges if n != node.join_node]))
                    key = (data, join_children, children)

                if key in d:
                    __merge_identical_nodes(node, d[key])
                    changed = True
                else:
                    d[key] = node

def make_counter_renamer(exclude: Set[str]) -> Callable[[Node], None]:
    counter = 0
    def inner(n: Node) -> None:
        nonlocal counter
        while True:
            name = f'Event{counter}'
            counter += 1
            if name not in exclude:
                n.name = name
                return
    return inner

def make_compact_renamer(exclude: Set[str]) -> Callable[[Node], None]:
    chars = [chr(i) for i in range(0x20, 0x7f)]
    def to_str(num):
        if num < len(chars):
            return chars[num]
        else:
            return to_str(num // len(chars)) + chars[num % len(chars)]

    counter = 0
    def inner(n: Node) -> None:
        nonlocal counter
        while True:
            name = to_str(counter)
            counter += 1
            if name not in exclude:
                n.name = name
                return
    return inner

# optimize event/local root names
def optimize_names(roots: List[RootNode], make_renamer=make_counter_renamer) -> None:
    entrypoints = set(r.name for r in roots if not r.local)
    entrypoint_rename: Dict[str, str] = {}
    renamer = make_renamer(entrypoints)
    for root in roots:
        if root.local:
            old_name = root.name
            renamer(root)
            new_name = root.name
            entrypoints.add(new_name)
            entrypoint_rename[old_name] = new_name

    nodes: Set[Node] = set()
    for root in roots:
        for node in find_postorder(root):
            if node in nodes:
                continue
            if not isinstance(node, RootNode):
                renamer(node)
            if isinstance(node, SubflowNode):
                if node.ns == '':
                    if node.called_root_name in entrypoint_rename:
                        node.called_root_name = entrypoint_rename[node.called_root_name]
            nodes.add(node)
