from __future__ import annotations

from typing import Generator, Set
from bfevfl.nodes import Node

def __find_postorder_helper(root: Node, visited: Set[str]) -> Generator[Node, None, None]:
    for node in root.out_edges:
        if node.name not in visited:
            visited.add(node.name)
            yield from __find_postorder_helper(node, visited)
    yield root

def find_postorder(root: Node) -> Generator[Node, None, None]:
    yield from __find_postorder_helper(root, set())

