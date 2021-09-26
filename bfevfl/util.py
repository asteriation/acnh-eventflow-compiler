from __future__ import annotations

from typing import Generator, Set
from .nodes import Node

def __find_postorder_helper(root: Node, visited: Set[Node]) -> Generator[Node, None, None]:
    for node in root.out_edges:
        if node not in visited:
            visited.add(node)
            yield from __find_postorder_helper(node, visited)
    yield root

def find_postorder(root: Node) -> Generator[Node, None, None]:
    yield from __find_postorder_helper(root, set())

