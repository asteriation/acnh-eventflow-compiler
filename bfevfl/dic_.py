from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from bitstring import BitStream, pack

from .str_ import String, StringPool
from .block import DataBlock, ContainerBlock

class _DictionaryItem(DataBlock):
    def __init__(self, string: String) -> None:
        super().__init__(16)

        self.bit_index = 0xFFFFFFFF
        self.bit0 = 0
        self.bit1 = 0
        self.string = string
        self._add_pointer(8, string)

        self._rewrite_block_data()

    def _rewrite_block_data(self) -> None:
        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:32', self.bit_index))
            self.buffer.overwrite(pack('uintle:16', self.bit0))
            self.buffer.overwrite(pack('uintle:16', self.bit1))

    def update_indices(self, bit_index: int, bit0: int, bit1: int) -> None:
        self.bit_index, self.bit0, self.bit1 = bit_index, bit0, bit1
        self._rewrite_block_data()

    def alignment(self) -> int:
        return 8

class _DictionaryHeader(DataBlock):
    def __init__(self, num_nodes: int) -> None:
        super().__init__(8)

        with self._at_offset(0):
            self.buffer.overwrite(b'DIC ')
            self.buffer.overwrite(pack('uintle:32', num_nodes))

    def alignment(self) -> int:
        return 8

def _get_bit(s: str, bit: int) -> bool:
    x, i = (bit >> 3), (bit & 7)
    if x >= len(s):
        return False
    return (ord(s[~x]) & (1 << i)) != 0

def _next_diff(s1: str, s2: str, start: int) -> int:
    while _get_bit(s1, start) == _get_bit(s2, start):
        start += 1
    return start

class _PTrieNode:
    def __init__(self, name: str, cmp_name: str, min_index: int) -> None:
        self.name = name
        self.index = _next_diff(cmp_name, name, min_index)
        self.branches: List[Optional[_PTrieNode]] = [None, None]
        self.branches[_get_bit(self.name, self.index)] = self

    def insert(self, s: str) -> _PTrieNode:
        check = _get_bit(s, self.index)
        branch = self.branches[check]
        if branch is None:
            node = _PTrieNode(s, '', self.index + 1)
        elif branch.index <= self.index:
            node = _PTrieNode(s, branch.name, self.index + 1)
            node.branches[_get_bit(branch.name, node.index)] = branch
        else:
            return branch.insert(s)

        self.branches[check] = node
        return node

def _compute_indices(strings: List[str]) -> List[Tuple[int, int, int]]:
    if not strings:
        return [(0xFFFFFFFF, 0, 0)]

    root = 0
    nodes = [_PTrieNode(strings[0], '', 0)]
    for i, s in enumerate(strings[1:]):
        diff = _next_diff(nodes[root].name, s, 0)
        if diff < nodes[root].index:
            node = _PTrieNode(s, nodes[root].name, 0)
            node.branches[_get_bit(nodes[root].name, node.index)] = nodes[root]
            root = i + 1
        else:
            node = nodes[root].insert(s)

        nodes.append(node)

    stoi = {s: i + 1 for i, s in enumerate(strings)}

    return [(0xFFFFFFFF, root + 1, 0)] + [(
        n.index,
        stoi[n.branches[0].name] if n.branches[0] is not None else 0,
        stoi[n.branches[1].name] if n.branches[1] is not None else 0,
    ) for n in nodes]

class Dictionary(ContainerBlock):
    def __init__(self, strings: List[str], pool: StringPool) -> None:
        assert len(set(strings)) == len(strings), 'dictionary cannot have duplicate strings'

        self.header = _DictionaryHeader(len(strings))
        self.root = _DictionaryItem(pool.empty)
        self.items = OrderedDict(
            (s, _DictionaryItem(pool.strings[s])) for s in strings
        )

        indices = _compute_indices(strings)
        self.root.update_indices(*indices[0])
        for item, inds in zip(self.items.values(), indices[1:]):
            item.update_indices(*inds)

        super().__init__([self.header, self.root] + list(self.items.values()))

