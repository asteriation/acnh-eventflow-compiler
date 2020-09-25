from __future__ import annotations

from typing import List, Set

from bitstring import BitStream, pack

from .actors import Actor
from .nodes import Node, RootNode
from .block import DataBlock, ContainerBlock
from .str_ import StringPool
from .dic_ import Dictionary
from .array import BlockPtrArray
from .flowchart import Flowchart
from .relt import RelocationTable

class _RELTPlaceholder(DataBlock):
    def __init__(self):
        super().__init__(0)

    def alignment(self) -> int:
        return 8

class _FileHeader(DataBlock):
    def __init__(self, filename: str, flowcharts: BlockPtrArray[Flowchart], flowchart_dic: Dictionary,
            timeline_dic: Dictionary, relt: _RELTPlaceholder, pool: StringPool) -> None:
        super().__init__(0x48)

        self.filename = pool[filename].c_str
        self.relt = relt

        with self._at_offset(0):
            self.buffer.overwrite(b'BFEVFL\0\0')
            self.buffer.overwrite(b'\x00\x03\x00\x00\xff\xfe\x03\x00')
            self.buffer.overwrite(b'\x00\x00\x00\x00\x00\x00\x90\x00')

        with self._at_offset(0x20):
            self.buffer.overwrite(b'\x01\x00')

        self._add_pointer(0x28, flowcharts)
        self._add_pointer(0x30, flowchart_dic)
        self._add_pointer(0x38, None)
        self._add_pointer(0x40, timeline_dic)

    def set_file_size(self, fsize: int) -> None:
        with self._at_offset(0x1c):
            self.buffer.overwrite(pack('uintle:32', fsize))

    def prepare_bitstream(self) -> BitStream:
        with self._at_offset(0x10):
            self.buffer.overwrite(pack('uintle:32', self.filename.offset))
        with self._at_offset(0x18):
            self.buffer.overwrite(pack('uintle:32', self.relt.offset))
        return super().prepare_bitstream()

    def alignment(self) -> int:
        return 8

class File(ContainerBlock):
    def __init__(self, filename: str, actors: List[Actor], nodes: List[Node]) -> None:
        entry_nodes = [n for n in nodes if isinstance(n, RootNode)]
        other_nodes = [n for n in nodes if not isinstance(n, RootNode)]

        pooled_strings: Set[str] = set()
        pooled_strings.add(filename)
        pooled_strings.update(n.name for n in nodes)
        for actor in actors:
            add_actor = False
            for n, a in actor.actions.items():
                if a.used:
                    add_actor = True
                    pooled_strings.add(a.name)
                    pooled_strings.update(p.name for p in a.params)
            for n, q in actor.queries.items():
                if q.used:
                    add_actor = True
                    pooled_strings.add(q.name)
                    pooled_strings.update(p.name for p in q.params)
            if add_actor:
                pooled_strings.add(actor.name)
        pool = StringPool(sorted(list(pooled_strings)))

        flowchart_dic = Dictionary([filename], pool)
        timeline_dic = Dictionary([], pool)

        flowchart = Flowchart(filename, actors, other_nodes, entry_nodes, pool)
        flowchart_ptrs = BlockPtrArray[Flowchart]([flowchart])

        self.relt = _RELTPlaceholder()
        self.header = _FileHeader(filename, flowchart_ptrs, flowchart_dic, timeline_dic, self.relt, pool)

        super().__init__([
            self.header,
            flowchart_ptrs,
            flowchart_dic,
            timeline_dic,
            flowchart,
            pool,
            self.relt
        ])

    def prepare_bitstream(self) -> BitStream:
        relt = RelocationTable(self.get_all_pointers())
        relt.set_offset(self.relt.offset)
        self.header.set_file_size(len(self) + len(relt))
        return super().prepare_bitstream() + relt.prepare_bitstream()

    def alignment(self) -> int:
        return 8

