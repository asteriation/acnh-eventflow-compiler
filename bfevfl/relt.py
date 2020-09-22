from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from bitstring import BitStream, pack

from .block import DataBlock, ContainerBlock, Block

class _RelocationTableHeader(DataBlock):
    def __init__(self) -> None:
        super().__init__(16)

        with self._at_offset(0):
            self.buffer.overwrite(b'RELT')
            self.buffer.overwrite(pack('uintle:32', 0))
            self.buffer.overwrite(pack('uintle:32', 1))

    def prepare_bitstream(self) -> BitStream:
        with self._at_offset(4):
            self.buffer.overwrite(pack('uintle:32', self.offset + 8))
        return super().prepare_bitstream()

    def alignment(self) -> int:
        return 8

class _RelocationTableSectionHeader(DataBlock):
    def __init__(self, num_entries: int) -> None:
        super().__init__(0x18)

        with self._at_offset(20):
            self.buffer.overwrite(pack('uintle:32', num_entries))

    def prepare_bitstream(self) -> BitStream:
        with self._at_offset(12):
            self.buffer.overwrite(pack('uintle:32', self.offset - 8))
        return super().prepare_bitstream()

    def alignment(self) -> int:
        return 8

class _RelocationTableEntry(DataBlock):
    def __init__(self, offset: int, bitfield: int) -> None:
        super().__init__(8)

        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:32', offset))
            self.buffer.overwrite(pack('uintle:32', bitfield))

    def alignment(self) -> int:
        return 4

class RelocationTable(ContainerBlock):
    def __init__(self, pointers: List[int]) -> None:
        pointers.sort()

        entries: List[Block] = []
        start = -1
        bitfield = 0
        for ptr in pointers:
            if start != -1:
                if (ptr - start) < 0x100:
                    bitfield = bitfield | (1 << ((ptr - start) >> 3))
                else:
                    entries.append(_RelocationTableEntry(start, bitfield))
                    start = -1
            if start == -1:
                start = ptr
                bitfield = 1
        if start != -1:
            entries.append(_RelocationTableEntry(start, bitfield))

        super().__init__([
            _RelocationTableHeader(),
            _RelocationTableSectionHeader(len(entries))
        ] + entries)

