from __future__ import annotations

from typing import Optional, Sequence

from bitstring import BitStream, pack

from .str_ import String
from .block import Block, DataBlock, ContainerBlock

class BlockPtrArray(DataBlock):
    def __init__(self, items: Sequence[Block]) -> None:
        super().__init__(8 * len(items))

        self.n = len(items)
        for i, item in enumerate(items):
            self._add_pointer(8 * i, item)

    def alignment(self) -> int:
        return 8

class BlockArray(ContainerBlock):
    def __init__(self, items: Sequence[Block]) -> None:
        super().__init__(items)
        self.n = len(items)

    def alignment(self) -> int:
        return 8

class IntArray(DataBlock):
    def __init__(self, items: Sequence[int]) -> None:
        super().__init__(4 * len(items))

        self.n = len(items)
        for i, item in enumerate(items):
            self.buffer.overwrite(pack('uintle:32', item))

    def alignment(self) -> int:
        return 4

class BoolArray(DataBlock):
    # todo: verify true
    def __init__(self, items: Sequence[bool]) -> None:
        super().__init__(4 * len(items))

        self.n = len(items)
        for i, item in enumerate(items):
            self.buffer.overwrite(pack('uintle:32', 0x80000001 if item else 0x00000000))

    def alignment(self) -> int:
        return 4

class FloatArray(DataBlock):
    def __init__(self, items: Sequence[float]) -> None:
        super().__init__(4 * len(items))

        self.n = len(items)
        for i, item in enumerate(items):
            self.buffer.overwrite(pack('floatle:32', item))

    def alignment(self) -> int:
        return 4

class StringArray(ContainerBlock):
    def __init__(self, items: Sequence[str]) -> None:
        sobj = [String(item) for item in items]
        for s in sobj:
            s.alignment = lambda: 8

        self.n = len(items)
        super().__init__(sobj)

    def alignment(self) -> int:
        return 8

