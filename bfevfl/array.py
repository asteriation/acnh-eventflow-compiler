from __future__ import annotations

from typing import Generic, Optional, Sequence, TypeVar

from bitstring import BitStream, pack

from .str_ import String
from .block import Block, DataBlock, ContainerBlock

T = TypeVar('T', bound='Block')

class BlockPtrArray(DataBlock, Generic[T]):
    def __init__(self, items: Sequence[T]) -> None:
        super().__init__(8 * len(items))

        self.n = len(items)
        for i, item in enumerate(items):
            self._add_pointer(8 * i, item)

    def alignment(self) -> int:
        return 8

class BlockArray(ContainerBlock, Generic[T]):
    def __init__(self, items: Sequence[T]) -> None:
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
    class _String(String):
        def alignment(self) -> int:
            return 8

    def __init__(self, items: Sequence[str]) -> None:
        super().__init__([StringArray._String(item) for item in items])
        self.n = len(items)

    def alignment(self) -> int:
        return 8

