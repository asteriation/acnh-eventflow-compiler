from __future__ import annotations

import abc
from typing import Any, List, Sequence, Tuple

from bitstring import BitStream, pack

class bitstream_offset:
    def __init__(self, bs: BitStream, offset: int, offset_in_bits: bool = False, restore: bool = True) -> None:
        self.bs = bs
        self.offset = offset * (1 if offset_in_bits else 8)
        self.old_pos = 0
        self.restore = restore

    def __enter__(self) -> None:
        self.old_pos = self.bs.pos
        self.bs.pos = self.offset

    def __exit__(self, *args: Any) -> None:
        if self.restore:
            self.bs.pos = self.old_pos

class Block(abc.ABC):
    def __init__(self) -> None:
        self.offset = 0

    def set_offset(self, offset: int) -> None:
        self.offset = offset

    @abc.abstractmethod
    def prepare_bitstream(self) -> BitStream:
        pass

    @abc.abstractmethod
    def get_all_pointers(self) -> List[int]:
        pass

    @abc.abstractmethod
    def alignment(self) -> int:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

class DataBlock(Block):
    def __init__(self, length: int) -> None:
        super().__init__()

        self.buffer = BitStream(length=length * 8)
        self.pointers: List[Tuple[int, Block]] = []

    def _add_pointer(self, offset: int, block: Block) -> None:
        self.pointers.append((offset, block))

    def prepare_bitstream(self) -> BitStream:
        for offset, obj in self.pointers:
            with bitstream_offset(self.buffer, offset):
                self.buffer.overwrite(pack('uintle:64', obj.offset))
        return self.buffer

    def get_all_pointers(self) -> List[int]:
        return sorted([x + self.offset for x, _ in self.pointers])

    @abc.abstractmethod
    def alignment(self) -> int:
        pass

    def __len__(self) -> int:
        return len(self.buffer) // 8

class ContainerBlock(Block):
    def __init__(self, contained: Sequence[Block]) -> None:
        super().__init__()

        self.contained: List[Tuple[int, Block]] = []
        offset = 0
        for block in contained:
            alignment = block.alignment()
            if offset % alignment != 0:
                offset += alignment - offset % alignment

            self.contained.append((offset, block))
            offset += len(block)

        self.length = offset
        self.base_alignment = 8
        if contained:
            self.base_alignment = contained[0].alignment()

        self.set_offset(0)

    def set_offset(self, offset: int) -> None:
        super().set_offset(offset)
        for boffset, block in self.contained:
            block.set_offset(boffset + offset)

    def prepare_bitstream(self) -> BitStream:
        bs = BitStream(length=self.length * 8)
        for offset, block in self.contained:
            with bitstream_offset(bs, offset):
                bs.overwrite(block.prepare_bitstream())
        return bs

    def get_all_pointers(self) -> List[int]:
        return sum((x.get_all_pointers() for _, x in self.contained), [])

    def alignment(self) -> int:
        return self.base_alignment

    def __len__(self) -> int:
        return self.length

