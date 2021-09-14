from __future__ import annotations

from collections import OrderedDict
from typing import Any, List, Sequence

from bitstring import BitStream, pack

from .block import Block, DataBlock, ContainerBlock

class _CString(Block):
    def prepare_bitstream(self) -> BitStream:
        raise RuntimeError('not supported')

    def get_all_pointers(self) -> List[int]:
        raise RuntimeError('not supported')

    def alignment(self) -> int:
        raise RuntimeError('not supported')

    def __len__(self) -> int:
        raise RuntimeError('not supported')

class String(DataBlock):
    def __init__(self, string: str) -> None:
        # u16 len + string + null terminator
        super().__init__(3 + len(string.encode('utf-8')))
        self.string = string

        self.c_str = _CString()
        self.c_str.set_offset(2)

        with self._at_offset(0):
            b = string.encode('utf-8')
            self.buffer.overwrite(pack('uintle:16', len(b)))
            self.buffer.overwrite(b)

    def set_offset(self, offset: int) -> None:
        self.c_str.set_offset(offset + 2)
        super().set_offset(offset)

    def alignment(self) -> int:
        return 2

class _StringPoolHeader(DataBlock):
    def __init__(self, num_strings: int) -> None:
        super().__init__(20)

        with self._at_offset(0):
            self.buffer.overwrite(b'STR ')
        with self._at_offset(16):
            self.buffer.overwrite(pack('uintle:32', num_strings))

    def alignment(self) -> int:
        return 8

class StringPool(ContainerBlock):
    def __init__(self, strings: Sequence[str]) -> None:
        self.header = _StringPoolHeader(len(strings))
        self.empty = String('')
        self.strings = OrderedDict((s, String(s)) for s in strings)

        super().__init__([self.header, self.empty] + list(self.strings.values()))

    def __getitem__(self, key: Any) -> String:
        assert isinstance(key, str)
        if key == '':
            return self.empty
        else:
            return self.strings[key]
