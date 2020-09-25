from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

from bitstring import BitStream, pack

from .block import DataBlock, ContainerBlock

class String(DataBlock):
    def __init__(self, string: str) -> None:
        # u16 len + string + null terminator
        super().__init__(3 + len(string))
        self.string = string

        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:16', len(string)))
            self.buffer.overwrite(string.encode('ascii'))

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
    def __init__(self, strings: List[str]) -> None:
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
