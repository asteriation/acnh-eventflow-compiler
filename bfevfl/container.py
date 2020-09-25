from __future__ import annotations

from typing import Dict, List, Optional

from bitstring import BitStream, pack

from .datatype import TypedValue, Argument
from .block import DataBlock, ContainerBlock, Block
from .str_ import String, StringPool
from .dic_ import Dictionary
from .array import BlockPtrArray, IntArray, BoolArray, FloatArray, StringArray

class _ContainerItemHeader(DataBlock):
    def __init__(self, typ: int, n: int, dic: Optional[Dictionary]) -> None:
        super().__init__(0x10)

        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:8', typ))
        with self._at_offset(2):
            self.buffer.overwrite(pack('uintle:16', n))
        self._add_pointer(8, dic)

    def alignment(self) -> int:
        return 8

class _ContainerItem(ContainerBlock):
    pass

class ArgumentContainerItem(_ContainerItem):
    def __init__(self, arg: str) -> None:
        arr = StringArray([arg])
        super().__init__([
            _ContainerItemHeader(0, 1, None),
            BlockPtrArray([arr]),
            arr,
        ])

class ContainerContainerItem(_ContainerItem):
    def __init__(self, items: Dict[str, TypedValue], pool: StringPool) -> None:
        dic = Dictionary(list(items.keys()), pool)
        values: List[Block] = []
        for item in items.values():
            if item.type.type == 'bool':
                assert isinstance(item.value, bool)
                values.append(BoolContainerItem(item.value))
            elif item.type.type == 'float':
                assert isinstance(item.value, float)
                values.append(FloatContainerItem(item.value))
            elif item.type.type.startswith('int') or item.type.type.startswith('enum'):
                assert isinstance(item.value, int)
                values.append(IntContainerItem(item.value))
            elif item.type.type == 'str' or item.type.type.startswith('Enum'):
                assert isinstance(item.value, str)
                values.append(StringContainerItem(item.value))
            elif item.type.type == '_argument':
                assert isinstance(item.value, Argument)
                values.append(ArgumentContainerItem(item.value))
            else:
                raise ValueError(f'bad type: {item.type}')

        super().__init__([
            _ContainerItemHeader(1, len(values), dic),
            BlockPtrArray(values),
            dic,
        ] + values)

class IntContainerItem(_ContainerItem):
    def __init__(self, value: int) -> None:
        super().__init__([
            _ContainerItemHeader(2, 1, None),
            IntArray([value])
        ])

class BoolContainerItem(_ContainerItem):
    def __init__(self, value: bool) -> None:
        super().__init__([
            _ContainerItemHeader(3, 1, None),
            BoolArray([value])
        ])

class FloatContainerItem(_ContainerItem):
    def __init__(self, value: float) -> None:
        super().__init__([
            _ContainerItemHeader(4, 1, None),
            FloatArray([value])
        ])

class StringContainerItem(_ContainerItem):
    def __init__(self, value: str) -> None:
        arr = StringArray([value])
        super().__init__([
            _ContainerItemHeader(5, 1, None),
            BlockPtrArray([arr]),
            arr,
        ])

Container = ContainerContainerItem
