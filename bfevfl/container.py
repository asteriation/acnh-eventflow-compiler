from __future__ import annotations

from typing import Dict, List, Optional

from bitstring import BitStream, pack

from .datatype import TypedValue, Argument
from .block import DataBlock, ContainerBlock
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
        # TODO: verify
        super().__init__([
            _ContainerItemHeader(0, 1, None),
            String(arg),
        ])

class ContainerContainerItem(_ContainerItem):
    def __init__(self, items: Dict[str, TypedValue], pool: StringPool) -> None:
        dic = Dictionary(list(items.keys()), pool)
        values: List[_ContainerItem] = []
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
        super().__init__([
            _ContainerItemHeader(5, 1, None),
            StringArray([value])
        ])

class IntArrayContainerItem(_ContainerItem):
    def __init__(self, values: List[int]) -> None:
        super().__init__([
            _ContainerItemHeader(7, 1, None),
            IntArray(values)
        ])

class BoolArrayContainerItem(_ContainerItem):
    def __init__(self, values: List[bool]) -> None:
        super().__init__([
            _ContainerItemHeader(8, 1, None),
            BoolArray(values)
        ])

class FloatArrayContainerItem(_ContainerItem):
    def __init__(self, values: List[float]) -> None:
        super().__init__([
            _ContainerItemHeader(9, 1, None),
            FloatArray(values)
        ])

class StringArrayContainerItem(_ContainerItem):
    def __init__(self, values: List[str]) -> None:
        super().__init__([
            _ContainerItemHeader(10, 1, None),
            StringArray(values)
        ])

Container = ContainerContainerItem
