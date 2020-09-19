from __future__ import annotations

from dataclasses import dataclass
from typing import Any

class Argument(str):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        str.__init__(*args, **kwargs)

@dataclass
class Type:
    type: str

    def num_values(self) -> int:
        if self.type.startswith('int') and self.type != 'int':
            return int(self.type[3:])
        elif self.type == 'bool':
            return 2
        elif self.type.startswith('enum'):
            return len(self.type.split(','))
        else:
            return 999999999

    def __str__(self) -> str:
        return self.type

BoolType = Type('bool')
FloatType = Type('float')
IntType = Type('int')
StrType = Type('str')

@dataclass
class TypedValue:
    type: Type
    value: Any

