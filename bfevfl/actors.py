from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Union

from .datatype import Argument, IntType, StrType, Type, TypedValue

class Param(NamedTuple):
    name: str
    type: Type

class Function:
    def __init__(self, actor_name: str, name: str, params: List[Param]) -> None:
        self.actor_name = actor_name
        self.name = name
        self.params = params
        self.used = False

    def prepare_param_dict(self, params: List[TypedValue]) -> Dict[str, TypedValue]:
        assert len(self.params) == len(params), f'{self.name}: expected {len(self.params)} params, got {len(params)}'
        d = {}
        for param, value in zip(self.params, params):
            if param.name.startswith('EntryVariableKey'):
                assert isinstance(value.value, Argument), f'{self.name}: expected variable for {param.name}, got {value.type}'
                value.type = param.type
            elif isinstance(value.value, Argument):
                value.type = param.type
            elif param.type.type.startswith('enum['):
                if value.type == StrType:
                    fields = {s.strip(): i for i, s in enumerate(param.type.type[5:-1].split(','))}
                    assert value.value in fields, f'{self.name}: "{value.value}" is not a valid key for the enum type {param.type}'
                    value = TypedValue(value=fields[value.value], type=IntType)
                elif value.type == IntType:
                    assert 0 <= value.value < param.type.num_values(), f'{self.name}: {value.value} is out of bounds for the enum type {param.type}'
                else:
                    assert param.type == value.type, f'{self.name}: expected {param.type} for {param.name}, got {value.type}'
                d[param.name] = value
            else:
                assert param.type == value.type, f'{self.name}: expected {param.type} for {param.name}, got {value.type}'
                d[param.name] = value

        return d

    def mark_used(self) -> None:
        self.used = True

    def __str__(self) -> str:
        name = self.name
        return f'{name}(' + ', '.join(p.name for p in self.params) + ')'

class Action(Function):
    pass

class Query(Function):
    def __init__(self, actor_name: str, name: str, params: List[Param], rv: Type, inverted: bool = False) -> None:
        super().__init__(actor_name, name, params)
        self.rv = rv
        self.inverted = inverted
        self.num_values = rv.num_values()
        self.used = False

class Actor:
    def __init__(self, name: str, secondary_name: str = '') -> None:
        self.name = name
        self.secondary_name = secondary_name
        self.actions: Dict[str, Action] = {}
        self.queries: Dict[str, Query] = {}

    def register_action(self, action: Action) -> None:
        if action.name not in self.actions:
            self.actions[action.name] = action

    def use_action(self, action: Action) -> None:
        self.actions[action.name].mark_used()

    def register_query(self, query: Query) -> None:
        if query.name not in self.queries:
            self.queries[query.name] = query

    def use_query(self, query: Query) -> None:
        self.queries[query.name].mark_used()

    def __str__(self):
        return f'Actor {self.name}\n' + '\n'.join([
            'actions:',
            *[f'- {a}' for a in sorted(self.actions.values(), key=lambda x: x.name)],
            'queries:',
            *[f'- {q}' for q in sorted(self.queries.values(), key=lambda x: x.name)],
        ])

