from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Union

from datatype import AnyType, Type, TypedValue

class Param(NamedTuple):
    name: str
    type: Type = AnyType

class Action:
    def __init__(self, actor_name: str, name: str, params: List[Param]) -> None:
        self.actor_name = actor_name
        self.name = name
        self.params = params

    def prepare_param_dict(self, params: List[TypedValue]) -> Dict[str, TypedValue]:
        assert len(self.params) == len(params), f'{self.name}: expected {len(self.params)} params, got {len(params)}'
        d = {}
        for param, value in zip(self.params, params):
            assert param.type == value.type, f'{self.name}: expected {param.type} for {param.name}, got {value.type}'
            d[param.name] = value

        return d

    def __str__(self) -> str:
        name = self.name
        return f'{name}(' + ', '.join(p.name for p in self.params) + ')'

class Query:
    def __init__(self, actor_name: str, name: str, params: List[Param], rv: Type = AnyType, inverted: bool = False) -> None:
        self.actor_name = actor_name
        self.name = name
        self.params = params
        self.rv = rv
        self.inverted = inverted
        self.num_values = rv.num_values()

    def __str__(self) -> str:
        name = self.name
        return f'{name}(' + ', '.join(p.name for p in self.params) + ')'

class Actor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.actions: Dict[str, Action] = {}
        self.queries: Dict[str, Query] = {}
        self.locked = False

    def register_action(self, action: Action) -> None:
        if action.name not in self.actions or True:
            self.actions[action.name] = action
            # if self.locked:
                # action.auto = True

    def register_query(self, query: Query) -> None:
        if query.name not in self.queries or True: # TODO
            self.queries[query.name] = query
            # if self.locked:
                # query.auto = True

    def lock_registration(self) -> None:
        self.locked = True

    def __str__(self):
        return f'Actor {self.name}\n' + '\n'.join([
            'actions:',
            *[f'- {a}' for a in sorted(self.actions.values(), key=lambda x: x.name)],
            'queries:',
            *[f'- {q}' for q in sorted(self.queries.values(), key=lambda x: x.name)],
        ])

