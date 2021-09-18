from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union

from bitstring import BitStream, pack

from .datatype import TypedValue
from .actors import Actor
from .nodes import Node, RootNode, ActionNode, SwitchNode, ForkNode, JoinNode, SubflowNode
from .block import DataBlock, ContainerBlock, Block
from .str_ import StringPool, String
from .dic_ import Dictionary
from .array import BlockPtrArray, BlockArray, Uint16Array
from .container import Container

class _Actor(DataBlock):
    def __init__(self, name: str, sec_name: str, actions: Optional[BlockPtrArray[String]],
                 queries: Optional[BlockPtrArray[String]], pool: StringPool) -> None:
        super().__init__(55)
        self._add_pointer(0, pool[name])
        self._add_pointer(8, pool[sec_name])
        self._add_pointer(16, pool.empty)
        self._add_pointer(24, actions)
        self._add_pointer(32, queries)
        self._add_pointer(40, None)

        with self._at_offset(48):
            self.buffer.overwrite(pack('uintle:16', actions.n if actions is not None else 0))
            self.buffer.overwrite(pack('uintle:16', queries.n if queries is not None else 0))
            self.buffer.overwrite(pack('uintle:16', 0xFFFF))
            self.buffer.overwrite(pack('uintle:8', 1))

    def alignment(self) -> int:
        return 8

class _SwitchCase(DataBlock):
    def __init__(self, value: int, event_index: int) -> None:
        super().__init__(6)

        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:32', value))
            self.buffer.overwrite(pack('uintle:16', event_index))

    def alignment(self) -> int:
        return 8

class _SwitchData(ContainerBlock):
    def __init__(self, params: Optional[Container], cases: BlockArray[_SwitchCase]) -> None:
        super().__init__(([params] if params else []) + [cases]) # type: ignore

    def alignment(self) -> int:
        return 8

_EventData = Union[Container, Uint16Array, _SwitchData]

class _Event(DataBlock):
    def  __init__(self, name: str, event_type: int, pool: StringPool) -> None:
        super().__init__(0x28)

        self._add_pointer(0, pool[name])
        with self._at_offset(8):
            self.buffer.overwrite(pack('uintle:8', event_type))

    def alignment(self) -> int:
        return 8

class _ActionEvent(_Event):
    def __init__(self, name: str, next_index: int, actor_index: int, action_index: int,
                 params: Optional[Container], pool: StringPool) -> None:
        super().__init__(name, 0, pool)

        with self._at_offset(0xa):
            self.buffer.overwrite(pack('uintle:16', next_index))
            self.buffer.overwrite(pack('uintle:16', actor_index))
            self.buffer.overwrite(pack('uintle:16', action_index))

        self._add_pointer(0x10, params)

class _SwitchEvent(_Event):
    def __init__(self, name: str, actor_index: int, query_index: int,
                 params: Optional[Container], cases: BlockArray[_SwitchCase], pool: StringPool) -> None:
        super().__init__(name, 1, pool)

        with self._at_offset(0xa):
            self.buffer.overwrite(pack('uintle:16', cases.n))
            self.buffer.overwrite(pack('uintle:16', actor_index))
            self.buffer.overwrite(pack('uintle:16', query_index))

        self._add_pointer(0x10, params)
        self._add_pointer(0x18, cases)

class _ForkEvent(_Event):
    def __init__(self, name: str, forks: Uint16Array, join_index: int, pool: StringPool) -> None:
        super().__init__(name, 2, pool)

        with self._at_offset(0xa):
            self.buffer.overwrite(pack('uintle:16', forks.n))
            self.buffer.overwrite(pack('uintle:16', join_index))

        self._add_pointer(0x10, forks)

class _JoinEvent(_Event):
    def __init__(self, name: str, next_index: int, pool: StringPool) -> None:
        super().__init__(name, 3, pool)

        with self._at_offset(0xa):
            self.buffer.overwrite(pack('uintle:16', next_index))

class _SubflowEvent(_Event):
    def __init__(self, name: str, next_index: int, params: Optional[Container],
                 flowchart: str, entrypoint: str, pool: StringPool) -> None:
        super().__init__(name, 4, pool)

        with self._at_offset(0xa):
            self.buffer.overwrite(pack('uintle:16', next_index))

        self._add_pointer(0x10, params)
        self._add_pointer(0x18, pool[flowchart])
        self._add_pointer(0x20, pool[entrypoint])

class _VarDefData(DataBlock):
    def alignment(self) -> int:
        return 8

class _IntVarDefData(_VarDefData):
    def __init__(self, value: int) -> None:
        super().__init__(4)
        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:32', value))

class _BoolVarDefData(_VarDefData):
    def __init__(self, value: bool) -> None:
        super().__init__(4)
        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:32', 0x80000001 if value else 0))

class _FloatVarDefData(_VarDefData):
    def __init__(self, value: float) -> None:
        super().__init__(4)
        with self._at_offset(0):
            self.buffer.overwrite(pack('floatle:32', value))

class _VarDefFooter(DataBlock):
    def __init__(self, type_: int, count: int) -> None:
        # other types should be supported, but unsure how they
        # are handled -- it's probably a pointer, but where the pointed data
        # resides is unknown - ACNH has no such usage
        assert type_ in (2, 3, 4) # int, bool, float

        super().__init__(4)
        with self._at_offset(0):
            self.buffer.overwrite(pack('uintle:16', count))
            self.buffer.overwrite(pack('uintle:16', type_))

    def alignment(self) -> int:
        return 8

class _VarDef(ContainerBlock):
    def __init__(self, value: TypedValue) -> None:
        # data (64 bit align) + footer (64 bit align)
        # unsure how most types are handled
        parts: List[Block] = []
        if value.type.type.startswith('int') or value.type.type.startswith('enum'):
            assert isinstance(value.value, int)
            parts = [
                _IntVarDefData(value.value),
                _VarDefFooter(2, 1),
            ]
        elif value.type.type == 'bool':
            assert isinstance(value.value, bool)
            parts = [
                _BoolVarDefData(value.value),
                _VarDefFooter(3, 1),
            ]
        elif value.type.type == 'float':
            assert isinstance(value.value, float)
            parts = [
                _FloatVarDefData(value.value),
                _VarDefFooter(4, 1),
            ]
        else:
            raise ValueError(f'unsupported vardef type: {value.type}')

        super().__init__(parts)

    def alignment(self) -> int:
        return 8

class _Pad24(DataBlock):
    def __init__(self) -> None:
        super().__init__(24)

    def alignment(self) -> int:
        return 8

class _EntrypointVardefData(ContainerBlock):
    def __init__(self, vardef_names: Dictionary, vardefs: BlockArray[_VarDef]) -> None:
        super().__init__([vardef_names, vardefs])

# there is a 24-byte padding per entrypoint after SubflowIndexArray, unless
# the entrypoint has vardefs
_EntrypointData = Union[_EntrypointVardefData, _Pad24]

class _Entrypoint(DataBlock):
    def __init__(self, vardef_names: Optional[Dictionary], vardefs: Optional[BlockArray[_VarDef]],
                 subflow_indices: Optional[Uint16Array], main_index: int) -> None:
        super().__init__(0x1e)

        self._add_pointer(0, subflow_indices)
        self._add_pointer(8, vardef_names)
        self._add_pointer(16, vardefs)

        with self._at_offset(24):
            self.buffer.overwrite(pack('uintle:16', subflow_indices.n if subflow_indices else 0))
            self.buffer.overwrite(pack('uintle:16', vardefs.n if vardefs is not None else 0))
            self.buffer.overwrite(pack('uintle:16', main_index))

    def alignment(self) -> int:
        return 8

class _FlowchartHeader(DataBlock):
    def __init__(self, name: str, num_actions: int, num_queries: int, actors: BlockArray[_Actor],
            events: BlockArray[_Event], entrypoint_names: Dictionary,
            entrypoints: BlockArray[_Entrypoint], pool: StringPool) -> None:
        super().__init__(0x48)

        self.pool = pool
        with self._at_offset(0):
            self.buffer.overwrite(b'EVFL')
        with self._at_offset(0x10):
            self.buffer.overwrite(pack('uintle:16', actors.n))
            self.buffer.overwrite(pack('uintle:16', num_actions))
            self.buffer.overwrite(pack('uintle:16', num_queries))
            self.buffer.overwrite(pack('uintle:16', events.n))
            self.buffer.overwrite(pack('uintle:16', entrypoints.n))
        self._add_pointer(0x20, pool[name])
        self._add_pointer(0x28, actors)
        self._add_pointer(0x30, events)
        self._add_pointer(0x38, entrypoint_names)
        self._add_pointer(0x40, entrypoints)

    def prepare_bitstream(self) -> BitStream:
        with self._at_offset(4):
            self.buffer.overwrite(pack('uintle:32', self.pool.offset - self.offset))
        return super().prepare_bitstream()

    def alignment(self) -> int:
        return 8

class Flowchart(ContainerBlock):
    def __init__(self, name: str, actors_: List[Actor], events_: List[Node],
                 entrypoints_: List[RootNode], pool: StringPool) -> None:
        actors: List[_Actor] = []
        events: List[_Event] = []
        entrypoint_names: List[str] = [] # -> Dictionary
        entrypoints: List[_Entrypoint] = []
        event_data: List[_EventData] = []
        actor_fnames: List[BlockPtrArray[String]] = []
        entrypoint_data: List[_EntrypointData] = []

        num_actions = 0
        num_queries = 0

        actor_indices: Dict[str, int] = {}
        action_indices: Dict[str, Dict[str, int]] = {}
        query_indices: Dict[str, Dict[str, int]] = {}
        for actor in actors_:
            actions = [n for n, a in actor.actions.items() if a.used]
            queries = [n for n, q in actor.queries.items() if q.used]
            num_actions += len(actions)
            num_queries += len(queries)

            if not actions and not queries:
                continue

            actor_indices[actor.name] = len(actor_indices)
            action_indices[actor.name] = {n: i for i, n in enumerate(actions)}
            query_indices[actor.name] = {n: i for i, n in enumerate(queries)}

            action_array = BlockPtrArray[String]([pool[s] for s in actions]) if actions else None
            query_array = BlockPtrArray[String]([pool[s] for s in queries]) if queries else None
            actors.append(_Actor(actor.name, actor.secondary_name, action_array, query_array, pool))
            if action_array:
                actor_fnames.append(action_array)
            if query_array:
                actor_fnames.append(query_array)

        event_indices = {n: i for i, n in enumerate(events_)}
        entrypoint_calls: Dict[str, List[int]] = {n.name: [] for n in entrypoints_}
        for event in events_:
            ev: _Event
            evdata: Optional[_EventData]
            if isinstance(event, ActionNode):
                nxt = event_indices[event.out_edges[0]] if event.out_edges else 0xFFFF
                actor_index = actor_indices[event.action.actor_name]
                action_index = action_indices[event.action.actor_name][event.action.name]
                params = Container(event.params, pool) if event.params else None

                ev = _ActionEvent(event.name, nxt, actor_index, action_index, params, pool)
                evdata = params
            elif isinstance(event, SwitchNode):
                num_cases = len(event.cases)
                actor_index = actor_indices[event.query.actor_name]
                query_index = query_indices[event.query.actor_name][event.query.name]
                params = Container(event.params, pool) if event.params else None
                cases = BlockArray([
                    _SwitchCase(v, event_indices[n])
                    for n, vs in event.cases.items() for v in vs
                ])

                ev = _SwitchEvent(event.name, actor_index, query_index, params, cases, pool)
                evdata = _SwitchData(params, cases)
            elif isinstance(event, ForkNode):
                forks = Uint16Array([event_indices[e] for e in event.out_edges if e != event.join_node])
                join_index = event_indices[event.join_node]

                ev = _ForkEvent(event.name, forks, join_index, pool)
                evdata = forks
            elif isinstance(event, JoinNode):
                nxt = event_indices[event.out_edges[0]] if event.out_edges else 0xFFFF

                ev = _JoinEvent(event.name, nxt, pool)
                evdata = None
            elif isinstance(event, SubflowNode):
                nxt = event_indices[event.out_edges[0]] if event.out_edges else 0xFFFF
                flowchart_name = event.ns
                entrypoint_name = event.called_root_name
                params = Container(event.params, pool) if event.params else None

                ev = _SubflowEvent(event.name, nxt, params, flowchart_name, entrypoint_name, pool)
                evdata = params
            else:
                raise TypeError(f'{type(event).__name__} not supported')

            events.append(ev)
            if evdata:
                event_data.append(evdata)

        for entrypoint in entrypoints_:
            si = Uint16Array(entrypoint_calls[entrypoint.name]) or None

            vardef_names: Optional[Dictionary] = None
            vardefs: Optional[BlockArray[_VarDef]] = None
            epdata: _EntrypointData = _Pad24()
            if entrypoint.vardefs:
                vardef_names = Dictionary([v.name for v in entrypoint.vardefs], pool)
                vardefs = BlockArray[_VarDef]([
                    _VarDef(TypedValue(v.type, v.initial_value))
                    for v in entrypoint.vardefs
                ])
                epdata = _EntrypointVardefData(vardef_names, vardefs)
            start = event_indices[entrypoint.out_edges[0]] if entrypoint.out_edges else 0xFFFF
            ep = _Entrypoint(vardef_names, vardefs, si, start)

            entrypoint_names.append(entrypoint.name)
            entrypoints.append(ep)
            entrypoint_data.append(epdata)

        actor_array = BlockArray[_Actor](actors)
        event_array = BlockArray[_Event](events)
        entrypoint_dictionary = Dictionary(entrypoint_names, pool)
        entrypoint_array = BlockArray[_Entrypoint](entrypoints)
        event_data_array = BlockArray[_EventData](event_data)
        entrypoint_data_array = BlockArray[_EntrypointData](entrypoint_data)

        header = _FlowchartHeader(name, num_actions, num_queries, actor_array, event_array,
                entrypoint_dictionary, entrypoint_array, pool)

        super().__init__([
            header,
            actor_array,
            event_array,
            entrypoint_dictionary,
            entrypoint_array,
            event_data_array,
            *actor_fnames,
            entrypoint_data_array,
        ])

    def alignment(self) -> int:
        return 8
