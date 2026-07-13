#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensor_state.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/05/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for tensor states. A tensor state is composed of tensor values.
Intuitively, each state can be represented as a dictionary, mapping from feature names to tensors.
Semantically, this dictionary represents the state of the system (e.g., a scene, a collection of robot and object states in an environment, etc.).
"""

from jacinle.utils.printing import indent_text
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterable, Sequence, Tuple, List, Mapping, Dict

from mover._vendor.concepts.dsl.dsl_types import ObjectType, AutoType, ListType, QINDEX, get_format_context
from mover._vendor.concepts.dsl.tensor_value import TensorValue, concat_tvalues
from mover._vendor.concepts.dsl.value import ListValue


__all__ = [
    'StateObjectReference', 'StateObjectList', 'StateObjectDistribution',
    'MultidimensionalArrayInterface', 'TensorValueDict',
    'TensorStateBase',
    'NamedObjectStateMixin', 'TensorState', 'NamedObjectTensorState',
    'ObjectNameArgument', 'ObjectTypeArgument',
    'concat_states',
]


@dataclass
class StateObjectReference(object):
    """The StateObjectReference class represents a reference to an object in the state. It contains the name of the object and
    an index of the object in the state. Note that this index is depending on the type of the object. See the documentation for
    :class:`~mover._vendor.concepts.dsl.tensor_state.NamedObjectTensorState` for more details.
    """

    name: str
    index: int
    dtype: Optional[ObjectType] = None

    def clone(self, dtype: Optional[ObjectType] = None) -> 'StateObjectReference':
        return StateObjectReference(self.name, self.index, dtype or self.dtype)


class StateObjectList(ListValue):
    def __init__(self, dtype: ListType, values: Union[Sequence[StateObjectReference], slice]):
        if isinstance(values, slice):
            assert values is QINDEX, 'Only QINDEX is allowed for the values of StateObjectList.'
            super().__init__(dtype, tuple())
            self.values = QINDEX
        else:
            super().__init__(dtype, values)

    dtype: ListType

    values: Union[Tuple[StateObjectReference, ...], slice]

    @property
    def element_type(self) -> ObjectType:
        return self.dtype.element_type

    @property
    def is_qindex(self) -> bool:
        return self.values == QINDEX

    @property
    def array_accessor(self) -> Union[Sequence[int], slice]:
        if isinstance(self.values, slice):
            return self.values
        return [v.index for v in self.values]

    def clone(self, dtype: Optional[ListType] = None) -> 'StateObjectList':
        if dtype is None:
            return StateObjectList(self.dtype, self.values)

        assert isinstance(dtype, ListType) or dtype is None, 'dtype should be a ListType.'
        if self.is_qindex:
            return StateObjectList(dtype, QINDEX)
        return StateObjectList(dtype, tuple(v.clone(dtype.element_type) for v in self.values))

    def __str__(self):
        if self.values == QINDEX:
            elements_str = 'QINDEX'
        else:
            elements_str = ', '.join(str(v.name) for v in self.values)
        if get_format_context().object_format_type:
            return f'LV({{{elements_str}}}, dtype={self.dtype})'
        else:
            return f'{{{elements_str}}}'

    def __repr__(self):
        if self.values == QINDEX:
            return f'LV(QINDEX, dtype={self.dtype})'
        elements_str = ', '.join(str(v.name) for v in self.values)
        return f'LV({{{elements_str}}}, dtype={self.dtype})'


class StateObjectDistribution(StateObjectList):
    def __init__(self, dtype: ListType, values: Union[Sequence[StateObjectReference], slice], distribution: Any):
        super().__init__(dtype, values)
        self.distribution = distribution

    dtype: ListType

    values: Union[Tuple[StateObjectReference, ...], slice]

    distribution: Any
    """A distribution of the objects. Since it's a distribution over a list, the distribution is a list of probabilities, which can be represented as a numpy array, a list, etc."""

    @property
    def element_type(self) -> ObjectType:
        return self.dtype.element_type

    def __str__(self):
        elements_str = ', '.join(str(v.name) for v in self.values)
        if get_format_context().object_format_type:
            return f'LV({{{elements_str}}}, dtype={self.dtype}, distribution={self.distribution})'
        else:
            return f'{{{elements_str}, distribution={self.distribution}}}'

    def __repr__(self):
        elements_str = ', '.join(str(v.name) for v in self.values)
        return f'LV({{{elements_str}}}, dtype={self.dtype}, distribution={self.distribution})'


class MultidimensionalArrayInterface(object):
    """
    A multi-dimensional array inferface. At a high-level, this can be interpreted as a dictionary that maps
    feature names (keys) to multi-diemsntional tensors (value).
    """

    def __init__(self, all_feature_names: Iterable[str] = tuple()):
        self.all_feature_names = set(all_feature_names)

    def clone(self) -> 'MultidimensionalArrayInterface':
        """Clone the multidimensional array interface."""
        raise NotImplementedError()

    def get_feature(self, name: str) -> TensorValue:
        """Get the feature tensor with the given name."""
        raise NotImplementedError()

    def _set_feature_impl(self, name: str, feature: TensorValue):
        """Set the feature tensor with the given name. It is guaranteed that the name is in the all_feature_names."""
        raise NotImplementedError()

    def set_feature(self, name: str, feature: TensorValue):
        """Set the feature tensor with the given name."""
        if name not in self.all_feature_names:
            self.all_feature_names.add(name)
        self._set_feature_impl(name, feature)

    def update_feature(self, other_tensor_dict: Mapping[str, TensorValue]):
        """Update the feature tensors with the given tensor dict."""
        for key, value in other_tensor_dict.items():
            self.set_feature(key, value)

    def __contains__(self, item: str) -> bool:
        """Check if the given feature name is in the interface."""
        return item in self.all_feature_names

    def __getitem__(self, name: str) -> TensorValue:
        """Get the feature tensor with the given name."""
        return self.get_feature(name)

    def __setitem__(self, key, value):
        """Set the feature tensor with the given name."""
        self.set_feature(key, value)

    def keys(self) -> Iterable[str]:
        """Get the feature names."""
        return self.all_feature_names

    def values(self) -> Iterable[TensorValue]:
        """Get the feature tensors."""
        for key in self.all_feature_names:
            yield self.get_feature(key)

    def items(self) -> Iterable[Tuple[str, TensorValue]]:
        """Get the feature name-tensor pairs."""
        for key in self.all_feature_names:
            yield key, self.get_feature(key)


class TensorValueDict(MultidimensionalArrayInterface):
    """Basic tensor dict implementation."""

    def __init__(self, tensor_dict: Optional[Dict[str, TensorValue]] = None):
        if tensor_dict is None:
            tensor_dict = dict()

        super().__init__(tensor_dict.keys())
        self.tensor_dict = tensor_dict

    def clone(self) -> 'TensorValueDict':
        return type(self)({k: v.clone() for k, v in self.tensor_dict.items()})

    def get_feature(self, name: str) -> TensorValue:
        return self.tensor_dict[name]

    def _set_feature_impl(self, name, feature: TensorValue):
        self.tensor_dict[name] = feature


class TensorStateBase(object):
    """A state representation is essentially a mapping from feature names to tensors."""

    @property
    def batch_dims(self) -> int:
        raise NotImplementedError()

    @property
    def features(self) -> MultidimensionalArrayInterface:
        raise NotImplementedError()

    def clone(self) -> 'TensorStateBase':
        raise NotImplementedError()

    def __getitem__(self, name: str):
        return self.features[name]

    def __str__(self) -> str:
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()


ObjectNameArgument = Union[Iterable[str], Mapping[str, ObjectType]]
ObjectTypeArgument = Optional[Iterable[ObjectType]]


class NamedObjectStateMixin(object):
    """A state type mixin with named objects."""

    def __init__(self, object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None):
        """A state type mixin with named objects.
        The object names can be either a list of names, or a mapping from names to :class:`ObjectType`'s.

            - If the `object_names` is a list of names, then the user should also specify a list of object types.
            - If the `object_names` is a mapping from names to :class:`ObjectType`'s, then the `object_types` argument should be None.

        Args:
            object_names: the object names.
            object_types: the object types.
        """
        if isinstance(object_names, Mapping):
            assert object_types is None, 'object_types should be None if object_names is a mapping.'
            self.object_names = tuple(object_names.keys())
            self.object_types = tuple(object_names.values())
        else:
            assert object_types is not None, 'object_types should not be None if object_names is not a mapping.'
            self.object_names = tuple(object_names)
            self.object_types = tuple(object_types)

        self.object_type2name: Dict[str, List[str]] = dict()
        self.object_name2index: Dict[Tuple[str, str], int] = dict()
        self.object_name2defaultindex: Dict[str, Tuple[str, int]] = dict()

        for name, obj_type in zip(self.object_names, self.object_types):
            self.object_type2name.setdefault(obj_type.typename, list()).append(name)
            self.object_name2index[name, obj_type.typename] = len(self.object_type2name[obj_type.typename]) - 1
            self.object_name2defaultindex[name] = obj_type.typename, len(self.object_type2name[obj_type.typename]) - 1
            for t in obj_type.iter_parent_types():
                self.object_type2name.setdefault(t.typename, list()).append(name)
                self.object_name2index[name, t.typename] = len(self.object_type2name[t.typename]) - 1

    @property
    def nr_objects(self) -> int:
        """The number of objects in the current state."""
        return len(self.object_types)

    def get_typed_index(self, name: str, typename: Optional[str] = None) -> int:
        """Get the typed index of the object with the given name.
        There is an additional typename argument to specify the type of the object.
        Because the same object can have multiple types (due to inheritence), the object can have multiple typed indices, one for each type.
        When the typename is not specified, the default type of the object is used (i.e., the most specific type).

        Args:
            name: the name of the object.
            typename: the typename of the object. If not specified, the default type of the object is used (i.e. the most specific type).

        Returns:
            the typed index of the object.
        """
        if typename is None or typename == AutoType.typename:
            return self.object_name2defaultindex[name][1]
        return self.object_name2index[name, typename]

    def get_default_typename(self, name: str) -> str:
        """Get the typename of the object with the given name."""
        return self.object_name2defaultindex[name][0]

    def get_name(self, typename: str, index: int) -> str:
        """Get the name of the object with the given type and index."""
        return self.object_type2name[typename][index]

    def get_objects_by_type(self, typename: str) -> List[str]:
        """Get the names of the objects with the given type."""
        return self.object_type2name[typename]

    def get_nr_objects_by_type(self, typename: str) -> int:
        """Get the number of objects with the given type."""
        return len(self.object_type2name[typename])

    def get_state_object_reference(self, dtype: Union[ObjectType, str], index: Optional[int] = None, name: Optional[str] = None) -> StateObjectReference:
        """Get the object reference with the given type and index."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename

        if index is not None:
            return StateObjectReference(self.get_name(typename, index), index, dtype)
        if name is not None:
            return StateObjectReference(name, self.get_typed_index(name, typename), dtype)
        raise ValueError('Either indices or names should be specified.')

    def get_state_object_list(self, dtype: Union[ObjectType, str], indices: Optional[Union[Sequence[int], slice]] = None, names: Optional[Sequence[str]] = None) -> StateObjectList:
        """Get a list of object references with the given type and indices."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename
        if indices is not None:
            if isinstance(indices, slice):
                if indices != QINDEX:
                    raise ValueError('Only QINDEX is allowed for the indices of StateObjectList.')
                return StateObjectList(ListType(dtype), QINDEX)
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, index) for index in indices])
        if names is not None:
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, name=name) for name in names])
        raise ValueError('Either indices or names should be specified.')


class TensorState(TensorStateBase):
    """A state representation is essentially a mapping from feature names to tensors."""

    def __init__(self, features: Optional[Union[Mapping[str, Any], TensorValueDict]] = None, batch_dims: int = 0, internals: Optional[Dict[str, Any]] = None):
        """Initialize a state.

        Args:
            features: the features of the state.
            batch_dims: the number of batch dimensions.
            internals: the internal state of the state.
        """

        if features is None:
            features = dict()
        if internals is None:
            internals = dict()

        if isinstance(features, TensorValueDict):
            self._features = features
        else:
            self._features = TensorValueDict(features)
        self._batch_dims = batch_dims

        self._internals = dict(internals)

    @property
    def batch_dims(self) -> int:
        """The number of batchified dimensions. For the basic State, it should be 0."""
        return self._batch_dims

    @property
    def features(self) -> TensorValueDict:
        return self._features

    @property
    def internals(self) -> Dict[str, Any]:
        """Additional internal information about the state."""
        return self._internals

    def clone(self) -> 'TensorState':
        return type(self)(features=self._features.clone(), batch_dims=self._batch_dims, internals=self.clone_internals())

    def clone_internals(self):
        """Clone the internals."""
        return self.internals.copy()

    def summary_string(self) -> str:
        """Get a summary string of the state. The main difference between this and __str__ is that this function only formats the shape of intermediate tensors."""
        fmt = f'''{type(self).__name__}{{
  states:
'''
        for p in self.features.all_feature_names:
            feature = self.features[p]
            fmt += f'    {p}: {feature.format(content=False)}\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def __str__(self):
        fmt = f'''{type(self).__name__}{{\n'''
        fmt += self.extra_state_str_before()
        fmt += '  states:\n'
        for p in self.features.all_feature_names:
            tensor = self.features[p]
            fmt += f'    - {p}'
            fmt += ': ' + indent_text(str(tensor), level=2).strip() + '\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def extra_state_str_before(self) -> str:
        """Extra state string before the features."""
        return ''

    def extra_state_str_after(self) -> str:
        """Extra state string."""
        return ''


class NamedObjectTensorState(TensorState, NamedObjectStateMixin):
    """A state type with named objects."""

    def __init__(self, features: Optional[Union[Mapping[str, Any], MultidimensionalArrayInterface]], object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None, batch_dims: int = 0, internals: Optional[Mapping[str, Any]] = None):
        """Initialize the state.

        Args:
            features: the features of the state.
            object_types: the types of the objects.
            object_names: the names of the objects. If the object_names is a mapping, the object_types should be None.
            batch_dims: the number of batchified dimensions.
            internals: the internals of the state.
        """

        TensorState.__init__(self, features, batch_dims, internals)
        NamedObjectStateMixin.__init__(self, object_names, object_types)

    def clone(self) -> 'NamedObjectTensorState':
        return type(self)(features=self._features.clone(), object_types=self.object_types, object_names=self.object_names, batch_dims=self._batch_dims, internals=self.clone_internals())

    def extra_state_str_before(self) -> str:
        """Extra state string: add the objects."""
        if self.object_names is not None:
            typename2objects = dict()
            for name, dtype in zip(self.object_names, self.object_types):
                typename2objects.setdefault(dtype.typename, list()).append(name)
            objects_str = '; '.join([f'{typename}: [{", ".join(names)}]' for typename, names in typename2objects.items()])
        else:
            objects_str = ', '.join(self.object_names)
        return '  objects: ' + objects_str + '\n'


def concat_states(*args: TensorState) -> TensorState:
    """Concatenate a list of states into a batch state.

    Args:
        *args: a list of states.

    Returns:
        a new state, which is the concatenation of the input states.
        This new state will have a new batch dimension.
    """

    if len(args) == 0:
        raise ValueError('No states to concatenate.')

    all_features = list(args[0].features.all_feature_names)

    # 1. Sanity checks.
    for state in args[1:]:
        assert len(all_features) == len(state.features.all_feature_names)
        for feature in all_features:
            assert feature in state.features.all_feature_names

    # 2. Put the same feature into a list.
    features = {feature_name: list() for feature_name in all_features}
    for state in args:
        for key, value in state.features.tensor_dict.items():
            features[key].append(value)

    # 3. Actually, compute the features.
    feature_names = list(features.keys())
    for feature_name in feature_names:
        features[feature_name] = concat_tvalues(*features[feature_name])

    # 4. Create the new state.
    state = args[0]
    kwargs: Dict[str, Any] = dict()
    if isinstance(state, NamedObjectTensorState):
        kwargs = dict(object_types=state.object_types, object_names=state.object_names)

    kwargs['features'] = features
    kwargs['batch_dims'] = args[0].batch_dims + 1
    return type(state)(**kwargs)
