#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dsl_types.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/09/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for custom types in a DSL.

All types extend the base class :class:`TypeBase`.

:class:`TypeBase` has four derived types::

    - :class:`ObjectType`: corresponds to an object in the physical world.
    - :class:`ValueType`: corresponds to an encoding of a feature of the object.
    - :class:`Sequence`: corresponds to a tuple type, a list type, or a multidimensional list type.
    - :class:`~mover._vendor.concepts.dsl.dsl_functions.FunctionType`: corresponds to a function over the encodings.
    - :class:`~mover._vendor.concepts.dsl.dsl_functions.OverloadedFunctionType`: corresponds to a collection of functions that share the same name.
    - :class:`UnionType`: corresponds to a union of types.

Furthermore, :class:`ValueType` has four derived types::

    - :class:`ConstantType`: corresponds to a constant.
    - :class:`TensorValueTypeBase`: corresponds to a tensor-based encoding.
    - :class:`PyObjValueType`: corresponds to an arbitrary Python object.


:class:`TensorValueTypeBase` has three derived types::

    - :class:`ScalarValueType`: corresponds to a scalar value.
    - :class:`VectorValueType`: corresponds to a vector.
    - :class:`NamedTensorValueType`: corresponds to a named tensor.

:class:`SequenceType` has two derived types::

    - :class:`TupleType`: corresponds to a tuple.
    - :class:`ListType`: corresponds to a list. This type is also of type :class:`VariableSizedSequenceType`.
    - :class:`MultidimensionalListType`: corresponds to a multidimensional list. This type is also of type :class:`VariableSizedSequenceType`.
"""

import torch
from typing import TYPE_CHECKING, Optional, Union, Iterable, Sequence, Tuple

import jacinle
from jacinle.logging import get_logger
from jacinle.utils.defaults import option_context

if TYPE_CHECKING:
    from mover._vendor.concepts.dsl.dsl_functions import FunctionType
    from mover._vendor.concepts.dsl.value import ListValue
    from mover._vendor.concepts.dsl.tensor_state import StateObjectReference, StateObjectList

logger = get_logger(__file__)

__all__ = [
    'FormatContext', 'get_format_context',
    'TypeBase', 'AliasType',
    'UnionType',
    'AutoType', 'AnyType', 'UnionTyping', 'is_any_type', 'is_auto_type',  # Special types and type constructors.
    'ObjectType',
    'ValueType', 'ConstantType', 'PyObjValueType', 'TensorValueTypeBase', 'ScalarValueType', 'STRING', 'BOOL', 'INT64', 'FLOAT32', 'VectorValueType', 'NamedTensorValueType',  # Value types.
    'SequenceType', 'TupleType', 'UniformSequenceType', 'ListType', 'BatchedListType',  # Sequence types.
    'QINDEX', 'Variable', 'ObjectConstant', 'UnnamedPlaceholder'  # Placeholder-like objects.
]


class FormatContext(option_context(
    '_FormatContext',
    type_format_cls=False, object_format_type=True, function_format_lambda=False, expr_max_length=120, type_varname_sep=': '
)):
    """FormatContext is a context manager that controls the format of types and objects."""

    type_format_cls: bool
    """If True, the type will be formatted as ``Type[typename]``."""

    object_format_type: bool
    """If True, the object format will include the type."""

    function_format_lambda: bool
    """If True, the function format will be lambda-function styled. Otherwise, it will be python-function styled."""

    expr_max_length: int
    """The maximum length of the expression per line."""

    type_varname_sep: str
    """The separator between the type and the variable name."""


get_format_context = FormatContext.get_default


class TypeBase(object):
    """Base class for all types."""

    def __init__(self, typename: str, alias: Optional[str] = None, parent_type: Optional['TypeBase'] = None):
        """Initialize the type.

        Args:
            typename: The name of the type.
            alias: The alias of the type.
        """

        self._typename = typename
        self._alias = alias
        self._parent_type = parent_type

    @property
    def typename(self) -> str:
        """The (full) typename of the type."""
        return self._typename

    @property
    def alias(self) -> Optional[str]:
        """An optional alias of the type."""
        return self._alias

    @property
    def parent_type(self) -> Optional['TypeBase']:
        """The parent type of the type."""
        return self._parent_type

    @property
    def element_type(self) -> Optional['TypeBase']:
        """The element type of the type."""
        raise TypeError(f'Type {self.typename} does not have an element type.')

    def set_parent_type(self, parent_type: 'TypeBase'):
        self._parent_type = parent_type

    @property
    def parent_typename(self):
        """Return the typename of the parent type."""
        return self._parent_type.typename

    @property
    def base_typename(self):
        """Return the typename of the base type."""
        if self._parent_type is None:
            return self.typename
        return self._parent_type.base_typename

    @property
    def is_wrapped_value_type(self) -> bool:
        return False

    @property
    def is_object_type(self) -> bool:
        """Return whether the type is an object type."""
        return False

    @property
    def is_value_type(self) -> bool:
        """Return whether the type is a value type."""
        return False

    @property
    def is_tensor_value_type(self) -> bool:
        """Return whether the type is a tensor value type."""
        return False

    @property
    def is_scalar_value_type(self) -> bool:
        return False

    @property
    def is_vector_value_type(self) -> bool:
        return False

    @property
    def is_pyobj_value_type(self) -> bool:
        """Return whether the type is a Python object value type."""
        return False

    @property
    def is_sequence_type(self) -> bool:
        """Return whether the type is a sequence type."""
        return False

    @property
    def is_tuple_type(self) -> bool:
        """Return whether the type is a tuple type."""
        return False

    @property
    def is_uniform_sequence_type(self) -> bool:
        return False

    @property
    def is_list_type(self) -> bool:
        """Return whether the type is a list type."""
        return False

    @property
    def is_batched_list_type(self) -> bool:
        """Return whether the type is a multidimensional list type."""
        return False

    def __str__(self) -> str:
        if get_format_context().type_format_cls:
            return self.long_str()
        else:
            return self.short_str()

    __repr__ = jacinle.repr_from_str

    def short_str(self) -> str:
        """Return the short string representation of the type."""
        if self.alias is not None:
            return self.alias
        return self.typename

    def long_str(self) -> str:
        """Return the long string representation of the type."""
        return f'Type[{self.short_str()}]'

    def assignment_type(self) -> 'TypeBase':
        """Return the value type for assignment."""
        return self

    def downcast_compatible(self, other: 'TypeBase', allow_self_list: bool = False, allow_list: bool = False) -> bool:
        """Check if the type is downcast-compatible with the other type; that is, if this type is a subtype of the other type.

        Args:
            other: the other type.
            allow_self_list: if True, this type can be a list type derived from the other type.
            allow_list: if True, the other type can be a list type derived from the type.
        """
        if self.typename == other.typename or other == AnyType or self == AutoType:
            return True
        if self.parent_type is not None and self.parent_type.downcast_compatible(other, allow_self_list=allow_self_list, allow_list=allow_list):
            return True
        if self.is_uniform_sequence_type and other.is_uniform_sequence_type:
            if not self.element_type.downcast_compatible(other.element_type, allow_self_list=False, allow_list=False):
                return False
            if self.is_batched_list_type and other.is_batched_list_type:
                if len(self.index_dtypes) != len(other.index_dtypes):
                    return False
                for i, j in zip(self.index_dtypes, other.index_dtypes):
                    if not i.downcast_compatible(j, allow_self_list=False, allow_list=False):
                        return False
                return True
            return True
        if allow_self_list and self.is_uniform_sequence_type:
            if self.element_type.downcast_compatible(other, allow_self_list=False, allow_list=False):
                return True
        if allow_list and self.is_uniform_sequence_type:
            if self.downcast_compatible(other.element_type, allow_self_list=False, allow_list=False):
                return True
        return False

    def unwrap_alias(self):
        return self

    def __eq__(self, other: 'TypeBase') -> bool:
        return self.typename == other.typename

    def __ne__(self, other: 'TypeBase') -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.typename)


class AliasType(TypeBase):
    def __init__(self, typename: str, parent_type: TypeBase, alias: Optional[str] = None):
        super().__init__(typename, alias=alias, parent_type=parent_type)

    @property
    def is_wrapped_value_type(self):
        return True

    @property
    def is_object_type(self) -> bool:
        return self.parent_type.is_object_type

    @property
    def is_value_type(self) -> bool:
        return self.parent_type.is_value_type

    @property
    def is_tensor_value_type(self) -> bool:
        return self.parent_type.is_tensor_value_type

    @property
    def is_scalar_value_type(self) -> bool:
        return self.parent_type.is_scalar_value_type

    @property
    def is_vector_value_type(self) -> bool:
        return self.parent_type.is_vector_value_type

    @property
    def is_pyobj_value_type(self) -> bool:
        return self.parent_type.is_pyobj_value_type

    @property
    def is_sequence_type(self) -> bool:
        return self.parent_type.is_sequence_type

    @property
    def is_tuple_type(self) -> bool:
        return self.parent_type.is_tuple_type

    @property
    def is_uniform_sequence_type(self) -> bool:
        return self.parent_type.is_uniform_sequence_type

    @property
    def is_list_type(self) -> bool:
        return self.parent_type.is_list_type

    @property
    def is_batched_list_type(self) -> bool:
        return self.parent_type.is_batched_list_type

    def downcast_compatible(self, other: TypeBase, allow_self_list: bool = False, allow_list: bool = False) -> bool:
        return self.parent_type.downcast_compatible(other, allow_self_list=allow_self_list, allow_list=allow_list)

    def assignment_type(self) -> TypeBase:
        return self.parent_type.assignment_type()

    @property
    def element_type(self) -> Optional['TypeBase']:
        return self.parent_type.element_type

    def short_str(self) -> str:
        return f'{self.typename} alias {self.parent_type.short_str()}'

    def unwrap_alias(self):
        return self.parent_type.unwrap_alias()

    def __eq__(self, other: TypeBase) -> bool:
        if isinstance(other, AliasType):
            return other.parent_type == self.parent_type
        return self.parent_type == other

    def __hash__(self) -> int:
        return hash(self.parent_type)


"""AnyType corresponds to the union of all types."""
AnyType = TypeBase('AnyType')


"""AutoType means the type will be automatically inferred later."""
AutoType = TypeBase('AutoType')


def is_any_type(t: TypeBase) -> bool:
    """Check if the type is AnyType."""
    return t == AnyType


def is_auto_type(t: TypeBase) -> bool:
    """Check if the type is AutoType."""
    return t == AutoType


class UnionType(TypeBase):
    """The UnionType is a type that is the union of multiple types."""

    def __init__(self, *types: TypeBase, alias: Optional[str] = None):
        """Initialize the union type.

        Args:
            types: The types in the union.
            alias: The alias of the union type.
        """
        self.types = tuple(types)
        super().__init__(self.long_str(), alias=alias)

    types: Tuple[TypeBase, ...]
    """The underlying types of the union type."""

    def short_str(self) -> str:
        return ' | '.join(t.short_str() for t in self.types)

    def long_str(self) -> str:
        return 'Union[' + ', '.join(t.long_str() for t in self.types) + ']'

    def downcast_compatible(self, other: TypeBase, allow_self_list: bool = False, allow_list: bool = False) -> bool:
        raise NotImplementedError('Cannot downcast to a union type.')


class _UnionTypingSugar(object):
    def __getitem__(self, item: Tuple[TypeBase]):
        assert isinstance(item, tuple)
        return UnionType(*item)


UnionTyping = _UnionTypingSugar()


class ObjectType(TypeBase):
    """The ObjectType corresponds to the type of "real-world" objects."""

    def __init__(self, typename: str, parent_types: Optional[Sequence['ObjectType']] = None, alias: Optional[str] = None):
        """Initialize the object type.

        Args:
            typename: The name of the object type.
            alias: The alias of the object type.
        """
        super().__init__(typename, alias=alias)

        self.parent_types = tuple(parent_types) if parent_types is not None else tuple()

    @property
    def is_object_type(self):
        return True

    parent_types: Tuple['ObjectType', ...]
    """The parent types of the object type."""

    def iter_parent_types(self) -> Iterable['ObjectType']:
        """Iterate over all parent types.

        Yields:
            the parent types following the inheritance order.
        """
        for parent_type in self.parent_types:
            yield parent_type
            yield from parent_type.iter_parent_types()

    def long_str(self) -> str:
        if len(self.parent_types) == 0:
            return f'OT[{self.typename}]'

        return f'OT[{self.typename}, parent={", ".join(t.typename for t in self.parent_types)}]'


class ValueType(TypeBase):
    """The ValueType corresponds to a value of a certain type."""

    @property
    def is_value_type(self):
        return True

    def long_str(self) -> str:
        return f'VT[{self.typename}]'


class ConstantType(ValueType):
    """The ConstantType corresponds to a constant value."""

    def long_str(self) -> str:
        return f'CT[{self.typename}]'


class PyObjValueType(ValueType):
    def __init__(self, pyobj_type: Union[type, str], typename: Optional[str] = None, alias: Optional[str] = None, parent_type: Optional['PyObjValueType'] = None):
        self.pyobj_type = pyobj_type
        if typename is None:
            if isinstance(pyobj_type, type):
                typename = pyobj_type.__name__
            else:
                typename = str(pyobj_type)
        super().__init__(typename, alias=alias, parent_type=parent_type)

    @property
    def is_pyobj_value_type(self):
        return True

    pyobj_type: Union[type, str]
    """The underlying Python object type."""


class TensorValueTypeBase(ValueType):
    """A value type refers to a type of some features associated with physical objects."""

    def __init__(self, typename: str, quantized: bool, alias: Optional[str] = None):
        super().__init__(typename, alias)
        self._quantized = quantized

    @property
    def is_tensor_value_type(self):
        return True

    @property
    def quantized(self) -> bool:
        """Whether the tensor is quantized."""
        return self._quantized

    def ndim(self) -> int:
        """Return the number of dimensions of the value type."""
        raise NotImplementedError()

    def size(self) -> int:
        """Return the total size of the value type."""
        raise NotImplementedError()

    def size_tuple(self) -> Tuple[int]:
        """Return the size of the value type as a tuple."""
        raise NotImplementedError()

    def tensor_dtype(self) -> torch.dtype:
        """Return the corresponding PyTorch tensor dtype."""
        raise NotImplementedError()

    def is_intrinsically_quantized(self) -> bool:
        """Return whether the value type can be quantized.
        In this project, quantizable means we can transform the value into a single integer scalar.
        Therefore, only BOOL-valued and INTxx-valued tensors are quantizable.

        Returns:
            True if the value type can be quantized.
        """
        raise NotImplementedError()

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'TensorValueTypeBase':
        """Create a value type from a pytorch tensor.

        Args:
            tensor: the tensor.

        Returns:
            the tensor value type.
        """
        if tensor.dtype in (torch.float32, torch.float64):
            return FLOAT32
        elif tensor.dtype in (torch.int32, torch.int64):
            return INT64
        elif tensor.dtype == torch.bool:
            return BOOL
        else:
            raise TypeError(f'Unsupported tensor type: {tensor.dtype}.')


class ScalarValueType(TensorValueTypeBase):
    TENSOR_DTYPE_MAPPING = {
        'int32': torch.int32,
        'int64': torch.int64,
        'float32': torch.float32,
        'float64': torch.float64,
        'bool': torch.int64,
    }

    TENSOR_DTYPE_QUANTIZED = {
        'int32': True,
        'int64': True,
        'float32': False,
        'float64': False,
        'bool': True
    }

    def __init__(self, typename):
        if typename not in type(self).TENSOR_DTYPE_MAPPING:
            raise TypeError(f'Unknown scalar value type: {typename}.')
        super().__init__(typename, quantized=type(self).TENSOR_DTYPE_QUANTIZED[typename])

    @property
    def is_scalar_value_type(self):
        return True

    def ndim(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def size_tuple(self) -> Tuple[int]:
        return tuple()

    def tensor_dtype(self) -> torch.dtype:
        return type(self).TENSOR_DTYPE_MAPPING[self.typename]

    INTRINSICALLY_QUANTIZED_DTYPES = ['bool', 'int32', 'int64']

    def is_intrinsically_quantized(self) -> bool:
        return self.typename in self.INTRINSICALLY_QUANTIZED_DTYPES

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ScalarValueType):
            return self.typename == o.typename
        return False

    def __hash__(self):
        return hash(self.typename)


BOOL = ScalarValueType('bool')
INT64 = ScalarValueType('int64')
FLOAT32 = ScalarValueType('float32')
STRING = PyObjValueType(str, 'string')


class VectorValueType(TensorValueTypeBase):
    def __init__(self, dtype: ScalarValueType, dim: int = 0, choices: int = 0, factors: int = 1, alias: Optional[str] = None):
        assert isinstance(dtype, ScalarValueType), 'Currently only support 1D vectors.'
        self.dtype = dtype
        self.dim = dim
        self.choices = choices
        self.factors = factors

        typename = self._gen_typename()
        quantized = choices > 0
        super().__init__(typename, quantized=quantized, alias=alias)

    dtype: ScalarValueType
    """The scalar value type of the vector."""

    dim: int
    """The dimension of the vector."""

    choices: int
    """The number of choices for vector values per factor."""

    factors: int
    """The number of factors of the vector."""

    @property
    def element_type(self) -> ScalarValueType:
        return self.dtype

    @property
    def is_vector_value_type(self):
        return True

    def _gen_typename(self) -> str:
        inner_string = f'{self.dtype.typename}'
        if self.dim > 0:
            inner_string += f', dim={self.dim}'
        if self.choices > 0:
            inner_string += f', choices={self.choices}'
        if self.factors > 1:
            inner_string += f', factors={self.factors}'
        return f'vector[{inner_string}]'

    def ndim(self) -> int:
        return 1

    def size(self) -> int:
        return self.dim * self.factors

    def size_tuple(self) -> Tuple[int]:
        return (self.size(), )

    def assignment_type(self) -> 'VectorValueType':
        if self._quantized:
            return VectorValueType(FLOAT32, self.choices, 0, self.factors)
        return self

    def tensor_dtype(self) -> torch.dtype:
        return self.dtype.tensor_dtype()

    def is_intrinsically_quantized(self) -> bool:
        return False

    def long_str(self):
        if self.alias is not None:
            return self.alias + f'({self.typename})'
        return self.typename


class NamedTensorValueType(TensorValueTypeBase):
    def __init__(self, typename: str, parent_type: TensorValueTypeBase):
        super().__init__(typename, parent_type._quantized)
        self.set_parent_type(parent_type)

    def ndim(self) -> int:
        return self.parent_type.ndim()

    def size(self) -> int:
        return self.parent_type.size()

    def size_tuple(self) -> Tuple[int]:
        return self.parent_type.size_tuple()

    def assignment_type(self) -> TensorValueTypeBase:
        return self

    def tensor_dtype(self) -> torch.dtype:
        return self.parent_type.tensor_dtype()

    def is_intrinsically_quantized(self) -> bool:
        return self.parent_type.is_intrinsically_quantized()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NamedTensorValueType):
            return self.typename == o.typename
        return False

    def __hash__(self):
        return hash(self.typename)


class SequenceType(TypeBase):
    """The basic sequence type. It has two forms: ListType and TupleType."""

    @property
    def is_sequence_type(self) -> bool:
        return True


class TupleType(SequenceType):
    def __init__(self, element_types: Sequence[TypeBase], alias: Optional[str] = None):
        super().__init__(f'Tuple[{", ".join(t.typename for t in element_types)}]', alias=alias)
        self.element_types = tuple(element_types)

    element_types: Tuple[TypeBase, ...]
    """The element types of the tuple."""

    @property
    def is_tuple_type(self) -> bool:
        return True


class UniformSequenceType(SequenceType):
    def __init__(self, typename: str, element_type: TypeBase, alias: Optional[str] = None):
        super().__init__(typename, alias=alias)
        self._element_type = element_type

    _element_type: TypeBase
    """The element type of the list."""

    @property
    def element_type(self) -> TypeBase:
        return self._element_type

    @property
    def is_uniform_sequence_type(self) -> bool:
        return True

    @property
    def is_object_type(self) -> bool:
        return self.element_type.is_object_type

    @property
    def is_value_type(self) -> bool:
        return self.element_type.is_value_type


class ListType(UniformSequenceType):
    def __init__(self, element_type: TypeBase, alias: Optional[str] = None):
        typename = f'List[{element_type.typename}]'
        super().__init__(typename, element_type, alias=alias)

    @property
    def is_list_type(self) -> bool:
        return True


class BatchedListType(UniformSequenceType):
    def __init__(self, element_type: TypeBase, index_dtypes: Sequence[ObjectType], alias: Optional[str] = None):
        typename = f'{element_type.typename}[{", ".join(t.typename for t in index_dtypes)}]'
        super().__init__(typename, element_type, alias=alias)
        self.index_dtypes = tuple(index_dtypes)

    element_type: TypeBase
    """The element type of the list."""

    index_dtypes: Tuple[ObjectType, ...]
    """The index types of the list."""

    def ndim(self) -> int:
        """The number of dimensions of the list."""
        return len(self.index_dtypes)

    @property
    def is_batched_list_type(self) -> bool:
        return True

    def iter_element_type(self) -> TypeBase:
        """Return the element type if we iterate over the list. Basically type(value[0])."""
        if len(self.index_dtypes) == 1:
            return self.element_type
        return BatchedListType(self.element_type, index_dtypes=self.index_dtypes[1:])


QINDEX = slice(None)
"""The "quantified index" object, used to indicate a quantified dimension in a tensor value type. Internally it is `slice(None)`."""


class _Placeholder(object):
    def __init__(self, name: Union[str, 'StateObjectReference', 'StateObjectList'], dtype: Union[ObjectType, ValueType, 'FunctionType']):
        self.name = name
        self.dtype = dtype

    name: Union[str, 'StateObjectReference', 'StateObjectList']
    """The name of the placeholder."""

    dtype: Union[ObjectType, ValueType, 'FunctionType']
    """The data type of the placeholder."""

    @property
    def typename(self):
        return self.dtype.typename

    def __str__(self):
        sep = get_format_context().type_varname_sep
        return f'{self.name}{sep}{self.dtype}'

    __repr__ = jacinle.repr_from_str

    def __eq__(self, other):
        return self.name == other.name and self.dtype == other.dtype

    def __hash__(self):
        return hash((self.name, self.dtype))


class Variable(_Placeholder):
    """The class representing a variable in a function."""

    name: str
    """The name of the variable."""

    dtype: Union[ObjectType, ValueType, 'FunctionType']
    """The data type of the variable."""

    scope: int = -1
    """An additional integer indicating the scope identifier of the variable."""

    quantifier_flag: Optional[str] = None
    """Additional quantifier flag for the variable. This flag will be set by the parser indicating the quantifier scope of this variable.
    Currently this is only used in PDSketch for "quantified variable" in regression rules. """

    def clone_with_scope(self, scope: int):
        new_var = Variable(self.name, self.dtype)
        new_var.scope = scope
        new_var.quantifier_flag = self.quantifier_flag
        return new_var

    def set_scope(self, scope: int):
        self.scope = scope

    def set_quantifier_flag(self, flag: str):
        self.quantifier_flag = flag

    def __str__(self):
        if self.scope > -1:
            if self.quantifier_flag is not None:
                return f'({self.quantifier_flag} {super().__str__()})@{self.scope}'
            return super().__str__() + f'@{self.scope}'
        if self.quantifier_flag is not None:
            return f'({self.quantifier_flag} {super().__str__()})'
        return super().__str__()


class ObjectConstant(_Placeholder):
    """The class representing a constant object in a DSL."""

    name: Union[str, 'StateObjectReference', 'ListValue']
    """The name of the constant."""

    dtype: Union[ObjectType, ValueType, SequenceType]
    """The data type of the constant."""

    def clone(self, dtype: Union[ObjectType, ValueType, SequenceType]):
        from mover._vendor.concepts.dsl.tensor_state import StateObjectReference, StateObjectList
        from mover._vendor.concepts.dsl.value import ListValue

        if isinstance(self.name, str):
            return ObjectConstant(self.name, dtype)
        if isinstance(self.name, StateObjectReference):
            return ObjectConstant(self.name.clone(dtype), dtype)
        if isinstance(self.name, StateObjectList):
            return ObjectConstant(self.name.clone(dtype), dtype)
        if isinstance(self.name, ListValue):
            raise DeprecationWarning('ListValue is deprecated as an ObjectConstant name.')


class UnnamedPlaceholder(_Placeholder):
    def __init__(self, dtype: Union[ObjectType, ValueType]):
        super().__init__('_', dtype)

    name: str
    """The name of the unnamed placeholder. Always equal to '_'."""

    dtype: Union[ObjectType, ValueType]
    """The data type of the unnamed placeholder."""

    def __str__(self):
        return '??' + str(self.dtype)

