#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensor_value.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/02/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures and simple shape-related operations for tensor values. Internally, we use torch.Tensor to represent tensor values."""

from dataclasses import dataclass
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple, Iterable, Iterator, Sequence, List

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from mover._vendor import torch_index
from jacinle.utils.printing import indent_text

from mover._vendor.concepts.dsl.dsl_types import TensorValueTypeBase, BOOL, INT64, FLOAT32, NamedTensorValueType, ScalarValueType, VectorValueType, PyObjValueType, BatchedListType, QINDEX
from mover._vendor.concepts.dsl.value import ValueBase

if TYPE_CHECKING:
    from mover._vendor.concepts.dsl.constraint import OptimisticValue


__all__ = [
    'TensorizedPyObjValues', 'MaskedTensorStorage', 'TensorValue',  # Basic tensorized representations.
    'from_tensor', 'vector_values', 'scalar',  # Tensor value creation.
    'concat_tvalues', 'index_tvalue', 'set_index_tvalue', 'expand_as_tvalue', 'expand_tvalue', 'expand_pyobj_value',
    'is_tvalue_simple_quantizable', 'simple_quantize_tvalue'
]


class TensorizedPyObjValues(object):
    """A value object with nested lists of Python objects. Note that this implies that we can not use list as the Python object type when shape is not specified."""

    def __init__(self, dtype: PyObjValueType, values: Union[Any, List[Any], List[List[Any]], List[List[List[Any]]]], shape: Optional[Tuple[int, ...]] = None):
        self.dtype = dtype
        if isinstance(values, np.ndarray):
            self.values = values
            self.shape = shape if shape is not None else self._infer_shape()
        elif shape is None:
            self.values = np.array(values, dtype=object)
            self.shape = shape if shape is not None else self._infer_shape()
        else:
            self.values = np.empty(shape, dtype=object)
            _recursive_assign(self.values, values)
            self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self) -> int:
        return int(np.prod(self.shape))

    def item(self):
        assert self.ndim == 0
        return self.values.item()

    def fast_index(self, arguments: Tuple[int, ...]):
        # assert len(arguments) == self.ndim
        # value = self.values
        # for i in arguments:
        #     value = value[i]
        # return value
        return self.values[arguments]

    def fast_set_index(self, arguments: Tuple[int, ...], value):
        self.values[arguments] = value

    def __getitem__(self, item):
        return TensorizedPyObjValues(self.dtype, self.values[item])

    def __setitem__(self, key, value: 'TensorizedPyObjValues'):
        self.values[key] = value.values

    def _infer_shape(self):
        """Infer the shape of the values by recursively checking the first element."""

        # if self.dtype.pyobj_type is list:
        #     raise TypeError('BatchedPyObjValue can not have list as the Python object type when shape is not specified.')

        # def _infer_shape_recursive(values):
        #     if type(values) is list:
        #         return (len(values), ) + _infer_shape_recursive(values[0])
        #     else:
        #         return tuple()

        # return _infer_shape_recursive(self.values)

        return self.values.shape

    @classmethod
    def make_empty(cls, dtype: PyObjValueType, shape: Iterable[int]):
        shape = tuple(shape)
        values = np.empty(shape, dtype=object)
        return cls(dtype, values, shape)

        # def _make_empty_recursive(s):
        #     if len(s) == 0:
        #         return None
        #     return [_make_empty_recursive(s[1:]) for _ in range(s[0])]
        # return cls(dtype, _make_empty_recursive(shape), shape)

    def clone(self, deep_copy: bool = False):
        """Recursively clone the values."""

        if deep_copy:
            return deepcopy(self)
        else:
            return type(self)(self.dtype, self.values.copy(), self.shape)

        # if deep_copy:
        #     new_values = deepcopy(self.values)
        # else:
        #     def _clone_recursive(values):
        #         if type(values) is list:
        #             return [_clone_recursive(v) for v in values]
        #         else:
        #             return values
        #     new_values = _clone_recursive(self.values)

        # return TensorizedPyObjValues(self.dtype, new_values, self.shape)

    def permute(self, indices: Tuple[int, ...]):
        """Permute the axes in the nested list of values."""
        return type(self)(self.dtype, np.transpose(self.values, indices), tuple(self.shape[i] for i in indices))

    def expand(self, target_size: Tuple[int, ...]):
        return type(self)(self.dtype, np.broadcast_to(self.values, target_size), target_size)

    def unsqueeze(self, dim: int):
        return type(self)(self.dtype, np.expand_dims(self.values, dim), (1, ) + self.shape)

    def __str__(self):
        if self.values.ndim == 0:
            return str(self.values.item())
        return f'TensorizedPyObjValues[{self.dtype}, shape={self.shape}]'

    def __repr__(self):
        return self.__str__()


def _recursive_assign(np_array: np.ndarray, values: Union[Any, List[Any], List[List[Any]], List[List[List[Any]]]]):
    if len(np_array.shape) == 0:
        np_array[()] = values
    elif len(np_array.shape) == 1:
        for i, v in enumerate(values):
            np_array[i] = v
    else:
        for i, v in enumerate(values):
            _recursive_assign(np_array[i], v)


@dataclass
class MaskedTensorStorage(object):
    """A storage for quantized tensors."""

    value: Union[torch.Tensor, TensorizedPyObjValues]
    """The unquantized tensor."""

    mask: Optional[torch.Tensor] = None
    """The mask of the value. If not None, entry = 1 if the value is valid, and 0 otherwise."""

    optimistic_values: Optional[torch.Tensor] = None
    """The optimistic values for the tensor. 0 for non-optimistic values."""

    quantized_values: Optional[torch.Tensor] = None
    """The quantized values for the tensor. -1 for non-quantized values."""


def _canonicalize_batch_variables(batch_variables: Union[Iterable[str], int]) -> Tuple[str, ...]:
    if isinstance(batch_variables, int):
        batch_variables = tuple([f'#{i}' for i in range(batch_variables)])
    assert all(isinstance(v, str) for v in batch_variables), 'All batch variables should be strings.'
    return tuple(batch_variables)


class TensorValue(ValueBase):
    """A value object with an internal :class:`torch.Tensor` storage."""

    def __init__(
        self,
        dtype: Union[TensorValueTypeBase, PyObjValueType, BatchedListType],
        batch_variables: Union[Iterable[str], int],
        tensor: Union[torch.Tensor, TensorizedPyObjValues, MaskedTensorStorage],
        batch_dims: int = 0,
        quantized: Optional[bool] = None,
        *,
        _check_tensor: bool = False,
        _mask_certified_flag: bool = True
    ):
        """Instantiate a Value object for storing intermediate computation results.

        The tensor is assumed to have the following layout: ``tensor[B1, B2, B3, ..., V1, V2, V3, ..., D1, D2, D3, ...]``.

        - The first `batch_dims` dimensions are "batch".
        - The next `len(batch_variables)` dimensions are "variables".
        - The next `dtype.ndim()` dimensions are data dimensions (e.g., images, vectors).
        - A special case is that `dtype.ndim()` can be zero (scalar).

        Args:
            dtype: The data type of the Value object.
            batch_variables: A sequence of variables that are processed in "batch." This typically corresponds to "quantified variables."
                It can also be a single integer, indicating the number of batched variables.
            tensor: The actual tensor
            batch_dims: The additional batch dimensions at the beginning. Defaults to 0.
            _check_tensor: internal flag, whether to run the tensor shape/type sanity check.
            _mask_certified_flag: internal flag, indicating whether self.tensor_mask is guaranteed to be the correct mask.
                This flag will be marked false when we do expand_as.
        """
        super().__init__(dtype)

        self.batch_variables = _canonicalize_batch_variables(batch_variables)
        self.batch_dims = batch_dims

        self.tensor_mask = None
        self.tensor_optimistic_values = None
        self.tensor_quantized_values = None
        self.quantized = quantized if quantized is not None else self._guess_quantized()

        if quantized is not None:
            if not hasattr(TensorValue, '_deprecated_quantized_warning'):
                TensorValue._deprecated_quantized_warning = True
                warnings.warn('TensorValue.quantized is deprecated. Please remove any usage of this flag. This flag will be removed by 2023/12/31', DeprecationWarning)

        if isinstance(tensor, MaskedTensorStorage):
            self.tensor = tensor.value
            self.tensor_mask = tensor.mask
            self.tensor_optimistic_values = tensor.optimistic_values
            self.tensor_quantized_values = tensor.quantized_values
        elif isinstance(tensor, TensorizedPyObjValues):
            assert isinstance(self.dtype, PyObjValueType)
            self.tensor = tensor
        elif isinstance(tensor, torch.Tensor):
            # TODO (Joy Hsu @ 2023/10/04): assert TensorValueTypeBase, PyObjValueType.
            self.tensor = tensor

        assert isinstance(self.tensor, (torch.Tensor, TensorizedPyObjValues))
        if _check_tensor:
            self._check_tensor('tensor')
            self._check_tensor('tensor_optimistic_values', is_index=True)
            self._check_tensor('tensor_quantized_values', is_index=True)
            self._check_tensor('tensor_mask', is_index=True)

        self._mask_certified_flag = _mask_certified_flag

        self._backward_function = None
        self._backward_args = tuple()
        self.tensor_grad = None

    dtype: Union[TensorValueTypeBase, PyObjValueType, BatchedListType]
    """The data type of the Value object."""

    batch_variables: Tuple[str, ...]
    """The list of batch variable names."""

    batch_dims: int
    """Additional batch dimensions at the beginning."""

    quantized: bool
    """Whether the values in self.tensor is quantized."""

    tensor: Union[torch.Tensor, TensorizedPyObjValues]
    """The internal tensor storage."""

    tensor_mask: Optional[torch.Tensor]
    """A mask of the tensor, indicating which elements are valid."""

    tensor_optimistic_values: Optional[torch.Tensor]
    """The optimistic values for the tensor. 0 for non-optimistic values."""

    tensor_quantized_values: Optional[torch.Tensor]
    """The quantized values for the tensor. -1 for non-quantized values."""

    def _guess_quantized(self) -> bool:
        return isinstance(self.dtype, TensorValueTypeBase) and self.dtype.is_intrinsically_quantized()

    @property
    def total_batch_dims(self):
        return self.batch_dims + len(self.batch_variables)

    @property
    def nr_variables(self):
        return len(self.batch_variables)

    @property
    def is_torch_tensor(self):
        return isinstance(self.tensor, torch.Tensor)

    @property
    def is_tensorized_pyobj(self):
        return isinstance(self.tensor, TensorizedPyObjValues)

    @property
    def variable_shape(self):
        return self.tensor.shape[self.batch_dims:self.total_batch_dims]

    def get_variable_size(self, variable_name_or_index: Union[str, int]) -> int:
        if isinstance(variable_name_or_index, str):
            variable_name_or_index = self.batch_variables.index(variable_name_or_index)
        if self.is_torch_tensor:
            return self.tensor.size(variable_name_or_index + self.batch_dims)
        return self.tensor.shape[variable_name_or_index + self.batch_dims]

    def to_masked_tensor_storage(self, clone_tensor: bool = False) -> MaskedTensorStorage:
        if clone_tensor:
            return MaskedTensorStorage(_maybe_clone_tensor(self.tensor), _maybe_clone_tensor(self.tensor_mask), _maybe_clone_tensor(self.tensor_optimistic_values), _maybe_clone_tensor(self.tensor_quantized_values))
        return MaskedTensorStorage(self.tensor, self.tensor_mask, self.tensor_optimistic_values, self.tensor_quantized_values)

    def to_masked_tensor_storage_with_padding(self, target_shape: Tuple[int, ...]):
        # TODO(Jiayuan Mao @ 2025/01/07): figure out the default value for tensor_mask.
        return MaskedTensorStorage(
            _maybe_pad_tensor(self.tensor, target_shape, self.dtype, self.batch_dims),
            _maybe_pad_tensor(self.tensor_mask, target_shape, self.dtype, self.batch_dims),
            _maybe_pad_tensor(self.tensor_optimistic_values, target_shape, self.dtype, self.batch_dims),
            _maybe_pad_tensor(self.tensor_quantized_values, target_shape, self.dtype, self.batch_dims)
        )

    def clone(self, clone_tensor=True, dtype: Optional[Union[TensorValueTypeBase, PyObjValueType, BatchedListType]] = None) -> 'TensorValue':
        storage = self.to_masked_tensor_storage(clone_tensor=clone_tensor)
        return type(self)(
            dtype if dtype is not None else self.dtype, self.batch_variables, storage, self.batch_dims,
            _check_tensor=False, _mask_certified_flag=self._mask_certified_flag
        )

    def pad(self, target_shape: Tuple[int, ...]) -> 'TensorValue':
        storage = self.to_masked_tensor_storage_with_padding(target_shape)
        return type(self)(
            self.dtype, self.batch_variables, storage, self.batch_dims,
            _check_tensor=False, _mask_certified_flag=self._mask_certified_flag
        )

    def rename_batch_variables(self, new_variables: Sequence[str], dtype: Optional[Union[TensorValueTypeBase, PyObjValueType, BatchedListType]] = None, force: bool = False, clone: bool = False):
        if not force:
            assert len(self.batch_variables) == len(new_variables)
        rv = self.clone() if clone else self
        rv.dtype = dtype if dtype is not None else self.dtype
        rv.batch_variables = _canonicalize_batch_variables(new_variables)
        return rv

    def init_tensor_optimistic_values(self):
        if self.tensor_optimistic_values is None:
            if self.is_torch_tensor:
                self.tensor_optimistic_values = torch.zeros(self.tensor.shape[:self.total_batch_dims], dtype=torch.int64, device=self.tensor.device)
            else:
                self.tensor_optimistic_values = torch.zeros(self.tensor.shape, dtype=torch.int64)

    # Simple creation.

    @classmethod
    def from_tensor(cls, value: torch.Tensor, dtype: Optional[Union[TensorValueTypeBase, BatchedListType]] = None, batch_variables: Optional[Iterable[str]] = None, batch_dims: int = 0) -> 'TensorValue':
        return from_tensor(value, dtype, batch_variables, batch_dims)

    @classmethod
    def from_tensorized_pyobj(cls, value: TensorizedPyObjValues, dtype: Optional[PyObjValueType] = None, batch_variables: Optional[Iterable[str]] = None, batch_dims: int = 0) -> 'TensorValue':
        return from_tensorized_pyobj(value, dtype, batch_variables, batch_dims)

    @classmethod
    def from_values(cls, *args: Any, dtype: Optional[TensorValueTypeBase] = None) -> 'TensorValue':
        return vector_values(*args, dtype=dtype)

    @classmethod
    def from_scalar(cls, value: Any, dtype: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None) -> 'TensorValue':
        return scalar(value, dtype=dtype)

    @classmethod
    def make_empty(cls, dtype: Union[TensorValueTypeBase, PyObjValueType, BatchedListType], batch_variables: Iterable[str] = tuple(), batch_sizes: Iterable[int] = tuple(), batch_dims: int = 0):
        if isinstance(dtype, TensorValueTypeBase):
            tensor = torch.zeros(tuple(batch_sizes) + dtype.size_tuple(), dtype=dtype.tensor_dtype())
        else:
            tensor = TensorizedPyObjValues.make_empty(dtype, batch_sizes)
        return cls(dtype, batch_variables, tensor, batch_dims=batch_dims)

    @classmethod
    def from_optimistic_value(cls, value: 'OptimisticValue') -> 'TensorValue':
        rv = cls.make_empty(value.dtype)
        rv.init_tensor_optimistic_values()
        rv.tensor_optimistic_values[...] = value.identifier
        return rv

    @classmethod
    def from_optimistic_value_int(cls, identifier: int, dtype: Union[TensorValueTypeBase, PyObjValueType]) -> 'TensorValue':
        rv = cls.make_empty(dtype)
        rv.init_tensor_optimistic_values()
        rv.tensor_optimistic_values[...] = identifier
        return rv

    # Simple tensor access.

    @property
    def is_scalar(self):
        return self.total_batch_dims == 0 and (isinstance(self.dtype, PyObjValueType) or self.dtype.ndim() == 0)

    def item(self) -> Union[torch.Tensor, Any, 'OptimisticValue']:
        from .constraint import is_optimistic_value, OptimisticValue

        assert self.is_scalar
        if self.tensor_optimistic_values is not None and is_optimistic_value(self.tensor_optimistic_values.item()):
            return OptimisticValue(self.dtype, int(self.tensor_optimistic_values.item()))
        if self.is_tensorized_pyobj:
            return self.tensor.values.item()
        return self.tensor.item()

    @property
    def is_single_elem(self):
        return self.total_batch_dims == 0

    def single_elem(self) -> Union[torch.Tensor, Any, 'OptimisticValue']:
        from .constraint import is_optimistic_value, OptimisticValue

        assert self.is_single_elem
        if self.tensor_optimistic_values is not None and is_optimistic_value(self.tensor_optimistic_values.item()):
            return OptimisticValue(self.dtype, int(self.tensor_optimistic_values.item()))
        if self.is_tensorized_pyobj:
            return self.tensor.values.item()
        return self.tensor

    def has_optimistic_value(self) -> bool:
        return self.tensor_optimistic_values is not None and self.tensor_optimistic_values.any()

    def is_single_optimistic_value(self) -> bool:
        from .constraint import is_optimistic_value

        return self.tensor_optimistic_values is not None and self.is_single_elem and is_optimistic_value(self.tensor_optimistic_values.item())

    def fast_index(self, arguments: Tuple[int, ...], wrap: bool = True) -> Union[torch.Tensor, Any, 'OptimisticValue']:
        from .constraint import is_optimistic_value, OptimisticValue

        arguments = tuple(arguments)
        assert len(arguments) == self.total_batch_dims
        if self.tensor_optimistic_values is not None and is_optimistic_value(self.tensor_optimistic_values[arguments]):
            return OptimisticValue(self.dtype, int(self.tensor_optimistic_values[arguments]))
        if self.is_tensorized_pyobj:
            if wrap:
                return TensorValue.from_scalar(self.tensor.fast_index(arguments), self.dtype)
            return self.tensor.fast_index(arguments)
        if wrap:
            return TensorValue.from_scalar(self.tensor[arguments], self.dtype)
        return self.tensor[arguments]

    def fast_set_index(self, arguments: Tuple[int, ...], value: Union[torch.Tensor, Any, 'OptimisticValue']):
        from .constraint import OptimisticValue
        if isinstance(value, OptimisticValue):
            value = value.identifier
            self.init_tensor_optimistic_values()
            self.tensor_optimistic_values[arguments] = value
            return

        if self.is_tensorized_pyobj:
            self.tensor.fast_set_index(arguments, value)
        else:
            self.tensor[arguments] = value

    def __bool__(self):
        if self.dtype == BOOL:
            return bool(self.item())
        else:
            raise TypeError('Cannot convert Value object {} into bool.'.format(self))

    # Value indexing.

    def index(self, key: 'ValueIndexType') -> 'TensorValue':
        if self.is_torch_tensor:
            return index_tvalue(self, key)
        else:
            return index_pyobj_value(self, key)

    def set_index(self, key: 'ValueIndexType', value: 'ValueSetIndexType') -> 'TensorValue':
        if self.is_torch_tensor:
            return set_index_tvalue(self, key, value)
        else:
            return set_index_pyobj_value(self, key, value)

    def iter_batched_indexing(self, dim: int = 0) -> Iterator['TensorValue']:
        """Iterate over one of the batch dimensions."""
        assert isinstance(self.dtype, BatchedListType)
        assert 0 <= dim < self.dtype.ndim()

        new_dtype = self.dtype.iter_element_type()
        index_target = self.batch_variables.index(f'@{dim}')
        indices = [slice(None) for _ in range(self.total_batch_dims)]
        for i in range(self.get_variable_size(index_target)):
            indices[dim] = i
            this_rv = self.index(tuple(indices))
            this_rv.dtype = new_dtype
            yield this_rv

    def __getitem__(self, item) -> 'TensorValue':
        return self.index(item)

    def __setitem__(self, key: 'ValueIndexType', value: 'ValueSetIndexType'):
        self.set_index(key, value)

    # Value shape manipulation.

    def expand(self, batch_variables: Iterable[str], batch_sizes: Iterable[int]) -> 'TensorValue':
        # NB(Jiayuan Mao @ 2023/08/16): expand_tvalue work for both pyobj and torch tensor.
        return expand_tvalue(self, batch_variables, batch_sizes)

    def expand_as(self, other: 'TensorValue') -> 'TensorValue':
        return expand_as_tvalue(self, other)

    # Simple value quantization.

    def is_simple_quantizable(self) -> bool:
        return is_tvalue_simple_quantizable(self)

    def simple_quantize(self) -> 'TensorValue':
        return simple_quantize_tvalue(self)

    # Stringify functions.

    STR_MAX_TENSOR_SIZE = 100

    @classmethod
    def set_print_options(cls, max_tensor_size=STR_MAX_TENSOR_SIZE):
        cls.STR_MAX_TENSOR_SIZE = max_tensor_size

    def format(self, content: bool = True, short: bool = False) -> str:
        from .constraint import is_optimistic_value, optimistic_value_id

        if short:
            if self.total_batch_dims == 0 and self.tensor_optimistic_values is not None and is_optimistic_value(self.tensor_optimistic_values.item()):
                return f'@{optimistic_value_id(self.tensor_optimistic_values.item())}'

            if self.is_scalar:
                if self.is_torch_tensor:
                    return str(self.item())
                else:
                    value = self.tensor.values.item()
                    return str(f'{type(value).__name__}@{hex(id(value))}')
            if content:
                if self.is_torch_tensor:
                    return str(self.tensor.tolist())
                else:
                    return str(self.tensor.values.tolist())
            else:
                axes = ', '.join(('B', ) * self.batch_dims + self.batch_variables)
                return f'{self.dtype}<{axes}>'

        if self.total_batch_dims == 0 and self.tensor_optimistic_values is not None and is_optimistic_value(self.tensor_optimistic_values.item()):
            return f'OptValue[{self.dtype}]{{@{optimistic_value_id(self.single_elem().identifier)}}}'

        axes = ', '.join(('B', ) * self.batch_dims + self.batch_variables)
        quantized_flag = ', quantized' if self.quantized else ''
        backward_flag = ', backward=' + str(self._backward_function) if self._backward_function is not None else ''
        tensor_content = ''

        if content:
            if self.tensor_grad is not None:
                tensor_content = str(self.tensor) + '\ngrad:\n' + str(self.tensor_grad)
            else:
                tensor_content = str(self.tensor)

            if self.tensor.numel() < type(self).STR_MAX_TENSOR_SIZE:
                if self.tensor.numel() < 10 and self.tensor.ndim <= 1:
                    tensor_content = '{' + tensor_content.replace('\n', ' ') + '}'
                else:
                    tensor_content = '{\n' + indent_text(tensor_content) + '\n}'

            return f'Value[{self.dtype}, axes=[{axes}], tdtype={self.tensor.dtype}, tdshape={tuple(self.tensor.shape)}{quantized_flag}{backward_flag}]{tensor_content}'

        return f'Value[{self.dtype}, axes=[{axes}], tdtype={self.tensor.dtype}, tdshape={tuple(self.tensor.shape)}{quantized_flag}{backward_flag}]'

    def short_str(self):
        return self.format(content=True, short=True)

    def __str__(self):
        return self.format(content=True)

    def __repr__(self):
        return self.__str__()

    # Internal dtype and shape sanity check.

    def _check_tensor(self, tensor_name: str, is_index: Optional[bool] = False):
        tensor = getattr(self, tensor_name)
        if tensor is None:
            return
        try:
            if isinstance(self.dtype, NamedTensorValueType):
                dtype = self.dtype.parent_type
            else:
                dtype = self.dtype

            if isinstance(dtype, ScalarValueType):
                if self.total_batch_dims == 0:
                    if dtype in (BOOL, INT64):
                        assert isinstance(tensor, (bool, int, torch.Tensor))
                        if isinstance(tensor, (bool, int)):
                            setattr(self, tensor_name, torch.tensor(tensor, dtype=torch.int64))
                        else:
                            if dtype == BOOL and tensor.dtype == torch.bool:
                                pass
                            elif dtype == BOOL and tensor.dtype == torch.int64:
                                setattr(self, tensor_name, tensor.to(torch.bool))
                            elif dtype == INT64 and tensor.dtype == torch.int64:
                                pass
                            else:
                                assert tensor.dtype == torch.float32
                    else:
                        raise TypeError('Unsupported dtype: {}.'.format(self.dtype))
                else:
                    assert torch.is_tensor(tensor)
                    assert tensor.ndim == len(self.batch_variables) + self.batch_dims
                    if dtype in (BOOL, INT64):
                        if dtype == BOOL and tensor.dtype == torch.bool:
                            pass
                        elif dtype == BOOL and tensor.dtype == torch.int64:
                            setattr(self, tensor_name, tensor.to(torch.bool))
                        elif dtype == INT64 and tensor.dtype == torch.int64:
                            pass
                        else:
                            assert tensor.dtype == torch.float32
                    else:
                        raise TypeError('Unsupported dtype: {}.'.format(self.dtype))
            elif isinstance(dtype, VectorValueType):
                assert torch.is_tensor(tensor)
                if is_index:
                    assert tensor.ndim == len(self.batch_variables) + self.batch_dims
                else:
                    assert tensor.ndim == len(self.batch_variables) + dtype.ndim() + self.batch_dims
            elif isinstance(dtype, PyObjValueType):
                assert isinstance(tensor, TensorizedPyObjValues)
                assert tensor.dtype == dtype
                assert tensor.ndim == len(self.batch_variables) + self.batch_dims
            else:
                raise NotImplementedError('Unsupported dtype: {}.'.format(self.dtype))
        except AssertionError as e:
            axes = ', '.join(self.batch_variables)
            raise ValueError(f'Tensor {tensor_name} shape/dtype mismatch for Value[{self.dtype}, axes=[{axes}], batch_dims={self.batch_dims}, tdtype={tensor.dtype}, tdshape={tuple(tensor.shape)}]') from e


    TRUE: 'TensorValue'
    FALSE: 'TensorValue'


def _maybe_clone_tensor(tensor: Optional[Union[torch.Tensor, TensorizedPyObjValues]]) -> Optional[torch.Tensor]:
    return tensor if tensor is None else tensor.clone()


def _maybe_pad_tensor(tensor: Optional[torch.Tensor], target_shape: Tuple[int, ...], dtype: Optional[TensorValueTypeBase], batch_dims: int, constant_value: Optional[Union[int, float]] = None) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return _pad_tensor(tensor, target_shape, dtype=dtype, batch_dims=batch_dims, constant_value=constant_value)


def from_tensor(value: torch.Tensor, dtype: Optional[Union[TensorValueTypeBase, BatchedListType]] = None, batch_variables: Optional[List[str]] = None, batch_dims: int = 0) -> TensorValue:
    if dtype is None:
        dtype = TensorValueTypeBase.from_tensor(value)
    if batch_variables is None:
        batch_variables = list() if not isinstance(dtype, BatchedListType) else [f'@{i}' for i in range(dtype.ndim())]
    return TensorValue(dtype, batch_variables, value, batch_dims=batch_dims)


def from_tensorized_pyobj(value: TensorizedPyObjValues, dtype: Optional[PyObjValueType] = None, batch_variables: Optional[List[str]] = None, batch_dims: int = 0) -> TensorValue:
    if dtype is None:
        dtype = PyObjValueType(value.dtype.pyobj_type, value.dtype.typename)
    if batch_variables is None:
        batch_variables = len(value.shape)
    return TensorValue(dtype, batch_variables, value, batch_dims=batch_dims)


def vector_values(*args: Any, dtype: Optional[TensorValueTypeBase] = None) -> TensorValue:
    if dtype is None:
        dtype = VectorValueType(FLOAT32, len(args), 0)
    assert isinstance(dtype, TensorValueTypeBase)
    tensor_dtype = dtype.tensor_dtype()
    return TensorValue(dtype, [], torch.tensor(args, dtype=tensor_dtype))


def scalar(value: Union[TensorValue, torch.Tensor, bool, float, int, str, Any], dtype: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None) -> TensorValue:
    if isinstance(value, TensorValue):
        return value

    if dtype is None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        dtype = TensorValueTypeBase.from_tensor(value)

    if torch.is_tensor(value):
        return TensorValue(dtype, [], value)

    if isinstance(dtype, PyObjValueType):
        if isinstance(value, int):
            raise RuntimeError('Invalid branch: report to developers.')
        else:
            return TensorValue(dtype, [], TensorizedPyObjValues(dtype, value, shape=()))

    assert isinstance(dtype, TensorValueTypeBase)
    tensor_dtype = dtype.tensor_dtype()
    return TensorValue(dtype, [], torch.tensor(value, dtype=tensor_dtype))


def t(value: torch.Tensor, dtype: Optional[Union[TensorValueTypeBase, BatchedListType]] = None, batch_variables: Optional[List[str]] = None) -> TensorValue:
    """Alias for :func:`from_tensor`."""
    return from_tensor(value, dtype, batch_variables)


def v(*args: Any, dtype: Optional[TensorValueTypeBase] = None) -> TensorValue:
    """Alias for :func:`values`."""
    return vector_values(*args, dtype=dtype)


def s(value: Any, dtype: Optional[Union[TensorValueTypeBase, PyObjValueType]] = None) -> TensorValue:
    """Alias for :func:`scalar`."""
    return scalar(value, dtype)


TensorValue.TRUE = TensorValue.from_scalar(True, BOOL)
TensorValue.FALSE = TensorValue.from_scalar(False, BOOL)



def concat_tvalues(*args: TensorValue):  # will produce a Value with batch_dims == 1, but the input can be either 0-batch or 1-batch.
    assert len(args) > 0
    include_tensor_optimistic_values = any([v.tensor_optimistic_values is not None for v in args])
    include_tensor_quantized_values = any([v.tensor_quantized_values is not None for v in args])

    # Sanity check.
    for value in args[1:]:
        assert value.is_torch_tensor
        assert value.dtype == args[0].dtype
        assert value.batch_variables == args[0].batch_variables
        if include_tensor_optimistic_values:
            pass  # we have default behavior for None.
        if include_tensor_quantized_values:
            assert value.tensor_quantized_values is not None

    device = args[0].tensor.device

    # Collect all tensors.
    all_tensor = [v.tensor for v in args]
    all_tensor_mask = [v.tensor_mask for v in args]
    all_tensor_optimistic_values = [v.tensor_optimistic_values for v in args]
    all_tensor_quantized_values = [v.tensor_quantized_values for v in args]

    target_shape = tuple([
        max([v.get_variable_size(i) for v in args])
        for i in range(args[0].nr_variables)
    ])
    for i in range(len(args)):
        tensor, tensor_mask, tensor_optimistic_values, tensor_quantized_values = all_tensor[i], all_tensor_mask[i], all_tensor_optimistic_values[i], all_tensor_quantized_values[i]
        all_tensor[i] = _pad_tensor(tensor, target_shape, args[i].dtype, args[i].batch_dims)

        if tensor_mask is None:
            tensor_mask = torch.ones(target_shape, dtype=torch.bool, device=device)
        else:
            tensor_mask = _pad_tensor(tensor_mask, target_shape, args[i].dtype, args[i].batch_dims)
        all_tensor_mask[i] = tensor_mask

        if include_tensor_optimistic_values:
            if tensor_optimistic_values is None:
                tensor_optimistic_values = torch.zeros(target_shape, dtype=torch.int64, device=device)
            else:
                tensor_optimistic_values = _pad_tensor(tensor_optimistic_values, target_shape, args[i].dtype, args[i].batch_dims)
            all_tensor_optimistic_values[i] = tensor_optimistic_values

        if include_tensor_quantized_values:
            tensor_quantized_values = _pad_tensor(tensor_quantized_values, target_shape, args[i].dtype, args[i].batch_dims)
            all_tensor_quantized_values[i] = tensor_quantized_values

        if args[0].batch_dims == 0:
            all_tensor[i] = all_tensor[i].unsqueeze(0)
            all_tensor_mask[i] = all_tensor_mask[i].unsqueeze(0)
            all_tensor_optimistic_values[i] = all_tensor_optimistic_values[i].unsqueeze(0) if all_tensor_optimistic_values[i] is not None else None
            all_tensor_quantized_values[i] = all_tensor_quantized_values[i].unsqueeze(0) if all_tensor_quantized_values[i] is not None else None
        else:
            assert args[0].batch_dims == 1

    masked_tensor_storage = MaskedTensorStorage(
        torch.cat(all_tensor, dim=0),
        torch.cat(all_tensor_mask, dim=0),
        torch.cat(all_tensor_optimistic_values, dim=0) if include_tensor_optimistic_values else None,
        torch.cat(all_tensor_quantized_values, dim=0) if include_tensor_quantized_values else None
    )
    return TensorValue(args[0].dtype, args[0].batch_variables, masked_tensor_storage, batch_dims=1)


def _pad_tensor(tensor: torch.Tensor, target_shape: Iterable[int], dtype: TensorValueTypeBase, batch_dims: int, constant_value: float = 0.0):
    target_shape = tuple(target_shape)
    paddings = list()
    for size, max_size in zip(tensor.size()[batch_dims:], target_shape):
        paddings.extend((max_size - size, 0))
    if tensor.dim() - batch_dims == len(target_shape):
        pass
    elif tensor.dim() - batch_dims == len(target_shape) + dtype.ndim():
        paddings.extend([0 for _ in range(dtype.ndim() * 2)])
    else:
        raise ValueError('Shape error during tensor padding.')
    paddings.reverse()  # no need to add batch_dims.
    return F.pad(tensor, paddings, "constant", constant_value)


ValueIndexElementType = Union[int, slice, List[int], torch.Tensor]
ValueIndexType = Optional[Union[ValueIndexElementType, Tuple[ValueIndexElementType, ...]]]
ValueSetIndexType = Union[bool, int, float, np.ndarray, torch.Tensor, TensorValue]


def index_tvalue(value: TensorValue, indices: ValueIndexType) -> TensorValue:
    assert value.is_torch_tensor, 'Use index_pyobj_value for non-tensor values.'
    if isinstance(indices, (int, list, torch.Tensor)) or indices == QINDEX:
        indices = (indices, )
    assert indices is None or isinstance(indices, tuple)

    if indices is None:
        return value.clone()

    if len(indices) < value.nr_variables:
        indices = indices + (slice(None), ) * (value.nr_variables - len(indices))
    assert len(indices) == value.nr_variables
    batch_variables = list()
    for i, idx in enumerate(indices):
        if idx == QINDEX:
            batch_variables.append(value.batch_variables[i])

    if value.batch_dims == 0:
        return type(value)(
            value.dtype, batch_variables, MaskedTensorStorage(
                value.tensor[indices] if value.tensor is not None else None,
                value.tensor_mask[indices] if value.tensor_mask is not None else None,
                value.tensor_optimistic_values[indices] if value.tensor_optimistic_values is not None else None,
                value.tensor_quantized_values[indices] if value.tensor_quantized_values is not None else None
            ), batch_dims=value.batch_dims, _mask_certified_flag=value._mask_certified_flag
        )
    elif value.batch_dims == 1:
        indices = (torch_index.batch,) + indices
        return type(value)(
            value.dtype, batch_variables, MaskedTensorStorage(
                torch_index.bvindex(value.tensor)[indices][0] if value.tensor is not None else None,
                torch_index.bvindex(value.tensor_mask)[indices][0] if value.tensor_mask is not None else None,
                torch_index.bvindex(value.tensor_optimistic_values)[indices][0] if value.tensor_optimistic_values is not None else None,
                torch_index.bvindex(value.tensor_quantized_values)[indices][0] if value.tensor_quantized_values is not None else None
            ), batch_dims=value.batch_dims, _mask_certified_flag=value._mask_certified_flag
        )
    else:
        raise NotImplementedError('Unsupported batched dims: {}.'.format(value.batch_dims))


def set_index_tvalue(lhs: TensorValue, indices: ValueIndexType, rhs: ValueSetIndexType) -> TensorValue:
    if isinstance(indices, (int, torch.Tensor)) or indices == QINDEX:
        indices = (indices, )
    if isinstance(indices, list):
        raise RuntimeError('Undefined behavior for list indices. Use either tuple([...], ) for single-dimension indexing or tuple(..., ...) for multi-dimension indexing.')
    assert indices is None or isinstance(indices, tuple)

    optimistic_values, quantized_values = None, None
    if isinstance(rhs, (bool, int, float)):
        rhs = torch.tensor(rhs, dtype=lhs.tensor.dtype, device=lhs.tensor.device)
    elif isinstance(rhs, np.ndarray):
        rhs = torch.from_numpy(rhs).to(lhs.tensor.dtype).to(lhs.tensor.device)
    if isinstance(rhs, TensorValue):
        optimistic_values, quantized_values = rhs.tensor_optimistic_values, rhs.tensor_quantized_values
        rhs = rhs.tensor

    if lhs.batch_dims == 0:
        if indices is None or len(indices) == 0:
            lhs.tensor = rhs
            lhs.tensor_optimistic_values = optimistic_values
            lhs.tensor_quantized_values = quantized_values
        else:
            # We have to this cloning. Consider the following case:
            # v[0] = v[1]
            indices = indices[0] if len(indices) == 1 else indices
            if not isinstance(indices, tuple) or not any((isinstance(x, (tuple, list)) and len(x) == 0) for x in indices):
                lhs.tensor = lhs.tensor.clone()
                lhs.tensor[indices] = rhs
                if optimistic_values is not None:
                    lhs.init_tensor_optimistic_values()
                    lhs.tensor_optimistic_values = lhs.tensor_optimistic_values.clone()
                    lhs.tensor_optimistic_values[indices] = optimistic_values
                if lhs.tensor_quantized_values is not None:
                    assert quantized_values is not None
                    lhs.tensor_quantized_values = lhs.tensor_quantized_values.clone()
                    lhs.tensor_quantized_values[indices] = quantized_values
    else:
        raise NotImplementedError('Unsupported batched dims: {}.'.format(lhs.batch_dims))

    return lhs


def index_pyobj_value(value: TensorValue, indices: ValueIndexType) -> TensorValue:
    if isinstance(indices, (int, slice)):
        indices = (indices, )

    assert indices is None or isinstance(indices, tuple)

    if indices is None:
        return value.clone()

    assert len(indices) == value.nr_variables
    batch_variables = list()
    sizes = list()
    for i, idx in enumerate(indices):
        if idx == QINDEX:
            batch_variables.append(value.batch_variables[i])
            sizes.append(value.get_variable_size(i))

    if value.batch_dims == 0:
        values = _recursive_index_pyobj_value(value.tensor.values, indices)
        values = TensorizedPyObjValues(value.dtype, values, sizes)
        return type(value)(value.dtype, batch_variables, values, batch_dims=value.batch_dims)
    elif value.batch_dims == 1:
        values = [_recursive_index_pyobj_value(value.tensor.values[i], [indexer[i] for indexer in indices]) for i in range(len(value.tensor.values))]
        assert isinstance(values, np.ndarray)
        values = TensorizedPyObjValues(value.dtype, values, values.shape)
        return type(value)(value.dtype, batch_variables, values, batch_dims=value.batch_dims)
    else:
        raise NotImplementedError('Unsupported batched dims: {}.'.format(value.batch_dims))


def _recursive_index_pyobj_value(np_array, indices):
    return np_array[indices]
    # if len(indices) == 1:
    #     if isinstance(indices[0], int):
    #         return nested_list[indices[0]]
    #     elif indices[0] == QINDEX:
    #         return nested_list
    #     else:
    #         raise ValueError(f'Invalid index: {indices[0]}.')

    # if isinstance(indices[0], int):
    #     return _recursive_index_pyobj_value(nested_list[indices[0]], indices[1:])
    # elif indices[0] == QINDEX:
    #     return [_recursive_index_pyobj_value(nested_list[i], indices[1:]) for i in range(len(nested_list))]
    # else:
    #     raise ValueError(f'Invalid index: {indices[0]}.')


def set_index_pyobj_value(lhs: TensorValue, indices: ValueIndexType, rhs: ValueSetIndexType) -> TensorValue:
    if isinstance(indices, (int, slice)):
        indices = (indices, )
    assert indices is None or isinstance(indices, tuple)

    is_scalar_rhs = not isinstance(rhs, TensorValue)

    if lhs.batch_dims == 0:
        _recursive_set_index_pyobj_value(lhs.tensor.values, indices, rhs, is_scalar_rhs)
    else:
        raise NotImplementedError('Unsupported batched dims: {}.'.format(lhs.batch_dims))
    return lhs


def _recursive_set_index_pyobj_value(np_array, indices: Tuple[Union[int, slice], ...], rhs, is_scalar_rhs):
    if isinstance(rhs, TensorValue):
        rhs = rhs.tensor.values
    np_array[indices] = rhs

    # if len(indices) == 1:
    #     if isinstance(indices[0], int):
    #         nested_list[indices[0]] = rhs
    #     elif indices[0] == QINDEX:
    #         if is_scalar_rhs:
    #             for i in range(len(nested_list)):
    #                 nested_list[i] = rhs
    #         else:
    #             assert len(nested_list) == len(rhs.tensor.values)
    #             for i in range(len(nested_list)):
    #                 nested_list[i] = rhs[i]
    #     else:
    #         raise ValueError(f'Invalid index: {indices[0]}.')
    # else:
    #     if isinstance(indices[0], int):
    #         _recursive_set_index_pyobj_value(nested_list[indices[0]], indices[1:], rhs, is_scalar_rhs)
    #     elif indices[0] == QINDEX:
    #         for i in range(len(nested_list)):
    #             _recursive_set_index_pyobj_value(nested_list[i], indices[1:], rhs[i] if not is_scalar_rhs else rhs, is_scalar_rhs)
    #     else:
    #         raise ValueError(f'Invalid index: {indices[0]}.')


def expand_as_tvalue(value: TensorValue, other: TensorValue) -> TensorValue:
    return expand_tvalue(value, other.batch_variables, other.tensor.size()[other.batch_dims: other.batch_dims + len(other.batch_variables)])


def expand_tvalue(value: TensorValue, batch_variables: Iterable[str], batch_sizes: Iterable[int]) -> TensorValue:
    batch_variables = tuple(batch_variables)
    batch_sizes = tuple(batch_sizes)

    data = value.tensor
    masked_tensor_storage_kwargs = {
        'mask': value.tensor_mask,
        'optimistic_values': value.tensor_optimistic_values,
        'quantized_values': value.tensor_quantized_values,
    }

    current_batch_variables = list(value.batch_variables)
    for var in batch_variables:
        if var not in current_batch_variables:
            data = data.unsqueeze(value.batch_dims)
            for k, v in masked_tensor_storage_kwargs.items():
                if v is not None:
                    masked_tensor_storage_kwargs[k] = v.unsqueeze(value.batch_dims)
            current_batch_variables.insert(0, var)

    indices = list()
    sizes = list()

    # process the first "batch" dims.
    for i in range(value.batch_dims):
        indices.append(i)
        sizes.append(data.size(i))

    # process the next "variables" dims.
    for var, size in zip(batch_variables, batch_sizes):
        indices.append(current_batch_variables.index(var) + value.batch_dims)
        sizes.append(size)

    for k, v in masked_tensor_storage_kwargs.items():
        if v is not None:
            # corner case for "scalar" storage.
            if v.ndim > 0:
                v = v.permute(indices)
            masked_tensor_storage_kwargs[k] = v.expand(sizes)

    # process the last "data" dims.
    if isinstance(value.dtype, TensorValueTypeBase):
        for i in range(value.dtype.ndim()):
            indices.append(value.batch_dims + len(batch_variables))
            sizes.append(data.size(value.batch_dims + len(batch_variables) + i))

    # corner case for "scalar" storage.
    if len(indices) > 0:
        data = data.permute(indices)
    data = data.expand(sizes)

    rv = TensorValue(
        value.dtype, batch_variables,
        MaskedTensorStorage(data, **masked_tensor_storage_kwargs),
        batch_dims=value.batch_dims, _mask_certified_flag=False
    )

    return rv


def expand_pyobj_value(value: TensorValue, batch_variables: Iterable[str], batch_sizes: Iterable[int]) -> TensorValue:
    raise NotImplementedError('expand_pyobj_value is not implemented yet.')


def is_tvalue_simple_quantizable(value: TensorValue) -> bool:
    """A value is simple quantizable if it is either quantized, or it has tensor_quantized_values, or it has intrinsically quantized dtypes."""
    return (
        value.tensor_quantized_values is not None or
        value.dtype.is_intrinsically_quantized()
    )


def simple_quantize_tvalue(value: TensorValue) -> TensorValue:
    """Quantize a value into a quantized value using the simple rules.

    The simple rules are:
        1. If the value is already quantized, return the value itself.
        2. If the value has tensor_quantized_values, return the tensor_indices.
        3. If the value has intrinsically quantized dtypes, quantize the value.
            - For BOOL, quantize the value with (value > 0.5).to(torch.int64).
            - For INT64, quantize the value with value.to(torch.int64).

    Args:
        value: the value to be quantized.

    Returns:
        the quantized value.
    """
    assert value.dtype.is_intrinsically_quantized() or value.tensor_quantized_values is not None

    if value.tensor_quantized_values is not None:
        rv = value.tensor_quantized_values
    elif value.dtype == BOOL:
        rv = (value.tensor > 0.5).to(torch.int64)
    elif value.dtype == INT64:
        rv = value.tensor.to(torch.int64)
    elif isinstance(value.dtype, NamedTensorValueType) and value.dtype.parent_type == BOOL:
        rv = (value.tensor > 0.5).to(torch.int64)
    elif isinstance(value.dtype, NamedTensorValueType) and value.dtype.parent_type == INT64:
        rv = value.tensor.to(torch.int64)
    else:
        raise TypeError('Unable to quantize value. Need either tensor_indices, or intrinsically quantized dtypes: {}.'.format(str(value)))

    rv = MaskedTensorStorage(rv, value.tensor_mask)
    return type(value)(value.dtype, value.batch_variables, rv, value.batch_dims, _mask_certified_flag=value._mask_certified_flag)


