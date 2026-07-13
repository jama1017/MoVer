#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : value_quantizers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/15/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import collections
import contextlib
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple, Iterable, Sequence, List, Mapping, Dict

import torch

from mover._vendor.concepts.dsl.dsl_types import TensorValueTypeBase, ScalarValueType, NamedTensorValueType, PyObjValueType
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.tensor_value import TensorValue, scalar
from mover._vendor.concepts.dsl.tensor_state import TensorState

if TYPE_CHECKING:
    from mover._vendor.concepts.dsl.executors.tensor_value_executor import TensorValueExecutorBase


__all__ = ['ValueQuantizer', 'PyObjectStore']


class ValueQuantizer(object):
    values: Dict[str, Union[List[Any], Dict[Any, int]]]
    """the value dictionary is a mapping from typename to a list of values. When we quantize a value `v`, we basically return the index of `v` in the list."""

    def __init__(self, executor: 'TensorValueExecutorBase'):
        """Initialize a value quantizer.

        Args:
            executor: the executor for the domain to use. We need the executor to access the underlying domain and
                the corresponding hash / equality functions.
        """

        self.executor = executor
        self.domain = executor.domain
        self.values = dict()

    def quantize(self, typename: str, value: Union[torch.Tensor, TensorValue]) -> int:
        """Quantize a single value. This API is used to quantize a single value, and it is the lowest level API in this class.
        Most of the time, you should use other higher-level APIs such as :meth:`quantize_tensor` and :meth:`quantize_value` instead.

        Args:
            typename: the typename of the value.
            value: the value to quantize.

        Returns:
            the quantized value, as a single integer.
        """

        if not isinstance(value, TensorValue):
            value = TensorValue(self.domain.types[typename], [], value)
        use_hash = self.executor.has_function_implementation(f'type::{typename}::hash')
        if typename not in self.values:
            self.values[typename] = dict() if use_hash else list()

        if use_hash:
            hash_value = self.executor.get_function_implementation(f'type::{typename}::hash')(value, wrap_rv=False)
            if hash_value not in self.values[typename]:
                self.values[typename][hash_value] = len(self.values[typename])
            return self.values[typename][hash_value]
        else:
            for i, v in enumerate(self.values[typename]):
                if bool(self.executor.get_function_implementation(f'type::{typename}::equal')(v, value, wrap_rv=False)):
                    return i
            self.values[typename].append(value)
            return len(self.values[typename]) - 1

    def quantize_tensor(self, dtype: NamedTensorValueType, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a PyTorch tensor. The main difference between this function and the :meth:`quantize` function is that
        this function will quantize all the elements in the tensor.

        Args:
            dtype: the dtype of the tensor.
            tensor: the tensor to quantize.

        Returns:
            the quantized tensor.
        """

        if dtype.parent_type.typename in ScalarValueType.INTRINSICALLY_QUANTIZED_DTYPES:
            raise RuntimeError('This branch should not be reached. Please report this bug to the developers.')
            # if torch.dtype in (torch.float16, torch.float32, torch.float64):
            #     return torch.round(tensor).to(torch.int64)

        tensor_flatten = tensor.reshape((-1,) + dtype.size_tuple())
        quantized_values = [self.quantize(dtype.typename, v) for v in tensor_flatten]
        quantized_tensor = torch.tensor(quantized_values, dtype=torch.int64, device=tensor_flatten.device)
        quantized_tensor = quantized_tensor.reshape(tensor.shape[:-dtype.ndim()])
        return quantized_tensor

    def quantize_value(self, value: TensorValue) -> TensorValue:
        """Quantize a single TensorValue object.

        Args:
            value: the value to quantize.

        Returns:
            the quantized value.
        """
        if value.dtype.is_intrinsically_quantized():
            return value.simple_quantize()
        assert isinstance(value.dtype, NamedTensorValueType)
        return TensorValue(value.dtype, value.batch_variables, self.quantize_tensor(value.dtype, value.tensor), value.batch_dims, quantized=True)

    def quantize_dict_list(self, continuous_values: Mapping[str, Sequence[Union[torch.Tensor, TensorValue]]]) -> Mapping[str, Sequence[TensorValue]]:
        """Quantize a dictionary of lists of values. The return is a dictionary that maps from the same keys to a list of quantized values.
        Note that the return type is a dictionary of lists of :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue`, not a dictionary of lists of integers.

        Args:
            continuous_values: the dictionary of lists of values to quantize.

        Returns:
            the quantized dictionary of lists of values.
        """
        output_dict = dict()
        for typename, values in continuous_values.items():
            output_dict[typename] = set()
            for v in values:
                output_dict[typename].add(self.quantize(typename, v))
            output_dict[typename] = [TensorValue(self.domain.types[typename], [], x, quantized=True) for x in output_dict[typename]]
        return output_dict

    def quantize_state(self, state: TensorState, includes=None, excludes=None) -> TensorState:
        """Quantize a TensorState object. Users can specify which variables to quantize by using the `includes` and `excludes` arguments.
        Meanwhile, this function will read the function definition in the domain to determine whether the feature is a state variable.
        If the feature is not a state variable, it will not be quantized.

        Args:
            state: the state to quantize.
            includes: the variables to include in the quantization. If this argument is not None, only the variables in this list will be quantized.
            excludes: the variables to exclude in the quantization. If this argument is not None, the variables in this list will not be quantized.

        Returns:
            the quantized state.
        """
        state = state.clone()
        for feature_name in state.features.all_feature_names:
            if (includes is not None and feature_name not in includes) or (excludes is not None and feature_name in excludes):
                rv = state.features[feature_name]
            else:
                function = self.domain.functions[feature_name]
                if hasattr(function, 'is_state_variable') and not function.is_state_variable:
                    rv = state.features[feature_name]
                else:
                    rv = self.quantize_value(state.features[feature_name])
            state.features[feature_name] = rv
        return state

    def unquantize(self, typename: str, value: int) -> TensorValue:
        """The lowest-level API to unquantize a single value. Most of the time, you should use other higher-level APIs.

        Args:
            typename: the typename of the value.
            value: the value to unquantize.

        Returns:
            the unquantized value.
        """
        return self.values[typename][value]

    def unquantize_tensor(self, dtype: NamedTensorValueType, tensor: torch.Tensor) -> torch.Tensor:
        """Unquantize a PyTorch tensor. The main difference between this function and the :meth:`unquantize` function is that
        this function will unquantize all the elements in the tensor.

        Args:
            dtype: the dtype of the tensor.
            tensor: the tensor to unquantize.

        Returns:
            the unquantized tensor.
        """

        if dtype.is_intrinsically_quantized():
            return tensor

        assert dtype.typename in self.values, f'Unknown typename: {dtype.typename}.'
        lookup_table = self.values[dtype.typename]
        output = [lookup_table[x].tensor for x in tensor.flatten().tolist()]
        output = torch.stack(output, dim=0).reshape(tensor.shape + dtype.size_tuple())
        return output

    def unquantize_value(self, value: TensorValue) -> TensorValue:
        """Unquantize a single TensorValue object.

        Args:
            value: the value to unquantize.

        Returns:
            the unquantized value.
        """
        dtype = value.dtype
        if isinstance(dtype, PyObjValueType):
            return value

        assert isinstance(dtype, TensorValueTypeBase)
        if dtype.is_intrinsically_quantized():
            return value
        else:
            assert isinstance(dtype, NamedTensorValueType)
            return TensorValue(dtype, value.batch_variables, self.unquantize_tensor(dtype, value.tensor), value.batch_dims, quantized=False)

    # def unquantize_csp(self, csp: ConstraintSatisfactionProblem) -> ConstraintSatisfactionProblem:
    #     """Unquantize a ConstraintSatisfactionProblem object. Essnetially this function will unquantize all the determined values in the CSP.

    #     Args:
    #         csp: the CSP to unquantize.

    #     Returns:
    #         the unquantized CSP.
    #     """
    #     def _cvt(arg):
    #         if isinstance(arg, DeterminedValue):
    #             if not arg.quantized:
    #                 return arg
    #             elif arg.dtype.quantized:
    #                 if arg.dtype == BOOL:
    #                     return DeterminedValue(BOOL, bool(arg.value), True)
    #                 return DeterminedValue(arg.dtype, int(arg.value), True)
    #             else:
    #                 assert isinstance(arg.dtype, NamedTensorValueType) and isinstance(arg.value, int)
    #                 return DeterminedValue(arg.dtype, self.unquantize(arg.dtype.typename, arg.value), False)
    #         else:
    #             return arg

    #     csp = csp.clone()
    #     for i, c in enumerate(csp.constraints):
    #         new_args = tuple(map(_cvt, c.arguments))
    #         new_rv = _cvt(c.rv)
    #         csp.constraints[i] = Constraint(c.function, new_args, new_rv, note=c.note)
    #     return csp

    @contextlib.contextmanager
    def checkpoint(self):
        """A context manager that can be used to checkpoint all the quantized values. This is useful when you performs a series of executions and want to restore the quantized values to save memory."""
        old_values = self.values.copy()
        yield
        self.values = old_values


class PyObjectStore(object):
    """A store for Python objects. This class is used to map Python objects to integers (so that they can be stored in a TensorValue object) and vice versa."""

    executor: 'TensorValueExecutorBase'
    """The executor that uses this object store."""

    domain: DSLDomainBase
    """The underlying domain for the PyObjectStore."""

    def __init__(self, executor: 'TensorValueExecutorBase'):
        """Initialize the PyObjectStore object.

        Args:
            executor: the executor that uses this object store.
        """
        self.executor = executor
        self.domain = executor.domain
        self.objects = collections.defaultdict(list)

    def add(self, typename: str, python_object: Any) -> int:
        """The lowest-API to add a Python object to the store. Most of the time, you should use other higher-level APIs such as :meth:`make_value` and :meth:`make_batched_value`.

        Args:
            typename: the typename of the Python object.
            python_object: the Python object to add.

        Returns:
            the index of the Python object in the store.
        """
        self.objects[typename].append(python_object)
        return len(self.objects[typename]) - 1

    def retrieve(self, typename: str, index: int) -> Any:
        """The lowest-API to retrieve a Python object from the store. Most of the time, you should use other higher-level APIs such as :meth:`retrieve_value`."""
        return self.objects[typename][index]

    def retrieve_value(self, value: TensorValue) -> Union[Any, List[Any]]:
        """Retrieve one or multiple Python object from the store. When the value is a batched value, this function will return a list of (or nested lists of) Python objects."""
        return _tensor2pyobj_list(self, value.dtype.typename, value.tensor)

    def make_value(self, typename: str, python_object: Any) -> TensorValue:
        """Make a TensorValue object from a single Python object.

        Args:
            typename: the typename of the Python object.
            python_object: the Python object to add.

        Returns:
            the TensorValue object.
        """
        index = self.add(typename, python_object)
        return scalar(index, self.domain.types[typename])

    def make_batched_value(self, typename: str, pyobj_list: List[Any], batch_variables: Optional[Sequence[str]] = None) -> TensorValue:
        """Make a TensorValue object from a list of (or nested lists of) Python objects.

        Args:
            typename: the typename of the Python object.
            pyobj_list: the list of (or nested lists of) Python objects.
            batch_variables: the batch variables of the TensorValue object.
        """

        sizes = _get_pyobj_list_size(pyobj_list)
        tensor = torch.zeros(sizes, dtype=torch.int64)

        for indices, pyobj in _iterate_pyobj_list(pyobj_list):
            tensor[indices] = self.add(typename, pyobj)

        return TensorValue(
            self.domain.types[typename],
            batch_variables if batch_variables is not None else len(sizes),
            tensor, batch_dims=0, quantized=True
        )

    @contextlib.contextmanager
    def checkpoint(self):
        """A context manager that can be used to checkpoint all the stored Python objects. This is useful when you performs a series of executions and want to restore the objects to save memory."""
        old_objects = self.objects.copy()
        yield
        self.objects = old_objects


def _tensor2pyobj_list(pyobj_store: PyObjectStore, typename: str, indices_tensor: torch.Tensor) -> Union[Any, List[Any]]:
    if indices_tensor.dim() == 0:
        return pyobj_store.retrieve(typename, indices_tensor.item())
    else:
        return [_tensor2pyobj_list(pyobj_store, typename, indices_tensor[i]) for i in range(indices_tensor.size(0))]


def _get_pyobj_list_size(pyobj_list: Union[Any, List[Any]]) -> Tuple[int, ...]:
    if type(pyobj_list) is list:
        assert len(pyobj_list) > 0
        return (len(pyobj_list), ) + _get_pyobj_list_size(pyobj_list[0])
    else:
        return tuple()


def _iterate_pyobj_list(pyobj_list: Union[Any, List[Any]], indices: Tuple[int, ...] = tuple()) -> Iterable[Tuple[Tuple[int, ...], Any]]:
    """Given a nested list of pyobjs, yield a tuple of (indices, pyobj) for each pyobj."""
    if type(pyobj_list) is list:
        for i, pyobj in enumerate(pyobj_list):
            yield from _iterate_pyobj_list(pyobj, indices + (i,))
    else:
        yield indices, pyobj_list


