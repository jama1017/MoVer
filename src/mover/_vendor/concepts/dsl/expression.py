#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : expression.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/19/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for expressions in a DSL.

All classes extend the basic :class:`Expression` class. They can be categorized into the following groups:

- :class:`ObjectOrValueOutputExpression` is the base class for expressions that output objects or values. This is only used for type hinting.
- :class:`ObjectOutputExpression` and :class:`ValueOutputExpression` are the expressions that output objects or values.
- :class:`VariableExpression` which is the expression that refers to a variable.
- :class:`VariableAssignmentExpression` which assigns a value to a variable.

Under the :class:`ValueOutputExpression` category, there are a few sub-categories:

- :class:`NullExpression` which is the expression that outputs a null value.
- :class:`ConstantExpression` which is the expression that outputs a constant value.
- :class:`ListCreationExpression` which is the expression that creates a list.
- :class:`ListExpansionExpression` which is the expression that expands a list into a sequence of values (e.g., plan steps).
- :class:`FunctionApplicationExpression` which represents the application of a function.
- :class:`ListFunctionApplicationExpression` which represents the application of a function to a list of arguments.
- :class:`BoolExpression` which represents Boolean operations (and, or, not).
- :class:`QuantificationExpression` which represents quantification (forall, exists).
- :class:`GeneralizedQuantificationExpression` which represents generalized quantification (iota, all, counting quantifiers).
- :class:`FindOneExpression` which represents the find-one quantification.
- :class:`FindAllExpression` which represents the find-all quantification.
- :class:`PredicateEqualExpression` which represents the equality test between a state variable and a value.
- :class:`ObjectCompareExpression` which represents the comparison between two objects.
- :class:`ValueCompareExpression` which represents the comparison between two values.
- :class:`ConditionExpression` which represents the ternary conditional expression.
- :class:`ConditionalSelectExpression` which represents the conditional selection for some computed value.
- :class:`DeicticSelectExpression` which represents the deictic selection for some computed value (i.e., forall quantifiers).
- :class:`BatchedExpression` which is the expression that assigns values to state variables with batched expressions.

Under the :class:`ObjectOutputExpression` category, there are a few sub-categories:

- :class:`ObjectConstantExpression` which is the expression that outputs a constant object.

Under the :class:`VariableAssignmentExpression` category, there are a few sub-categories:

- :class:`AssignExpression` which is the expression that assigns a value to a state variable.
- :class:`ConditionalAssignExpression` which is the expression that assigns a value to a state variable conditionally.
- :class:`DeicticAssignExpression` which is the expression that assigns values to state variables with deictic expressions (i.e., forall quantifiers).

The most important classes are: :class:`VariableExpression`, :class:`ObjectConstantExpression`, :class:`ConstantExpression`, and :class:`FunctionApplicationExpression`.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Iterable, Tuple, Sequence, List, Dict, Callable
from functools import lru_cache

import torch
import jacinle

from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.defaults import wrap_custom_as_default, gen_get_default

from mover._vendor.concepts.dsl.dsl_types import FormatContext, get_format_context
from mover._vendor.concepts.dsl.dsl_types import TypeBase, ObjectType, ValueType, SequenceType, ListType, BatchedListType, AutoType, TensorValueTypeBase, PyObjValueType, BOOL, FLOAT32, INT64, STRING, ObjectConstant, Variable
from mover._vendor.concepts.dsl.dsl_functions import FunctionType, Function, FunctionArgumentUnset, AnonymousFunctionArgumentGenerator
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.value import ValueBase, Value, ListValue
from mover._vendor.concepts.dsl.tensor_value import TensorValue
from mover._vendor.concepts.dsl.tensor_state import StateObjectReference, StateObjectList

try:
    from typing import TypeGuard
except ImportError:
    class _DummyTypeGuard:
        def __getitem__(self, item):
            return bool
    TypeGuard = _DummyTypeGuard()


__all__ = [
    'DSLExpressionError', 'Expression', 'ExpressionDefinitionContext', 'get_expression_definition_context',
    'ObjectOutputExpression', 'ValueOutputExpression', 'NullExpression', 'VariableExpression', 'VariableAssignmentExpression',
    'ObjectOrValueOutputExpression', 'VariableOrObjectOutputExpression', 'VariableOrValueOutputExpression',
    'ObjectConstantExpression', 'ConstantExpression', 'ListCreationExpression',
    'FunctionApplicationError', 'FunctionApplicationExpression', 'ListFunctionApplicationExpression', 'ListExpansionExpression',
    'ConditionalSelectExpression', 'DeicticSelectExpression',
    'BoolOpType', 'BoolExpression', 'AndExpression', 'OrExpression', 'NotExpression', 'XorExpression', 'ImpliesExpression',
    'QuantificationOpType', 'QuantificationExpression', 'GeneralizedQuantificationExpression', 'ForallExpression', 'ExistsExpression', 'BatchedExpression', 'FindOneExpression', 'FindAllExpression',
    'CompareOpType', 'ObjectCompareExpression', 'ValueCompareExpression', 'ConditionExpression',
    'PredicateEqualExpression', 'AssignExpression', 'ConditionalAssignExpression', 'DeicticAssignExpression',
    'cvt_expression', 'cvt_expression_list', 'get_type', 'get_types',
    'is_null_expression', 'is_object_output_expression', 'is_value_output_expression', 'is_variable_assignment_expression',
    'is_constant_bool_expr', 'is_and_expr', 'is_or_expr', 'is_not_expr', 'is_xor_expr', 'is_implies_expr', 'is_forall_expr', 'is_exists_expr',
]


class DSLExpressionError(Exception):
    pass


class Expression(ABC):
    """Expression is an abstract class for all expressions in the DSL.
    An important note about Expression is that the class itself does not contain any "implementation."

    For example, the expression `and(x, y, z)` does not contain any information about how to compute
    the conjunction (e.g., taking the product of the three values).
    The actual implementation of the expression will be provided by the `Executor` classes.
    """

    @property
    @abstractmethod
    def return_type(self) -> Optional[Union[ObjectType, ValueType, FunctionType, SequenceType]]:
        raise NotImplementedError()

    def check_arguments(self):
        ctx = get_expression_definition_context()
        if ctx is None or ctx.check_arguments:
            self._check_arguments()

    def _check_arguments(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str

    @lru_cache(maxsize=10)
    def cached_string(self, max_length: Optional[int] = None):
        if max_length is None:
            return str(self)
        else:
            with FormatContext(expr_max_length=max_length).as_default():
                return str(self)

    @property
    def is_null_expression(self) -> bool:
        return is_null_expression(self)

    @property
    def is_object_output_expression(self) -> bool:
        return is_object_output_expression(self)

    @property
    def is_value_output_expression(self) -> bool:
        return is_value_output_expression(self)

    @property
    def is_variable_assignment_expression(self) -> bool:
        return is_variable_assignment_expression(self)

    @property
    def is_constant_bool_expr(self) -> bool:
        return is_constant_bool_expr(self)

    @property
    def is_and_expr(self) -> bool:
        return is_and_expr(self)

    @property
    def is_or_expr(self) -> bool:
        return is_or_expr(self)

    @property
    def is_not_expr(self) -> bool:
        return is_not_expr(self)

    @property
    def is_xor_expr(self) -> bool:
        return is_xor_expr(self)

    @property
    def is_implies_expr(self) -> bool:
        return is_implies_expr(self)

    @property
    def is_forall_expr(self) -> bool:
        return is_forall_expr(self)

    @property
    def is_exists_expr(self) -> bool:
        return is_exists_expr(self)


class ExpressionDefinitionContext(object):
    def __init__(
        self, *variables: Variable,
        domain: Optional['DSLDomainBase'] = None,
        scope: Optional[str] = None,
        is_effect_definition: bool = False,
        slot_functions_are_sgc: bool = False,
        allow_auto_predicate_def: bool = True,
        check_arguments: bool = True,
    ):
        """Initialize the context.

        Args:
            variables: The variables that are available in the expression.
            domain: the domain of the expression.
            scope: the current definition scope (e.g., in a function). This variable will be used to generate unique names for the functions.
            is_effect_definition: whether the expression is defined in an effect of an operator.
            slot_functions_are_sgc: whether the slot functions are SGC functions (state-goal-constraints functions).
            allow_auto_predicate_def: whether to enable automatic predicate definition.
            check_arguments: whether to check the arguments of the functions.
        """

        self.variables = list(variables)
        self.variable_name2obj = {v.name: v for v in self.variables}
        self.domain = domain
        self.scope = scope
        self.anonymous_argument_generator = AnonymousFunctionArgumentGenerator()
        self.is_effect_definition_stack = [is_effect_definition]
        self.slot_functions_are_sgc = slot_functions_are_sgc
        self.allow_auto_predicate_def = allow_auto_predicate_def
        self.check_arguments = check_arguments

    OPTION_NAMES = ['allow_auto_predicate_def', 'check_arguments']

    variables: List[Variable]
    """The list of variables."""

    variable_name2obj: Dict[str, Variable]
    """The mapping from variable names to variables."""

    domain: Optional['DSLDomainBase']
    """The domain of the expression."""

    scope: Optional[str]
    """The current definition scope (e.g., in a function). This variable will be used to generate unique names for the functions."""

    anonymous_argument_generator: AnonymousFunctionArgumentGenerator
    """The anonymous argument generator."""

    is_effect_definition_stack: List[bool]
    """Whether the expression is defined in an effect of an operator."""

    slot_functions_are_sgc: bool
    """Whether the slot functions are SGC functions (state-goal-constraints functions)."""

    allow_auto_predicate_def: bool
    """Whether to enable automatic predicate definition."""

    check_arguments: bool
    """Whether to check the arguments of the functions."""

    @wrap_custom_as_default
    def as_default(self):
        yield self

    def has_variable(self: 'ExpressionDefinitionContext', variable: Union[str, Variable]) -> bool:
        if isinstance(variable, Variable):
            return variable.name in self.variable_name2obj
        return variable in self.variable_name2obj

    def get_variable(self, variable: Union[str, Variable]) -> Variable:
        if isinstance(variable, Variable):
            return variable
        if variable not in self.variable_name2obj:
            raise ValueError(f'Unknown variable: {variable}; available variables: {self.variables}.')
        return self.variable_name2obj[variable]

    def __getitem__(self, variable: Union[str, Variable]) -> 'VariableExpression':
        return self.wrap_variable(variable)

    def wrap_variable(self, variable: Union[str, Variable]) -> 'VariableExpression':
        if isinstance(variable, Variable):
            return VariableExpression(variable)
        variable_name = variable
        if variable_name == '??':
            return VariableExpression(Variable('??', AutoType))
        if variable_name not in self.variable_name2obj:
            raise ValueError('Unknown variable: {}; available variables: {}.'.format(variable_name, self.variables))
        return VariableExpression(self.variable_name2obj[variable_name])

    def gen_random_named_variable(self, dtype) -> Variable:
        """Generate a variable expression with a random name. This utility is useful in "flatten_expression". See the doc for that function for details."""
        name = self.anonymous_argument_generator.gen()
        return Variable(name, dtype)

    @contextlib.contextmanager
    def with_variables(self, *args: Variable):
        """Reset the list of variables."""
        old_variables = self.variables.copy()
        self.variables = list(args)
        self.variable_name2obj = {v.name: v for v in self.variables}
        yield
        self.variables = old_variables
        self.variable_name2obj = {v.name: v for v in self.variables}

    @contextlib.contextmanager
    def new_variables(self, *args: Variable):
        """Adding a list of new variables. This function is a context manager, and the variables will be removed after the context is closed."""
        for arg in args:
            if arg.name in self.variable_name2obj:
                raise ValueError(f'Variable {arg.name} already exists.')
            self.variables.append(arg)
            self.variable_name2obj[arg.name] = arg
        yield self
        for arg in reversed(args):
            self.variables.pop()
            del self.variable_name2obj[arg.name]

    def add_variables(self, *args: Variable):
        """Adding a list of new variables. Unlike :meth:`new_variables`, the variables will be directly added to the current context."""
        for arg in args:
            if arg.name in self.variable_name2obj:
                raise ValueError(f'Variable {arg.name} already exists.')
            self.variables.append(arg)
            self.variable_name2obj[arg.name] = arg

    @contextlib.contextmanager
    def mark_is_effect_definition(self, is_effect_definition: bool):
        """`is_effect_definition` is a boolean flag that indicates whether the current expression is defined in an effect of an operator. This function is a context manager and is only used in planning tasks."""
        self.is_effect_definition_stack.append(is_effect_definition)
        yield self
        self.is_effect_definition_stack.pop()

    @property
    def is_effect_definition(self) -> bool:
        return self.is_effect_definition_stack[-1]

    @contextlib.contextmanager
    def options(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.OPTION_NAMES:
                raise ValueError(f'Unknown option {k}.')
        old_options = {k: getattr(self, k) for k in kwargs}
        for k, v in kwargs.items():
            setattr(self, k, v)
        yield self
        for k, v in old_options.items():
            setattr(self, k, v)


get_expression_definition_context: Callable[[], Optional[ExpressionDefinitionContext]] = gen_get_default(ExpressionDefinitionContext)


class ObjectOrValueOutputExpression(Expression, ABC):
    @property
    def return_type(self) -> Union[ObjectType, ValueType, SequenceType]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class ObjectOutputExpression(ObjectOrValueOutputExpression, ABC):
    @property
    def return_type(self) -> Union[ObjectType, SequenceType]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class ValueOutputExpression(ObjectOrValueOutputExpression, ABC):
    @property
    def return_type(self) -> Union[ValueType, SequenceType]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class NullExpression(ObjectOrValueOutputExpression):
    def __init__(self, dtype: Union[ObjectType, ValueType, SequenceType]):
        self.dtype = dtype

    dtype: Union[ObjectType, ValueType, SequenceType]
    """The type of the null expression."""

    @property
    def return_type(self) -> Union[ObjectType, ValueType, SequenceType]:
        return self.dtype

    def __str__(self) -> str:
        return 'null'


class VariableExpression(ObjectOrValueOutputExpression):
    def __init__(self, variable: Variable):
        self.variable = variable

    variable: Variable
    """The variable."""

    @property
    def name(self) -> str:
        return self.variable.name

    @property
    def dtype(self) -> Union[ObjectType, ValueType, FunctionType, ListType]:
        return self.variable.dtype

    @property
    def return_type(self) -> Union[ObjectType, ValueType, FunctionType, ListType]:
        return self.variable.dtype

    def __str__(self) -> str:
        return f'V::{self.name}'


VariableOrValueOutputExpression = Union[VariableExpression, ValueOutputExpression]
VariableOrObjectOutputExpression = Union[VariableExpression, ObjectOutputExpression]


class VariableAssignmentExpression(Expression):
    @property
    def return_type(self):
        return None

    def __str__(self):
        raise NotImplementedError()


class ObjectConstantExpression(ObjectOutputExpression):
    def __init__(self, constant: Union[ObjectConstant, StateObjectReference]):
        if isinstance(constant, StateObjectReference):
            assert constant.dtype is not None, 'StateObjectReference must have a dtype.'
            constant = ObjectConstant(constant.name, constant.dtype)
        if isinstance(constant, ListValue):
            raise ValueError('ObjectConstantExpression does not accept ListValue.')
        self.constant = constant

    constant: ObjectConstant
    """The object constant."""

    @property
    def name(self) -> str:
        """The name of the object."""
        if not isinstance(self.constant, ObjectConstant):
            raise TypeError('ObjectConstantExpression.name is only available for ObjectConstant.')
        return self.constant.name

    @property
    def dtype(self) -> Union[ObjectType, ListType]:
        """The type of the object."""
        return self.constant.dtype

    @property
    def is_constant_list(self):
        """Whether the object is a constant list."""
        return isinstance(self.constant.name, ListValue)

    @property
    def return_type(self) -> Union[ObjectType, ListType]:
        return self.constant.dtype

    def __str__(self) -> str:
        if self.is_constant_list:
            if isinstance(self.constant.name.values, slice):
                return 'O::[...]'
            constant_str = ' '.join(x.name for x in self.constant.name.values)
            return f'O::{{{constant_str}}}'
        return f'O::{self.name}'


class ConstantExpression(ValueOutputExpression):
    """Constant expression always returns a constant value."""

    constant: ValueBase
    """The constant."""

    def __init__(self, value: Union[bool, int, float, str, torch.Tensor, Any, ValueBase], dtype: Optional[ValueType] = None):
        if isinstance(value, ValueBase):
            self.constant = value
        else:
            assert dtype is not None
            if isinstance(dtype, (TensorValueTypeBase, PyObjValueType)):
                self.constant = TensorValue.from_scalar(value, dtype)
            else:
                self.constant = Value(dtype, value)

    @property
    def return_type(self) -> Union[ValueType, ListType]:
        if isinstance(self.constant, ListValue):
            assert isinstance(self.constant.dtype.element_type, ValueType)
            return self.constant.dtype
        assert isinstance(self.constant.dtype, ValueType)
        return self.constant.dtype

    @classmethod
    def true(cls):
        return cls(torch.tensor(1, dtype=torch.int64), BOOL)

    @classmethod
    def false(cls):
        return cls(torch.tensor(0, dtype=torch.int64), BOOL)

    @classmethod
    def int64(cls, value):
        return cls(torch.tensor(value, dtype=torch.int64), INT64)

    @classmethod
    def float32(cls, value):
        return cls(torch.tensor(value, dtype=torch.float32), FLOAT32)

    @classmethod
    def string(cls, value):
        return cls(value, STRING)

    @classmethod
    def from_value(cls, value, dtype: Optional[ValueType] = None):
        if isinstance(value, bool):
            return cls(torch.tensor(bool(value), dtype=torch.int64), dtype if dtype is not None else BOOL)
        elif isinstance(value, int):
            return cls(torch.tensor(value, dtype=torch.int64), dtype if dtype is not None else INT64)
        elif isinstance(value, float):
            return cls(torch.tensor(value, dtype=torch.float32), dtype if dtype is not None else FLOAT32)
        elif isinstance(value, str):
            return cls(value, dtype if dtype is not None else STRING)
        else:
            raise ValueError(f'Unknown value type: {type(value)}.')

    def __str__(self):
        if isinstance(self.constant, TensorValue) and self.constant.is_single_elem:
            return f'C::{self.constant.single_elem()}'
        return f'C::{self.constant}'


ConstantExpression.TRUE = ConstantExpression.true()
ConstantExpression.FALSE = ConstantExpression.false()


class ListCreationExpression(ObjectOrValueOutputExpression):
    def __init__(self, arguments: Sequence[ValueOutputExpression], element_type: Optional[TypeBase] = None):
        self.arguments = tuple(arguments)

        if len(self.arguments) == 0:
            assert element_type is not None, 'Must specify the element type if the list is empty.'
            self.element_type = element_type
        else:
            self.element_type = element_type if element_type is not None else self.arguments[0].return_type

        self.check_arguments()

    @property
    def return_type(self) -> ListType:
        return ListType(self.element_type)

    def _check_arguments(self):
        for i, arg in enumerate(self.arguments):
            if arg.return_type != self.element_type:
                raise TypeError(f'Argument #{i} has type {arg.return_type}, which does not match the list type {self.element_type}.')

    def __str__(self) -> str:
        return f'{{{", ".join([str(arg) for arg in self.arguments])}}}'


class ListExpansionExpression(Expression):
    def __init__(self, expression: ValueOutputExpression):
        self.expression = expression
        self.check_arguments()
        self.element_type = self.expression.return_type.element_type

    expression: ValueOutputExpression
    """The expression."""

    element_type: TypeBase
    """The element type."""

    def _check_arguments(self):
        assert isinstance(self.expression, ValueOutputExpression) and self.expression.return_type.is_list_type, \
            f'ListExpansionExpression only accepts ValueOutputExpressions with list-typed return, got {self.expression} which returns {self.expression.return_type}.'

    @property
    def return_type(self) -> ListType:
        return self.expression.return_type

    def __str__(self) -> str:
        return f'... {str(self.expression)}'


class FunctionApplicationError(Exception):
    def __init__(self, index: int, expect, got):
        msg = f'Argument #{index} type does not match: expect {expect}, got {got}.'
        super().__init__(msg)

        self.index = index
        self.expect = expect
        self.got = got


class FunctionApplicationExpression(ValueOutputExpression):
    """Function application expression represents the application of a function over a list of arguments."""

    def __init__(self, function: Function, arguments: Iterable[ObjectOrValueOutputExpression], batch: str = 'mixed'):
        self.function = function
        self.arguments = tuple(arguments)
        self.batch = batch
        self._return_type = AutoType

        assert batch in ['mixed', 'inner', 'outer'], f'Unknown batch type: {batch}. Allowed values are "mixed", "inner", "outer".'

        self.check_arguments()

    function: Function
    """The function to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The list of arguments to the function."""

    def _check_arguments(self):
        try:
            if len(self.function.arguments) != len(self.arguments):
                raise TypeError('Argument number mismatch: expect {}, got {}.'.format(len(self.function.arguments), len(self.arguments)))

            is_simple_arguments = self.function.ftype.is_simple_arguments

            is_inner = self.batch == 'inner'
            is_outer = self.batch == 'outer'
            if is_simple_arguments:
                found_multi_arity = False
                for arg in self.arguments:
                    if arg.return_type.is_uniform_sequence_type:
                        if isinstance(arg, ObjectConstantExpression) and isinstance(arg.constant, StateObjectList) and arg.constant.is_qindex:
                            pass
                        else:
                            if arg.return_type.is_batched_list_type and arg.return_type.ndim > 1:
                                found_multi_arity = True
                                break
                if is_outer and found_multi_arity:
                    raise ValueError('Outer batch does not support multi-arity arguments.')
            else:
                assert not is_inner and not is_outer, 'Batched function does not support inner/outer batch.'

            inner_batched_arguments = None
            outer_batched_arguments = list()
            for i, (arg_def, arg) in enumerate(zip(self.function.arguments, self.arguments)):
                # arg_def: the definition of the argument in the function signature
                # arg: the actual argument being returned
                if not isinstance(arg_def, Variable):
                    raise TypeError('Unknown argument definition type: {}.'.format(type(arg_def)))
                if not isinstance(arg, ObjectOrValueOutputExpression):
                    raise TypeError('Unknown argument value type: {}.'.format(type(arg)))

                argdef_dtype: TypeBase = arg_def.dtype
                arg_dtype: TypeBase = arg.return_type

                if is_simple_arguments:
                    if not arg_dtype.downcast_compatible(argdef_dtype, allow_self_list=True):
                        raise FunctionApplicationError(i, argdef_dtype, arg_dtype)
                    if arg_dtype.is_uniform_sequence_type:
                        if isinstance(arg, ObjectConstantExpression) and isinstance(arg.constant.name, StateObjectList) and arg.constant.name.is_qindex:
                            if is_inner:
                                new_inner_batched_arguments = (argdef_dtype, )
                                if inner_batched_arguments is None:
                                    inner_batched_arguments = new_inner_batched_arguments
                                elif inner_batched_arguments != (AutoType, ) and inner_batched_arguments != new_inner_batched_arguments:
                                    raise ValueError(f'Inconsistent inner batched arguments: prev={inner_batched_arguments} vs this={new_inner_batched_arguments}.')
                            else:  # Outer or Mixed
                                outer_batched_arguments.append(argdef_dtype)
                        else:
                            if is_inner:
                                if arg_dtype.is_batched_list_type:
                                    new_inner_batched_arguments = (argdef_dtype, )
                                    if inner_batched_arguments is None:
                                        inner_batched_arguments = new_inner_batched_arguments
                                    elif inner_batched_arguments != (AutoType, ) and inner_batched_arguments != new_inner_batched_arguments:
                                        raise ValueError(f'Inconsistent inner batched arguments: prev={inner_batched_arguments} vs this={new_inner_batched_arguments}.')
                                elif arg_dtype.is_list_type:
                                    new_inner_batched_arguments = (arg_dtype.element_type, )
                                    if inner_batched_arguments is None:
                                        inner_batched_arguments = new_inner_batched_arguments
                                    elif len(inner_batched_arguments) != 1:
                                        raise ValueError(f'Inconsistent inner batched arguments: prev={inner_batched_arguments} vs this={new_inner_batched_arguments}.')
                                else:
                                    raise ValueError(f'Unknown inner batched argument type: {arg_dtype}.')
                            else:  # Outer or Mixed
                                if arg_dtype.is_batched_list_type:
                                    arg_dtype: BatchedListType
                                    outer_batched_arguments.extend(arg_dtype.index_dtypes)
                                elif arg_dtype.is_list_type:
                                    outer_batched_arguments.append(arg_dtype.element_type)
                                else:
                                    raise ValueError(f'Unknown outer batched argument type: {arg_dtype}.')
                else:
                    if not arg_dtype.downcast_compatible(argdef_dtype):
                        raise FunctionApplicationError(i, argdef_dtype, arg_dtype)

            if is_simple_arguments:
                if is_inner:
                    if inner_batched_arguments is None:
                        self._return_type = self.function.return_type
                    else:
                        if inner_batched_arguments == (AutoType, ):
                            self._return_type = ListType(self.function.return_type)
                        else:
                            self._return_type = BatchedListType(self.function.return_type, inner_batched_arguments)
                elif is_outer:
                    if len(outer_batched_arguments) == 0:
                        self._return_type = self.function.return_type
                    else:
                        self._return_type = BatchedListType(self.function.return_type, outer_batched_arguments)
                else:
                    if inner_batched_arguments is None:
                        if len(outer_batched_arguments) == 0:
                            self._return_type = self.function.return_type
                        else:
                            self._return_type = BatchedListType(self.function.return_type, outer_batched_arguments)
                    else:
                        if len(outer_batched_arguments) == 0:
                            if inner_batched_arguments == (AutoType, ):
                                self._return_type = ListType(self.function.return_type)
                            else:
                                self._return_type = BatchedListType(self.function.return_type, inner_batched_arguments)
                        else:
                            self._return_type = BatchedListType(self.function.return_type, inner_batched_arguments + tuple(outer_batched_arguments))
            else:
                self._return_type = self.function.return_type
        except (TypeError, FunctionApplicationError) as e:
            error_header = 'Error during applying {}.\n'.format(str(self.function))
            try:
                arguments_str = ', '.join(str(arg) for arg in self.arguments)
                error_header += '  Arguments: {}\n'.format(arguments_str)
            except Exception:  # noqa
                pass
            print(error_header)
            print(e)
            # import ipdb; ipdb.set_trace()
            raise TypeError(error_header + str(e)) from e

    @property
    def return_type(self) -> ValueType:
        return self._return_type

    def __str__(self) -> str:
        if self.batch == 'mixed':
            fmt = self.function.name + '('
        elif self.batch == 'inner':
            fmt = self.function.name + '[[inner]]('
        elif self.batch == 'outer':
            fmt = self.function.name + '[[outer]]('
        else:
            raise ValueError(f'Unknown batch type: {self.batch}.')
        arg_fmt = [str(x) for x in self.arguments]
        arg_fmt_len = [len(x) for x in arg_fmt]

        ctx = get_format_context()

        # The following criterion is just an approximation. A more principled way is to pass the current indent level
        # to the recursive calls to str(x).
        if ctx.expr_max_length > 0 and (sum(arg_fmt_len) + len(fmt) + 1 > ctx.expr_max_length):
            if sum(arg_fmt_len) > ctx.expr_max_length:
                fmt += '\n' + ',\n'.join([indent_text(x) for x in arg_fmt]) + '\n'
            else:
                fmt += '\n' + ', '.join(arg_fmt) + '\n'
        else:
            fmt += ', '.join(arg_fmt)
        fmt += ')'
        return fmt


class ListFunctionApplicationExpression(ValueOutputExpression):
    """Function application expression represents the application of a function over a list of arguments."""

    def __init__(self, function: Function, arguments: Iterable[ObjectOrValueOutputExpression]):
        self.function = function
        self.arguments = tuple(arguments)
        self.check_arguments()

    function: Function
    """The function to be applied."""

    arguments: Tuple[ObjectOrValueOutputExpression, ...]
    """The list of arguments to the function."""

    def _check_arguments(self):
        try:
            if len(self.function.arguments) != len(self.arguments):
                raise TypeError('Argument number mismatch: expect {}, got {}.'.format(len(self.function.arguments), len(self.arguments)))
            for i, (arg_def, arg) in enumerate(zip(self.function.arguments, self.arguments)):
                if isinstance(arg_def, Variable):
                    if isinstance(arg, VariableExpression):
                        if not arg.dtype.downcast_compatible(arg_def.dtype, allow_self_list=True):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.dtype)
                    elif isinstance(arg, ObjectConstantExpression):
                        if not arg.dtype.downcast_compatible(arg_def.dtype, allow_self_list=True):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.dtype)
                    elif isinstance(arg, ConstantExpression):
                        if not arg.return_type.downcast_compatible(arg_def.dtype, allow_self_list=True):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.return_type)
                    elif isinstance(arg, (FunctionApplicationExpression, ListFunctionApplicationExpression, GeneralizedQuantificationExpression)):
                        if not arg.return_type.downcast_compatible(arg_def.dtype, allow_self_list=True):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.return_type)
                    elif isinstance(arg, ListCreationExpression):
                        if not arg.return_type.downcast_compatible(arg_def.dtype, allow_self_list=True):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.return_type)
                    else:
                        raise FunctionApplicationError(i, 'VariableExpression or ObjectConstantExpression or ConstantExpression or FunctionApplication', type(arg))
                elif isinstance(arg_def, ValueType):
                    if isinstance(arg, ValueOutputExpression):
                        pass
                    elif isinstance(arg, VariableExpression) and isinstance(arg.return_type, ValueType):
                        pass
                    elif isinstance(arg, ConstantExpression):
                        pass
                    else:
                        raise FunctionApplicationError(i, 'ValueOutputExpression', type(arg))
                    if arg_def != arg.return_type:
                        raise FunctionApplicationError(i, arg_def, arg.return_type)
                else:
                    raise TypeError('Unknown argument definition type: {}.'.format(type(arg_def)))
        except (TypeError, FunctionApplicationError) as e:
            error_header = 'Error during applying {}.\n'.format(str(self.function))
            try:
                arguments_str = ', '.join(str(arg) for arg in self.arguments)
                error_header += ' Arguments: {}\n'.format(arguments_str)
            except Exception:  # noqa
                pass
            raise TypeError(error_header + str(e)) from e

    @property
    def return_type(self) -> Union[ValueType, ListType]:
        for arg in self.arguments:
            if arg.return_type.is_list_type:
                return ListType(self.function.return_type)
        return self.function.return_type

    def __str__(self) -> str:
        fmt = self.function.name + '[list]('
        arg_fmt = [str(x) for x in self.arguments]
        arg_fmt_len = [len(x) for x in arg_fmt]

        ctx = get_format_context()

        # The following criterion is just an approximation. A more principled way is to pass the current indent level
        # to the recursive calls to str(x).
        if ctx.expr_max_length > 0 and (sum(arg_fmt_len) + len(fmt) + 1 > ctx.expr_max_length):
            if sum(arg_fmt_len) > ctx.expr_max_length:
                fmt += '\n' + ',\n'.join([indent_text(x) for x in arg_fmt]) + '\n'
            else:
                fmt += '\n' + ', '.join(arg_fmt) + '\n'
        else:
            fmt += ', '.join(arg_fmt)
        fmt += ')'
        return fmt


class ConditionalSelectExpression(ValueOutputExpression):
    """Conditional select expression represents the selection of a value based on a condition."""

    def __init__(self, predicate: ValueOutputExpression, condition: ValueOutputExpression):
        self.predicate = predicate
        self.condition = condition
        self.check_arguments()

    predicate: ValueOutputExpression
    """The predicate expression."""

    condition: ValueOutputExpression
    """The condition expression."""

    def _check_arguments(self):
        if isinstance(self.condition, ValueOutputExpression) and self.condition.return_type == BOOL:
            pass
        elif isinstance(self.condition, VariableExpression) and self.condition.return_type.downcast_compatible(BOOL):
            pass
        else:
            raise TypeError('Condition must be a boolean expression.')

    @property
    def return_type(self) -> ValueType:
        return self.predicate.return_type

    def __str__(self):
        predicate_str = str(self.predicate)
        condition_str = str(self.condition)

        if len(predicate_str) + len(condition_str) + 2 < 80:
            return f'cond-select({predicate_str} if {condition_str})'
        return f'cond-select({predicate_str} if\n{indent_text(condition_str)})'


class DeicticSelectExpression(ValueOutputExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        self.variable = variable
        self.expression = expr
        self.check_arguments()

    variable: Variable
    """The new quantified variable."""

    expression: ValueOutputExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self) -> ValueType:
        return self.expression.return_type

    def __str__(self):
        return f'deictic-select({self.variable}: {self.expression})'


class BoolOpType(JacEnum):
    AND = 'and'
    OR = 'or'
    NOT = 'not'
    XOR = 'xor'
    IMPLIES = 'implies'


class BoolExpression(ValueOutputExpression):
    OpType = BoolOpType

    def __init__(self, bool_op_type: BoolOpType, arguments: Sequence[ValueOutputExpression]):
        self.bool_op = bool_op_type
        self.arguments = tuple(arguments)
        self.check_arguments()

    bool_op: BoolOpType
    """The boolean operation. Can be AND, OR, NOT, XOR, IMPLIES."""

    arguments: Tuple[ValueOutputExpression, ...]
    """The list of arguments."""

    def _check_arguments(self):
        if self.bool_op is BoolOpType.NOT:
            assert len(self.arguments) == 1, f'Number of arguments for NotOp should be 1, got: {len(self.arguments)}.'
        if self.bool_op is BoolOpType.IMPLIES:
            assert len(self.arguments) == 2, f'Number of arguments for ImpliesOp should be 2, got: {len(self.arguments)}.'
        for i, arg in enumerate(self.arguments):
            assert isinstance(arg, (VariableExpression, ValueOutputExpression)), f'BoolOp only accepts ValueOutputExpressions, got argument #{i} of type {type(arg)}.'

    @property
    def return_type(self) -> ValueType:
        return self.arguments[0].return_type

    def __str__(self):
        argument_strings = [str(arg) for arg in self.arguments]
        if sum(len(x) for x in argument_strings) < 80:
            return f'{self.bool_op.value}({", ".join(argument_strings)})'
        arguments = ',\n'.join([indent_text(x) for x in argument_strings])
        return f'{self.bool_op.value}(\n{arguments}\n)'


class AndExpression(BoolExpression):
    bool_op: BoolOpType
    """The boolean operation. Must be :py:attr:`BoolOpType.AND`."""

    arguments: Tuple[ValueOutputExpression, ...]

    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.AND, arguments)


class OrExpression(BoolExpression):
    bool_op: BoolOpType
    """The boolean operation. Must be :py:attr:`BoolOpType.OR`."""

    arguments: Tuple[ValueOutputExpression, ...]

    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.OR, arguments)


class NotExpression(BoolExpression):
    bool_op: BoolOpType
    """The boolean operation. Must be :py:attr:`BoolOpType.NOT`."""

    arguments: Tuple[ValueOutputExpression]
    """The list of arguments. Must contain exactly one argument."""

    def __init__(self, arg: ValueOutputExpression):
        super().__init__(BoolOpType.NOT, [arg])


class XorExpression(BoolExpression):
    bool_op: BoolOpType
    """The boolean operation. Must be :py:attr:`BoolOpType.XOR`."""

    arguments: Tuple[ValueOutputExpression, ...]

    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.XOR, arguments)


class ImpliesExpression(BoolExpression):
    bool_op: BoolOpType
    """The boolean operation. Must be :py:attr:`BoolOpType.IMPLIES`."""

    arguments: Tuple[ValueOutputExpression, ValueOutputExpression]

    def __init__(self, lhs: ValueOutputExpression, rhs: ValueOutputExpression):
        super().__init__(BoolOpType.IMPLIES, [lhs, rhs])


class QuantificationOpType(JacEnum):
    FORALL = 'forall'
    EXISTS = 'exists'
    BATCHED = 'batched'


class QuantificationExpression(ValueOutputExpression):
    OpType = QuantificationOpType

    def __init__(self, quantification_op: QuantificationOpType, variable: Variable, expr: ValueOutputExpression):
        self.quantification_op = quantification_op
        self.variable = variable
        self.expression = expr

        self.check_arguments()

    quantification_op: QuantificationOpType
    """The quantification operation. Can be FORALL or EXISTS."""

    variable: Variable
    """The quantified variable."""

    expression: ValueOutputExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.expression, ValueOutputExpression), f'QuantificationOp only accepts ValueOutputExpressions, got type {type(self.expression)}.'
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self) -> ValueType:
        return self.expression.return_type

    def __str__(self):
        return f'{self.quantification_op.value}({self.variable}: {self.expression})'


class GeneralizedQuantificationExpression(ValueOutputExpression):
    def __init__(self, quantification_op: Any, variable: Variable, expr: ValueOutputExpression, return_type: Optional[ValueType] = None):
        self.quantification_op = quantification_op
        self.variable = variable
        self.expression = expr

        self._return_type = return_type if return_type is not None else self.expression.return_type

        self.check_arguments()

    quantification_op: Any
    """The quantification operation. It can be any data type."""

    variable: Variable
    """The quantified variable."""

    expression: ValueOutputExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.expression, ValueOutputExpression), f'QuantificationOp only accepts ValueOutputExpressions, got type {type(self.expression)}.'
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self) -> ValueType:
        return self._return_type

    def __str__(self):
        return f'{self.quantification_op}({self.variable}: {self.expression})'


class ForallExpression(QuantificationExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantificationOpType.FORALL, variable, expr)

    quantification_op: QuantificationOpType
    """The quantification operation. Must be :py:attr:`QuantificationOpType.FORALL`."""

    variable: Variable
    expression: ValueOutputExpression


class ExistsExpression(QuantificationExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantificationOpType.EXISTS, variable, expr)

    quantification_op: QuantificationOpType
    """The quantification operation. Must be :py:attr:`QuantificationOpType.EXISTS`."""

    variable: Variable
    expression: ValueOutputExpression


class BatchedExpression(QuantificationExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantificationOpType.BATCHED, variable, expr)

    quantification_op: QuantificationOpType
    """The quantification operation. Must be :py:attr:`QuantificationOpType.BATCHED`."""

    variable: Variable
    expression: ValueOutputExpression


class FindOneExpression(ObjectOutputExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        self.variable = variable
        self.expression = expr
        self.check_arguments()

    variable: Variable
    """The quantified variable."""

    expression: ValueOutputExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.expression, ValueOutputExpression), f'FindAllOp only accepts ValueOutputExpressions, got type {type(self.expression)}.'
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self) -> ObjectType:
        return self.variable.dtype

    def __str__(self):
        return f'findone({self.variable}: {self.expression})'


class FindAllExpression(ObjectOutputExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        self.variable = variable
        self.expression = expr
        self.check_arguments()

    variable: Variable
    """The quantified variable."""

    expression: ValueOutputExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.expression, ValueOutputExpression), f'FindAllOp only accepts ValueOutputExpressions, got type {type(self.expression)}.'
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self) -> ListType:
        return ListType(self.variable.dtype)

    def __str__(self):
        return f'findall({self.variable}: {self.expression})'


class CompareOpType(JacEnum):
    EQ = '=='
    NEQ = '!='
    LT = '<'
    LEQ = '<='
    GT = '>'
    GEQ = '>='


class _CompareExpressionBase(ValueOutputExpression, ABC):
    OpType = CompareOpType

    def __init__(self, compare_op: CompareOpType, lhs: ObjectOrValueOutputExpression, rhs: ObjectOrValueOutputExpression):
        self.compare_op = compare_op
        self.lhs = lhs
        self.rhs = rhs

        self.check_arguments()

    @property
    def arguments(self) -> Tuple[ObjectOrValueOutputExpression, ObjectOrValueOutputExpression]:
        return self.lhs, self.rhs

    @property
    def return_type(self) -> ValueType:
        return BOOL

    def __str__(self):
        return f'({self.lhs} {self.compare_op.value} {self.rhs})'


class ObjectCompareExpression(_CompareExpressionBase):
    def _check_arguments(self):
        assert self.compare_op in (CompareOpType.EQ, CompareOpType.NEQ), f'ObjectCompareExpression only accepts EQ and NEQ, got {self.compare_op}.'
        assert isinstance(self.lhs.return_type, ObjectType), f'lhs of ObjectCompareExpression must be of type ObjectType, got {self.lhs.return_type}.'
        assert isinstance(self.rhs.return_type, ObjectType), f'rhs of ObjectCompareExpression must be of type ObjectType, got {self.rhs.return_type}.'

    compare_op: CompareOpType
    """The comparison operation."""

    lhs: Union[ObjectOutputExpression, VariableExpression]
    """The left-hand side of the comparison."""

    rhs: Union[ObjectOutputExpression, VariableExpression]
    """The right-hand side of the comparison."""


class ValueCompareExpression(_CompareExpressionBase):
    def _check_arguments(self):
        assert isinstance(self.lhs.return_type, ValueType), f'lhs of ValueCompareExpression must be of type ValueType, got {self.lhs.return_type}.'
        assert isinstance(self.rhs.return_type, ValueType), f'rhs of ValueCompareExpression must be of type ValueType, got {self.rhs.return_type}.'

    compare_op: CompareOpType
    """The comparison operation."""

    lhs: ValueOutputExpression
    """The left-hand side of the comparison."""

    rhs: ValueOutputExpression
    """The right-hand side of the comparison."""


class ConditionExpression(ValueOutputExpression):
    def __init__(self, condition: ValueOutputExpression, true_value: ValueOutputExpression, false_value: ValueOutputExpression):
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value
        self.check_arguments()

    condition: ValueOutputExpression
    """The condition expression."""

    true_value: ValueOutputExpression
    """The true value expression."""

    false_value: ValueOutputExpression
    """The false value expression."""

    def _check_arguments(self):
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.return_type == BOOL, f'Condition must be a boolean expression, got {self.condition}.'
        assert self.true_value.return_type == self.false_value.return_type, f'True value and false value must have the same type, got {self.true_value.return_type} and {self.false_value.return_type}.'

    @property
    def return_type(self) -> ValueType:
        return self.true_value.return_type

    def __str__(self):
        condition_str = str(self.condition)
        true_value_str = str(self.true_value)
        false_value_str = str(self.false_value)

        if len(condition_str) + len(true_value_str) + len(false_value_str) + 4 < 80:
            return f'cond({condition_str} ? {true_value_str} : {false_value_str})'
        return f'cond({condition_str} ?\n{indent_text(true_value_str)}\n{indent_text(false_value_str)})'


class _PredicateValueExpression(Expression, ABC):
    def __init__(self, predicate: Union[VariableExpression, FunctionApplicationExpression], value: ValueOutputExpression):
        self.predicate = predicate
        self.value = value
        self.check_arguments()

    def _check_arguments(self):
        try:
            rtype = self.predicate.return_type
            if rtype.is_uniform_sequence_type:
                element_rtype = rtype.element_type
            else:
                element_rtype = rtype
            parent_type = element_rtype.assignment_type().parent_type
            if parent_type is not None and parent_type.typename in ('bool', 'int64', 'float32', 'string'):
                if self.value.return_type.typename == parent_type.typename:
                    return
            if parent_type is not None and parent_type.is_vector_value_type and self.value.return_type.is_list_type and self.value.return_type.element_type == parent_type.dtype:
                return
            if parent_type is not None and parent_type.is_vector_value_type and self.value.return_type.is_vector_value_type:
                return
            if parent_type is not None and parent_type.is_vector_value_type and self.value.return_type.is_scalar_value_type:
                # Handle cases such as x[:] = True
                return
            if element_rtype.assignment_type() != self.value.return_type:
                raise FunctionApplicationError(0, f'{self.predicate.return_type}(assignment type is {rtype.assignment_type()})', self.value.return_type)
        except TypeError as e:
            raise e
        except FunctionApplicationError as e:
            error_header = 'Error during _PredicateValueExpression checking: feature = {} value = {}.\n'.format(str(self.predicate), str(self.value))
            raise TypeError(
                error_header +
                f'Value type does not match: expect: {e.expect}, got {e.got}.'
            ) from e


class PredicateEqualExpression(ValueOutputExpression, _PredicateValueExpression):
    """This function is obsolete and should be replaced by :class:`ValueCompareExpression`."""

    predicate: Union[VariableExpression, FunctionApplicationExpression]
    """The predicate expression."""

    value: ValueOutputExpression
    """The value expression."""

    def _check_arguments(self):
        super()._check_arguments()
        if not isinstance(self.predicate, (VariableExpression, FunctionApplicationExpression)):
            raise TypeError(f'PredicateEqualOp only support dest type VariableExpression or FunctionApplication, got {type(self.predicate)}.')

    @property
    def return_type(self):
        return BOOL

    def __str__(self):
        return f'equal({self.predicate}, {self.value})'


class AssignExpression(_PredicateValueExpression, VariableAssignmentExpression):
    def __init__(self, predicate: FunctionApplicationExpression, value: ValueOutputExpression):
        _PredicateValueExpression.__init__(self, predicate, value)

    predicate: FunctionApplicationExpression
    """The predicate expression, must be a :class:`FunctionApplicationExpression` which refers to a state variable."""

    value: ValueOutputExpression
    """The expression for the value to assign to the state variable."""

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.predicate, (ListFunctionApplicationExpression, FunctionApplicationExpression)), 'AssignOp only support dest type FunctionApplication, got {}.'.format(type(self.predicate))

    def __str__(self):
        return f'assign{{{self.predicate}: {self.value}}}'


class ConditionalAssignExpression(_PredicateValueExpression, VariableAssignmentExpression):
    def __init__(self, feature: FunctionApplicationExpression, value: ValueOutputExpression, condition: ValueOutputExpression):
        self.condition = condition
        _PredicateValueExpression.__init__(self, feature, value)

    predicate: FunctionApplicationExpression
    """The predicate expression, must be a :class:`FunctionApplicationExpression` which refers to a state variable."""

    value: ValueOutputExpression
    """The expression for the value to assign to the state variable."""

    condition: ValueOutputExpression
    """The condition expression."""

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.return_type == BOOL

    def __str__(self):
        return f'cond-assign{{{self.predicate}: {self.value} if {self.condition}}}'


class DeicticAssignExpression(VariableAssignmentExpression):
    def __init__(self, variable: Variable, expr: Union[VariableAssignmentExpression]):
        self.variable = variable
        self.expression = expr
        self.check_arguments()

    variable: Variable
    """The quantified variable."""

    expression: VariableAssignmentExpression
    """The internal expression."""

    def _check_arguments(self):
        assert isinstance(self.variable.dtype, ObjectType)

    def __str__(self):
        return f'deictic-assign{{{self.variable}: {self.expression}}}'


ExpressionCompatible = Union[Expression, Variable, str, ObjectConstant, bool, int, float, torch.Tensor, ValueBase]


def cvt_expression(expr: ExpressionCompatible, dtype: Optional[Union[ObjectType, ValueType]] = None) -> Expression:
    """Convert an expression compatible object to an expression. Acceptable types are:

    * :class:`Expression`.
    * :class:`Variable`: return a :class:`VariableExpression`.
    * :class:`str`: return a :class:`ConstantExpression` with the given constant string name, or a :class:`ObjectConstantExpression` if the dtype is a :class:`ObjectType`.
    * :class:`ObjectConstant`: return a :class:`ObjectConstantExpression`.
    * :class:`bool`, :class:`int`, :class:`float`, :class:`torch.Tensor`: return a :class:`ConstantExpression`.
    * :class:`ValueBase`: return a :class:`ConstantExpression`.

    Args:
        expr: the expression compatible object.
        dtype: the expected data type of the expression. If not given, the dtype will be inferred from the given object.

    Returns:
        the converted expression.

    Raises:
        TypeError: if the given object is not an expression compatible object.
    """

    if isinstance(expr, Expression):
        return expr
    elif isinstance(expr, Variable):
        return VariableExpression(expr)
    elif isinstance(expr, str):
        if isinstance(dtype, ObjectType):
            return ObjectConstantExpression(ObjectConstant(expr, dtype or AutoType))
        elif isinstance(dtype, ValueType):
            return ConstantExpression(Value(dtype or AutoType, expr))
    elif isinstance(expr, ObjectConstant):
        return ObjectConstantExpression(expr)
    elif isinstance(expr, bool):
        return ConstantExpression(torch.tensor(int(expr), dtype=torch.int64), dtype or BOOL)
    elif isinstance(expr, int):
        return ConstantExpression(torch.tensor(expr, dtype=torch.int64), dtype or INT64)
    elif isinstance(expr, float):
        return ConstantExpression(torch.tensor(expr, dtype=torch.float32), dtype or FLOAT32)
    elif isinstance(expr, torch.Tensor):
        if expr.dtype == torch.int64:
            return ConstantExpression(expr, dtype or INT64)
        elif expr.dtype == torch.float32:
            return ConstantExpression(expr, dtype or FLOAT32)
        else:
            raise TypeError(f'Unsupported tensor type: {expr.dtype}.')
    elif isinstance(expr, ValueBase):
        if isinstance(expr.dtype, ValueType):
            return ConstantExpression(expr, expr.dtype)
        else:
            raise TypeError(f'Unsupported value type: {expr.dtype}.')
    else:
        raise TypeError(f'Non-compatible expression type {type(expr)} for expression "{expr}".')


def cvt_expression_list(arguments: Sequence[ExpressionCompatible], dtypes: Optional[Sequence[Union[ObjectType, ValueType]]] = None) -> List[Expression]:
    """Convert a list of expression compatible objects to a list of expressions.

    Args:
        arguments: the list of expression compatible objects.
        dtypes: the list of expected data types of the expressions. If not given, the dtypes will be inferred from the given objects.
            It can be a single data type, in which case all the expressions will be converted to this data type.

    Returns:
        the list of converted expressions.
    """
    if dtypes is None:
        arguments = [cvt_expression(arg) for arg in arguments]
    else:
        arguments = [cvt_expression(arg, dtype) for arg, dtype in zip(arguments, dtypes)]
    return arguments


def get_type(value: Any) -> Union[TypeBase, Tuple[TypeBase, ...]]:
    """Get the type of the given value."""
    if value is FunctionArgumentUnset:
        return FunctionArgumentUnset
    elif isinstance(value, Function):
        return value.ftype
    elif isinstance(value, Expression):
        return value.return_type
    elif isinstance(value, ValueBase):
        return value.dtype
    elif isinstance(value, (bool, int, float, complex, str)):
        return AutoType
    else:
        raise ValueError(f'Unknown value type: {type(value)}.')


def get_types(args=None, kwargs=None):
    """Get the types of the given arguments and keyword arguments."""
    ret = list()
    if args is not None:
        ret.append(tuple(get_type(v) for v in args))
    if kwargs is not None:
        ret.append({k: get_type(v) for k, v in kwargs.items()})
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def is_null_expression(expr: Expression) -> bool:
    return isinstance(expr, NullExpression)


def is_object_output_expression(expr: Expression) -> TypeGuard[ObjectOutputExpression]:
    return isinstance(expr, ObjectOutputExpression) or (isinstance(expr, VariableExpression) and isinstance(expr.variable.dtype, ObjectType))


def is_value_output_expression(expr: Expression) -> TypeGuard[ValueOutputExpression]:
    return isinstance(expr, ValueOutputExpression)


def is_variable_assignment_expression(expr: Expression) -> bool:
    return isinstance(expr, VariableAssignmentExpression)


def is_and_expr(expr: Expression) -> TypeGuard[AndExpression]:
    return isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.AND


def is_or_expr(expr: Expression) -> TypeGuard[OrExpression]:
    return isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.OR


def is_not_expr(expr: Expression) -> TypeGuard[NotExpression]:
    return isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT


def is_xor_expr(expr: Expression) -> TypeGuard[XorExpression]:
    return isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.XOR


def is_implies_expr(expr: Expression) -> TypeGuard[ImpliesExpression]:
    return isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.IMPLIES


def is_constant_bool_expr(expr: Expression) -> TypeGuard[ConstantExpression]:
    if isinstance(expr, ConstantExpression) and expr.return_type == BOOL:
        return True
    return False


def is_forall_expr(expr: Expression) -> TypeGuard[ForallExpression]:
    return isinstance(expr, QuantificationExpression) and expr.quantification_op is QuantificationOpType.FORALL


def is_exists_expr(expr: Expression) -> TypeGuard[ExistsExpression]:
    return isinstance(expr, QuantificationExpression) and expr.quantification_op is QuantificationOpType.EXISTS

