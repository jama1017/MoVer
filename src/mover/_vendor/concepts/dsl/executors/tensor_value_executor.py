#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensor_value_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/03/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Tensor-based expression executor.

The high-level interface for tensor-based expression is that we can execute an expression with a given state and a set of
bounded variables. The executor will return a tensor value.

The state is represented using :class:`mover._vendor.concepts.dsl.tensor_state.TensorState` or :class:`mover._vendor.concepts.dsl.tensor_state.NamedObjectTensorState`, which internally stores a dictionary
mapping from string (the state variable name, e.g., ``is_hot``) to a :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue` class.

The bounded variables are essentially a dictionary mapping from strings (the variable name, e.g., ``x``) to its value. There are
two types of values: (1) a :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue` class, which represents an actual value (e.g., a vector representation);
(2) a :class:`StateObjectReference` instance or a QINDEX (a.k.a., ``slice(None)``), which represents a reference to an object in the state.

With the bounded variables, the expressions can have variables, which are essentially placeholders for the actual values. For example,

.. code-block:: python

    domain = FunctionDomain()
    # Define an object type `person`.
    domain.define_type(ObjectType('person'))
    # Define a state variable `is_friend` with type `person x person -> bool`.
    domain.define_function(Function('is_friend', FunctionType([ObjectType('person'), ObjectType('person')], BOOL)))

    x = VariableExpression(Variable('x', ObjectType('person')))
    y = VariableExpression(Variable('y', ObjectType('person')))
    relation = FunctionApplication(domain.functions['is_friend'], [x, y])

Then we can execute the expression with a given state and bounded variables:

.. code-block:: python

    # See the documentation for namedObjectTensorState for more details.
    state = NamedObjectTensorState({
        'is_friend': TensorValue(BOOL, ['x', 'y'], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=torch.bool))
    }, object_names={
        'Alice': ObjectType('person'),
        'Bob': ObjectType('person'),
        'Charlie': ObjectType('person'),
    })
    executor = SimpleFunctionTensorValueExecutor(domain)

    # For both of the following lines, the result is a tensor value with value `True`.
    # Use the constructed expression:
    executor.execute(relation, state, {'x': 'Alice', 'y': 'Bob'})
    # To use the default parser:
    executor.execute('is_friend(x, y)', state, {'x': 'Alice', 'y': 'Bob'})
"""

import contextlib
from typing import Optional, Union, Tuple, Sequence, Dict

import torch

from mover._vendor.concepts.dsl.dsl_types import ObjectType, ValueType, TensorValueTypeBase, NamedTensorValueType, PyObjValueType, ListType, BatchedListType, ObjectConstant, Variable, UnnamedPlaceholder, QINDEX
from mover._vendor.concepts.dsl.dsl_types import BOOL, INT64, FLOAT32
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.function_domain import FunctionDomain
from mover._vendor.concepts.dsl.value import ListValue
from mover._vendor.concepts.dsl.constraint import Constraint, SimulationFluentConstraintFunction
from mover._vendor.concepts.dsl.tensor_value import TensorValue, scalar
from mover._vendor.concepts.dsl.tensor_state import StateObjectReference, TensorState, NamedObjectTensorState
from mover._vendor.concepts.dsl.expression import Expression, VariableExpression, ObjectConstantExpression, ConstantExpression, FunctionApplicationExpression, ValueCompareExpression, BoolOpType, QuantificationOpType
from mover._vendor.concepts.dsl.constraint import OptimisticValue, ConstraintSatisfactionProblem, OPTIM_MAGIC_NUMBER
from mover._vendor.concepts.dsl.parsers.parser_base import ParserBase
from mover._vendor.concepts.dsl.parsers.function_expression_parser import FunctionExpressionParser
from mover._vendor.concepts.dsl.executors.executor_base import DSLExecutorBase
from mover._vendor.concepts.dsl.executors.value_quantizers import ValueQuantizer, PyObjectStore

__all__ = [
    'BoundedVariablesDict', 'BoundedVariablesDictCompatible',
    'compose_bvdict', 'compose_bvdict_args', 'get_bvdict',
    'TensorValueExecutorReturnType', 'TensorValueExecutorBase', 'FunctionDomainTensorValueExecutor'
]


BoundedVariablesDict = Dict[str, Dict[str, Union[StateObjectReference, slice, TensorValue]]]
"""Internal representation of a bounded variable dictionary. It stores a nested two-layer dictionary, where the first layer
stores the type of the object, and the second layer stores the name of the object. The value can be either a :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue`
or a :class:`StateObjectReference` instance (representing the reference to a single object)."""

BoundedVariablesDictCompatibleKeyType = Union[str, Variable]
BoundedVariablesDictCompatibleValueType = Union[str, int, slice, bool, float, torch.Tensor, TensorValue, ObjectConstant, StateObjectReference]

BoundedVariablesDictCompatible = Union[
    None, Sequence[Variable],
    Dict[BoundedVariablesDictCompatibleKeyType, BoundedVariablesDictCompatibleValueType],
    BoundedVariablesDict
]
"""Compatible types with :class:`BoundedVariablesDict`. They can be converted to :class:`BoundedVariablesDict` using :func:`compose_bvdict`."""


def _get_state_object_reference(state, dtype, value):
    if isinstance(value, int):
        assert isinstance(state, NamedObjectTensorState)
        value = StateObjectReference(state.object_type2name[dtype.typename][value], value, dtype)
        return value
    elif isinstance(value, str):
        assert isinstance(state, NamedObjectTensorState)
        value = StateObjectReference(value, state.get_typed_index(value), dtype)
        return value
    elif isinstance(value, ObjectConstant):
        assert isinstance(state, NamedObjectTensorState)
        value = StateObjectReference(value.name, state.get_typed_index(value.name, typename=value.dtype.typename), value.dtype)
        return value
    elif isinstance(value, slice):
        return value
    elif isinstance(value, StateObjectReference):
        return value
    else:
        raise TypeError(f'Invalid object reference type: {type(value)}.')


def compose_bvdict(input_dict: BoundedVariablesDictCompatible, state: Optional[TensorState] = None) -> BoundedVariablesDict:
    """Compose a bounded variable dict from raw inputs.

    Args:
        input_dict: the input dict. There are three types of inputs:

            1. A sequence of :class:`mover._vendor.concepts.dsl.dsl_types.Variable` instances, which represents a set of variables with no values.
            2. A dictionary mapping from :class:`mover._vendor.concepts.dsl.dsl_types.Variable` instances to the actual value.
            3. A dictionary mapping from strings (the name of the variables) to values.

            Acceptable values are:

            1. A :class:`str`, which represents a reference to an object in the state (so the state must be object-named).
            2. An integer, which represents a reference to an object in the state (so the state must be object-named).
            3. A QINDEX (a.k.a., ``slice(None)``), which represents all objects in the state of a given type (so the state must be object-named).
            4. A :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue` instance, which represents an actual value.
            5. A :class:`StateObjectReference` instance, which represents a reference to an object in the state (so the state must be object-named).
            6. A :class:`bool`, :class:`int`, :class:`float`, or :class:`torch.Tensor` instance, which represents an actual value. They will be converted to a :class:`mover._vendor.concepts.dsl.tensor_value.TensorValue` instance.

        state: the state.

    Returns:
        a dictionary mapping from strings (the typename) to a dictionary mapping from strings (the name of the variables) to values.
    """

    if input_dict is None:
        return dict()

    if isinstance(input_dict, dict):
        if len(input_dict) == 0:
            return {}

        sample_value = next(iter(input_dict.values()))
        if isinstance(sample_value, dict):
            return {k: v.copy() for k, v in input_dict.items()}

        output_dict = dict()
        for var, value in input_dict.items():
            if isinstance(var, Variable):
                # Part 1: the variable corresponds to an object.
                if isinstance(var.dtype, ObjectType):
                    output_dict.setdefault(var.typename, dict()).setdefault(var.name, _get_state_object_reference(state, var.dtype, value))
                elif isinstance(var.dtype, ListType):
                    assert isinstance(value, ListValue)
                    output_dict.setdefault(var.dtype.typename, {})[var.name] = value
                # Part 2: the variable corresponds to a Python object.
                elif isinstance(var.dtype, PyObjValueType):
                    if isinstance(value, TensorValue):
                        pass
                    else:
                        value = TensorValue.from_scalar(value, var.dtype)
                    typename = var.dtype.typename
                    output_dict.setdefault(typename, {})[var.name] = value
                # Part 3: the variable corresponds to a PyTorch tensor.
                elif isinstance(var.dtype, TensorValueTypeBase):
                    if isinstance(value, TensorValue):
                        pass
                    elif isinstance(value, (bool, int, float, torch.Tensor)):
                        value = TensorValue.from_scalar(value, var.dtype)
                    elif isinstance(value, UnnamedPlaceholder):
                        value = TensorValue.from_optimistic_value_int(OPTIM_MAGIC_NUMBER, var.dtype)  # Just a placeholder.
                    else:
                        raise TypeError(f'Invalid value type for variable {var}: {type(value)}.')
                    output_dict.setdefault(var.dtype.typename, {})[var.name] = value
                elif isinstance(var.dtype, ListType):
                    assert isinstance(value, ListValue)
                    if isinstance(var.dtype.element_type, ObjectType):
                        if value.values == QINDEX:
                            pass
                        else:
                            value = ListValue(var.dtype, [_get_state_object_reference(state, var.dtype.element_type, v) for v in value.values])
                    else:
                        pass
                    output_dict.setdefault(var.dtype.typename, {})[var.name] = value
                elif isinstance(var.dtype, BatchedListType):
                    assert isinstance(value, TensorValue)
                    output_dict.setdefault(var.dtype.typename, {})[var.name] = value
                else:
                    raise TypeError(f'Invalid variable type: {var.dtype}.')
            elif isinstance(var, OptimisticValue):
                raise RuntimeError('Invalid branch; OptimisticValue should be handled in the previous branch. Report a bug to the developers.')
            elif isinstance(var, str) and isinstance(value, str):
                assert state is not None
                typename, value_index = state.get_typename(value), state.get_typed_index(value)
                value = StateObjectReference(value, value_index)
                output_dict.setdefault(typename, dict()).setdefault(var, value)
            elif isinstance(var, str) and isinstance(value, ObjectConstant):
                assert state is not None
                typename = value.typename
                value_index = state.get_typed_index(value.name, typename)
                value = StateObjectReference(value.name, value_index, value.dtype)
                output_dict.setdefault(typename, dict()).setdefault(var, value)
            elif isinstance(var, str) and isinstance(value, StateObjectReference):
                assert state is not None
                assert value.dtype is not None
                output_dict.setdefault(value.dtype.typename, dict()).setdefault(var, value)
            elif isinstance(var, str) and isinstance(value, ListValue):
                output_dict.setdefault(value.dtype.typename, dict()).setdefault(var, value)
            elif isinstance(var, str) and isinstance(value, TensorValue):
                output_dict.setdefault(value.dtype.typename, dict()).setdefault(var, value)
            else:
                raise TypeError(f'Invalid KV pair: {var} -> {value}.')
        return output_dict
    elif isinstance(input_dict, (list, tuple)):
        # The input dict is a list of variables.
        assert isinstance(input_dict, (list, tuple))
        output_dict = dict()
        for var in input_dict:
            assert isinstance(var, Variable)
            output_dict.setdefault(var.typename, dict()).setdefault(var.name, QINDEX)
        return output_dict
    else:
        raise TypeError(f'Invalid input type: {type(input_dict)}.')


def compose_bvdict_args(arguments_def: Sequence[Variable], arguments: Sequence[BoundedVariablesDictCompatibleValueType], state: Optional[TensorState] = None) -> BoundedVariablesDict:
    """Compose a bounded variable dict, but from a list of arguments. This function is useful when we want to compose a bounded variable dict from a list of arguments to a function.

    Args:
        arguments_def: the definition of the arguments, including their name and dtypes.
        arguments: the actual arguments.
        state: the state.

    Returns:
        a bounded variable dictionary.
    """
    return compose_bvdict(dict(zip(arguments_def, arguments)), state=state)


def get_bvdict(bvdict: BoundedVariablesDict, variable: Variable) -> Union[StateObjectReference, slice, TensorValue]:
    """Get the value of a variable from a bounded variable dict.

    Args:
        bvdict: the bounded variable dict.
        variable: the variable.

    Returns:
        the value of the variable.
    """
    return bvdict[variable.typename][variable.name]


TensorValueExecutorReturnTypeElem = Union[TensorValue, slice, StateObjectReference, ListValue, None]
TensorValueExecutorReturnType = Union[TensorValueExecutorReturnTypeElem, Tuple[TensorValueExecutorReturnTypeElem, ...]]


class TensorValueExecutorBase(DSLExecutorBase):
    """The base class for tensor value executors."""

    def __init__(self, domain: DSLDomainBase, parser: Optional[ParserBase] = None):
        """Initialize the base class for tensor value executors.

        Args:
            domain: the domain of the executor.
            parser: the parser to use. If None, no parser will be used.
        """
        super().__init__(domain)

        self._parser = parser

        self._state = None
        self._bounded_variables = dict()

    @property
    def parser(self) -> Optional[ParserBase]:
        """The parser for the domain."""
        return self._parser

    @property
    def state(self) -> Optional[TensorState]:
        """The current state of the environment."""
        return self._state

    @property
    def bounded_variables(self) -> BoundedVariablesDict:
        """The bounded variables for the execution. Note that most of the time you should use the :meth:`get_bounded_variable` method to get values for the bounded variable."""
        return self._bounded_variables

    @property
    def value_quantizer(self) -> ValueQuantizer:
        """The value quantizer."""
        return self._value_quantizer

    @property
    def pyobj_store(self) -> PyObjectStore:
        """The Python object store."""
        return self._pyobj_store

    @contextlib.contextmanager
    def with_state(self, state: Optional[TensorState] = None):
        """A context manager to temporarily set the state of the executor."""
        old_state = self._state
        self._state = state
        yield
        self._state = old_state

    @contextlib.contextmanager
    def with_bounded_variables(self, bvdict: BoundedVariablesDictCompatible, bypass_bounded_variable_check: bool = False):
        """A context manager to set the bounded variables for the executor.

        Args:
            bvdict: the bounded variables.
            bypass_bounded_variable_check: whether to bypass the check for the bounded variables. If True, the input bounded variables will be used directly without any modification. Otherwise, the input bounded variables will be composed with the current state of the executor.
        """
        old_bvdict = self._bounded_variables
        if not bypass_bounded_variable_check:
            bvdict = compose_bvdict(bvdict, state=self._state)
        self._bounded_variables = bvdict
        yield
        self._bounded_variables = old_bvdict

    @contextlib.contextmanager
    def new_bounded_variables(self, bvdict: BoundedVariablesDictCompatible):
        """A context manager to add additional bounded variables to the executor.

        Args:
            bvdict: the new bounded variables.
        """
        bvdict = compose_bvdict(bvdict, state=self._state)
        for typename, variables in bvdict.items():
            for name, value in variables.items():
                if typename not in self._bounded_variables:
                    self._bounded_variables[typename] = dict()
                assert name not in self._bounded_variables[typename], f'Variable {name} already exists in bounded variables.'
                self._bounded_variables[typename][name] = value
        yield
        for typename, variables in bvdict.items():
            for name in variables:
                del self._bounded_variables[typename][name]

    def retrieve_bounded_variable_by_name(self, name: str) -> Union[TensorValue, slice, StateObjectReference]:
        """Retrieve a bounded variable by its name.

        Args:
            name: the name of the variable.

        Returns:
            the value of the variable.
        """
        for variables in self._bounded_variables.values():
            if name in variables:
                return variables[name]
        raise KeyError(f'Variable {name} not found in the bounded variables.')

    def get_bounded_variable(self, variable: Variable) -> Union[TensorValue, slice, StateObjectReference]:
        """Get the value of a bounded variable.

        Args:
            variable: the variable.

        Returns:
            the value of the variable.
        """
        return get_bvdict(self._bounded_variables, variable)

    def set_parser(self, parser: ParserBase):
        """Set the parser for the executor.

        Args:
            parser: the parser.
        """
        self._parser = parser

    def parse(self, expression: Union[Expression, str]):
        """Parse an expression.

        Args:
            expression: the expression to parse. When the input is already an expression, it will be returned directly.

        Returns:
            the parsed expression.
        """
        if isinstance(expression, Expression):
            return expression
        if self._parser is None:
            raise ValueError('No parser is set for the executor.')
        return self._parser.parse_expression(expression)

    def execute(
        self, expression: Union[Expression, str],
        state: Optional[TensorState] = None,
        bounded_variables: Optional[BoundedVariablesDictCompatible] = None,
    ) -> TensorValueExecutorReturnType:
        """Execute an expression.

        Args:
            expression: the expression to execute.
            state: the state to use. If None, the current state of the executor will be used.
            bounded_variables: the bounded variables to use. If None, the current bounded variables of the executor will be used.

        Returns:
            the TensorValue object.
        """
        if isinstance(expression, str):
            expression = self.parse(expression)

        state = state if state is not None else self._state
        bounded_variables = bounded_variables if bounded_variables is not None else self._bounded_variables
        with self.with_state(state), self.with_bounded_variables(bounded_variables):
            return self._execute(expression)

    def _execute(self, expression: Expression) -> TensorValueExecutorReturnType:
        raise NotImplementedError()

    def check_constraint(self, constraint: Constraint, state: Optional[TensorState] = None):
        if constraint.function is BoolOpType.NOT:
            return constraint.arguments[0].item() == (not constraint.rv.item())
        elif constraint.function in (QuantificationOpType.FORALL, BoolOpType.AND):
            return all([x.item() for x in constraint.arguments]) == constraint.rv.item()
        elif constraint.function in (QuantificationOpType.EXISTS, BoolOpType.OR):
            return any([x.item() for x in constraint.arguments]) == constraint.rv.item()
        elif constraint.function is BoolOpType.IMPLIES:
            return (not constraint.arguments[0].item()) or constraint.arguments[1].item() == constraint.rv.item()
        elif constraint.function is BoolOpType.XOR:
            return sum([x.item() for x in constraint.arguments]) % 2 == constraint.rv.item()

        if constraint.is_equal_constraint:
            if constraint.arguments[0].dtype in (BOOL, INT64, FLOAT32):
                return (constraint.arguments[0].item() == constraint.arguments[1].item()) == constraint.rv.item()
            else:
                return self.check_eq_constraint(constraint.arguments[0].dtype, constraint.arguments[0], constraint.arguments[1], constraint.rv.item(), state)


        if isinstance(constraint.function, SimulationFluentConstraintFunction):
            return False

        # assert isinstance(c.function, CrowFunctionBase)
        # # NB(Jiayuan Mao @ 09/05): for generator placeholders, they can only be set true through the corresponding generators.
        # if isinstance(c.function, CrowFunction) and c.function.is_generator_placeholder:
        #     return False

        argument_values = list()
        for argument, argv in zip(constraint.function.arguments, constraint.arguments):
            if isinstance(argument.dtype, ObjectType):
                assert isinstance(argv, StateObjectReference)
                argument_values.append(ObjectConstantExpression(ObjectConstant(argv, argument.dtype)))
            elif isinstance(argument.dtype, ValueType):
                argument_values.append(ConstantExpression(argv, argument.dtype))
            else:
                raise TypeError(f'Unsupported argument type: {argument.dtype}.')

        func = FunctionApplicationExpression(constraint.function, argument_values)
        with self.with_state(state):
            rv = self._execute(func)
        if rv.dtype == BOOL:
            return (rv.item() > 0.5) == constraint.rv.item()
        else:
            return self.check_eq_constraint(rv.dtype, rv, constraint.rv.item(), True, state)

    def check_eq_constraint(self, dtype: TensorValueTypeBase, x: TensorValue, y: TensorValue, target: bool, state: Optional[TensorState] = None) -> bool:
        expr = ValueCompareExpression(ValueCompareExpression.OpType.EQ, ConstantExpression(x, dtype), ConstantExpression(y, dtype))
        with self.with_state(state):
            return self._execute(expr).item() == target


class FunctionDomainTensorValueExecutor(TensorValueExecutorBase):
    """Similar to :class:`~mover._vendor.concepts.dsl.executors.function_domain_executor.FunctionDomainExecutor`, but works for :class:`~mover._vendor.concepts.dsl.tensor_value.TensorValue`.
    The two of the main differences are:

    1. The :meth:`execute` method returns a :class:`~mover._vendor.concepts.dsl.tensor_value.TensorValue` object instead of a :class:`~mover._vendor.concepts.dsl.value.Value` object.
    2. The class supports binding variables to values during execution. See the documentation for this file and tutorials for details.
    """

    def __init__(self, domain: FunctionDomain, parser: Optional[ParserBase] = None):
        """Initialize a tensor value executor for a function domain.

        Args:
            domain: the domain of the executor.
            parser: the parser to use. If not specified, no parser will be used.
        """

        if parser is None:
            parser = FunctionExpressionParser(domain, allow_variable=True, escape_string=True)

        super().__init__(domain, parser)

    _domain: FunctionDomain

    @property
    def domain(self) -> FunctionDomain:
        """The function domain of the executor."""
        return self._domain

    def _execute(self, expr: Expression) -> TensorValueExecutorReturnType:
        if isinstance(expr, VariableExpression):
            variable = expr.variable
            return self._bounded_variables[variable.dtype.typename][variable.name]
        elif isinstance(expr, ObjectConstantExpression):
            if isinstance(expr.constant.name, StateObjectReference):
                return expr.constant.name
            assert isinstance(self._state, NamedObjectTensorState)
            constant = expr.constant
            return StateObjectReference(
                constant.name,
                self._state.get_typed_index(constant.name, constant.dtype.typename),
                constant.dtype
            )
        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.constant, TensorValue)
            return expr.constant
        elif isinstance(expr, FunctionApplicationExpression):
            assert isinstance(self._state, NamedObjectTensorState)
            func = expr.function
            args = [self._execute(arg) for arg in expr.arguments]
            if func.name in self._state.features:
                args = [arg.index if isinstance(arg, StateObjectReference) else arg for arg in args]
                return self._state.features[func.name][tuple(args)]
            else:
                assert self.has_function_implementation(func.name)
                return self.get_function_implementation(func.name)(*args)
        else:
            raise ValueError(f'Unsupported expression type: {type(expr)}.')

