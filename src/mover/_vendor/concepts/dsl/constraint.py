#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : constraint.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/10/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures for Constraint Satisfaction Problems (CSPs).

In general, a CSP can be represented as a tuple of variables, domains (for each variable), and constraints.
Here, we only consider constraints of the following form: ``function(*arguments) == rv``. Internally:

- Each variable can either be a determined :class:`~mover._vendor.concepts.dsl.tensor_value.TensorValue` or an :class:`OptimisticValue` (i.e., undetermined value).
    For :class:`TensorValue`, the representation contains its data type and the actual value.
    For :class:`OptimisticValue`, the representation contains its data type and a single integer, representing the index of the variable.
- Each constraint is represented as a :class:`Constraint` object, which contains the function, arguments, and the return value.
    The function is represented as a string (only for equality constraints), or BoolOpType (and, or, not), or QuantificationOpType (forall, exists),
        or :class:`~mover._vendor.concepts.dsl.dsl_functions.Function` objects. Both arguments and return values can be :class:`DeterminedValue` or :class:`OptimisticValue`.
- The index of optimistic values starts from -2147483647 (i.e., ``OPTIM_MAGIC_NUMBER``). Therefore, they can be directly stored in tensors without any conversion,
    which is helpful for implementing CSP-based expression executors. There are a few helper functions to convert between optimistic values and their indices.
"""

import contextlib
from enum import IntEnum
from dataclasses import dataclass
from os import times
from typing import Any, Optional, Union, Sequence, Tuple, List, Set, Dict

import torch
import jacinle
from jacinle.utils.printing import indent_text
from jacinle.utils.enum import JacEnum

from mover._vendor.concepts.dsl.dsl_types import BOOL, INT64, TensorValueTypeBase, PyObjValueType, ValueType, Variable
from mover._vendor.concepts.dsl.dsl_functions import Function
from mover._vendor.concepts.dsl.tensor_value import TensorValue
from mover._vendor.concepts.dsl.tensor_state import StateObjectReference
from mover._vendor.concepts.dsl.expression import BoolOpType, QuantificationOpType, ValueOutputExpression

__all__ = [
    'OPTIM_MAGIC_NUMBER', 'OPTIM_MAGIC_NUMBER_UPPER', 'OPTIM_MAGIC_NUMBER_MAGIC',
    'is_optimistic_value', 'optimistic_value_id', 'is_optimistic_value',
    'OptimisticValue', 'cvt_opt_value', 'SimulationFluentConstraintFunction',
    'Constraint', 'EqualityConstraint', 'GroupConstraint', 'OptimisticValueRecord', 'ConstraintSatisfactionProblem', 'NamedConstraintSatisfactionProblem',
    'AssignmentType', 'Assignment', 'AssignmentDict', 'print_assignment_dict', 'ground_assignment_value'
]


OPTIM_MAGIC_NUMBER = -2147483647
OPTIM_MAGIC_NUMBER_UPPER = (-2147483648) / 2
OPTIM_MAGIC_NUMBER_MAGIC = -2147483648


def is_optimistic_value(v: Union[int, torch.Tensor]):
    """Check if a value is an optimistic identifier."""
    return v < OPTIM_MAGIC_NUMBER_UPPER


def optimistic_value_id(v: Union[int, 'OptimisticValue']):
    """Get the optimistic identifier from an optimistic value."""
    if isinstance(v, OptimisticValue):
        return v.identifier - OPTIM_MAGIC_NUMBER
    return v - OPTIM_MAGIC_NUMBER


def maybe_optimistic_string(v: int):
    """If `v` is an optimistic value, return a string representation of the optimistic identifier, otherwise return `str(v)`."""
    if is_optimistic_value(v):
        return '@' + str(optimistic_value_id(v))
    return str(v)


class OptimisticValue(object):
    """An optimistic value object holds a pair of data type and an optimistic identifier."""

    def __init__(self, dtype: ValueType, identifier: int):
        """Initialize the OptimisticValue object.

        Args:
            dtype: the type of the value.
            identifier: the optimistic identifier.
        """

        if not isinstance(dtype, ValueType) and dtype != BOOL and dtype != INT64:
            raise TypeError('OptimisticValue only supports NamedTensorValueType, PyObjValueType, BOOL, and INT64.')

        self.dtype = dtype
        self.identifier = identifier
        self.__post_init__()

    dtype: ValueType
    """The dtype of the optimistic value."""

    identifier: int
    """The optimistic identifier."""

    # TODO(Jiayuan Mao @ 2023/08/16): remove this check.
    def __post_init__(self):
        assert isinstance(self.identifier, int)

    def __str__(self) -> str:
        return f'O[{self.dtype}]{{@{optimistic_value_id(self.identifier)}}}'

    def __repr__(self) -> str:
        return self.__str__()


class GroupTypeClass(object):
    pass


GroupType = GroupTypeClass()


class OptimisticValueRecord(object):
    def __init__(
        self, identifier: int, dtype: Union[TensorValueTypeBase, PyObjValueType, GroupTypeClass],
        domain: Optional[Set[Any]] = None, actionable: bool = False,
        name: Optional[str] = None, is_constant: bool = False,
        group_id: int = 0
    ):
        self.identifier = identifier
        self.dtype = dtype
        self.domain = domain
        self.actionable = actionable
        self.name = name
        self.is_constant = is_constant
        self.group_id = group_id

    identifier: int
    """The identifier of the optimistic value."""

    dtype: Union[TensorValueTypeBase, PyObjValueType, GroupTypeClass]
    """The dtype of the optimistic value."""

    domain: Optional[Set[Any]]
    """The domain of the optimistic value."""

    actionable: bool
    """Whether the variable is actionable."""

    name: Optional[str]
    """The name of the variable."""

    is_constant: bool
    """Whether the variable is constant."""

    group_id: int
    """The group id of the variable. -1 if the variable is not a group."""

    @property
    def is_group(self) -> bool:
        """Check if the optimistic value is a group."""
        return isinstance(self.dtype, GroupTypeClass)

    @property
    def is_in_group(self) -> bool:
        return not self.is_group and self.group_id != 0


def cvt_opt_value(value: Union[OptimisticValue, TensorValue, StateObjectReference, bool, int], dtype: Optional[TensorValueTypeBase] = None) -> Union[TensorValue, StateObjectReference, OptimisticValue]:
    """Convert a value to OptimisticValue or a (determined) TensorValue. Acceptable types are:

    - OptimisticValue
    - TensorValue, which will be converted to OptimisticValue if it corresponds to an optimistic value.
    - bool, which will be converted to TensorValue.
    - int, which will be converted to OptimisticValue if it is an optimistic identifier, otherwise TensorValue.

    Args:
        value: the value to be converted.
        dtype: the data type of the value. If not specified (and the input is a TensorValue), the data type of the value will be used.

    Returns:
        The converted value.
    """

    if isinstance(value, OptimisticValue):
        return value
    elif isinstance(value, TensorValue):
        assert dtype is None or value.dtype == dtype
        v = value.single_elem()
        if isinstance(v, OptimisticValue):
            return v
        return value
    elif isinstance(value, StateObjectReference):
        return value
    elif isinstance(value, bool):
        assert dtype is None or dtype == BOOL
        return TensorValue.from_scalar(value, BOOL)
    elif isinstance(value, int):
        if is_optimistic_value(value):
            assert dtype is not None, 'dtype must be specified for optimistic value'
            return OptimisticValue(dtype, value)
        if dtype == BOOL:
            raise RuntimeError('Should not use int to represent bool. This could be a bug in the code; report to the developers.')
            # return TensorValue.from_scalar(bool(value), BOOL)
        assert dtype == INT64
        return TensorValue.from_scalar(value, INT64)
    else:
        raise TypeError('Unknown value type: {} (type={}).'.format(value, type(value)))


@dataclass
class SimulationFluentConstraintFunction(object):
    """SimulationConstraint is a special kind of constraint that asserts the return value of the function is the grounding
    of a predicate after executing action ``state_index`` in the simulation. This is a special kind of constraint that
    has to be evaluated in a simulation environment. Therefore it is listed separately from the other domain-general constraints.
    """

    state_index: int
    """The index of the state in simulation."""

    predicate: Function
    """The predicate to be grounded."""

    arguments: Tuple[int, ...]
    """The arguments to the predicate."""

    is_execution_constraint: bool
    """The mode of the constraint. Can be one of the following: SIMULATION, EXECUTION. When it is simulation this flag is set to False."""

    @property
    def name(self) -> str:
        if self.is_execution_constraint:
            return f'ExecutionFluentConstraint(t={self.state_index}, {self.predicate.name}, args={self.arguments})'
        return f'SimulationFluentConstraint(t={self.state_index}, {self.predicate.name}, args={self.arguments})'


class Constraint(object):
    """A constraint is basically a equality expression: ``function(*arguments) == rv``."""

    EQUAL = '__EQ__'
    """Magic name for equality constraints."""

    def __init__(
        self, function: Union[str, BoolOpType, QuantificationOpType, Function, SimulationFluentConstraintFunction],
        arguments: Sequence[Union[TensorValue, StateObjectReference, OptimisticValue]],
        rv: Union[TensorValue, OptimisticValue],
        note: Any = None,
        group: Optional['GroupConstraint'] = None,
        timestamp: int = 0
    ):
        """Initialize a constraint. Each constraint takes the form of:

        .. code-block:: python

            function(*arguments) == rv

        Args:
            function: the function name, or a BoolOpType or QuantifierType, or a Function object.
            arguments: the arguments of the function.
            rv: the expected return value of the function.
            note: an optional note for the constraint.
        """

        self.function = function
        self.arguments = tuple(map(cvt_opt_value, arguments))
        self.rv = cvt_opt_value(rv)
        self.note = note
        self.group = group
        self.timestamp = timestamp

    function: Union[str, BoolOpType, QuantificationOpType, Function, SimulationFluentConstraintFunction]
    """The function identifier: either a string (currently only for equality constraints), or a BoolOpType or QuantificationOpType (for Boolean expressions), or a Function object."""

    arguments: Tuple[Union[TensorValue, StateObjectReference, OptimisticValue], ...]
    """The arguments to the function."""

    rv: Union[TensorValue, OptimisticValue]
    """The expected return value of the function."""

    note: Any
    """An optional note for the constraint."""

    group: Optional['GroupConstraint']
    """The group of the constraint. None if the constraint is not in a group."""

    timestamp: int
    """The timestamp of the constraint."""

    def constraint_str(self):
        """Return the string representation of the constraint."""
        argument_strings = [x.format(short=True) if isinstance(x, TensorValue) else str(x) for x in self.arguments]
        if self.is_equal_constraint and isinstance(self.rv, TensorValue):
            if self.rv.item():
                return f'__EQ__@{self.timestamp}(' + ', '.join(argument_strings) + ')'
            else:
                return f'__NEQ__@{self.timestamp}(' + ', '.join(argument_strings) + ')'
        else:
            if isinstance(self.function, (str, JacEnum)):
                name = str(self.function)
            else:
                name = self.function.name
            name = f'{name}@{self.timestamp}'
            return name + '(' + ', '.join(argument_strings) + ') == ' + (self.rv.format(short=True) if isinstance(self.rv, TensorValue) else str(self.rv))

    def __str__(self):
        ret = self.constraint_str()
        if self.note is not None:
            ret += '  # ' + str(self.note)
        if self.group is not None:
            ret += f' (group=g{hex(id(self.group))})'
        return ret

    __repr__ = jacinle.repr_from_str

    @property
    def is_group_constraint(self) -> bool:
        """Return True if the constraint is a group constraint."""
        return False

    @property
    def is_equal_constraint(self) -> bool:
        """Check if the constraint is an equality constraint."""
        return isinstance(self.function, str) and self.function == Constraint.EQUAL

    @property
    def is_in_group(self) -> bool:
        """Check if the constraint is in a group."""
        return self.group is not None

    def set_group(self, group: 'GroupConstraint'):
        """Set the group of the constraint."""
        assert self.group is None
        self.group = group

    @classmethod
    def from_function(cls, function: Union[Function], args: Sequence[Union[bool, int, torch.Tensor, TensorValue, Any]], rv: Union[bool, int, torch.Tensor, TensorValue, Any]) -> 'Constraint':
        """Create a constraint given a function, arguments, and return value.

        Args:
            function: the function object.
            args: the arguments of the function. The arguments can be bool, int, or torch.Tensor.
            rv: the return value of the function. The return value can be bool, int, or torch.Tensor.

        Returns:
            The created constraint.
        """
        _cvt = cvt_opt_value
        if isinstance(function, Function):
            args = [_cvt(x, var.dtype if isinstance(var, Variable) else var) for x, var in zip(args, function.arguments)]
            rv = _cvt(rv, function.return_type)
        else:
            args = [_cvt(x) for x in args]
            rv = _cvt(rv)
        return cls(function, args, rv)

    def replace_constants(self, assignments: Dict[int, TensorValue]) -> 'Constraint':
        def _cvt(x):
            if isinstance(x, OptimisticValue):
                return assignments.get(x.identifier, x)
            return x
        return Constraint(self.function, [_cvt(x) for x in self.arguments], _cvt(self.rv), self.note)


class GroupConstraint(object):
    def __init__(self, expression: ValueOutputExpression, note: Any = None):
        self.expression = expression
        self.candidate_generators = list()
        self.note = note

    expression: ValueOutputExpression
    """The expression of the group constraint."""

    candidate_generators: List[Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]]

    note: Any
    """An optional note for the constraint."""

    @property
    def is_group_constraint(self) -> bool:
        """Return True if the constraint is a group constraint."""
        return True

    def __str__(self):
        ret = 'GroupConstraint@' + hex(id(self)) + ': '
        ret += str(self.expression)
        if self.note is not None:
            ret += '  # ' + str(self.note)
        return ret

    __repr__ = jacinle.repr_from_str


class EqualityConstraint(Constraint):
    """A special constraint for equality constraints. It is equivalent to:

    .. code-block:: python

        Constraint(Constraint.EQUAL, [left, right], rv)

    Basically, it states

    .. code-block:: python

        (left == right) == rv

    Therefore, when rv is True, it states that left and right are equal, and when rv is False, it states that left and right are not equal.
    """

    def __init__(self, left: Union[TensorValue, OptimisticValue], right: Union[TensorValue, OptimisticValue], rv: Optional[Union[TensorValue, OptimisticValue]] = None, timestamp: int = 0):
        """Initialize an equality constraint.

        Args:
            left: the left hand side of the equality constraint.
            right: the right hand side of the equality constraint.
            rv: the expected return value of the equality constraint. If None, it will be set to True (i.e., left == right).
            timestamp: the timestamp of the constraint.
        """
        super().__init__(Constraint.EQUAL, [left, right], rv if rv is not None else TensorValue.TRUE, timestamp=timestamp)

    function: str
    """The function identifier, which is always ``Constraint.EQUAL``."""

    arguments: Tuple[Union[TensorValue, OptimisticValue], Union[TensorValue, OptimisticValue]]
    rv: Union[TensorValue, OptimisticValue]
    note: Any
    group: Optional['GroupConstraint']
    timestamp: int

    @classmethod
    def from_bool(cls, left: Union[bool, int, OptimisticValue], right: Union[bool, int, OptimisticValue], rv: Optional[Union[bool, int, OptimisticValue]] = None, timestamp: int = 0) -> 'EqualityConstraint':
        """Create an equality constraint from bool or optimistic values (represented as integers).

        Args:
            left: the left hand side of the equality constraint.
            right: the right hand side of the equality constraint.
            rv: the expected return value of the equality constraint. If None, it will be set to True (i.e., left == right).
            timestamp: the timestamp of the constraint.

        Returns:
            The created equality constraint.
        """

        def _cvt(x):
            if x is None:
                return x
            if isinstance(x, OptimisticValue):
                return x
            elif isinstance(x, bool):
                return cvt_opt_value(x, BOOL)
            else:
                assert isinstance(x, int) and is_optimistic_value(x)
                # TODO(Jiayuan Mao @ 2023/11/17): remove this.
                return OptimisticValue(BOOL, x)
        return cls(_cvt(left), _cvt(right), _cvt(rv), timestamp=timestamp)


class ConstraintSatisfactionProblem(object):
    """A constraint satisfaction problem.

    A constraint satisfaction problem is a set of constraints, and a set of variables.
    The solution to a constraint satisfaction problem is to find a set of values for the variables that satisfy all the constraints.
    """

    def __init__(
        self,
        index2record: Optional[Dict[int, OptimisticValueRecord]] = None,
        index_order: Optional[List[Tuple[int, int]]] = None,
        constraints: Optional[List[Union[Constraint, GroupConstraint]]] = None,
        counter: int = 0,
        state_timestamp: int = 0
    ):
        """Initialize a constraint satisfaction problem.

        Args:
            index2record: a mapping from variable indices to the variable objects.
            index_order: a list of tuples (index, index), indicating that the first index should be solved earlier than the second index.
            constraints: a list of constraints.
            counter: the counter for generating new variable indices.
        """

        self.index2record = index2record if index2record is not None else dict()
        self.index_order = index_order if index_order is not None else list()
        self.constraints = constraints if constraints is not None else list()
        self._optim_var_counter = counter
        self._current_group = None
        self._state_timestamp = state_timestamp

    index2record: Dict[int, OptimisticValueRecord]
    """A mapping from variable indices to the variable objects."""

    index_order: List[Tuple[int, int]]
    """A list of tuples (index, index), indicating that the first index should be solved earlier than the second index."""

    constraints: List[Union[Constraint, GroupConstraint]]
    """A list of constraints."""

    _current_group: Optional['GroupConstraint']
    """A temporary variable that can be used to annotate the current group. It will be triggered by the `with_group` context manager."""

    _state_timestamp: int
    """The timestamp of the current state. When we create a new constraint, we will use this timestamp."""

    def empty(self):
        return len(self.constraints) == 0

    def get_state_timestamp(self):
        return self._state_timestamp

    def increment_state_timestamp(self):
        self._state_timestamp += 1

    def set_state_timestamp(self, timestamp: int):
        self._state_timestamp = timestamp

    def clone(self, constraints: Optional[List[Constraint]] = None) -> 'ConstraintSatisfactionProblem':
        """Clone the constraint satisfaction problem.

        Args:
            constraints: the constraints to be replaced into the cloned constraint satisfaction problem. If None, the constraints of the original constraint satisfaction problem will be used.
        """

        if constraints is None:
            constraints = self.constraints.copy()
        return type(self)(self.index2record.copy(), self.index_order.copy(), constraints, self._optim_var_counter, self._state_timestamp)

    def new_actionable_var(self, dtype: Union[TensorValueTypeBase, PyObjValueType], wrap: bool = False, **kwargs) -> Union[int, OptimisticValue]:
        """Create a new actionable variable.

        Args:
            dtype: the type of the variable.
            wrap: whether to wrap the variable index into an OptimisticValue.
        """
        return self.new_var(dtype, actionable=True, wrap=wrap, **kwargs)

    def new_var(
        self, dtype: Union[TensorValueTypeBase, PyObjValueType], domain: Optional[Set[Any]] = None, actionable: bool = False,
        name: Optional[str] = None, is_constant: bool = False,
        wrap: bool = False
    ) -> Union[int, OptimisticValue]:
        """Create a new variable.

        Args:
            dtype: the type of the variable.
            domain: the domain of the variable. If None, it will be assumed to the full domain of the type.
            actionable: whether the variable is actionable.
            name: the name of the variable.
            is_constant: whether the variable is constant.
            wrap: whether to wrap the variable index into an OptimisticValue.

        Returns:
            The index of the new variable (int) if wrap is False, or the wrapped OptimisticValue if wrap is True.
        """

        identifier = OPTIM_MAGIC_NUMBER + self._optim_var_counter
        self.index2record[identifier] = OptimisticValueRecord(identifier, dtype, domain=domain, actionable=actionable, name=name, is_constant=is_constant)
        self._optim_var_counter += 1

        if wrap:
            return OptimisticValue(dtype, identifier)
        return identifier

    def get_type(self, identifier: int) -> Union[TensorValueTypeBase, PyObjValueType]:
        """Get the type of a variable."""
        return self.index2record[identifier].dtype

    def get_domain(self, identifier: int) -> Optional[Set[Any]]:
        """Get the domain of a variable."""
        return self.index2record[identifier].domain

    @contextlib.contextmanager
    def with_group(self, expression: ValueOutputExpression, note: Any = None):
        """Add a group constraint."""
        group = GroupConstraint(expression, note)
        self.constraints.append(group)
        assert self._current_group is None
        self._current_group = group
        yield group
        self._current_group = None

    def add_domain_value(self, identifier: int, value: Any):
        """Add a value to the domain of a variable."""
        variable = self.index2record[identifier]
        if variable.domain is None:
            variable.domain = set()
        variable.domain.add(value)

    def add_constraint(self, c: Constraint, note: Optional[Any] = None):
        """Add a constraint to the constraint satisfaction problem.

        Args:
            c: the constraint to be added.
            note: the note of the constraint.
        """
        if note is not None:
            c.note = note
        if self._current_group is not None:
            c.set_group(self._current_group)
        c.timestamp = self._state_timestamp
        self.constraints.append(c)
        return self

    def add_equal_constraint(self, left: Union[TensorValue, OptimisticValue], right: Optional[Union[TensorValue, OptimisticValue, bool]] = None, note: Optional[Any] = None):
        """Add an equality constraint.

        Args:
            left: the left hand side of the equality constraint.
            right: the right hand side of the equality constraint.
            note: the note of the constraint.
        """
        if right is None:
            right = TensorValue.TRUE
        if isinstance(right, bool):
            right = TensorValue.TRUE if right else TensorValue.FALSE
        self.add_constraint(EqualityConstraint(left, right), note=note)
        return self

    def __str__(self) -> str:
        fmt = 'ConstraintSatisfactionProblem{\n'
        fmt += '  Variables:'
        for _, variable in sorted(self.index2record.items(), key=lambda x: x[0]):
            fmt += f'\n    @{variable.identifier - OPTIM_MAGIC_NUMBER} - {variable.dtype} (actionable={variable.actionable})'
        fmt += '\n'
        fmt += '  Constraints:\n'
        for c in self.constraints:
            fmt += f'{indent_text(str(c), 2)}\n'
        fmt += '}'
        return fmt

    def __repr__(self):
        return self.__str__()

    def print_assignment_dict(self, assignments: 'AssignmentDict'):
        print_assignment_dict(assignments, self)

    def ground_assignment_value(self, assignments: 'AssignmentDict', identifier: int) -> Any:
        return ground_assignment_value(assignments, identifier, self)

    def ground_assignment_value_partial(self, assignments: 'AssignmentDict', identifier: int) -> Any:
        return ground_assignment_value(assignments, identifier, self, allow_partial=True)


class NamedConstraintSatisfactionProblem(ConstraintSatisfactionProblem):
    def __init__(
        self,
        index2record: Optional[Dict[int, OptimisticValueRecord]] = None,
        name2optimistic_value: Optional[Dict[str, OptimisticValue]] = None,
        index_order: Optional[List[Tuple[int, int]]] = None,
        constraints: Optional[List[Constraint]] = None,
        counter: int = 0,
        state_timestamp: int = 0
    ):
        super().__init__(index2record, index_order, constraints, counter, state_timestamp)
        self.name2optimistic_value = name2optimistic_value if name2optimistic_value is not None else dict()

    index2record: Dict[int, OptimisticValueRecord]
    index_order: List[Tuple[int, int]]
    constraints: List[Constraint]

    name2optimistic_value: Dict[str, OptimisticValue]
    """A mapping from variable names to variable indices."""

    def get_name(self, identifier: int, default: str) -> str:
        if identifier in self.index2record:
            if self.index2record[identifier].name is not None:
                return self.index2record[identifier].name
        return default

    def clone(self, constraints: Optional[List[Constraint]] = None) -> 'ConstraintSatisfactionProblem':
        """Clone the constraint satisfaction problem.

        Args:
            constraints: the constraints to be replaced into the cloned constraint satisfaction problem. If None, the constraints of the original constraint satisfaction problem will be used.
        """

        if constraints is None:
            constraints = self.constraints.copy()
        return type(self)(
            index2record=self.index2record.copy(),
            name2optimistic_value=self.name2optimistic_value.copy(),
            index_order=self.index_order.copy(),
            constraints=constraints,
            counter=self._optim_var_counter,
            state_timestamp=self._state_timestamp
        )

    def new_constant_var(self, name: str, dtype: Union[TensorValueTypeBase, PyObjValueType], domain: Optional[Set[Any]] = None, wrap: bool = False) -> Union[int, OptimisticValue]:
        identifier = self.new_var(dtype, domain=domain, name=name, is_constant=True, wrap=False)
        self.name2optimistic_value[name] = opt_value = OptimisticValue(dtype, identifier)
        if wrap:
            return opt_value
        return identifier

    def new_named_actionable_var(self, name: str, dtype: Union[TensorValueTypeBase, PyObjValueType], domain: Optional[Set[Any]] = None, wrap: bool = False) -> Union[int, OptimisticValue]:
        identifier = self.new_actionable_var(dtype, domain=domain, name=name, wrap=False)
        self.name2optimistic_value[name] = opt_value = OptimisticValue(dtype, identifier)
        if wrap:
            return opt_value
        return identifier

    @property
    def all_constants_resolved(self) -> bool:
        return not any(self.index2record[x.identifier].is_constant for x in self.name2optimistic_value.values())

    def replace_constants(self, assignments: Dict[str, TensorValue]) -> 'NamedConstraintSatisfactionProblem':
        constant2index = {k: v.identifier for k, v in self.name2optimistic_value.items() if self.index2record[v.identifier].is_constant}
        assignments = {constant2index[k]: v for k, v in assignments.items()}
        return NamedConstraintSatisfactionProblem(
            index2record=self.index2record.copy(),
            name2optimistic_value={k: v for k, v in self.name2optimistic_value.items() if v.identifier not in assignments},
            index_order=self.index_order.copy(),
            constraints=[c.replace_constants(assignments) for c in self.constraints],
            counter=self._optim_var_counter
        )

    def ground_assignment_value(self, assignments: 'AssignmentDict', identifier: Union[int, str]) -> Any:
        return ground_assignment_value(assignments, identifier, self)


class AssignmentType(IntEnum):
    """See class `Assignment`."""
    EQUAL = 0
    VALUE = 1
    IGNORE = 2


@dataclass
class Assignment(object):
    """An assignment of a variable."""

    t: AssignmentType
    """the type of the assignment. There are three types:

        EQUAL: The variable is equal to another variable.
        VALUE: The variable is equal to a value.
        IGNORE: The variable is ignored. This happens when the variable does not appear in any constraint.
    """

    d: Union[bool, int, None, TensorValue]
    """the value of the variable.

        - EQUAL: The variable is equal to another variable.
        - VALUE: The variable is equal to a value.
        - IGNORE: The variable is ignored. This happens when the variable does not appear in any constraint.
    """

    generator_index: Optional[Tuple[str, int]] = None

    # TODO(Jiayuan Mao @ 2023/08/16): remove this check.
    def __post_init__(self):
        if self.t == AssignmentType.EQUAL:
            assert isinstance(self.d, int)
        elif self.t == AssignmentType.VALUE:
            assert isinstance(self.d, (TensorValue, bool)), f'Invalid type of d: {type(self.d)}'
            if isinstance(self.d, TensorValue) and self.d.is_single_elem and self.d.dtype == BOOL:
                self.d = bool(self.d.item())
        else:
            assert self.d is None

    @property
    def assignment_type(self):
        """Alias of `assignment.t`."""
        return self.t

    @property
    def data(self):
        """Alias of `assignment.d`."""
        return self.d


AssignmentDict = Dict[int, Assignment]
"""A mapping from variable indices to assignment values."""


def print_assignment_dict(assignments: AssignmentDict, csp: Optional[ConstraintSatisfactionProblem] = None):
    """Print an assignment dictionary.

    Args:
        assignments: the assignment dictionary.
        csp: the constraint satisfaction problem.
    """

    def _name(x):
        default_name = f'@{optimistic_value_id(x)}'
        if csp is not None and isinstance(csp, NamedConstraintSatisfactionProblem):
            return csp.get_name(x, default_name)
        return default_name

    print('AssignmentDict{')
    for k, v in assignments.items():
        # while v.t is AssignmentType.EQUAL and v.d in assignments:
        #     v = assignments[v.d]
        if v.t is AssignmentType.VALUE:
            print('  {}: {}'.format(_name(k), v.d))
        elif v.t is AssignmentType.EQUAL:
            print('  {}: {}'.format(_name(k), _name(v.d)))
        elif v.t is AssignmentType.IGNORE:
            print('  {}: {}'.format(_name(k), 'IGNORE'))
        else:
            raise ValueError(f'Unknown assignment type: {v.t}')
    print('}')


def ground_assignment_value(assignments: AssignmentDict, identifier: Union[int, str], csp: Optional[ConstraintSatisfactionProblem] = None, allow_partial: bool = False) -> Any:
    """Get the value of a variable based on the assignment dictionary. It will follow the EQUAL assignment types.

    Args:
        assignments: the assignment dictionary.
        identifier: the identifier of the variable.
        csp: the constraint satisfaction problem. If None, the identifier must be an integer.
        allow_partial: whether to allow partial assignment. If True, the function will return an optimistic value if the variable is not assigned.

    Returns:
        the value of the variable in the assignment dict.
    """
    if isinstance(identifier, str):
        if csp is None or not isinstance(csp, NamedConstraintSatisfactionProblem):
            raise RuntimeError('Cannot ground a variable by name without a NamedConstraintSatisfactionProblem.')
        if identifier not in csp.name2optimistic_value:
            raise KeyError(f'Unknown variable name: {identifier}')
        identifier = csp.name2optimistic_value[identifier].identifier

    while identifier in assignments and assignments[identifier].t is AssignmentType.EQUAL:
        identifier = assignments[identifier].d
    if not allow_partial:
        assert identifier in assignments
    else:
        if identifier not in assignments:
            return OptimisticValue(csp.get_type(identifier), identifier)
    return assignments[identifier].d

