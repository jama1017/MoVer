#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dsl_functions.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/19/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Data structures representing functions in a DSL.

Most importantly, this file contains the following classes:

- :class:`FunctionType`: the underlying type of a function, including argument types and return types.
- :class:`Function`: the function object, which is a callable object that can be used in expressions. They have names and types.

This file also implements a data structure for overloaded functions: :class:`OverloadedFunctionType`.
Internally, it contains a list of :class:`FunctionType` objects, and it is used to represent overloaded functions.
There are a few argument resolution methods implemented for both :class:`FunctionType` and :class:`OverloadedFunctionType`.
"""

import itertools
import contextlib
import inspect
import re
from typing import TYPE_CHECKING, Any, Union, Sequence, Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

import jacinle
from jacinle.utils.cache import cached_property
from jacinle.utils.defaults import option_context
from jacinle.utils.printing import indent_text

from mover._vendor.concepts.dsl.dsl_types import TypeBase, ObjectType, ValueType, SequenceType, TupleType, UniformSequenceType, ConstantType, UnionType, Variable
from mover._vendor.concepts.dsl.dsl_types import FormatContext, get_format_context

if TYPE_CHECKING:
    from mover._vendor.concepts.dsl.expression import Expression, FunctionApplicationExpression

__all__ = [
    'FunctionArgumentResolutionError', 'FunctionArgumentResolutionContext', 'get_function_argument_resolution_context',
    'FunctionArgumentUnset', 'AnonymousFunctionArgumentGenerator',
    'FunctionArgumentType', 'FunctionArgumentListType', 'FunctionReturnType', 'FunctionType',
    'OverloadedFunctionResolution', 'OverloadedFunctionAmbiguousResolutions', 'OverloadedFunctionType',
    'FunctionTyping',
    'FunctionOverriddenCallList', 'FunctionDerivedExpressionList', 'FunctionResolvedFromRecord', 'Function',
]


class FunctionArgumentResolutionError(Exception):
    """Exception raised when the function argument resolution fails."""
    pass


class FunctionArgumentResolutionContext(option_context(
    '_FunctionArgumentResolutionContext',
    check_missing=True, check_type=True, check_overloaded_ambiguity=True, exc_verbose=True
)):
    """A context manager for controlling the function argument resolution.

    Attributes:
        check_missing (bool): whether to check if the function argument is missing.
        check_type (bool): whether to check if the function argument type is correct.
        check_overloaded_ambiguity (bool): whether to check if the function argument resolution is ambiguous.
        exc_verbose (bool):wWhether to print verbose error message.
    """

    @contextlib.contextmanager
    def exc(self, exc_type=None, from_=None):
        if self.exc_verbose:
            yield
        else:
            if exc_type is None:
                exc_type = FunctionArgumentResolutionError
            if from_ is not None:
                raise exc_type() from from_
            raise exc_type()


get_function_argument_resolution_context: Callable[[], FunctionArgumentResolutionContext] = FunctionArgumentResolutionContext.get_default
"""Get the current function argument resolution context."""


FunctionArgumentUnset = object()
"""A placeholder indicating that the argument is not specified."""


class AnonymousFunctionArgumentGenerator(object):
    """A generator for anonymous function arguments."""

    def __init__(self, template='_t{i:d}'):
        self.template = template
        self.counter = 0

    @property
    def nr_generated(self) -> int:
        return self.counter

    def gen(self, n: Optional[int] = None) -> Union[str, List[str]]:
        if n is None:
            self.counter += 1
            return self.template.format(i=self.counter)
        return [self.gen() for _ in range(n)]


FunctionArgumentType = Union[ObjectType, ValueType, 'FunctionType', SequenceType]
"""Acceptable types for function arguments. See the documentation of `FunctionType` for more details."""

FunctionArgumentListType = Union[Sequence[FunctionArgumentType], Sequence[Variable], Dict[str, FunctionArgumentType]]
"""Acceptable types for function argument lists. See the documentation of `FunctionType` for more details."""

FunctionReturnType = Union[ObjectType, ValueType, 'FunctionType', SequenceType]
"""Acceptable types for function return types. See the documentation of `FunctionType` for more details."""


class FunctionType(TypeBase):
    """FunctionType defines the signature of a function."""

    argument_types: Tuple[FunctionArgumentType, ...]
    """The types of the arguments."""

    argument_names: Tuple[str, ...]
    """The names of the arguments."""

    arguments: Tuple[Variable, ...]
    """The argument list composed of `Variable` instances."""

    arguments_dict: Dict[str, FunctionArgumentType]
    """The arguments as a dict, as mappings from argument names to argument types."""

    arguments_name2index: Dict[str, int]
    """The mapping from argument names to argument indices."""

    return_type: FunctionReturnType
    """The return type of the function type."""

    return_name: Optional[Union[str, Tuple[str, ...]]]
    """The name of the return value."""

    is_generator_function: bool
    """Whether the function is a generator function."""

    is_simple_arguments: bool
    """Whether all arguments are not batched-list types."""

    is_singular_return: bool
    """Whether there is only one return value."""

    is_cacheable: bool
    """Whether the function is cacheable."""

    def __init__(
        self,
        arguments: FunctionArgumentListType,
        return_type: FunctionReturnType,
        argument_names: Optional[Sequence[str]] = None,
        return_name: Optional[Union[str, Sequence[str]]] = None,
        is_generator_function: bool = False,
        alias: Optional[str] = None
    ):
        """Initialize the function type.

        There are four ways to specify the arguments of the function type:

            1. A list of types, in which case the name of each argument is the index of the argument, using the format `#{index}`.
            2. A list of types as the `arguments`, and a list of names as the `argument_names`.
            3. A list of variables, in which case the name of each argument is the name of the variable.
            4. A dictionary of {name: type}, in which case the order of the arguments is the order of the keys.

        The return type can be either a single type or a tuple of types (multi-return types).

        Args:
            arguments: The arguments of the function type.
                When it is a list, the name of the arguments will be automatically generated.
                When it is a dict, the name of the arguments will be the keys of the dict.
            return_type: The return type of the function type.
            argument_names: The names of the arguments.
            return_name: The name of the return value.
            is_generator_function: Whether the function is a generator function.
            alias: The alias name of the function type.
        """

        self.arguments = None  # noqa
        self.arguments_dict = None  # noqa

        if isinstance(arguments, (list, tuple)):
            if len(arguments) == 0:
                self.arguments = tuple()
                self.arguments_dict = dict()
                self.argument_names = tuple()
                self.argument_types = tuple()
            elif isinstance(arguments[0], Variable):
                assert argument_names is None, 'Cannot specify both `arguments` and `argument_names`.'
                self.argument_types = tuple(arg.dtype.unwrap_alias() for arg in arguments)
                self.argument_names = tuple(arg.name for arg in arguments)
                self.arguments = arguments
            else:
                if argument_names is None:
                    argument_names = tuple('_' + str(i) for i in range(len(arguments)))
                else:
                    assert len(arguments) == len(argument_names), 'The length of `arguments` and `argument_names` must be the same.'
                self.argument_types = tuple([arg.unwrap_alias() for arg in arguments])
                self.argument_names = tuple(argument_names)
        elif isinstance(arguments, dict):
            assert argument_names is None, 'Cannot specify both `arguments` and `argument_names`.'
            self.argument_names = tuple(arguments.keys())
            self.argument_types = tuple([arg.unwrap_alias() for arg in arguments.values()])
            self.arguments_dict = arguments.copy()
        else:
            raise TypeError(f'Invalid argument types: {arguments}. Must be a list or a dict.')

        if self.arguments is None:
            self.arguments = tuple(Variable(name, dtype.unwrap_alias()) for name, dtype in zip(self.argument_names, self.argument_types))
        if self.arguments_dict is None:
            self.arguments_dict = {name: dtype for name, dtype in zip(self.argument_names, self.argument_types)}

        if isinstance(return_type, TupleType):
            self.return_type = return_type
            if return_name is None:
                self.return_name = None
            else:
                self.return_name = tuple(return_name)
                assert len(return_type.element_types) == len(self.return_name), 'The length of `return_type` and `return_name` must be the same.'

            # NB(Jiayuan Mao @ 2024/03/10): I commented out this behavior but I'm not sure if it's correct.
            # if len(return_type.element_types) == 1:
            #     self.return_type = self.return_type.element_types[0]
            #     if self.return_name is not None:
            #         self.return_name = self.return_name[0]
        else:
            if return_name is not None and not isinstance(return_name, str):
                raise ValueError(f'Invalid return name: {return_name}. Must be a string when the function has a singular return type')
            self.return_type = return_type
            self.return_name = return_name

        self.is_generator_function = is_generator_function
        self.is_simple_arguments = all(not isinstance(arg, UniformSequenceType) for arg in self.argument_types)
        self.is_singular_return = not isinstance(self.return_type, TupleType)
        self.is_cacheable = self._gen_is_cacheable()

        super().__init__(self._gen_typename(), alias=alias)

    def _gen_typename(self) -> str:
        if self.is_generator_function:
            return '(' + ', '.join([str(arg) for arg in self.arguments]) + ') -> Gen[' + str(self.return_type) + ']'
        return '(' + ', '.join([str(arg) for arg in self.arguments]) + ') -> ' + str(self.return_type)

    def _gen_is_cacheable(self):
        if self.is_generator_function:
            return False
        for arg_def in self.arguments:
            if isinstance(arg_def, ValueType):
                return False
        return True

    @property
    def nr_arguments(self) -> int:
        """Return the number of arguments."""
        return len(self.argument_types)

    @cached_property
    def nr_object_arguments(self) -> int:
        """Return the number of arguments that are ObjectType-ed."""
        return len(list(filter(lambda x: isinstance(x, ValueType), self.argument_types)))

    @cached_property
    def nr_value_arguments(self) -> int:
        """Return the number of arguments that are ValueType-ed."""
        return len(list(filter(lambda x: isinstance(x, ValueType), self.argument_types)))

    @cached_property
    def nr_variable_arguments(self) -> int:
        """Return the number of arguments that are VariableType-ed (i.e. is ValueType-ed but not ConstantType-ed."""
        return len(list(filter(lambda x: isinstance(x, ValueType) and not isinstance(x, ConstantType), self.argument_types)))

    @cached_property
    def nr_constant_arguments(self) -> int:
        """Return the number of arguments that are ConstantType-ed."""
        return len(list(filter(lambda x: isinstance(x, ConstantType), self.argument_types)))

    @cached_property
    def arguments_name2index(self):
        assert self.argument_names is not None
        return {v: k for k, v in enumerate(self.argument_names)}

    @classmethod
    def from_annotation(cls, function: Callable, sig: Optional[inspect.Signature] = None) -> Union['FunctionType', 'OverloadedFunctionType']:
        """Create a FunctionType from a function annotation.

        Args:
            function: The function.
            sig: The signature of the function.

        Returns:
            Union[FunctionType, OverloadedFunctionType]: The function type.
        """

        if sig is None:
            sig = inspect.signature(function)

        argument_types = list()
        argument_names = list()
        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0 and name == 'self':
                continue  # is an instancemethod.
            if i == 0 and name == 'cls':
                continue  # is a classmethod.
            argument_names.append(name)
            argument_types.append(param.annotation)

        return_type = sig.return_annotation

        if inspect._empty in argument_types or return_type is inspect._empty:
            raise FunctionArgumentResolutionError(f'Incomplete argument and return type annotation for {function}.')

        function_type = cls(argument_types, return_type, argument_names=argument_names)

        for arg_type in function_type.argument_types:
            if isinstance(arg_type, UnionType):
                return OverloadedFunctionType.from_function_type_with_union_arguments(function_type)

        return function_type

    def resolve_args(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Resolve the arguments to the function type.

        If you want to specify a specific "positional" argument by its index, use `_{index}` as the name of the argument.

        Args:
            *args: The positional arguments.
            **kwargs: The keyword arguments.

        Returns:
            A list of argument values.
        """

        resolution_context = get_function_argument_resolution_context()

        # Construct a mapping from the name of arguments to their indices.
        name2index = {f'#{i}': i for i in range(self.nr_arguments)}
        name2index.update(self.arguments_name2index)

        arguments = [FunctionArgumentUnset for _ in range(self.nr_arguments)]
        if len(args) + len(kwargs) > self.nr_arguments:
            with resolution_context.exc():
                raise FunctionArgumentResolutionError(f'Function {self} takes {len(self.argument_types)} arguments, got {len(args) + len(kwargs)}.')

        for i in range(len(args)):
            arguments[i] = args[i]
        for k, v in kwargs.items():
            if k not in name2index:
                with resolution_context.exc():
                    raise FunctionArgumentResolutionError(f'Got unknown keyword argument: {k} when invoking function {self}.')
            i = name2index[k]
            if arguments[i] is not FunctionArgumentUnset:
                with resolution_context.exc():
                    raise FunctionArgumentResolutionError(f'Got duplicated argument for keyword argument: {k} when invoking function {self}.')
            arguments[i] = v

        if resolution_context.check_missing:
            for i in range(self.nr_arguments):
                if arguments[i] is FunctionArgumentUnset:
                    with resolution_context.exc():
                        raise FunctionArgumentResolutionError(f'Missing argument {self.argument_names[i]} when invoking function {self}.')

        if resolution_context.check_type:
            from mover._vendor.concepts.dsl.expression import get_types
            argument_types = get_types(arguments)
            for i in range(self.nr_arguments):
                if argument_types[i] is not FunctionArgumentUnset and not argument_types[i].downcast_compatible(self.argument_types[i]):
                    with resolution_context.exc():
                        raise FunctionArgumentResolutionError(f'Typecheck failed for argument {self.argument_names[i]} while invoking the function {self}.\nInvoked with types: {argument_types}.')

        return arguments


@dataclass
class OverloadedFunctionResolution(object):
    """The data structure for storing the result of resolving an overloaded function."""

    type_index: int
    """The index of the function type that matches the expected signature."""

    ftype: FunctionType
    """The function type that matches the expected signature."""

    arguments: List[Any]
    """The resolved arguments."""


class OverloadedFunctionAmbiguousResolutions(list, List[OverloadedFunctionResolution]):
    pass


class OverloadedFunctionType(TypeBase):
    types: Tuple[FunctionType]

    def __init__(
        self,
        types: Sequence[Union['OverloadedFunctionType', FunctionType]],
        alias: Optional[str] = None
    ):
        types_flatten: List[FunctionType] = list()
        for ftype in types:
            if isinstance(ftype, OverloadedFunctionType):
                types_flatten.extend(ftype.types)
            else:
                assert isinstance(type, FunctionType)
                types_flatten.append(ftype)

        self.types = tuple(types_flatten)
        super().__init__(self._gen_typename(), alias=alias)

    def _gen_typename(self) -> str:
        return 'Overloaded{' + ','.join([x.typename for x in self.types]) + '}'

    @property
    def nr_types(self) -> int:
        """Return the number of sub-types."""
        return len(self.types)

    @classmethod
    def from_function_type_with_union_arguments(cls, function_type: FunctionType):
        """Create an OverloadedFunctionType from a FunctionType with Union-Typed arguments."""
        product_bases = list()
        for arg_type in function_type.argument_types:
            if isinstance(arg_type, UnionType):
                product_bases.append(arg_type.types)
            else:
                product_bases.append([arg_type])

        product_types = tuple(
            FunctionType(
                arg_type, function_type.return_type,
                argument_names=function_type.argument_names
            ) for arg_type in itertools.product(*product_bases)
        )
        return cls(product_types, alias=function_type.typename)

    def resolve_type_and_args(self, *args: Any, **kwargs: Any) -> Union[OverloadedFunctionResolution, OverloadedFunctionAmbiguousResolutions]:
        """Resolve the exact sub-function type being called and the argument list.

        Args:
            *args: The positional arguments.
            **kwargs: The keyword arguments.

        Returns:
            A :class:`OverloadedFunctionResolution` object if the resolution is unambiguous, or a :class:`OverloadedFunctionAmbiguousResolutions` object if the resolution is ambiguous.
            The ambiguity resolution object will only be returned if the ``check_overloaded_ambiguity`` flag is set to False in the :class:`FunctionArgumentResolutionContext`.

            - The :class:`OverloadedFunctionResolution` object contains the index of the sub-function type being called, the sub-function type, and the resolved argument list.
            - The :class:`OverloadedFunctionAmbiguousResolutions` object contains a list of :class:`OverloadedFunctionResolution` objects.
        """
        resolution_context = get_function_argument_resolution_context()

        success_results = list()
        exceptions = list()
        for i, ftype in enumerate(self.types):
            try:
                arguments = ftype.resolve_args(*args, **kwargs)
                success_results.append(OverloadedFunctionResolution(i, ftype, arguments))
            except FunctionArgumentResolutionError as e:
                exceptions.append(e)

        if len(success_results) == 1:
            return success_results[0]
        elif len(success_results) == 0:
            with resolution_context.exc():
                fmt = 'Failed to resolve overloaded function{}.\n'.format('' if self.typename is None else ' ' + self.typename)
                fmt += 'Detailed messages are:\n'
                for ftype, r in zip(self.types, exceptions):
                    this_fmt = 'Trying ' + str(ftype) + ':\n'
                    this_fmt += indent_text(str(r))
                    fmt += indent_text(this_fmt) + '\n'
                raise FunctionArgumentResolutionError(fmt.rstrip())
        else:
            if resolution_context.check_overloaded_ambiguity:
                with resolution_context.exc():
                    fmt = 'Got ambiguous application of overloaded function{}.\n'.format('' if self.typename is None else ' ' + self.typename)
                    fmt += 'Candidates are:\n'
                    for r in success_results:
                        fmt += indent_text(str(r[1])) + '\n'
                    fmt += 'Invoked with arguments: {}.'.format(str(success_results[0][2]))
                    raise FunctionArgumentResolutionError(fmt)
            else:
                return OverloadedFunctionAmbiguousResolutions(success_results)


class _FunctionTypingSugarInner(object):
    def __init__(self, return_type):
        self.return_type = return_type

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return FunctionType(tuple(), self.return_type)
        elif len(args) != 0:
            assert len(kwargs) == 0, 'Only support all positional arguments or all positional keyword arguments.'
            return FunctionType(args, self.return_type)
        elif len(kwargs) != 0:
            assert len(args) == 0, 'Only support all positional arguments or all positional keyword arguments.'
            return FunctionType(tuple(kwargs.values()), self.return_type, tuple(kwargs.keys()))
        raise ValueError('Unreachable.')


class _FunctionTypingSugar(object):
    def __getitem__(self, return_type):
        return _FunctionTypingSugarInner(return_type)


"""FunctionTyping is a language-sugar constructor for function types.
For example: `FunctionTypingp[BOOL](INT64, INT64)` creates a function type with two INT64 arguments and a BOOL return type."""
FunctionTyping = _FunctionTypingSugar()


class FunctionOverriddenCallList(list, List[Callable]):
    """A data structure that holds multiple overridden __call__ implementations for a function.

    This is only useful when we are partial evaluating a function (and when the actual function type can not be resolved.)
    """
    pass


class FunctionDerivedExpressionList(list, List['Expression']):
    """A data structure that holds multiple derived expressions for a function."""
    pass


@dataclass
class FunctionResolvedFromRecord(object):
    function: Callable
    ftype_index: Union[int, Tuple[int, ...]]


class Function(object):
    """A function object holds a function type and an optional overridden __call__.

    The function object holds an additional field called `overridden_call`, which isa callable function.
    This field is used to override the __call__ method of the function object.

    By default, the __call__ function returns a FunctionApplication object, which contains the name of the function
    and a list of arguments. However, when `overridden_call` is set, the __call__ method will return the result of
    calling `overridden_call` with the arguments.
    """

    def __init__(
        self,
        name: str,
        ftype: Union[FunctionType, OverloadedFunctionType],
        derived_expression: Optional[Union['Expression', FunctionDerivedExpressionList]] = None,
        overridden_call: Optional[Union[Callable, FunctionOverriddenCallList]] = None,
        resolved_from: Optional[FunctionResolvedFromRecord] = None,
        function_body: Optional[Union[Callable, Sequence[Callable]]] = None
    ):
        """
        Args:
            name: the name of the function.
            ftype: the function type.
            derived_expression: the expression that this function is derived from.
            overridden_call: the overridden call function.
            resolved_from: the record of the function that this function is resolved from. This is used for handling
                partial evaluation and function specialization (for overloadded functions).
            function_body: the function body.
        """

        self.ftype = ftype
        self.derived_expression = derived_expression
        self.overridden_call = overridden_call
        self.resolved_from = resolved_from

        if isinstance(self.ftype, OverloadedFunctionType) and isinstance(self.overridden_call, FunctionOverriddenCallList):
            assert self.ftype.nr_types == len(self.overridden_call)

        self.name = name
        self.function_body = function_body  # the function body defined during the declaration.

        if self.derived_expression is None and self.overridden_call is not None:
            if self.is_overloaded:
                self.derived_expression = FunctionDerivedExpressionList()
                for i in range(self.ftype.nr_types):
                    self.derived_expression.append(_gen_expression_from_overridden_call(self.ftype.types[i], self.overridden_call[i]))
            else:
                self.derived_expression = _gen_expression_from_overridden_call(ftype, self.overridden_call)

        self.is_derived = self.derived_expression is not None

    def set_function_name(self, function_name: str):
        """Set the function name."""
        self.name = function_name

    def set_function_body(self, function_body: Callable):
        """Set the function body."""
        self.function_body = function_body

    """Argument and return type of the function (when the function is not an overloaded one)."""

    @property
    def arguments(self) -> Tuple[Variable]:
        assert not self.is_overloaded
        return self.ftype.arguments

    @property
    def nr_arguments(self) -> int:
        assert not self.is_overloaded
        return self.ftype.nr_arguments

    @property
    def return_type(self) -> FunctionReturnType:
        assert not self.is_overloaded
        return self.ftype.return_type

    @property
    def is_generator_function(self) -> bool:
        return self.ftype.is_generator_function

    """When the function is overloaded, the following functions are used for get the "overridden calls" for each function type."""

    @property
    def is_overloaded(self) -> bool:
        """Return True if the function is overloaded."""
        return isinstance(self.ftype, OverloadedFunctionType)

    def get_overridden_call(self, ftype_index: Optional[int] = None) -> Optional[Callable]:
        """Get the overridden call function."""
        if isinstance(self.overridden_call, FunctionOverriddenCallList):
            assert ftype_index is not None
            return self.overridden_call[ftype_index]
        return self.overridden_call

    def get_sub_function(self, ftype_index: int) -> 'Function':
        assert self.is_overloaded
        assert 0 <= ftype_index < self.ftype.nr_types
        return type(self)(
            self.name,
            self.ftype.types[ftype_index],
            self.get_overridden_call(ftype_index),
            resolved_from=FunctionResolvedFromRecord(self, ftype_index),
            function_body=self.function_body[ftype_index] if self.function_body is not None else None
        )

    @cached_property
    def all_sub_functions(self) -> List['Function']:
        assert self.is_overloaded
        return [self.get_sub_function(i) for i in range(self.ftype.nr_types)]

    @classmethod
    def from_function(cls, function: Callable, implementation: bool = True, sig: Optional[inspect.Signature] = None):
        """Create a function object from an actual Python function.

        Args:
            function: The function.
            implementation: Whether the function is an implementation. Defaults to True.
            sig: The signature of the function. Defaults to None.
        """
        ftype = FunctionType.from_annotation(function, sig=sig)
        return cls(function.__name__, ftype, function_body=function if implementation else None)

    def __call__(self, *args, **kwargs):
        if self.overridden_call is not None:
            if isinstance(self.ftype, OverloadedFunctionType):
                ftype_index, function_type, resolved_args = self.ftype.resolve_type_and_args(*args, **kwargs)
                return self.get_overridden_call(ftype_index)(*resolved_args)
            else:
                resolved_args = self.ftype.resolve_args(*args, **kwargs)
                return self.overridden_call(*resolved_args)

        if isinstance(self.ftype, OverloadedFunctionType):
            ftype_index, function_type, resolved_args = self.ftype.resolve_type_and_args(*args, **kwargs)
            function = Function(
                self.name + f'_{ftype_index}',
                function_type,
                overridden_call=None,  # Must be none.
                resolved_from=FunctionResolvedFromRecord(self, ftype_index),
            )
        else:
            resolved_args = self.ftype.resolve_args(*args, **kwargs)
            function = self

        from mover._vendor.concepts.dsl.expression import FunctionApplicationExpression, cvt_expression_list
        return FunctionApplicationExpression(function, cvt_expression_list(resolved_args, function.ftype.argument_types))

    def __str__(self):
        if self.is_derived and not self.is_overloaded:
            with FormatContext(type_format_cls=False).as_default():
                if get_format_context().function_format_lambda:
                    fmt = ''.join(['lam ' + str(n.name) + '.' for n in self.arguments])
                    with FormatContext(object_format_type=False).as_default():
                        fmt += str(self.derived_expression)
                else:
                    fmt = 'def ' + self.name + '(' + ', '.join([str(x) for x in self.arguments]) + '): '
                    with FormatContext(object_format_type=False).as_default():
                        fmt += 'return ' + indent_text(str(self.derived_expression)).lstrip()
        else:
            if isinstance(self.ftype, OverloadedFunctionType):
                fmt = '\n'.join([f'{func_type}' for func_type in self.ftype.types])
                if self.name is not None:
                    fmt = re.sub(r'^' + re.escape(self.name) + ' (overloaded): ', '', fmt, flags=re.MULTILINE)
                    fmt = self.name + ': ' + '\n' + indent_text(fmt)
            else:
                fmt = f'{self.name}{self.ftype}'

        return fmt

    __repr__ = jacinle.repr_from_str

    def remap_arguments(self, remapping: List[int]) -> 'Function':
        """
        Generate a new Function object with a different argument order.
        Specifically, remapping is a permutation. The i-th argument to the new function will be the remapping[i]-th
        argument in the old function.

        Args:
            remapping: The remapping.

        Returns:
            The new function.
        """
        if isinstance(self.ftype, OverloadedFunctionType):
            raise NotImplementedError('Argument remapping for overloaded functions are not implemented.')

        new_argument_types = [self.ftype.argument_types[i] for i in remapping]

        def new_overridden_call(*args):
            remapped_args = [None for _ in range(len(args))]
            for i, arg in enumerate(args):
                remapped_args[remapping[i]] = arg
            return self(*remapped_args)

        return Function(
            self.name,
            FunctionType(
                new_argument_types, self.ftype.return_type,
            ), overridden_call=new_overridden_call, resolved_from=self.resolved_from
        )

    def partial(self, *args, execute_fully_bound_functions=False, **kwargs) -> Union['Function', 'FunctionApplicationExpression']:
        if self.name == '__lambda__':
            new_name = '__lambda__'
        else:
            new_name = f'{self.name}_partial'
        new_overridden_call = None
        new_resolved_from = None

        if self.is_overloaded:
            with FunctionArgumentResolutionContext(
                check_missing=False,
                check_overloaded_ambiguity=False
            ).as_default():
                types_and_arguments = self.ftype.resolve_type_and_args(*args, **kwargs)

            if not isinstance(types_and_arguments, OverloadedFunctionAmbiguousResolutions):
                ftype_index, function_type, resolved_args = types_and_arguments
                unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is FunctionArgumentUnset]
                if len(unmapped_arguments) == 0:
                    return self._apply_with_resolved_args(resolved_args, ftype_index, function_type)
                new_type = _gen_partial_function_type(function_type, unmapped_arguments)
                new_resolved_from = FunctionResolvedFromRecord(self, ftype_index)
            else:
                # Block BEGIN {{{
                # If there is one specific resolution s.t. all variables are grounded, use it.

                all_grounded_resolutions = list()
                for ftype_index, function_type, resolved_args in types_and_arguments:
                    unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is FunctionArgumentUnset]
                    if len(unmapped_arguments) == 0:
                        all_grounded_resolutions.append((ftype_index, function_type, resolved_args))

                if len(all_grounded_resolutions) == 1:
                    ftype_index, function_type, resolved_args = all_grounded_resolutions[0]
                    return self._apply_with_resolved_args(resolved_args, ftype_index, function_type)
                elif len(all_grounded_resolutions) > 1:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Got ambiguous application of overloaded function{}.\n'.format(
                            '' if self.name is None else ' ' + self.name)
                        fmt += 'Candidates are:\n'
                        for r in all_grounded_resolutions:
                            fmt += indent_text(str(r[1])) + '\n'
                        fmt += 'Invoked with arguments: {}.'.format(str(all_grounded_resolutions[0][2]))
                        raise FunctionArgumentResolutionError(fmt)

                # }}} Block END.

                possible_resolution_ids = list()
                possible_resolutions = list()
                possible_overridden_calls = list()
                for ftype_index, function_type, resolved_args in types_and_arguments:
                    unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is FunctionArgumentUnset]
                    new_subtype = _gen_partial_function_type(
                        function_type, unmapped_arguments
                    )
                    possible_resolution_ids.append(ftype_index)
                    possible_resolutions.append(new_subtype)
                    possible_overridden_calls.append(_gen_partial_overriden_call(
                        new_subtype, resolved_args, self
                    ))
                new_type = OverloadedFunctionType(possible_resolutions)
                new_resolved_from = FunctionResolvedFromRecord(self, tuple(possible_resolution_ids))
                new_overridden_call = FunctionOverriddenCallList(possible_overridden_calls)
        else:
            with FunctionArgumentResolutionContext(check_missing=False).as_default():
                resolved_args = self.ftype.resolve_args(*args, **kwargs)
            unmapped_arguments = [i for i, arg in enumerate(resolved_args) if arg is FunctionArgumentUnset]

            if execute_fully_bound_functions:
                if len(unmapped_arguments) == 0:
                    return self._apply_with_resolved_args(resolved_args)

            new_type = _gen_partial_function_type(self.ftype, unmapped_arguments)
            new_overridden_call = _gen_partial_overriden_call(
                new_type, resolved_args, self
            )

        return Function(
            new_name, new_type,
            overridden_call=new_overridden_call, resolved_from=new_resolved_from
        )

    def _apply_with_resolved_args(
        self, resolved_args,
        resolved_ftype_id=None, resolved_function_type=None
    ):

        if self.overridden_call is not None:
            return self.get_overridden_call(resolved_ftype_id)(*resolved_args)

        if isinstance(self.ftype, OverloadedFunctionType):
            function = Function(
                self.name + f'_{resolved_ftype_id}',
                resolved_function_type,
                overridden_call=None,  # Must be none.
                resolved_from=FunctionResolvedFromRecord(self, resolved_ftype_id)
            )
        else:
            function = self

        from mover._vendor.concepts.dsl.expression import FunctionApplicationExpression, cvt_expression_list
        return FunctionApplicationExpression(function, cvt_expression_list(resolved_args, function.ftype.argument_types))


def _gen_expression_from_overridden_call(ftype: FunctionType, overridden_call: Callable):
    from mover._vendor.concepts.dsl.expression import ExpressionDefinitionContext
    ctx = ExpressionDefinitionContext()
    with ctx.as_default():
        arguments = [ctx[arg] for arg in ftype.arguments]
        return overridden_call(*arguments)


def _gen_partial_function_type(old_type: FunctionType, unmapped_arguments):
    new_argument_types = [old_type.argument_types[i] for i in unmapped_arguments]
    new_return_type = old_type.return_type
    new_argument_names = None
    if old_type.argument_names is not None:
        new_argument_names = [old_type.argument_names[i] for i in unmapped_arguments]
    new_type = FunctionType(new_argument_types, new_return_type, argument_names=new_argument_names)
    return new_type


def _gen_partial_overriden_call(new_type, resolved_args, call):
    assert isinstance(new_type, FunctionType)

    def partial_overriden_call(*new_args, **new_kwargs):
        new_resolved_args = new_type.resolve_args(*new_args, **new_kwargs)
        new_full_args = resolved_args.copy()
        j = 0
        for i in range(len(resolved_args)):
            if new_full_args[i] is FunctionArgumentUnset:
                new_full_args[i] = new_resolved_args[j]
                j += 1
        return call(*new_full_args)

    return partial_overriden_call
