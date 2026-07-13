#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dsl_domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The base calss for domain definitions: containing custom types, constants, and functions."""

import inspect
from typing import Optional, Union, Dict, Callable
from mover._vendor.concepts.dsl.dsl_types import ObjectType, ValueType, ObjectConstant
from mover._vendor.concepts.dsl.dsl_functions import Function
from mover._vendor.concepts.dsl.value import Value

__all__ = ['DSLDomainBase']


class DSLDomainBase(object):
    """The baseclass for the domain definition of a domain-specific language."""

    def __init__(self, name: Optional[str] = None):
        """Initialize the domain.

        Args:
            name: the name of the domain. If not specified, the name of the class will be used.
        """
        if name is None:
            name = type(self).__name__

        self.name = name
        self.types = dict()
        self.functions = dict()
        self.constants = dict()

    name: str
    """The name of the domain."""

    types: Dict[str, Union[ObjectType, ValueType]]
    """The types defined in the domain, as a dictionary from type names to types."""

    functions: Dict[str, Function]
    """The functions defined in the domain, as a dictionary from function names to functions."""

    constants: Dict[str, Union[ObjectConstant, Value]]
    """The constants defined in the domain, as a dictionary from the name to the :class:`~mover._vendor.concepts.dsl.value.Value` objects."""

    def define_type(self, t: Union[ObjectType, ValueType]) -> Union[ObjectType, ValueType]:
        """Define a type in the domain.

        Args:
            t: the type to be defined.

        Returns:
            the type that is defined.
        """
        if not isinstance(t, (ObjectType, ValueType)):
            raise TypeError('Types can only be object types and value types.')
        if t.typename in self.types:
            raise NameError(f'Type {t.typename} already defined.')
        self.types[t.typename] = t

        return t

    def define_function(self, function: Union[Function, Callable], implementation: bool = True) -> Function:
        """Define a function in the domain.

        Args:
            function: the function to be defined.
            implementation: whether to store the function body of `function` as the implementation of the function.

        Returns:
            the function that is defined.
        """
        if not isinstance(function, Function):
            assert inspect.isfunction(function)
            function = Function.from_function(function, implementation=implementation)
        if function.name in self.functions:
            raise ValueError(f'Function {function.name} already defined.')
        self.functions[function.name] = function

        return function

    def define_const(self, dtype: Union[ObjectType, ValueType], value: str):
        """Define a constant with the given type and value.

        Args:
            dtype: the type of the constant.
            value: the value of the constant. The value should be a string that is the name of the constant.
        """
        if value in self.constants:
            raise ValueError(f'Constant {value} already defined.')

        if isinstance(dtype, ObjectType):
            self.constants[value] = ObjectConstant(dtype, value)
        else:
            self.constants[value] = Value(dtype, value)

    def has_function(self, name: str) -> bool:
        """Check whether the domain has a function with the given name.

        Args:
            name: the name of the function.

        Returns:
            whether the domain has a function with the given name.
        """
        return name in self.functions

    def get_function(self, name: str) -> Function:
        """Get the function with the given name.

        Args:
            name: the name of the function.

        Returns:
            the function with the given name.
        """
        if name not in self.functions:
            raise KeyError(f'Function {name} not found.')
        return self.functions[name]

    def __str__(self) -> str:
        return f'{type(self).__name__}({self.name})'

    def __repr__(self) -> str:
        return str(self)

