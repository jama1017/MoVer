#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : executor_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The baseclass for all executors of domain-specific languages."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Callable

from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.expression import Expression

__all__ = ['DSLExecutorBase', 'DSLExecutionError']


class DSLExecutorBase(ABC):
    """The baseclass for all executors of domain-specific languages."""

    def __init__(self, domain: DSLDomainBase):
        """Initialize the executor.

        Args:
            domain: the domain of the executor.
        """

        self._domain = domain
        self._function_implementations = dict()

    @property
    def domain(self) -> DSLDomainBase:
        """The domain of the executor."""
        return self._domain

    @property
    def function_implementations(self) -> Dict[str, Callable]:
        """The implementations of functions, which is a mapping from function names to implementations."""
        return self._function_implementations

    def register_function(self, name: str, func: Callable):
        """Register an implementation for a function to the executor. Alias for :meth:`register_function_implementation`.

        Args:
            name: the name of the function.
            func: the implementation of the function.
        """
        self.register_function_implementation(name, func)

    def register_function_implementation(self, name: str, func: Callable):
        """Register an implementation for a function.

        Args:
            name: the name of the function.
            func: the implementation of the function.
        """
        self._function_implementations[name] = func

    def get_function_implementation(self, name: str) -> Callable:
        """Get the implementation of a function. When the executor does not have an implementation for the function,
        the implementation of the function in the domain will be returned. If that is also None, a `KeyError` will be
        raised.

        Args:
            name: the name of the function.

        Returns:
            the implementation of the function.
        """

        if name in self._function_implementations:
            return self._function_implementations[name]
        elif name in self._domain.functions:
            function_body = self._domain.functions[name].function_body
            if function_body is None:
                raise KeyError(f'No implementation for function {name}.')
            return function_body
        raise KeyError(f'No implementation for function {name}.')

    def has_function_implementation(self, name: str) -> bool:
        """Check whether the executor has an implementation for a function.

        Args:
            name: the name of the function.

        Returns:
            whether the executor has an implementation for the function.
        """
        return name in self._function_implementations or (name in self.domain.functions and self.domain.functions[name].function_body is not None)

    @abstractmethod
    def execute(self, expression: Expression) -> Any:
        """Execute an expression.

        Args:
            expression: the expression to execute.

        Returns:
            the result of the execution.
        """
        raise NotImplementedError()


class DSLExecutionError(Exception):
    """The exception raised when an execution error occurs."""

