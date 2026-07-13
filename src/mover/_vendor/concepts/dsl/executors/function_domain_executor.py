#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : function_domain_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A simple executor for a domain that only contains function applications. See the tutorial for more details."""

import functools
import contextlib
from typing import Any, Optional, Union, Callable
from jacinle.logging import get_logger

from mover._vendor.concepts.dsl.dsl_functions import FunctionType, Function
from mover._vendor.concepts.dsl.value import ValueBase, Value
from mover._vendor.concepts.dsl.expression import Expression, FunctionApplicationExpression, ConstantExpression
from mover._vendor.concepts.dsl.function_domain import FunctionDomain
from mover._vendor.concepts.dsl.parsers.parser_base import ParserBase
from mover._vendor.concepts.dsl.parsers.function_expression_parser import FunctionExpressionParser
from mover._vendor.concepts.dsl.executors.executor_base import DSLExecutorBase

logger = get_logger(__file__)

__all__ = ['FunctionDomainExecutor']


class FunctionDomainExecutor(DSLExecutorBase):
    """The executor for simple function DSLs."""

    def __init__(self, domain: FunctionDomain, parser: Optional[ParserBase] = None):
        """Initialize the executor.

        Args:
            domain: the function domain.
            parser: the parser of the DSL. If not specified, the default
                :class:`~mover._vendor.concepts.dsl.parsers.function_expression_parser.FunctionExpressionParser` will be used.
        """
        super().__init__(domain)
        self.parser = parser

        for function_name, function in domain.functions.items():
            assert not function.is_overloaded
            if not function.is_derived:
                if hasattr(self, function_name):
                    self.register_function(function_name, self.unwrap_values(getattr(self, function_name)))
                    # logger.info('Function {} automatically registered.'.format(function_name))

        self._grounding = None

    _domain: FunctionDomain

    @property
    def domain(self) -> FunctionDomain:
        """The function domain of the executor."""
        return self._domain

    @property
    def grounding(self) -> Any:
        """The grounding of the current execution."""
        return self._grounding

    parser: ParserBase
    """The parser of the DSL."""

    @contextlib.contextmanager
    def with_grounding(self, grounding: Any):
        """A context manager for setting the grounding of the executor.

        Args:
            grounding: the grounding to set.
        """
        old_grounding = self.grounding
        self._grounding = grounding
        try:
            yield
        finally:
            self._grounding = old_grounding

    def parse_expression(self, string: str) -> Expression:
        """Parse an expression from a string.

        Args:
            string: the string to parse.

        Returns:
            The parsed expression.
        """
        if self.parser is None:
            self.parser = FunctionExpressionParser(self.domain)
        return self.parser.parse_expression(string)

    def execute(self, expr: Union[str, Expression], grounding: Optional[Any] = None) -> Value:
        """Execute an expression.

        Args:
            expr: the expression to execute.
            grounding: the grounding of the expression.

        Returns:
            The result of the execution.
        """

        if isinstance(expr, str):
            expr = self.parse_expression(expr)

        grounding = grounding if grounding is not None else self._grounding
        with self.with_grounding(grounding):
            return self._execute(expr)

    def execute_function(self, function: Function, *args: Value, grounding: Optional[Any] = None) -> Value:
        """Execute a function with a list of arguments.

        Args:
            function: the function to execute.
            *args: the arguments of the function.
            grounding: the grounding of the function.

        Returns:
            The result of the execution.
        """

        expression = function(*args)
        return self.execute(expression, grounding)

    def _execute(self, expr: Expression) -> Value:
        """Internal implementation of the executor. This method will be called by the public method :meth:`execute`.
        This function basically implements a depth-first search on the expression tree.

        Args:
            expr: the expression to execute.

        Returns:
            The result of the execution.
        """
        if isinstance(expr, FunctionApplicationExpression):
            func = self.function_implementations[expr.function.name]
            args = [self._execute(arg) for arg in expr.arguments]
            return func(*args)
        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.constant, Value)
            return expr.constant
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')

    def unwrap_values(self, func_or_ftype: Union[Callable, FunctionType]) -> Callable:
        """A function decorator that automatically unwraps the values of the arguments of the function.
        Basically, this decorator will unwrap the values of the arguments of the function, and then wrap the result with the
        :class:`mover._vendor.concepts.dsl.value.Value` class.

        There are two ways to use this decorator. The first way is to use it as a function decorator:
        In this case, the wrapped function should have the same name as the DSL function it implements.

            >>> domain = FunctionDomain()
            >>> # Assume domain has a function named "add" with two arguments.
            >>> executor = FunctionDomainExecutor(domain)
            >>> @executor.unwrap_values
            >>> def add(a, b):
            >>>     return a + b
            >>> executor.register_function('add', add)

        The second way is to use it as function that generates a function decorator:

            >>> domain = FunctionDomain()
            >>> # Assume domain has a function named "add" with two arguments.
            >>> executor = FunctionDomainExecutor(domain)
            >>> @executor.unwrap_values(domain.functions['add'].ftype)
            >>> def add(a, b):
            >>>     return a + b
            >>> executor.register_function('add', executor.unwrap_values(add))

        Args:
            func_or_ftype: the function to wrap, or the function type of the function to wrap.

        Returns:
            The decorated function or a function decorator.
        """

        if isinstance(func_or_ftype, FunctionType):
            ftype = func_or_ftype
        else:
            if func_or_ftype.__name__ not in self.domain.functions:
                raise NameError(f'Function {func_or_ftype.__name__} is not registered in the domain.')
            ftype = self.domain.functions[func_or_ftype.__name__].ftype

        def wrapper(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                args = [arg.value if isinstance(arg, Value) else arg for arg in args]
                kwargs = {k: v.value if isinstance(v, Value) else v for k, v in kwargs.items()}
                rv = func(*args, **kwargs)

                if isinstance(ftype.return_type, tuple):
                    return tuple(
                        Value(ftype.return_type[i], rv[i]) if not isinstance(rv[i], ValueBase) else rv[i] for i in range(len(rv))
                    )
                elif isinstance(rv, ValueBase):
                    return rv
                else:
                    return Value(ftype.return_type, rv)
            return wrapped

        if isinstance(func_or_ftype, FunctionType):
            return wrapper
        else:
            return wrapper(func_or_ftype)

