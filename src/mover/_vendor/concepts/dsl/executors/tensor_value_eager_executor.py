#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensor_value_eager_executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/29/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

from mover._vendor.concepts.dsl.dsl_functions import Function
from mover._vendor.concepts.dsl.tensor_value import TensorValue
from mover._vendor.concepts.dsl.function_domain import FunctionDomain
from mover._vendor.concepts.dsl.parsers.parser_base import ParserBase
from mover._vendor.concepts.dsl.parsers.function_expression_parser import FunctionExpressionParser
from mover._vendor.concepts.dsl.executors.tensor_value_executor import TensorValueExecutorBase, TensorValueExecutorReturnType


class TensorValueEagerExecutor(TensorValueExecutorBase):
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

    def _execute_and(self, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

    def _execute_or(self, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

    def _execute_not(self, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

    def _execute_forall(self, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

    def _execute_exists(self, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

    def _execute_function(self, function: Function, *args: TensorValue) -> TensorValue:
        raise NotImplementedError()

