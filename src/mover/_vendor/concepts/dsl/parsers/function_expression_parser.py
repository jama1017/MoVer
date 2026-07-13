#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : function_expression_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The parser for simple function expressions."""

import lark
import json

from dataclasses import dataclass
from mover._vendor.concepts.dsl.dsl_types import ObjectType, Variable, ObjectConstant
from mover._vendor.concepts.dsl.function_domain import FunctionDomain
from mover._vendor.concepts.dsl.value import Value
from mover._vendor.concepts.dsl.expression import Expression, ConstantExpression, VariableExpression, ObjectConstantExpression, FunctionApplicationExpression
from mover._vendor.concepts.dsl.parsers.parser_base import ParserBase

# lark.v_args
inline_args = lark.v_args(inline=True)

__all__ = ['FunctionExpressionTransformer', 'FunctionExpressionParser']


@dataclass
class _Placeholder(object):
    name: str
    placeholder_type: str


class FunctionExpressionTransformer(lark.Transformer):
    """The lark transformer for the simple function expression parser."""

    def __init__(self, domain: FunctionDomain, escape_string: bool = True):
        """Initialize the transformer.

        Args:
            domain: The domain to use.
            escape_string: Whether to escape the string.
        """
        super().__init__()
        self.domain = domain
        self.escape_string = escape_string

    def start(self, args):
        return args[0]

    @inline_args
    def function_application(self, function_name, *args):
        function = self.domain.get_function(function_name)

        if function.is_overloaded:
            raise NotImplementedError()
        if function.nr_arguments != len(args):
            raise ValueError(f'Function {function_name} expects {function.nr_arguments} arguments, but {len(args)} are given.')

        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, _Placeholder):
                arg_type = function.ftype.argument_types[i]
                if arg.placeholder_type == 'c':
                    if isinstance(arg_type, ObjectType):
                        args[i] = ObjectConstantExpression(ObjectConstant(arg.name, arg_type))
                    else:
                        args[i] = ConstantExpression(Value(arg_type, arg.name))
                elif arg.placeholder_type == 'v':
                    args[i] = VariableExpression(Variable(arg.name, arg_type))
                else:
                    raise ValueError(f'Unknown placeholder type {arg.placeholder_type}.')

        return FunctionApplicationExpression(function, args)

    def function_name(self, function_name):
        return function_name[0].value

    def argument(self, argument):
        return argument[0]

    def variable(self, variable):
        return _Placeholder(variable[0].value, 'v')

    def constant(self, constant):
        if self.escape_string:
            return _Placeholder(json.loads(constant[0].value), 'c')
        return _Placeholder(constant[0].value, 'c')


class FunctionExpressionParser(ParserBase):
    """The simple function expression parser.

    This parser works for simple function expressions, for example: ``f(x, y)``.

    Each function name is a string composed of letters, digits, and _, and each argument is an expression or a string.
    When ``escape_string`` is set to ``True``, the string should be escaped with double quotes.

        >>> from mover._vendor.concepts.dsl.function_domain import FunctionDomain
        >>> from mover._vendor.concepts.dsl.parsers.function_expression_parser import FunctionExpressionParser
        >>> domain = FunctionDomain()
        >>> parser = FunctionExpressionParser(domain)
        >>> parser.parse_expression('f(g(), "string")')

    """

    GRAMMAR = r"""
start: function_application
function_application: function_name "(" (argument ("," argument)*)? ")"
function_name: STRING

%import common.WS
%ignore WS

%import common.LETTER
%import common.DIGIT
STRING: LETTER ("_"|"-"|LETTER|DIGIT)*

%import common.ESCAPED_STRING
"""

    def __init__(self, domain: FunctionDomain, allow_variable: bool = False, escape_string: bool = True):
        """Initialize the parser.

        Args:
            domain: the domain to use.
            allow_variable: whether to allow variable.
            escape_string: whether to escape the string.
        """

        self.allow_variable = allow_variable
        self.escape_string = escape_string if escape_string is not None else allow_variable

        if self.allow_variable:
            assert self.escape_string, 'If allow_variable is True, escape_string must be True.'

        self.parser = lark.Lark(
            self.GRAMMAR +
            ('constant: ESCAPED_STRING\n' if escape_string else 'constant: STRING\n') +
            ('variable: STRING\nargument: function_application | constant | variable' if allow_variable else 'argument: function_application | constant'),
            parser='lalr', start='start'
        )
        self.transformer = FunctionExpressionTransformer(domain)

    def parse_expression(self, string: str) -> Expression:
        """Parse an expression from a string.

        Args:
            string: The string to parse.

        Returns:
            The parsed expression.
        """
        tree = self.parser.parse(string)
        return self.transformer.transform(tree)

