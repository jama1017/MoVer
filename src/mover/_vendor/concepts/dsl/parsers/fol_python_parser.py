#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : fol_python_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import ast
from typing import Optional, Sequence, Tuple, List
from mover._vendor.concepts.dsl.dsl_types import TypeBase, ObjectType, ValueType, BOOL, INT64, FLOAT32, Variable
from mover._vendor.concepts.dsl.dsl_functions import Function, FunctionType
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.expression import ExpressionDefinitionContext, ValueOutputExpression, get_expression_definition_context, get_types
from mover._vendor.concepts.dsl.expression import ConstantExpression, NotExpression, AndExpression, OrExpression, ForallExpression, ExistsExpression, FunctionApplicationExpression, ObjectCompareExpression, ValueCompareExpression, CompareOpType, VariableAssignmentExpression
from mover._vendor.concepts.dsl.parsers.parser_base import ParserBase


class FOLProgramAssignmentExpression(VariableAssignmentExpression):
    def __init__(self, variable: Variable, value: ValueOutputExpression):
        self.variable = variable
        self.value = value

    def __str__(self):
        return f'{self.variable} = {self.value}'


class FOLPythonParser(ParserBase):
    """A parser to parse first-order logic (FOL) expressions in Python syntax. Currrently supported features:

    - logic operations: and, or, not
    - quantifiers: forall, exists (see below for the syntax)
    - function application: ``f(x, y, z)``
    - function definition: ``def f(x, y, z): return x + y + z``

    The syntax for quantifiers is as follows:
        .. code-block:: python

            from typing import Any, Type, Callable

            def forall(dtype: Type, func: Callable[[Any], bool]) -> bool: ...
            def exists(dtype: Type, func: Callable[[Any], bool]) -> bool: ...


    Examples:
        .. code-block:: python

            from mover._vendor.concepts.dsl.dsl_types import ObjectType
            from mover._vendor.concepts.dsl.function_domain import FunctionDomain

            domain = FunctionDomain()
            domain.define_type(ObjectType('Person'))

            parser = FOLPythonParser(domain, inplace_definition=True)
            parser.parse_expression('exists(Person, lambda x: is_phd(x))')

            function_string = '''
            def is_grandfather(x: Person, y: Person) -> bool:
                # x is the grandfather of y
                return exists(Person, lambda z: is_father(x, z) and is_parent(y, z))
            '''
            parser.parse_function(function_string)
    """

    def __init__(self, domain: DSLDomainBase, inplace_definition: bool = False, inplace_polymorphic_function: bool = False, inplace_definition_type: bool = False):
        """Initialize the parser.

        Args:
            domain: the domain to use.
            inplace_definition: whether to allow expressions to contain functions that are not defined in the domain.
                If set to True, the parser will automatically define these functions.
            inplace_polymorphic_function: whether inplace functions are assumed to be polymorphic.
            inplace_definition_type: whether to allow expressions to contain types that are not defined in the domain.
                If set to True, the parser will automatically define these types.
        """
        self.domain = domain
        self.inplace_definition = inplace_definition
        self.inplace_polymorphic_function = inplace_polymorphic_function
        self.inplace_definition_type = inplace_definition_type

    domain: DSLDomainBase
    """The domain for types and functions."""

    inplace_definition: bool
    """Whether to allow functions to be defined in-place."""

    def parse_domain_string(self, string: str) -> DSLDomainBase:
        raise NotImplementedError('FOLPythonParser does not support parsing domain definition.')

    def parse_multiple_expressions(self, string: str, arguments: Sequence[Variable] = tuple()) -> Tuple[ValueOutputExpression, ...]:
        module = ast.parse(string)
        expressions = ast_get_multiple_expressions(module)
        with ExpressionDefinitionContext(*arguments, domain=self.domain).as_default():
            return tuple(self._parse_expression_inner(expression) for expression in expressions)

    def parse_expression(self, string: str, arguments: Sequence[Variable] = tuple()) -> ValueOutputExpression:
        module = ast.parse(string)
        expression = ast_get_expression(module)
        return self.parse_expression_ast(expression, arguments)

    def parse_expression_ast(self, expression: ast.AST, arguments: Sequence[Variable] = tuple()) -> ValueOutputExpression:
        with ExpressionDefinitionContext(*arguments, domain=self.domain).as_default():
            return self._parse_expression_inner(expression)

    def parse_function(self, string: str) -> Function:
        module = ast.parse(string)
        function = ast_get_function_definition(module)
        arguments, return_type, body = ast_get_simple_function(function)

        arguments = [Variable(arg.name, self._get_type_from_domain(self.domain, arg.dtype)) for arg in arguments]
        return_type = self._get_type_from_domain(return_type, return_type)

        return Function(
            function.name,
            FunctionType(arguments, return_type),
            self.parse_expression_ast(body, arguments)
        )

    def _parse_expression_inner(self, expression: ast.AST) -> ValueOutputExpression:
        ctx = get_expression_definition_context()
        if isinstance(expression, ast.Call):
            function_name = ast_get_literal_or_class_name(expression.func)
            if self._is_quantification_expression_name(function_name):
                return self._parse_quantification_expression(function_name, expression)
            else:  # function is a regular function.
                return self._parse_function_application(function_name, expression)
        elif isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return NotExpression(self._parse_expression_inner(expression.operand))
        elif isinstance(expression, ast.BoolOp):
            if isinstance(expression.op, ast.And):
                return AndExpression(*[self._parse_expression_inner(value) for value in expression.values])
            elif isinstance(expression.op, ast.Or):
                return OrExpression(*[self._parse_expression_inner(value) for value in expression.values])
            else:
                raise NotImplementedError(f'BoolOp {expression.op} is not supported.')
        elif isinstance(expression, ast.Assign):
            assert len(expression.targets) == 1, f'Expect one target for assignment, got {len(expression.targets)}.'
            assert isinstance(expression.targets[0], ast.Name), f'Expect a name for assignment, got {expression.targets[0]}.'
            var_name = expression.targets[0].id
            value = self._parse_expression_inner(expression.value)
            var_type = value.return_type
            var = Variable(var_name, var_type)
            ctx.add_variables(var)
            return FOLProgramAssignmentExpression(var, value)
        elif isinstance(expression, ast.Name):
            return ctx[expression.id]
        elif isinstance(expression, ast.Constant):
            if isinstance(expression.value, bool):
                return ConstantExpression(expression.value, BOOL)
            elif isinstance(expression.value, int):
                return ConstantExpression(expression.value, INT64)
            elif isinstance(expression.value, float):
                return ConstantExpression(expression.value, FLOAT32)
            else:
                raise NotImplementedError(f'Constant {expression.value} is not supported.')
        elif isinstance(expression, ast.Compare):
            left = self._parse_expression_inner(expression.left)
            right = self._parse_expression_inner(expression.comparators[0])
            op = expression.ops[0]

            if isinstance(left.return_type, ObjectType) and isinstance(right.return_type, ObjectType):
                if isinstance(op, ast.Eq):
                    return ObjectCompareExpression(CompareOpType.EQ, left, right)
                elif isinstance(op, ast.NotEq):
                    return ObjectCompareExpression(CompareOpType.NEQ, left, right)
                else:
                    raise NotImplementedError(f'Compare {op} is not supported.')
            elif isinstance(left.return_type, ValueType) and isinstance(right.return_type, ValueType):
                if isinstance(op, ast.Eq):
                    return ValueCompareExpression(CompareOpType.EQ, left, right)
                elif isinstance(op, ast.NotEq):
                    return ValueCompareExpression(CompareOpType.NEQ, left, right)
                elif isinstance(op, ast.Lt):
                    return ValueCompareExpression(CompareOpType.LT, left, right)
                elif isinstance(op, ast.LtE):
                    return ValueCompareExpression(CompareOpType.LEQ, left, right)
                elif isinstance(op, ast.Gt):
                    return ValueCompareExpression(CompareOpType.GT, left, right)
                elif isinstance(op, ast.GtE):
                    return ValueCompareExpression(CompareOpType.GEQ, left, right)
                else:
                    raise NotImplementedError(f'Compare {op} is not supported.')
            else:
                raise ValueError(f'Cannot compare {left.return_type} with {right.return_type}.')
        else:
            raise NotImplementedError(f'Expression {expression} is not supported. Full expression: {ast.dump(expression)}.')

    def _is_quantification_expression_name(self, function_name: str) -> bool:
        return function_name in ('forall', 'exists')

    def _parse_quantification_expression(self, function_name: str, expression: ast.Call) -> ValueOutputExpression:
        ctx = get_expression_definition_context()

        assert len(expression.args) in (2, 3), f'Expect two or three arguments for quantification expressions, got {len(expression.args)}: {ast.dump(expression)}'
        assert isinstance(expression.args[0], (ast.Name, ast.Constant)), f'Expect a variable type name for quantification expressions, got {expression.args[0]}: {ast.dump(expression)}'
        if len(expression.args) == 3:
            assert isinstance(expression.args[1], ast.Constant), f'Expect an integer for counting quantifiers, got {expression.args[1]}: {ast.dump(expression)}'
            assert isinstance(expression.args[1].value, int), f'Expect an integer for counting quantifiers, got {expression.args[1].value}: {ast.dump(expression)}'

        assert isinstance(expression.args[-1], ast.Lambda), f'Expect a lambda expression for quantification expressions, got {expression.args[1]}: {ast.dump(expression)}'
        var_type = ast_get_literal_or_class_name(expression.args[0])
        var_type = self._get_type_from_domain(ctx.domain, var_type)
        assert isinstance(var_type, ObjectType), f'Expect an object type for quantification expressions, got {var_type}.'

        lambda_expression = expression.args[-1]
        assert len(lambda_expression.args.args) == 1, f'Expect one argument for quantification expressions, got {len(lambda_expression.args.args)}.'
        assert lambda_expression.args.args[0].annotation is None, f'Expect no type annotation for lambda functions in quantification expressions, got {lambda_expression.args.args[0].annotation}.'

        var_name = lambda_expression.args.args[0].arg
        var = Variable(var_name, var_type)

        return self._parse_quantification_expression_inner(function_name, var, lambda_expression.body, counting_quantifier=expression.args[1].value if len(expression.args) == 3 else None)

    def _parse_quantification_expression_inner(self, function_name: str, var: Variable, lambda_body: ast.AST, counting_quantifier: Optional[int] = None) -> ValueOutputExpression:
        assert counting_quantifier is None, 'Counting quantifiers are not supported yet.'
        ctx = get_expression_definition_context()
        with ctx.new_variables(var):
            if function_name == 'forall':
                return ForallExpression(var, self._parse_expression_inner(lambda_body))
            else:
                return ExistsExpression(var, self._parse_expression_inner(lambda_body))

    def _parse_function_application(self, function_name: str, expression: ast.Call) -> ValueOutputExpression:
        ctx = get_expression_definition_context()

        parsed_args = [self._parse_expression_inner(arg) for arg in expression.args]
        function = None
        if function_name not in ctx.domain.functions:
            if self.inplace_definition:
                if self.inplace_polymorphic_function:
                    function_name = function_name + '_' + '_'.join([arg.return_type.typename for arg in parsed_args])

                if function_name in ctx.domain.functions:
                    function = ctx.domain.functions[function_name]
                else:
                    function = Function(function_name, FunctionType(get_types(parsed_args), BOOL))
                    ctx.domain.define_function(function)
            else:
                raise KeyError(f'Function {function_name} is not defined in the domain.')
        else:
            function = ctx.domain.functions[function_name]
        return FunctionApplicationExpression(function, parsed_args)

    def _get_type_from_domain(self, domain: DSLDomainBase, name: str) -> ObjectType:
        if name == 'bool':
            return BOOL
        elif name == 'int':
            return INT64
        else:
            if name not in domain.types:
                if self.inplace_definition_type:
                    domain.define_type(ObjectType(name))
                else:
                    raise ValueError(f'Undefined type {name}.')

            return domain.types[name]


def ast_get_literal_or_class_name(const: ast.AST) -> str:
    """Get the literal value or identifier name of a constant.

    Args:
        const: the constant, should be either :class:`ast.Constant` or :class:`ast.Name`.

    Returns:
        the literal value or identifier name.
    """
    if isinstance(const, ast.Constant):
        return const.value
    elif isinstance(const, ast.Name):
        return const.id
    else:
        raise TypeError(f'Expect an ast.Constant or a ast.Name, got {type(const)}.')


def ast_get_function_definition(module: ast.Module) -> ast.FunctionDef:
    """Get the single function definition in the module.

    Args:
        module: the module. It should contains exactly one function definition.

    Returns:
        the function definition.
    """
    assert len(module.body) == 1, f'Expect one single function definition, got {len(module.body)}.'
    assert isinstance(module.body[0], ast.FunctionDef), f'Expect a function definition, got {module.body[0]}.'
    return module.body[0]


def ast_get_expression(module: ast.Module) -> ast.AST:
    """Get the single expression in the module.

    Args:
        module: the module. It should contains exactly one expression.

    Returns:
        the expression.
    """
    assert len(module.body) == 1, f'Expect one single expression, got {len(module.body)}.'
    assert isinstance(module.body[0], (ast.Expr, ast.Assign)), f'Expect an expression, got {module.body[0]}.'
    return module.body[0].value if isinstance(module.body[0], ast.Expr) else module.body[0]


def ast_get_multiple_expressions(module: ast.Module) -> List[ast.AST]:
    """Get the multiple expressions in the module.

    Args:
        module: the module. It should contains multiple expressions.

    Returns:
        the expressions.
    """
    for node in module.body:
        assert isinstance(node, (ast.Expr, ast.Assign)), f'Expect an expression, got {node}.'
    return [node.value if isinstance(node, ast.Expr) else node for node in module.body]


def ast_get_simple_function(function: ast.FunctionDef) -> Tuple[Tuple[Variable, ...], TypeBase, ast.AST]:
    """Get the arguments, return type, and body of a simple function. This function only works for "simple functions".
    That is, the function body contains only a single return statement. This function imposes strong restrictions:

    - The function should have a single return statement.
    - All arguments and return type should be annotated with either a single class name or a string. (It does not support
      type hints like ``List[int]``.)

    Args:
        function: the function definition.

    Returns:
        - arguments: the arguments of the function as a tuple of :class:`~mover._vendor.concepts.dsl.dsl_types.Variable` (types are strings).
        - return_type: the return type of the function (string).
        - body: the body of the function.
    """
    arguments = list()
    for arg in function.args.args:
        assert arg.annotation is not None, f'Expect type annotation for argument {arg.arg}.'
        arguments.append(Variable(arg.arg, ast_get_literal_or_class_name(arg.annotation)))

    assert function.returns is not None, f'Expect return type annotation for function {function.name}.'
    return_type = ast_get_literal_or_class_name(function.returns)

    assert len(function.body) == 1, f'Expect one single return statement, got {len(function.body)}.'
    return_statement = function.body[0]
    assert isinstance(return_statement, ast.Return), f'Expect a return statement, got {return_statement}.'

    return tuple(arguments), return_type, return_statement.value

