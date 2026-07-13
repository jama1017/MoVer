#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : expression_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utilities for manipulating expressions."""

import contextlib
from typing import Optional, Union, Iterator, Tuple, Dict
from mover._vendor.concepts.dsl.dsl_types import ObjectConstant, Variable, ObjectType
from mover._vendor.concepts.dsl.dsl_functions import Function
from mover._vendor.concepts.dsl.value import ValueBase, ListValue
from mover._vendor.concepts.dsl.expression import ExpressionDefinitionContext, Expression, ObjectOrValueOutputExpression, ValueOutputExpression, VariableExpression, ObjectConstantExpression, ConstantExpression, ListCreationExpression, ListFunctionApplicationExpression, FunctionApplicationExpression, BoolExpression, QuantificationExpression
from mover._vendor.concepts.dsl.expression import ListExpansionExpression, GeneralizedQuantificationExpression, ObjectCompareExpression, PredicateEqualExpression, FindAllExpression, FindOneExpression
from mover._vendor.concepts.dsl.expression import ValueCompareExpression, VariableAssignmentExpression, AssignExpression, ConditionalSelectExpression, ConditionalAssignExpression, DeicticSelectExpression, DeicticAssignExpression
from mover._vendor.concepts.dsl.expression import is_and_expr, is_or_expr, is_not_expr, is_forall_expr, is_exists_expr, BoolOpType, QuantificationOpType
from mover._vendor.concepts.dsl.expression_visitor import IdentityExpressionVisitor

__all__ = [
    'iter_exprs', 'find_free_variables',
    'flatten_expression', 'FlattenExpressionVisitor',
    'surface_fol_downcast',
    'ground_fol_expression', 'ground_fol_expression_str',
    'is_simple_bool', 'split_simple_bool', 'get_simple_bool_predicate',
    'simplify_bool_expr'
]


def iter_exprs(expr: Expression) -> Iterator[Expression]:
    """Iterate over all sub-expressions of the input."""
    yield expr
    if isinstance(expr, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, ListCreationExpression):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, ListExpansionExpression):
        yield from iter_exprs(expr.expression)
    elif isinstance(expr, BoolExpression):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, QuantificationExpression):
        yield from iter_exprs(expr.expression)
    elif isinstance(expr, GeneralizedQuantificationExpression):
        yield from iter_exprs(expr.expression)
    elif isinstance(expr, PredicateEqualExpression):
        yield from iter_exprs(expr.predicate)
        yield from iter_exprs(expr.value)
    elif isinstance(expr, ValueCompareExpression):
        yield from iter_exprs(expr.lhs)
        yield from iter_exprs(expr.rhs)
    elif isinstance(expr, AssignExpression):
        yield from iter_exprs(expr.value)
    elif isinstance(expr, ConditionalSelectExpression):
        yield from iter_exprs(expr.predicate)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, ConditionalAssignExpression):
        yield from iter_exprs(expr.value)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, (DeicticSelectExpression, DeicticAssignExpression)):
        yield from iter_exprs(expr.expression)
    elif isinstance(expr, (VariableExpression, ConstantExpression, ObjectConstantExpression)):
        pass
    else:
        raise TypeError('Unknown expression type: {}.'.format(type(expr)))


def find_free_variables(expr: Expression) -> Tuple[Variable, ...]:
    free_variables = dict()
    bounded_variables = dict()

    def dfs(e: Expression):
        if isinstance(e, VariableExpression):
            if e.variable.name not in bounded_variables:
                free_variables[e.variable.name] = e.variable
        elif isinstance(e, ListCreationExpression):
            [dfs(arg) for arg in e.arguments]
        elif isinstance(e, ListExpansionExpression):
            dfs(e.expression)
        elif isinstance(e, (QuantificationExpression, GeneralizedQuantificationExpression)):
            bounded_variables[e.variable.name] = e.variable
            dfs(e.expression)
            del bounded_variables[e.variable.name]
        elif isinstance(e, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
            [dfs(arg) for arg in e.arguments]
        elif isinstance(e, BoolExpression):
            [dfs(arg) for arg in e.arguments]
        elif isinstance(e, (ObjectCompareExpression, ValueCompareExpression)):
            dfs(e.lhs)
            dfs(e.rhs)
        elif isinstance(e, PredicateEqualExpression):
            dfs(e.predicate)
            dfs(e.value)
        elif isinstance(e, ValueCompareExpression):
            dfs(e.lhs)
            dfs(e.rhs)
        elif isinstance(e, AssignExpression):
            dfs(e.value)
        elif isinstance(e, ConditionalSelectExpression):
            dfs(e.predicate)
            dfs(e.condition)
        elif isinstance(e, ConditionalAssignExpression):
            dfs(e.value)
            dfs(e.condition)
        elif isinstance(e, (DeicticSelectExpression, DeicticAssignExpression)):
            bounded_variables[e.variable.name] = e.variable
            dfs(e.expression)
            del bounded_variables[e.variable.name]
        elif isinstance(e, (ConstantExpression, ObjectConstantExpression)):
            pass
        else:
            raise TypeError('Unknown expression type: {}.'.format(type(e)))

    dfs(expr)
    return tuple(free_variables.values())


def flatten_expression(
    expr: Expression,
    mappings: Optional[Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ObjectOrValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
    deep: bool = True,
) -> Union[ObjectOrValueOutputExpression, VariableAssignmentExpression]:
    """Flatten an expression by replacing certain variables or function applications with sub-expressions.
    The input mapping is a dictionary of {expression: sub-expression}. There are two cases:

    - The expression is a :class:`~mover._vendor.concepts.dsl.expression.VariableExpression`, and the sub-expression is a
        :class:`~mover._vendor.concepts.dsl.dsl_types.Variable` or a :class:`~mover._vendor.concepts.dsl.expression.ValueOutputExpression`. In this case,
        the variable expression will is the sub-expression used for replacing the variable.
    - The expression is a :class:`~mover._vendor.concepts.dsl.expression.FunctionApplicationExpression`, and the sub-expression is a
        :class:`~mover._vendor.concepts.dsl.dsl_types.Variable`. Here, the function
        application expression must be a "simple" function application expression, i.e., it contains only variables
        as arguments. The Variable will replace the entire function application expression.

    Args:
        expr: the expression to flatten.
        mappings: a dictionary of {expression: sub-expression} to replace the expression with the sub-expression.
        ctx: a :class:`~mover._vendor.concepts.dsl.expression.ExpressionDefinitionContext`.
        deep: whether to flatten the expression recursively (expand derived functions). Default is True.

    Returns:
        the flattened expression.
    """

    if mappings is None:
        mappings = dict()

    if ctx is None:
        ctx = ExpressionDefinitionContext()

    with ctx.as_default():
        return FlattenExpressionVisitor(ctx, mappings, deep).visit(expr)


class FlattenExpressionVisitor(IdentityExpressionVisitor):
    def __init__(
        self,
        ctx: ExpressionDefinitionContext,
        mappings: Dict[Union[FunctionApplicationExpression, VariableExpression], Union[Variable, ValueOutputExpression]],
        deep: bool = True,
    ):
        self.ctx = ctx
        self.mappings = mappings
        self.deep = deep

    def visit_variable_expression(self, expr: VariableExpression) -> Union[VariableExpression, ValueOutputExpression]:
        rv = expr
        for k, v in self.mappings.items():
            if isinstance(k, VariableExpression):
                if k.name == expr.name:
                    if isinstance(v, Variable):
                        rv = VariableExpression(v)
                    else:
                        rv = v
                    break
        return rv

    def visit_function_application_expression(self, expr: Union[FunctionApplicationExpression, ListFunctionApplicationExpression]) -> Union[VariableExpression, ValueOutputExpression]:
        # Case 1: the function application will be replaced by something in the mappings.
        for k, v in self.mappings.items():
            if isinstance(k, FunctionApplicationExpression):
                if expr.function.name == k.function.name and all(
                    isinstance(a1, VariableExpression) and isinstance(a2, VariableExpression) and a1.name == a2.name for a1, a2 in zip(expr.arguments, k.arguments)
                ):
                    assert isinstance(v, Variable)
                    return VariableExpression(v)

        if not self.deep:
            return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

        # Case 2 contains three sub-cases:
        #   (1) the function is not a derived function
        #   (2) the function corresponds to a state variable
        #   (3) the function is a cacheable function and we do not want to flatten it.
        if not expr.function.is_derived:
            return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

        # Case 3: the function is a derived function and we want to flatten it.
        for arg in expr.function.arguments:
            if not isinstance(arg, Variable):
                raise TypeError(f'Cannot flatten function application {expr} because it contains non-variable arguments.')

        # (1) First resolve the arguments.
        argvs = [self.visit(e) for e in expr.arguments]

        # (2) Make a backup of the current context, and then create a new context using the arguments.
        old_mappings = self.mappings
        self.mappings = dict()
        for arg, argv in zip(expr.function.arguments, argvs):
            if isinstance(arg, Variable):
                self.mappings[VariableExpression(arg)] = argv

        # (3) Flatten the derived expression.
        with self.ctx.with_variables(*expr.function.arguments):
            rv = self.visit(expr.function.derived_expression)

        # (4) Restore the old context.
        self.mappings = old_mappings

        # (5) Flatten the result again, using the old context + mappings.
        return self.visit(rv)
        # return type(rv)(rv.function, [self.visit(e) for e in rv.arguments])

    def visit_list_creation_expression(self, expr: ListCreationExpression) -> ListCreationExpression:
        return type(expr)([self.visit(e) for e in expr.arguments])

    def visit_list_expansion_expression(self, expr: ListExpansionExpression) -> ListExpansionExpression:
        return type(expr)(self.visit(expr.expression))

    def visit_list_function_application_expression(self, expr: ListFunctionApplicationExpression) -> Union[VariableExpression, ValueOutputExpression]:
        return self.visit_function_application_expression(expr)

    def visit_bool_expression(self, expr: BoolExpression) -> BoolExpression:
        return BoolExpression(expr.bool_op, [self.visit(child) for child in expr.arguments])

    def visit_object_compare_expression(self, expr: ObjectCompareExpression) -> ObjectCompareExpression:
        return ObjectCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_value_compare_expression(self, expr: ValueCompareExpression) -> ValueCompareExpression:
        return ValueCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_quantification_expression(self, expr: QuantificationExpression) -> QuantificationExpression:
        if expr.quantification_op is QuantificationOpType.BATCHED:
            with self.ctx.with_variables(expr.variable):
                return QuantificationExpression(expr.quantification_op, expr.variable, self.visit(expr.expression))

        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return QuantificationExpression(expr.quantification_op, dummy_variable, self.visit(expr.expression))

    def visit_find_all_expression(self, expr: FindAllExpression) -> FindAllExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return FindAllExpression(dummy_variable, self.visit(expr.expression))

    def visit_find_one_expression(self, expr: FindOneExpression) -> FindOneExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return FindOneExpression(dummy_variable, self.visit(expr.expression))

    def visit_predicate_equal_expression(self, expr: PredicateEqualExpression) -> PredicateEqualExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_assign_expression(self, expr: AssignExpression) -> AssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_conditional_select_expression(self, expr: ConditionalSelectExpression) -> ConditionalSelectExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.condition))

    def visit_deictic_select_expression(self, expr: DeicticSelectExpression) -> DeicticSelectExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return type(expr)(dummy_variable, self.visit(expr.expression))

    def visit_conditional_assign_expression(self, expr: ConditionalAssignExpression) -> ConditionalAssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value), self.visit(expr.condition))

    def visit_deictic_assign_expression(self, expr: DeicticAssignExpression) -> DeicticAssignExpression:
        with self.make_dummy_variable(expr.variable) as dummy_variable:
            return type(expr)(dummy_variable, self.visit(expr.expression))

    def visit_constant_expression(self, expr: Expression) -> Expression:
        return expr

    def visit_object_constant_expression(self, expr: Expression) -> Expression:
        return expr

    @contextlib.contextmanager
    def make_dummy_variable(self, variable: Variable):
        dummy_variable = self.ctx.gen_random_named_variable(variable.dtype)
        dummy_variable_expr = VariableExpression(variable)

        old_mapping = self.mappings.get(dummy_variable_expr, None)
        self.mappings[dummy_variable_expr] = dummy_variable

        yield dummy_variable

        if old_mapping is None:
            del self.mappings[dummy_variable_expr]
        else:
            self.mappings[dummy_variable_expr] = old_mapping


def surface_fol_downcast(expression_1: ValueOutputExpression, expression_2: ValueOutputExpression) -> Optional[Dict[str, Union[Variable, ObjectConstant]]]:
    """Trying to downcast the `expression_1` to the same form as `expression_2`. Downcasting means that
    we try to replace variables in `expression_1` with constants in `expression_2` to make them the same.

    Args:
        expression_1: the first expression.
        expression_2: the second expression.

    Returns:
        the downcasted mapping if the downcasting is successful, otherwise None.
    """
    current_mapping = dict()

    # @jacinle.log_function(verbose=True)
    def dfs(expr1, expr2):
        nonlocal current_mapping

        if isinstance(expr1, VariableExpression):
            if expr1.name in current_mapping:
                expr2_name = _get_variable_or_constant_name(expr2)
                return expr2_name == current_mapping[expr1.name].name
            else:
                if isinstance(expr2, VariableExpression):
                    current_mapping[expr1.name] = expr2.variable
                    return True
                elif isinstance(expr2, ObjectConstantExpression):
                    current_mapping[expr1.name] = expr2.constant
                    return True
                elif isinstance(expr2, ConstantExpression):
                    current_mapping[expr1.name] = expr2.constant
                    return True
                elif isinstance(expr2, ListCreationExpression):
                    current_mapping[expr1.name] = ListValue(expr1.return_type, [_get_variable_or_constant_object(x) for x in expr2.arguments])
                    return True
                else:
                    return False
        elif isinstance(expr1, ObjectConstantExpression):
            if isinstance(expr2, ObjectConstantExpression):
                return expr1.name == expr2.name
            else:
                return False
        elif isinstance(expr1, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
            if not isinstance(expr2, (FunctionApplicationExpression, ListFunctionApplicationExpression)):
                return False
            if expr1.function.name != expr2.function.name:
                return False
            if len(expr1.arguments) != len(expr2.arguments):
                return False
            for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
                if not dfs(arg1, arg2):
                    return False
            return True
        elif isinstance(expr1, BoolExpression):
            if not isinstance(expr2, BoolExpression):
                return False
            if expr1.bool_op != expr2.bool_op:
                return False
            if len(expr1.arguments) != len(expr2.arguments):
                return False
            for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
                if not dfs(arg1, arg2):
                    return False
            return True
        elif isinstance(expr1, QuantificationExpression):
            if not isinstance(expr2, QuantificationExpression):
                return False
            if expr1.quantification_op != expr2.quantification_op:
                return False
            assert expr1.variable.name not in current_mapping
            current_mapping[expr1.variable.name] = expr2.variable
            try:
                return dfs(expr1.expression, expr2.expression)
            finally:
                del current_mapping[expr1.variable.name]
        elif isinstance(expr1, ValueCompareExpression):
            if not isinstance(expr2, ValueCompareExpression):
                return False
            if expr1.compare_op != expr2.compare_op:
                return False
            if not dfs(expr1.lhs, expr2.lhs):
                return False
            return dfs(expr1.rhs, expr2.rhs)
        else:
            raise TypeError(f'Unsupported expression type: {type(expr1)}')

    rv = dfs(expression_1, expression_2)
    if rv:
        return current_mapping
    return None


def ground_fol_expression(expression: ValueOutputExpression, variable_mapping: Dict[Variable, Union[ListValue, ObjectConstant, ValueBase, Variable, str]]) -> ValueOutputExpression:
    """Ground the given FOL expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.

    Returns:
        the grounded expression.
    """
    name2symbol = dict()
    for var, content in variable_mapping.items():
        if isinstance(content, ListValue):
            if isinstance(content.element_type, ObjectType):
                name2symbol[var.name] = ObjectConstantExpression(content)
            else:
                name2symbol[var.name] = ConstantExpression(content)
        elif isinstance(content, ObjectConstant):
            name2symbol[var.name] = ObjectConstantExpression(content)
        elif isinstance(content, ValueBase):
            name2symbol[var.name] = ConstantExpression(content)
        elif isinstance(content, Variable):
            name2symbol[var.name] = VariableExpression(content)
        elif isinstance(content, str):
            name2symbol[var.name] = ObjectConstantExpression(ObjectConstant(content, var.dtype))
        else:
            raise TypeError(f'Unsupported type: {type(content)}')

    bounded_variables = set()

    def dfs(e):
        if isinstance(e, VariableExpression):
            if e.name not in bounded_variables:
                return name2symbol[e.name]
            return e
        elif isinstance(e, ObjectConstantExpression):
            return e
        elif isinstance(e, ConstantExpression):
            return e
        elif isinstance(e, ListFunctionApplicationExpression):
            return ListFunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, FunctionApplicationExpression):
            return FunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, BoolExpression):
            return BoolExpression(e.bool_op, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, QuantificationExpression):
            bounded_variables.add(e.variable.name)
            rv = QuantificationExpression(e.quantification_op, e.variable, dfs(e.expression))
            bounded_variables.remove(e.variable.name)
            return rv
        else:
            raise TypeError(f'Unsupported expression type: {type(e)}')

    return dfs(expression)


def ground_fol_expression_str(expression: ValueOutputExpression, variable_mapping: Dict[str, Union[ListValue, ObjectConstant, ValueBase, Variable]]) -> ValueOutputExpression:
    """Ground the given FOL expression with the given variable mapping.

    Args:
        expression: the expression to ground.
        variable_mapping: the variable mapping, which is a mapping from the Variable object to the constant name.

    Returns:
        the grounded expression.
    """

    name2symbol = dict()
    for var, content in variable_mapping.items():
        if isinstance(content, ListValue):
            if isinstance(content.element_type, ObjectType):
                name2symbol[var] = ObjectConstantExpression(content)
            else:
                name2symbol[var] = ConstantExpression(content)
        elif isinstance(content, ObjectConstant):
            name2symbol[var] = ObjectConstantExpression(content)
        elif isinstance(content, ValueBase):
            name2symbol[var] = ConstantExpression(content)
        elif isinstance(content, Variable):
            name2symbol[var] = VariableExpression(content)
        else:
            raise TypeError(f'Unsupported type: {type(content)}')
    bounded_variables = set()

    def dfs(e):
        if isinstance(e, VariableExpression):
            if e.name not in bounded_variables:
                return name2symbol[e.name]
            return e
        elif isinstance(e, ObjectConstantExpression):
            return e
        elif isinstance(e, ListFunctionApplicationExpression):
            return ListFunctionApplicationExpression(e.function, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, FunctionApplicationExpression):
            arguments = [dfs(arg) for arg in e.arguments]
            return FunctionApplicationExpression(e.function, arguments)
        elif isinstance(e, BoolExpression):
            return BoolExpression(e.bool_op, [dfs(arg) for arg in e.arguments])
        elif isinstance(e, QuantificationExpression):
            bounded_variables.add(e.variable.name)
            rv = QuantificationExpression(e.quantification_op, e.variable, dfs(e.expression))
            bounded_variables.remove(e.variable.name)
            return rv
        else:
            raise TypeError(f'Unsupported expression type: {type(e)}')

    return dfs(expression)


def _get_variable_or_constant_name(expr: Union[VariableExpression, ObjectConstantExpression]) -> str:
    """Get the name of the given variable or constant expression."""
    if isinstance(expr, VariableExpression):
        return expr.name
    elif isinstance(expr, ObjectConstantExpression):
        return expr.name
    else:
        raise TypeError(f'Unsupported type: {type(expr)} for _get_variable_or_constant_name.')


def _get_variable_or_constant_object(expr: Union[VariableExpression, ObjectConstantExpression]) -> Union[Variable, ObjectConstant]:
    """Get the object of the given variable or constant expression."""
    if isinstance(expr, VariableExpression):
        return expr.variable
    elif isinstance(expr, ObjectConstantExpression):
        return expr.constant
    else:
        raise TypeError(f'Unsupported type: {type(expr)} for _get_variable_or_constant_object.')


def is_simple_bool(expr: Expression) -> bool:
    """Check if the expression is a simple Boolean expression. That is, it is either a Boolean non-derived function application
    or the negation of a Boolean non-derived function application.

    Args:
        expr: the expression to check.

    Returns:
        True if the expression is a simple boolean expression, False otherwise.
    """

    if isinstance(expr, FunctionApplicationExpression) and not expr.function.is_derived:
        return True
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return is_simple_bool(expr.arguments[0])
    return False


def split_simple_bool(expr: Expression, initial_negated: bool = False) -> Tuple[Optional[FunctionApplicationExpression], bool]:
    """
    If the expression is a simple Boolean expression (see :func:`is_simple_bool`),
    it returns the feature definition and a boolean indicating whether the expression is negated.

    Args:
        expr (Expression): the expression to be checked.
        initial_negated (bool, optional): whether outer context of the feature expression is a negated function.

    Returns:
        a tuple of the feature application and a boolean indicating whether the expression is negated.
        The first element is None if the feature is not a simple Boolean feature application.
    """
    if isinstance(expr, FunctionApplicationExpression) and not expr.function.is_derived:
        return expr, initial_negated
    if isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT:
        return split_simple_bool(expr.arguments[0], not initial_negated)
    return None, initial_negated


def get_simple_bool_predicate(expr: Expression) -> Function:
    """If the expression is a simple bool (see :func:`is_simple_bool`), it returns the underlying predicate.

    Args:
        expr: the expression, assumed to be a simple Boolean expression.

    Returns:
        the underlying predicate.
    """
    if isinstance(expr, FunctionApplicationExpression) and not expr.function.is_derived:
        return expr.function
    assert isinstance(expr, BoolExpression) and expr.bool_op is BoolOpType.NOT
    return get_simple_bool_predicate(expr.arguments[0])


def simplify_bool_expr(expr: ValueOutputExpression, propagate_negation: bool = False) -> ValueOutputExpression:
    """Simplify a Boolean expression. Currently only supports AND, OR, NOT, FORALL, EXISTS."""

    def merge_bool_expr(op: BoolOpType, args):
        if op == BoolOpType.AND:
            new_args = []
            for arg in args:
                if is_and_expr(arg):
                    new_args.extend(arg.arguments)
                else:
                    new_args.append(arg)
            return BoolExpression(BoolOpType.AND, new_args)
        elif op == BoolOpType.OR:
            new_args = []
            for arg in args:
                if is_or_expr(arg):
                    new_args.extend(arg.arguments)
                else:
                    new_args.append(arg)
            return BoolExpression(BoolOpType.OR, new_args)
        else:
            return BoolExpression(op, args)

    def dfs(e: ValueOutputExpression, negated: bool):
        if is_and_expr(e):
            if negated:
                return merge_bool_expr(BoolOpType.OR, [dfs(arg, True) for arg in e.arguments])
            else:
                return merge_bool_expr(BoolOpType.AND, [dfs(arg, False) for arg in e.arguments])
        elif is_or_expr(e):
            if negated:
                return merge_bool_expr(BoolOpType.AND, [dfs(arg, True) for arg in e.arguments])
            else:
                return merge_bool_expr(BoolOpType.OR, [dfs(arg, False) for arg in e.arguments])
        elif is_not_expr(e):
            if propagate_negation:
                return dfs(e.arguments[0], not negated)
            else:
                if is_not_expr(e.arguments[0]):
                    return dfs(e.arguments[0].arguments[0], negated)
                else:
                    return BoolExpression(BoolOpType.NOT, [dfs(e.arguments[0], not negated)])
        elif is_forall_expr(e):
            if negated:
                return QuantificationExpression(QuantificationOpType.EXISTS, e.variable, dfs(e.expression, True))
            else:
                return QuantificationExpression(QuantificationOpType.FORALL, e.variable, dfs(e.expression, False))
        elif is_exists_expr(e):
            if negated:
                return QuantificationExpression(QuantificationOpType.FORALL, e.variable, dfs(e.expression, True))
            else:
                return QuantificationExpression(QuantificationOpType.EXISTS, e.variable, dfs(e.expression, False))
        else:
            if negated:
                return BoolExpression(BoolOpType.NOT, [e])
            else:
                return e

    return dfs(expr, False)

