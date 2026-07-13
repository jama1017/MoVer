#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : expression_visitor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/30/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A visitor for iterating over expressions."""

from typing import Any, Union

import jacinle
import mover._vendor.concepts.dsl.expression as E
from mover._vendor.concepts.dsl.expression import Expression

__all__ = ['ExpressionVisitor', 'IdentityExpressionVisitor']


class ExpressionVisitor(object):
    """A visitor for iterating over expressions."""

    def visit(self, expr: Expression) -> Any:
        """The main entry point of the visitor. It will call the corresponding method for the given expression type.

        Args:
            expr: the expression to visit.

        Returns:
            the result of the visit.
        """
        if isinstance(expr, E.NullExpression):
            return self.visit_null_expression(expr)
        elif isinstance(expr, E.VariableExpression):
            return self.visit_variable_expression(expr)
        elif isinstance(expr, E.ConstantExpression):
            return self.visit_constant_expression(expr)
        elif isinstance(expr, E.ObjectConstantExpression):
            return self.visit_object_constant_expression(expr)
        elif isinstance(expr, E.ListCreationExpression):
            return self.visit_list_creation_expression(expr)
        elif isinstance(expr, E.ListExpansionExpression):
            return self.visit_list_expansion_expression(expr)
        elif isinstance(expr, E.ListFunctionApplicationExpression):
            return self.visit_list_function_application_expression(expr)
        elif isinstance(expr, E.FunctionApplicationExpression):
            return self.visit_function_application_expression(expr)
        elif isinstance(expr, E.ConditionalSelectExpression):
            return self.visit_conditional_select_expression(expr)
        elif isinstance(expr, E.DeicticSelectExpression):
            return self.visit_deictic_select_expression(expr)
        elif isinstance(expr, E.BoolExpression):
            return self.visit_bool_expression(expr)
        elif isinstance(expr, E.QuantificationExpression):
            return self.visit_quantification_expression(expr)
        elif isinstance(expr, E.GeneralizedQuantificationExpression):
            return self.visit_generalized_quantification_expression(expr)
        elif isinstance(expr, E.FindAllExpression):
            return self.visit_find_all_expression(expr)
        elif isinstance(expr, E.FindOneExpression):
            return self.visit_find_one_expression(expr)
        elif isinstance(expr, E.ObjectCompareExpression):
            return self.visit_object_compare_expression(expr)
        elif isinstance(expr, E.ValueCompareExpression):
            return self.visit_value_compare_expression(expr)
        elif isinstance(expr, E.ConditionExpression):
            return self.visit_condition_expression(expr)
        elif isinstance(expr, E.PredicateEqualExpression):
            return self.visit_predicate_equal_expression(expr)
        elif isinstance(expr, E.AssignExpression):
            return self.visit_assign_expression(expr)
        elif isinstance(expr, E.ConditionalAssignExpression):
            return self.visit_conditional_assign_expression(expr)
        elif isinstance(expr, E.DeicticAssignExpression):
            return self.visit_deictic_assign_expression(expr)
        else:
            raise TypeError(f'Unknown expression type: {type(expr)}.')

    def visit_null_expression(self, expr: E.NullExpression) -> Any:
        raise NotImplementedError()

    def visit_variable_expression(self, expr: E.VariableExpression) -> Any:
        raise NotImplementedError()

    def visit_object_constant_expression(self, expr: E.ObjectConstantExpression) -> Any:
        raise NotImplementedError()

    def visit_constant_expression(self, expr: E.ConstantExpression) -> Any:
        raise NotImplementedError()

    def visit_list_creation_expression(self, expr: E.ListCreationExpression) -> Any:
        raise NotImplementedError()

    def visit_list_expansion_expression(self, expr: E.ListExpansionExpression) -> Any:
        raise NotImplementedError()

    def visit_function_application_expression(self, expr: E.FunctionApplicationExpression) -> Any:
        raise NotImplementedError()

    def visit_list_function_application_expression(self, expr: E.ListFunctionApplicationExpression) -> Any:
        raise NotImplementedError()

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> Any:
        raise NotImplementedError()

    def visit_bool_expression(self, expr: E.BoolExpression) -> Any:
        raise NotImplementedError()

    def visit_quantification_expression(self, expr: E.QuantificationExpression) -> Any:
        raise NotImplementedError()

    def visit_generalized_quantification_expression(self, expr: E.GeneralizedQuantificationExpression) -> Any:
        raise NotImplementedError()

    def visit_find_one_expression(self, expr: E.FindOneExpression) -> Any:
        raise NotImplementedError()

    def visit_find_all_expression(self, expr: E.FindAllExpression) -> Any:
        raise NotImplementedError()

    def visit_object_compare_expression(self, expr: E.ObjectCompareExpression) -> Any:
        raise NotImplementedError()

    def visit_value_compare_expression(self, expr: E.ValueCompareExpression) -> Any:
        raise NotImplementedError()

    def visit_condition_expression(self, expr: E.ConditionExpression) -> Any:
        raise NotImplementedError()

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression) -> Any:
        raise NotImplementedError()

    def visit_assign_expression(self, expr: E.AssignExpression) -> Any:
        raise NotImplementedError()

    def visit_conditional_assign_expression(self, expr: E.ConditionalAssignExpression) -> Any:
        raise NotImplementedError()

    def visit_deictic_assign_expression(self, expr: E.DeicticAssignExpression) -> Any:
        raise NotImplementedError()


class IdentityExpressionVisitor(ExpressionVisitor):
    def visit_null_expression(self, expr: E.NullExpression) -> E.NullExpression:
        return expr

    def visit_variable_expression(self, expr: E.VariableExpression) -> E.VariableExpression:
        return type(expr)(expr.variable)

    def visit_function_application_expression(self, expr: Union[E.FunctionApplicationExpression, E.ListFunctionApplicationExpression]) -> Union[E.FunctionApplicationExpression, E.ListFunctionApplicationExpression]:
        return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

    def visit_list_creation_expression(self, expr: E.ListCreationExpression) -> E.ListCreationExpression:
        return type(expr)([self.visit(e) for e in expr.arguments])

    def visit_list_expansion_expression(self, expr: E.ListExpansionExpression) -> E.ListExpansionExpression:
        return type(expr)(self.visit(expr.expression))

    def visit_list_function_application_expression(self, expr: E.ListFunctionApplicationExpression) -> E.ListFunctionApplicationExpression:
        return type(expr)(expr.function, [self.visit(e) for e in expr.arguments])

    def visit_bool_expression(self, expr: E.BoolExpression) -> E.BoolExpression:
        return E.BoolExpression(expr.bool_op, [self.visit(child) for child in expr.arguments])

    def visit_quantification_expression(self, expr: E.QuantificationExpression) -> E.QuantificationExpression:
        return E.QuantificationExpression(expr.quantification_op, expr.variable, self.visit(expr.expression))

    def visit_generalized_quantification_expression(self, expr: E.GeneralizedQuantificationExpression) -> E.GeneralizedQuantificationExpression:
        return E.GeneralizedQuantificationExpression(expr.quantification_op, expr.variable, self.visit(expr.expression), return_type=expr.return_type)

    def visit_find_one_expression(self, expr: E.FindOneExpression) -> E.FindOneExpression:
        return E.FindOneExpression(expr.variable, self.visit(expr.expression))

    def visit_find_all_expression(self, expr: E.FindAllExpression) -> E.FindAllExpression:
        return E.FindAllExpression(expr.variable, self.visit(expr.expression))

    def visit_object_compare_expression(self, expr: E.ObjectCompareExpression) -> E.ObjectCompareExpression:
        return E.ObjectCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_value_compare_expression(self, expr: E.ValueCompareExpression) -> E.ValueCompareExpression:
        return E.ValueCompareExpression(expr.compare_op, self.visit(expr.lhs), self.visit(expr.rhs))

    def visit_condition_expression(self, expr: E.ConditionExpression) -> Any:
        return type(expr)(self.visit(expr.condition), self.visit(expr.true_value), self.visit(expr.false_value))

    def visit_predicate_equal_expression(self, expr: E.PredicateEqualExpression) -> E.PredicateEqualExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_assign_expression(self, expr: E.AssignExpression) -> E.AssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value))

    def visit_conditional_select_expression(self, expr: E.ConditionalSelectExpression) -> E.ConditionalSelectExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.condition))

    def visit_deictic_select_expression(self, expr: E.DeicticSelectExpression) -> E.DeicticSelectExpression:
        return type(expr)(expr.variable, self.visit(expr.expression))

    def visit_conditional_assign_expression(self, expr: E.ConditionalAssignExpression) -> E.ConditionalAssignExpression:
        return type(expr)(self.visit(expr.predicate), self.visit(expr.value), self.visit(expr.condition))

    def visit_deictic_assign_expression(self, expr: E.DeicticAssignExpression) -> E.DeicticAssignExpression:
        return type(expr)(expr.variable, self.visit(expr.expression))

    def visit_constant_expression(self, expr: Expression) -> Expression:
        return expr

    def visit_object_constant_expression(self, expr: Expression) -> Expression:
        return expr

