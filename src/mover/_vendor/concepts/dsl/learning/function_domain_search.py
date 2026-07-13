#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : function_domain_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/10/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""An enumerative search algorithm to generate candidate functions and expressions in a simple function domain."""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterable, Sequence, Tuple, List, Callable

from mover._vendor.concepts.dsl.dsl_types import ConstantType, ValueType
from mover._vendor.concepts.dsl.dsl_functions import FunctionType, Function
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.value import Value
from mover._vendor.concepts.dsl.expression import ConstantExpression, FunctionApplicationExpression, VariableExpression
from mover._vendor.concepts.dsl.function_domain import FunctionDomain
from mover._vendor.concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor

__all__ = [
    'FunctionDomainExpressionSearchResult', 'FunctionDomainExpressionEnumerativeSearcher', 'gen_merge_functions',
    'gen_expression_search_result_from_expressions',
    'FunctionArgumentStat', 'stat_function', 'canonicalize_function_parameters',
    'learn_expression_from_examples'
]


_Types = FunctionDomain.AllowedTypes


@dataclass
class FunctionDomainExpressionSearchResult(object):
    expression: Union[ConstantExpression, Function, FunctionApplicationExpression]
    """The expression that is enumerated."""

    depth: int
    """The depth of the expression."""

    nr_constant_arguments: int
    """The number of constant arguments in the expression."""

    nr_variable_arguments: int
    """The number of variable arguments in the expression."""

    nr_function_arguments: int
    """The number of function arguments in the expression."""


class FunctionDomainExpressionEnumerativeSearcher(object):
    """An enumerator of expressions and functions for a function domain."""

    def __init__(self, domain: DSLDomainBase):
        """Initialize the searcher.

        Args:
            domain: the domain of the semantics.
        """
        self.domain = domain
        self._constant_type_cache = self._gen_constant_type_cache()
        self._function_type_cache = self._gen_function_type_cache()

    def _gen_constant_type_cache(self):
        """Generate a dictionary that maps types to constants."""
        cache = defaultdict(list)
        for const in self.domain.constants.values():
            cache[const.dtype].append(const)
        return cache

    def _gen_function_type_cache(self):
        """Generate a dictionary that maps types to functions (by return types)."""
        cache = defaultdict(list)
        for func in self.domain.functions.values():
            if func.is_overloaded:
                for f in func.all_sub_functions:
                    cache[f.return_type].append(f)
            else:
                cache[func.ftype.return_type].append(func)
        return cache

    def gen(
        self,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None,
        *,
        max_depth: int = 3,
        max_variable_arguments: int = 2,
        max_constant_arguments: int = 1,
        max_function_arguments: int = 0,
        search_constants: bool = False,
        hash_function: Optional[Callable[[Union[Function, FunctionApplicationExpression]], Any]] = None,
        verbose: bool = False,
    ):
        return self.gen_constant_expressions(return_type) + self.gen_function_application_expressions(
            return_type, max_depth=max_depth, max_variable_arguments=max_variable_arguments,
            max_constant_arguments=max_constant_arguments, max_function_arguments=max_function_arguments,
            search_constants=search_constants, hash_function=hash_function, verbose=verbose
        )

    def gen_constant_expressions(
        self,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None
    ) -> List[FunctionDomainExpressionSearchResult]:
        """Generate constant expressions of a set of given types.

        Args:
            return_type: the return type of the expressions. If None, all types are allowed.
                It can be a single type, a tuple of types, or a list of types.

        Returns:
            A list of constant expressions.
        """
        constants = list()
        for c in self.domain.constants.values():
            if (
                return_type is None or
                (isinstance(return_type, ConstantType) and c.dtype == return_type) or
                (isinstance(return_type, (tuple, list)) and c.dtype in return_type)
            ):
                constants.append(FunctionDomainExpressionSearchResult(
                    ConstantExpression(c), 1, 0, 0, 0
                ))
        return constants

    def gen_function_application_expressions(
        self,
        return_type: Optional[Union[_Types, Tuple[_Types, ...], List[_Types]]] = None,
        *,
        max_depth: int = 3,
        max_variable_arguments: int = 2,
        max_constant_arguments: int = 1,
        max_function_arguments: int = 0,
        search_constants: bool = False,
        hash_function: Optional[Callable[[Union[Function, FunctionApplicationExpression]], Any]] = None,
        verbose: bool = False,
    ) -> List[FunctionDomainExpressionSearchResult]:
        """Generate functions and function application expressions of a set of given types.

        Args:
            max_depth: the maximum depth of the expressions.
            return_type: the return type of the expressions. If None, all types are allowed.
                It can be a single type, a tuple of types, or a list of types.
            max_variable_arguments: the maximum number of variable arguments of the functions.
            max_constant_arguments: the maximum number of constant arguments of the functions.
                Note that when ``search_constants`` is True, this parameter corresponds to the maximum number of
                constant arguments bound to the function / function application expression.
            max_function_arguments: the maximum number of arguments of the functions.
            search_constants: whether to search for constants.
            verbose: whether to print the search progress.

        Returns:
            A list of function application expressions.
        """

        ftcache = self._function_type_cache
        current = {i: defaultdict(list) for i in range(1, max_depth + 1)}

        def iter_ddl_values(dd):
            for v in dd.values():
                yield from v

        for rtype, functions in ftcache.items():
            for f in functions:
                current[1][f.ftype.return_type].append(gen_merge_functions(f))

        if max_function_arguments > 0:
            for ret_type in self.domain.types.values():
                current[1][ret_type].extend(self._gen_function_primitives(ret_type, max_function_arguments))

        if verbose:
            print('-' * 20, 'Depth', 1, '-' * 100)
            for rtype, functions in current[1].items():
                for f in functions:
                    print(rtype, '\t', f)

        for depth in range(2, max_depth + 1):
            for depth1 in range(1, depth):
                for f1 in list(iter_ddl_values(current[depth1])):
                    for i in range(f1.nr_arguments):
                        for depth2 in range(1, depth - depth1 + 1):
                            for f2 in current[depth2][f1.ftype.argument_types[i]]:
                                current[depth][f1.ftype.return_type].append(
                                    gen_merge_functions(f1, i, f2)
                                )

            if verbose:
                print('-' * 20, 'Depth', depth, '-' * 100)
                for rtype, functions in current[depth].items():
                    for f in functions:
                        print(rtype, '\t', f)

        expressions = list()
        for depth, vs in current.items():
            for f in iter_ddl_values(vs):
                if isinstance(f, Function):
                    stat = stat_function(f)
                    if stat.nr_constant_arguments <= max_constant_arguments and \
                            stat.nr_variable_arguments <= max_variable_arguments and \
                            stat.nr_function_arguments <= max_function_arguments:
                        expressions.append(FunctionDomainExpressionSearchResult(
                            canonicalize_function_parameters(f, ignore_permutation=True),
                            depth, stat.nr_constant_arguments, stat.nr_variable_arguments, stat.nr_function_arguments
                        ))
                else:
                    expressions.append(FunctionDomainExpressionSearchResult(
                        f, depth, 0, 0, 0
                    ))

        expressions = self._unique_function_expressions(expressions, hash_function=hash_function)

        if search_constants:
            expressions = self._bind_constants_to_expressions(expressions)

        if return_type is not None:
            output_expressions = list()
            for result in expressions:
                if _match_return_type(result.expression, return_type):
                    output_expressions.append(result)
            expressions = output_expressions

        return expressions

    def _gen_function_primitives(self, ret_type, nr_function_arguments):
        def function_call(func, *args):
            return func(*args)

        def gen():
            types = tuple(self.domain.types.values())
            for repeat in range(1, nr_function_arguments + 1):
                for arg_types in itertools.product(types, repeat=repeat):
                    yield Function(
                        '__lambda__',
                        FunctionType(
                            [FunctionType(arg_types, ret_type), ] + list(arg_types),
                            ret_type
                        ),
                        overridden_call=function_call,
                    )

        return tuple(gen())

    def _unique_function_expressions(
        self,
        functions: List[FunctionDomainExpressionSearchResult],
        hash_function: Optional[Callable[[Union[Function, FunctionApplicationExpression]], Any]] = None
    ) -> List[FunctionDomainExpressionSearchResult]:
        """Return a list of unique functions and function application expressions.

        Args:
            functions: a list of input expressions.
            hash_function: a custom function that generates a hash for a function or function application expression.
                If None, the default ``str`` function is used.

        Returns:
            A list of expressions without duplicates.
        """

        if hash_function is None:
            hash_function = str

        unique_functions = list()
        unique_function_hashes = set()
        for r in functions:
            h = hash_function(r.expression)
            if h not in unique_function_hashes:
                unique_functions.append(r)
                unique_function_hashes.add(h)

        return unique_functions

    def _bind_constants_to_expressions(
        self,
        functions: List[FunctionDomainExpressionSearchResult],
    ) -> List[FunctionDomainExpressionSearchResult]:
        output_functions = list()
        for result in functions:
            if not isinstance(result.expression, Function):
                output_functions.append(result)
                continue

            f = result.expression
            constant_denotations, constant_types = list(), list()
            for index, argument_type in enumerate(f.ftype.argument_types):
                if isinstance(argument_type, ConstantType):
                    constant_denotations.append(f'#{index}')
                    constant_types.append(argument_type)

            for const in itertools.product(*[
                self._constant_type_cache[t] for t in constant_types
            ]):
                const_mapping = {k: ConstantExpression(v) for k, v in zip(constant_denotations, const)}
                partial_func = f.partial(**const_mapping, execute_fully_bound_functions=True)
                new_r = partial_func
                output_functions.append(FunctionDomainExpressionSearchResult(
                    new_r,
                    result.depth,
                    result.nr_constant_arguments, result.nr_variable_arguments, result.nr_function_arguments
                ))

        return output_functions


def _match_return_type(f: Union[Function, FunctionApplicationExpression], return_types: Union[_Types, Tuple[_Types, ...], List[_Types]]) -> bool:
    """Check if the type of a function or a function application expression matches a set of given return type."""
    if not isinstance(return_types, (tuple, list)):
        return_types = (return_types, )

    if isinstance(f, Function):
        for return_type in return_types:
            if f.ftype.typename == return_type.typename:
                return True
    elif isinstance(f, FunctionApplicationExpression):
        return f.return_type in return_types
    else:
        raise TypeError(f'Expected Function or FunctionApplicationExpression, got {type(f)}.')


def gen_merge_functions(f1: Function, arg_index=None, f2: Optional[Function] = None) -> Function:
    """Generate merge functions. Specifically, given two functions f1 and f2,
    this function generates a new function. For example, given f1(x, y) and f2(z),
    with arg_index = 1, this function generates:

    .. code-block:: python

        def merged(x, z):
            return f1(x, f2(z))

    That is, we first apply f2 with the input arguments, and then apply f1 with the
    rest of the arguments and the output of f2.

    A special case is when f2 is None. In this case, we generate a function that
    is simply a wrapper of f1.

    Args:
        f1: the first function.
        arg_index: the index of the argument of f1 that we want to merge with f2.
        f2: the second function.

    Returns:
        A new function.
    """
    if arg_index is None:
        return Function(
            '__lambda__',
            FunctionType(f1.ftype.argument_types, f1.ftype.return_type), overridden_call=f1
        )
    else:
        f1_arg_types = f1.ftype.argument_types
        f2_arg_types = f2.ftype.argument_types
        arg_types = f2_arg_types + f1_arg_types[:arg_index] + f1_arg_types[arg_index + 1:]

        def new_function_call(*args):
            f2_args = args[:f2.nr_arguments]
            f1_args = list(args[f2.nr_arguments:])

            f2_ret = f2(*f2_args)
            f1_args.insert(arg_index, f2_ret)
            return f1(*f1_args)

        return Function(
            '__lambda__',
            FunctionType(arg_types, f1.ftype.return_type),
            overridden_call=new_function_call
        )


def gen_expression_search_result_from_expressions(expressions: Iterable[Union[ConstantExpression, Function, FunctionApplicationExpression]]) -> List[FunctionDomainExpressionSearchResult]:
    """Generate a list of FunctionDomainExpressionSearchResult from a list of expressions."""
    results = list()
    for expression in expressions:
        if isinstance(expression, (ConstantExpression, FunctionApplicationExpression)):
            results.append(FunctionDomainExpressionSearchResult(expression, 0, 0, 0, 0))
        else:
            stat = stat_function(expression)
            results.append(FunctionDomainExpressionSearchResult(expression, 0, stat.nr_constant_arguments, stat.nr_variable_arguments, stat.nr_function_arguments))
    return results


@dataclass
class FunctionArgumentStat(object):
    """Statistics for the argument list of a function."""

    nr_constant_arguments: int
    nr_variable_arguments: int
    nr_function_arguments: int


def stat_function(f: Function) -> FunctionArgumentStat:
    """Return the number of constants, variables, and functions in a function."""
    nr_variable_arguments = 0
    nr_constant_arguments = 0
    nr_function_arguments = 0
    for arg_t in f.ftype.argument_types:
        if isinstance(arg_t, ConstantType):
            nr_constant_arguments += 1
        elif isinstance(arg_t, FunctionType):
            nr_function_arguments += 1
            nr_variable_arguments += 1  # function arg is also variable arg
        else:
            nr_variable_arguments += 1
    return FunctionArgumentStat(
        nr_constant_arguments=nr_constant_arguments,
        nr_variable_arguments=nr_variable_arguments,
        nr_function_arguments=nr_function_arguments
    )


def canonicalize_function_parameters(
    f: Union[ConstantExpression, FunctionApplicationExpression, Function],
    ignore_permutation: bool = False
) -> Union[ConstantExpression, FunctionApplicationExpression, Function]:
    """Return a new function object with argument reordered: functions, variables, constants.

    Args:
        f: the function to be canonicalized. If the function is a function application expression or a constant expression,
            this function returns the same object.

    Returns:
        A new function object with argument reordered: functions, variables, constants.
    """
    if isinstance(f, (ConstantExpression, FunctionApplicationExpression)):
        return f
    assert isinstance(f, Function)

    # NB(Jiayuan Mao @ 2022/12/10): if the function is a primitive function, just return it.
    if f.overridden_call is None:
        return f

    assert not f.is_overloaded
    assert isinstance(f.derived_expression, FunctionApplicationExpression)
    function_args, variable_args, constant_args = list(), list(), list()

    if ignore_permutation:
        def walk(node: FunctionApplicationExpression):
            if isinstance(node.function, VariableExpression):
                new_index = f.ftype.argument_names.index(node.function.name)
                if new_index not in function_args:
                    function_args.append(new_index)
            for arg in node.arguments:
                if isinstance(arg, VariableExpression):
                    new_index = f.ftype.argument_names.index(arg.name)
                    if isinstance(arg.dtype, ConstantType):
                        if new_index not in constant_args:
                            constant_args.append(new_index)
                    elif isinstance(arg.dtype, ValueType):
                        if new_index not in variable_args:
                            variable_args.append(new_index)
                    else:
                        raise TypeError('Unknown type for anonymous argument #{}, type = {}.'.format(arg.name, arg.dtype))
                elif isinstance(arg, FunctionApplicationExpression):
                    walk(arg)
                else:
                    raise TypeError('Unknown type for anonymous argument type {}.'.format(type(arg)))
        walk(f.derived_expression)
    else:
        for arg_index, arg_t in enumerate(f.ftype.argument_types):
            if isinstance(arg_t, FunctionType):
                function_args.append(arg_index)
            elif isinstance(arg_t, ConstantType):
                constant_args.append(arg_index)
            elif isinstance(arg_t, ValueType):
                variable_args.append(arg_index)
            else:
                raise TypeError('Unknown type for argument {}, type = {}.'.format(arg_index, arg_t))
    new_argument_mapping = list(function_args) + list(variable_args) + list(constant_args)

    f = f.remap_arguments(new_argument_mapping)
    return f


def learn_expression_from_examples(
    domain: FunctionDomain,
    executor: FunctionDomainExecutor,
    input_output: Sequence[Tuple[Sequence[Value], Value, Any]],
    criterion: Callable[[Value, Value], bool],
    candidate_expressions: Optional[Iterable[FunctionDomainExpressionSearchResult]] = None,
    max_depth: int = 3,
    max_function_arguments: int = 0,
    search_constants: bool = True,
) -> Function:
    """Learn a function from examples.

    Args:
        domain: the function domain.
        executor: the executor of the function domain.
        input_output: a sequence of (input, output, grounding) pairs.
        criterion: a function that takes two values and returns True if they are equal.
        candidate_expressions: a sequence of candidate functions. If None, we will generate
            candidate functions automatically.
        max_depth: the maximum depth of the function. Only used when ``candidate_expressions`` is None.
        max_function_arguments: the maximum number of arguments of a function. Only used when
            ``candidate_expressions`` is None.
        search_constants: whether to search for constants. Only used when ``candidate_expressions`` is None.

    Returns:
        A function.
    """

    assert len(input_output) > 0, 'No input-output pairs are given.'
    sample_input, sample_output, _ = input_output[0]
    if len(sample_input) == 0:
        target_type = sample_output.dtype
    else:
        target_type = FunctionType([v.dtype for v in sample_input], sample_output.dtype)

    if candidate_expressions is None:
        domain = FunctionDomainExpressionEnumerativeSearcher(domain)
        candidate_expressions = domain.gen_function_application_expressions(
            target_type,
            max_depth=max_depth,
            max_function_arguments=max_function_arguments,
            search_constants=search_constants
        )

    if isinstance(target_type, FunctionType):
        def score_function(result: FunctionDomainExpressionSearchResult) -> float:
            f = result.expression
            if isinstance(f, FunctionApplicationExpression):
                return -1
            return sum(
                criterion(executor.execute_function(f, *input, grounding=grounding), output)
                for input, output, grounding in input_output
            )
    else:
        def score_function(result: FunctionDomainExpressionSearchResult) -> float:
            f = result.expression
            if isinstance(f, Function):
                return -1
            return sum(
                criterion(executor.execute(f, grounding), output)
                for input, output, grounding in input_output
            )
    return max(candidate_expressions, key=score_function).expression

