#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : function_domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Function domain contains the definition for a collection of types and function types."""

import inspect
import collections
import contextlib
from typing import Any, Optional, Union, List, Dict, Callable
from jacinle.logging import get_logger
from jacinle.utils.printing import indent_text

from mover._vendor.concepts.dsl.dsl_types import TypeBase, AutoType, ObjectType, ValueType, UnionType, Variable
from mover._vendor.concepts.dsl.dsl_functions import FunctionType, OverloadedFunctionType, Function, FunctionArgumentResolutionError, get_function_argument_resolution_context
from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.expression import ExpressionDefinitionContext, ConstantExpression, FunctionApplicationExpression, get_type
from mover._vendor.concepts.dsl.value import Value

logger = get_logger(__file__)

__all__ = ['FunctionDomain', 'resolve_lambda_function_type']


class FunctionDomain(DSLDomainBase):
    """A basic domain definition that contains a collection of types, constants, and functions."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.__define_overloaded_functions = collections.defaultdict(list)

    AllowedTypes = Union[ObjectType, ValueType, FunctionType]

    name: str
    types: Dict[str, Union[ObjectType, ValueType]]
    functions: Dict[str, Function]
    constants: Dict[str, Value]

    def define_overloaded_function(self, name: str, overloads: List[Union[Function, Callable]], implementation: bool = True):
        """Define a function with multiple overloads.

        Args:
            name: the name of the function.
            overloads: the overloads of the function. Each overload should be a python function or a :class:`Function` instance.
            implementation: whether to store the function body of `function` as the implementation of the function.
        """
        overloads_qs = list()
        overloads_function_body = list()
        for f in overloads:
            if isinstance(f, Function):
                overloads_qs.append(f.ftype)
                overloads_function_body.append(f.function_body)
            else:
                overloads_qs.append(FunctionType.from_annotation(f))
                overloads_function_body.append(f)

        self.functions[name] = Function(
            name, OverloadedFunctionType(overloads_qs),
            overridden_call=None, resolved_from=None,
        )

    def lam(self, lambda_expression: Callable, name: str = '__lambda__', typing_cues: Optional[Dict[str, Union[str, TypeBase]]] = None) -> Function:
        """Parse a lambda expression into a Function instance.

        Args:
            lambda_expression: the lambda expression.
            name: the name of the function.
            typing_cues: the typing cues for the function. It should be a dictionary that maps the argument names to the types.
            If you want to specify the return type, use the key ``'return'``.

        Returns:
            The parsed :class:`Function` instance.
        """

        # TODO (Jiayuan Mao): fix this.
        # sig = inspect.signature(lambda_expression)
        # argument_names = list(sig.parameters.keys())
        # if len(argument_names) == 0:
        #     return lambda_expression()

        try:
            function_type = FunctionType.from_annotation(lambda_expression)
        except FunctionArgumentResolutionError:
            function_type = resolve_lambda_function_type(self, lambda_expression, typing_cues)

        return Function(name, function_type, overridden_call=lambda_expression)

    def __getattr__(self, name: str) -> Union[TypeBase, Value, Function]:
        if name.startswith('t_'):
            return self.types[name[2:]]
        elif name.startswith('c_'):
            return self.constants[name[2:]]
        elif name.startswith('f_'):
            return self.functions[name[2:]]
        raise AttributeError(name)

    @contextlib.contextmanager
    def define(self, implementation=True):
        """A context manager that defines the types, constants, and functions in the domain.

        Usage:

        .. code-block:: python

            with domain.define():
                A = domain.define_type('A', ...)
                B = domain.define_type('B', ...)

                # Definition of constants. Equivalent to `domain.define_const(domain.types['A'], 'a')`.
                a: A
                b: B

                # Definition of functions.
                def f(a: A, b: B) -> A:
                    pass

                # Definition of overloaded functions.
                @domain.overload
                def g(a: A, b: B) -> A:
                    pass

                @domain.overload
                def g(b: B, a: A) -> B:
                    pass

        Note that the definition for constants is only allowed when the code is executed at the global scope.
        """
        locals_before = inspect.stack()[2][0].f_locals.copy()
        annotations_before = locals_before.get('__annotations__', dict()).copy()
        yield self
        locals_after = inspect.stack()[2][0].f_locals.copy()
        annotations_after = locals_after.get('__annotations__', dict()).copy()

        new_functions = {
            k for k in locals_after
            if k not in locals_before and not isinstance(locals_after[k], FunctionDomain)
        }
        new_annotations = {
            k for k in annotations_after
            if k not in annotations_before or annotations_after[k] != annotations_before[k]
        }

        if len(new_annotations) == 0:
            logger.warning('ts.define() for constants is only allowed at the global scope.')

        functions = list()
        for func_name in new_functions:
            var = locals_after[func_name]
            if not inspect.isfunction(var):
                raise ValueError('Support only function definitions in the DEFINE body, got {}.'.format(func_name))
            if func_name in self.__define_overloaded_functions:
                continue
            functions.append((var.__code__.co_firstlineno, var))

        for func_name, overloads in self.__define_overloaded_functions.items():
            lineno = overloads[0].__code__.co_firstlineno
            if len(overloads) > 1:
                functions.append((lineno, (func_name, overloads)))
            else:
                functions.append((lineno, overloads[0]))
        functions.sort()

        for _, f in functions:
            if isinstance(f, tuple):
                self.define_overloaded_function(*f, implementation=implementation)
            else:
                self.define_function(f, implementation=implementation)

        for const_name in new_annotations:
            variable_type = annotations_after[const_name]
            self.define_const(variable_type, const_name)

        self.__define_overloaded_functions = collections.defaultdict(list)

    def overload(self, function):
        """Overload a function. See the docstring for :meth:`define` for usage."""
        self.__define_overloaded_functions[function.__name__].append(function)
        return function

    def format_summary(self) -> str:
        """Format the summary of the domain."""
        fmt = 'Types:\n'
        for type in self.types.values():
            fmt += '  ' + str(type) + '\n'
        fmt += 'Constants:\n'
        for const in self.constants.values():
            fmt += '  ' + str(const) + '\n'
        fmt += 'Functions:\n'
        for function in self.functions.values():
            fmt += '  ' + str(function).replace('\n', '\n  ') + '\n'

        fmt = 'TypeSystem: {}\n'.format(self.name) + indent_text(fmt.rstrip())
        return fmt

    def print_summary(self):
        """Print the summary of the domain."""
        print(self.format_summary())

    def serialize(self, program: Union[FunctionApplicationExpression, Value]) -> Dict[str, Any]:
        """Serialize a program into a dictionary.

        Args:
            program: the program to be serialized.

        Returns:
            the serialized program.
        """

        def dfs(p):
            if isinstance(p, FunctionApplicationExpression):
                record = dict()
                record['__serialize_type__'] = 'function'
                record['function'] = p.function.name
                record['args'] = list()
                for arg in p.arguments:
                    record['args'].append(dfs(arg))
            elif isinstance(p, ConstantExpression):
                record = dict()
                record['__serialize_type__'] = 'object'
                record['type'] = p.constant.dtype.typename
                record['value'] = p.constant.value
            else:
                raise TypeError('Unserializable object: {}.'.format(type(p)))
            return record
        return dfs(program)

    def deserialize(self, dictionary: Dict[str, Any]) -> Union[FunctionApplicationExpression, Value]:
        """Deserialize a program from a dictionary.

        Args:
            dictionary: the dictionary to be deserialized.

        Returns:
            the deserialized program.
        """
        def dfs(d):
            assert '__serialize_type__' in d
            stype = d['__serialize_type__']
            if stype == 'function':
                func = self.functions[d['function']]
                args = list()
                for arg in d['args']:
                    args.append(dfs(arg))
                return func(*args)
            elif stype == 'object':
                return ConstantExpression(Value(self.types[d['type']], d['value']))
            else:
                raise TypeError('Unknown serialized type: {}.'.format(stype))
        return dfs(dictionary)


def _canonicalize_type(fn_domain: FunctionDomain, t: Optional[Union[str, TypeBase]]) -> Optional[TypeBase]:
    if t is None:
        return None
    if isinstance(t, UnionType):
        return UnionType(*tuple(_canonicalize_type(fn_domain, t) for t in t.types))
    if isinstance(t, TypeBase):
        return t

    assert isinstance(t, str)
    return fn_domain.types[t]


def _canonicalize_signature(fn_domain: FunctionDomain, signature: inspect.Signature) -> inspect.Signature:
    """canonicalize the signature of a function. Specifically, it converts the type names to the corresponding type objects.
    For example, given a domain with types ``A`` and ``B``, the signature ``def f(a: 'A', b: 'B') -> A`` will be converted to
    the signature of ``def f(a: A, b: B) -> A``.

    Args:
        fn_domain: the function domain.
        signature: the signature to be canonicalized.

    Returns:
        the canonicalized signature.
    """
    params = [
        inspect.Parameter(v.name, v.kind, default=v.default, annotation=_canonicalize_type(fn_domain, v.annotation))
        for k, v in signature.parameters.items()
    ]
    return_annotation = _canonicalize_type(fn_domain, signature.return_annotation)
    return inspect.Signature(params, return_annotation=return_annotation)


def resolve_lambda_function_type(fn_domain: FunctionDomain, lambda_expression: Callable, typing_cues: Dict[str, Union[str, TypeBase]]) -> FunctionType:
    """Resolve the function type of a lambda expression. It enumerates all possible types for the arguments and the return value,
    and then checks whether the lambda expression satisfies the type constraints. See the docstring for :meth:`FunctionDomain.lam` for usage.

    Args:
        fn_domain: The function domain.
        lambda_expression: The lambda expression.
        typing_cues: The typing cues. The key is the variable name, and the value is the type.
            If you want to specify the typing cue for the return type, use the key 'return'.

    Returns:
        FunctionType: The resolved function type.
    """
    sig = inspect.signature(lambda_expression)
    parameters = list(sig.parameters.keys())
    parameter_types = {k: AutoType for k in parameters}

    for i, (name, param) in enumerate(sig.parameters.items()):
        if param.annotation is not sig.empty:
            parameter_types[param.name] = param.annotation
    return_type = sig.return_annotation if sig.return_annotation is not sig.empty else AutoType

    if typing_cues is not None:
        for k, v in typing_cues.items():
            assert k in parameter_types
            parameter_types[k] = v

        if 'return' in typing_cues:
            return_type = typing_cues['return']

    ctx = ExpressionDefinitionContext()
    with ctx.as_default():
        for k in sig.parameters:
            if parameter_types[k] is AutoType:
                exceptions = list()
                success_types = list()

                for k_type in fn_domain.types.values():
                    parameter_types[k] = k_type
                    parameter_grounding = list()

                    for param in parameters:
                        t = parameter_types[param]
                        parameter_grounding.append(ctx.wrap_variable(Variable(param, t)))

                    try:
                        output = lambda_expression(*parameter_grounding)
                        success_types.append(k_type)

                        if return_type is AutoType:
                            return_type = get_type(output)

                    except FunctionArgumentResolutionError as e:
                        exceptions.append((k_type, e))

                if len(success_types) == 1:
                    parameter_types[k] = success_types[0]
                elif len(success_types) == 0:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Failed to infer argument type for {}.\n'.format(k)
                        fmt += 'Detailed messages are:\n'
                        for t, e in exceptions:
                            fmt += indent_text('Trying {}:\n'.format(t) + indent_text(str(e)) + '\n')
                        raise FunctionArgumentResolutionError(fmt.rstrip())
                else:
                    with get_function_argument_resolution_context().exc():
                        fmt = 'Got ambiguous type for {}.\n'.format(k)
                        fmt += 'Candidates are:\n'
                        for r in success_types:
                            fmt += indent_text(str(r)) + '\n'
                        raise FunctionArgumentResolutionError(fmt.strip())

    if return_type is AutoType and len(parameter_types) == 0:
        return_type = get_type(lambda_expression())

    return FunctionType([parameter_types[i] for i in parameters], return_type, parameters)

