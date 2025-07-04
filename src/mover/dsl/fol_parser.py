import ast
from typing import Optional

from concepts.dsl.dsl_types import Variable, BOOL, INT64
from concepts.dsl.dsl_functions import Function, FunctionType
from concepts.dsl.expression import GeneralizedQuantificationExpression, ValueOutputExpression, FunctionApplicationExpression, AndExpression, OrExpression, get_expression_definition_context, get_types, ConstantExpression
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser
from concepts.dsl.value import Value


class FOLParser(FOLPythonParser):
    def _is_quantification_expression_name(self, name: str) -> bool:
        return name in ['exists', 'forall', 'all', 'iota', 'describe', 'execute', 'point', 'count', 'view']

    def _parse_quantification_expression_inner(self, function_name: str, var: Variable, body: ast.Call, counting_quantifier: Optional[int] = None) -> ValueOutputExpression:
        ctx = get_expression_definition_context()
        if function_name in ['exists', 'forall']:
            assert var.dtype.typename in ['Object', 'Motion'], f'Quantification variable must be of type Object or Motion, got {var.dtype}.'
            rv = super()._parse_quantification_expression_inner(function_name, var, body)
            if rv.expression.return_type != BOOL:
                raise ValueError(f'Quantification expression must return a boolean, got {rv.expression.return_type}.')
            return rv
        elif function_name in ['all', 'iota']:
            if counting_quantifier is not None:
                function_name = (function_name, counting_quantifier)
            assert var.dtype.typename in ['Object', 'Motion'], f'Quantification variable must be of type Object or Motion, got {var.dtype}.'

            if var.dtype.typename == 'Object':
                if function_name == 'iota' and counting_quantifier is None:
                    # Conventional single-object iota.
                    return_type = self.domain.types['Object']
                else:
                    ## Jiaju: always return Object
                    # return_type = self.domain.types['ObjectSet']
                    return_type = self.domain.types['Object']

            elif var.dtype.typename == 'Motion':
                if function_name == 'iota' and counting_quantifier is None:
                    return_type = self.domain.types['Motion']
                else:
                    raise NotImplementedError('Does not support MotionSet')
            else:
                raise TypeError(f'Unknown type name: {var.dtype.typename}.')

            with ctx.new_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=return_type
                )
        elif function_name == 'describe':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for describe().'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        elif function_name == 'count':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for count().'
            assert var.dtype.typename == 'Object', f'Counting variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=INT64
                )
        elif function_name == 'execute':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for execute().'
            assert var.dtype.typename == 'Motion', f'Execute variable must be of type Motion, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=BOOL
                )
        elif function_name == 'point':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for point().'
            assert var.dtype.typename == 'Object', f'Point variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        elif function_name == 'view':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for view().'
            assert var.dtype.typename == 'Object', f'View variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        else:
            raise ValueError(f'Unknown quantification expression name: {function_name}.')

    def _parse_function_application(self, function_name: str, expression: ast.Call):
        if function_name == 'query':  # bypass query.
            assert len(expression.args) == 1, f'query() takes exactly one argument, got {len(expression.args)}: {ast.dump(expression)}'
            return self._parse_expression_inner(expression.args[0])
        else:
            return self._parse_function_application_simple(function_name, expression)

    def _parse_function_application_simple(self, function_name: str, expression: ast.Call) -> ValueOutputExpression:
        ctx = get_expression_definition_context()
        # print("-------ctx:{}------".format(ctx.domain.functions))
        # print(expression.args)

        # parsed_args = [self._parse_expression_inner(arg) for arg in expression.args]

        parsed_args = []
        for i, arg in enumerate(expression.args):
            if isinstance(arg, ast.Constant):
                if function_name not in ctx.domain.functions:
                    arg_type = ctx.domain.types['concept_name']
                else:
                    arg_type = ctx.domain.functions[function_name].ftype.argument_types[i]
                parsed_args.append(ConstantExpression(Value(arg_type, arg.value)))
            elif isinstance(arg, ast.List):
                arg_type = ctx.domain.functions[function_name].ftype.argument_types[i]
                if arg_type.alias == 'motion_direction':
                    value = []
                    for elt in arg.elts:
                        if isinstance(elt, ast.Constant):
                            value.append(elt.value)
                        elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                            value.append(-elt.operand.value)
                        else:
                            raise ValueError(f'Unsupported motion direction value: {ast.dump(elt)}')
                    parsed_args.append(ConstantExpression(Value(arg_type, value)))
                else:
                    raise NotImplementedError('Other List is not implemented.')
            else:
                parsed_args.append(self._parse_expression_inner(arg))

        function = None

        if function_name not in ctx.domain.functions:
            # NB(Jiayuan Mao @ 2023/05/10): added to handle parsing failures.
            if function_name == 'and_':
                return AndExpression(*parsed_args)
            elif function_name == 'or_':
                return OrExpression(*parsed_args)

            if self.inplace_definition or self.inplace_polymorphic_function:
                assert self.inplace_polymorphic_function

                # for arg in parsed_args:
                #     if not isinstance(arg.return_type, ObjectType):
                #         raise ValueError(f'In-place polymorphic function definition requires all arguments to be object-typed, got {arg.return_type}.')

                if self.inplace_polymorphic_function:
                    function_name = function_name + '_' + '_'.join([arg.return_type.typename for arg in parsed_args])

                if function_name in ctx.domain.functions:
                    function = ctx.domain.functions[function_name]
                else:
                    if self.inplace_definition:
                        function = Function(function_name, FunctionType(get_types(parsed_args), BOOL))
                        ctx.domain.define_function(function)
                    else:
                        raise KeyError(f'Function {function_name} is not defined in the domain.')
            else:
                raise KeyError(f'Function {function_name} is not defined in the domain.')
        else:
            function = ctx.domain.functions[function_name]
        return FunctionApplicationExpression(function, parsed_args)
