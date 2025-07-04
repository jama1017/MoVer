from concepts.dsl.dsl_types import ValueType, ConstantType, BOOL, FLOAT32, VectorValueType, ObjectType
from concepts.dsl.dsl_functions import Function, FunctionTyping
from concepts.dsl.function_domain import FunctionDomain
import ast
from enum import IntFlag

def init_domain():
    domain = FunctionDomain()

    # define types
    t_object = ObjectType('Object')
    t_object_set = ObjectType('Object_Set')
    t_concept_name = ConstantType('concept_name')
    t_shape = ValueType('shape')
    t_color = ValueType('color')
    t_size = VectorValueType(FLOAT32, 3, alias='size')
    t_motion = ObjectType('Motion')
    t_motion_type = ValueType('motion_type')
    t_motion_direction = VectorValueType(FLOAT32, 2, alias='motion_direction')
    t_position = VectorValueType(FLOAT32, 2, alias='position')
    t_magnitude = ValueType('magnitude')

    domain.define_type(t_object)
    domain.define_type(t_object_set)
    domain.define_type(t_concept_name)
    domain.define_type(t_shape)
    domain.define_type(t_color)
    domain.define_type(t_size)
    domain.define_type(t_motion)
    domain.define_type(t_motion_type)
    domain.define_type(t_motion_direction)
    domain.define_type(t_magnitude)
        
    # define functions
    domain.define_function(Function('scene', FunctionTyping[t_object_set]()))
    domain.define_function(Function('color', FunctionTyping[BOOL](t_object, t_concept_name)))
    domain.define_function(Function('shape', FunctionTyping[BOOL](t_object, t_concept_name)))
    domain.define_function(Function('id', FunctionTyping[BOOL](t_object, t_concept_name)))
    
    domain.define_function(Function('get_pos', FunctionTyping[t_position](t_object)))
    domain.define_function(Function('get_size', FunctionTyping[t_size](t_object)))
    domain.define_function(Function('get_color', FunctionTyping[t_concept_name](t_object)))
    domain.define_function(Function('get_shape', FunctionTyping[t_concept_name](t_object)))
    domain.define_function(Function('get_id', FunctionTyping[t_concept_name](t_object)))
    
    domain.define_function(Function('get_type', FunctionTyping[t_concept_name](t_motion)))
    domain.define_function(Function('get_agent', FunctionTyping[t_object](t_motion)))
    domain.define_function(Function('get_direction', FunctionTyping[t_motion_direction](t_motion)))
    domain.define_function(Function('get_origin', FunctionTyping[t_motion_direction](t_motion)))
    domain.define_function(Function('get_magnitude', FunctionTyping[t_magnitude](t_motion)))
    domain.define_function(Function('get_duration', FunctionTyping[t_magnitude](t_motion)))
    domain.define_function(Function('get_post', FunctionTyping[t_concept_name](t_motion)))

    
    domain.define_function(Function('type', FunctionTyping[BOOL](t_motion, t_motion_type)))
    domain.define_function(Function('direction', FunctionTyping[BOOL](t_motion, t_motion_direction)))
    domain.define_function(Function('origin', FunctionTyping[BOOL](t_motion, t_motion_direction)))
    domain.define_function(Function('agent', FunctionTyping[BOOL](t_motion, t_object)))
    domain.define_function(Function('magnitude', FunctionTyping[BOOL](t_motion, t_motion_direction)))
    domain.define_function(Function('duration', FunctionTyping[BOOL](t_motion, t_magnitude)))
    domain.define_function(Function('post', FunctionTyping[BOOL](t_motion, BOOL)))
    
    domain.define_function(Function('precedes', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('preceded_by', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('meets', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('met_by', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('overlaps', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('overlapped_by', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('finished_by', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('finishes', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('contains', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('during', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('starts', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('started_by', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('equals', FunctionTyping[BOOL](t_motion, t_motion)))
    
    domain.define_function(Function('t_before', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('t_while', FunctionTyping[BOOL](t_motion, t_motion)))
    domain.define_function(Function('t_after', FunctionTyping[BOOL](t_motion, t_motion)))
    
    domain.define_function(Function('s_precedes', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_preceded_by', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_meets', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_met_by', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_overlaps', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_overlapped_by', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_finished_by', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_finishes', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_contains', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_during', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_starts', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_started_by', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    domain.define_function(Function('s_equals', FunctionTyping[BOOL](t_object, t_object, t_concept_name)))
    
    domain.define_function(Function('s_top', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_bottom', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_left', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_right', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_top_left', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_top_right', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_bottom_left', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_bottom_right', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_intersect', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_border', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_top_border', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_bottom_border', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_left_border', FunctionTyping[BOOL](t_object, t_object)))
    domain.define_function(Function('s_right_border', FunctionTyping[BOOL](t_object, t_object)))
    
    return domain


### utils
transform_type_to_string = {
    "T": "translate",
    "S": "scale",
    "R": "rotate",
}


transform_string_to_type = {
    "translate": "T",
    "scale": "S",
    "rotate": "R",
}


class TransformType(IntFlag):
    NONE = 0
    
    TRANSLATE = 1 << 0
    ROTATE =    1 << 1
    SCALE =     1 << 2
    
    ANY = TRANSLATE | ROTATE | SCALE


transform_string_to_enum = {
    "translate": TransformType.TRANSLATE,
    "rotate": TransformType.ROTATE,
    "scale": TransformType.SCALE,
    "any": TransformType.ANY,
    "none": TransformType.NONE,
}


class BooleanTransformVisitor(ast.NodeVisitor):
    def visit_Expression(self, node):
        return self.visit(node.body)
    
    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Invert):
            return ~self.visit(node.operand)
        else:
            raise NotImplementedError(node.op.__doc__ + " Operator")
        
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.BitOr):
            return left | right
        elif isinstance(node.op, ast.BitAnd):
            return left & right
        else:
            raise NotImplementedError(node.op.__doc__ + " Operator")
        
    def visit_Name(self, node):
        try:
            return transform_string_to_enum[node.id]
        except KeyError:
            raise NameError(node.id)
        
    def generic_visit(self, node):
        raise RuntimeError("non-boolean expression")


def parse_motion_type_expr(expr):
    visitor = BooleanTransformVisitor()
    parse_tree = ast.parse(expr, mode='eval')
    return visitor.visit(parse_tree)


def parse_motion_type_string(str):
    res = TransformType.NONE
    for letter in str:
        type_str = transform_type_to_string[letter]
        type_enum = transform_string_to_enum[type_str]
        res |= type_enum
    return res