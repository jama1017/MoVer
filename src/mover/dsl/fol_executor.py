import torch
import torch.nn.functional as F
import math
import contextlib
import itertools as it
import operator
import numpy as np
from treelib import Node, Tree
from mover.dsl.utils import *
from mover.dsl.fol_domain import transform_type_to_string, transform_string_to_type, TransformType, parse_motion_type_expr, parse_motion_type_string
from typing import Any, Optional, Tuple, List, Dict

import concepts.dsl.expression as E

from concepts.dsl.dsl_types import BOOL, INT64, Variable
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.parsers.parser_base import ParserBase
from concepts.dsl.parsers.fol_python_parser import FOLProgramAssignmentExpression
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.dsl.executors.tensor_value_executor import TensorValueExecutorBase, TensorValueExecutorReturnType


class ExecutionTraceGetter(object):
    def __init__(self, trace_obj):
        self.trace_obj = trace_obj

    def get(self) -> List[Tuple[E.Expression, TensorValue]]:
        return self.trace_obj

def swap(x, y):
    return y, x

class FOLExecutor(FunctionDomainExecutor):
    def __init__(self, domain: FunctionDomain, parser: Optional[ParserBase] = None, allow_shift_grounding=False):
        super().__init__(domain, parser)
        self.allow_shift_grounding = allow_shift_grounding
        self.variable_stack = dict()
        self.variable_info = dict()
        self.persistent_variable_info = dict()
        self._record_execution_trace = False
        self._execution_trace = list()
        self.execution_tree = Tree()
        self.print_verbose = False
        
    variable_stack: Dict[str, Variable]
    variable_info: Dict[str, dict]

    
    ## DSL functions
    def scene(self):
        return self.grounding.anim_data
    

    ## utils
    def get_scene_info(self):
        anim_data = self.scene()
        objects = anim_data["info"]["objects"]
        steps = int(anim_data['info']['steps'])
        return anim_data, objects, steps
    
    
    def is_vector_in_direction(self, vector, direction):
        vector = np.array(vector)
        direction = np.array(direction)
        scalar_projection = np.dot(direction, vector) / np.linalg.norm(direction)

        return 0 if np.isclose(scalar_projection, 0) else scalar_projection > 0
    
    
    def is_on_right_side(self, x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a*x0 - b*y0
        return a*x + b*y + c >= 0


    def is_point_in_rect(self, point, vertices):
        x = point[0]
        y = point[1]
        num_vert = len(vertices)
        is_right = [self.is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right
    
    
    def is_intersect_aabb(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        return (x1_min <= x2_max and x1_max >= x2_min and 
                y1_min <= y2_max and y1_max >= y2_min)
    
    
    def AABB_by_dimension(self, AABB_a, AABB_b, dimension):
        if dimension == "x":
            start_a = AABB_a['left']
            end_a = AABB_a['right']
            start_b = AABB_b['left']
            end_b = AABB_b['right']
        elif dimension == "y":
            start_a = AABB_a['bottom']
            end_a = AABB_a['top']
            start_b = AABB_b['bottom']
            end_b = AABB_b['top']
            
        return start_a, end_a, start_b, end_b
    
    
    ## Object
    def color(self, obj_var, color_name):
        """
        Determines if there are any objects in the scene with the specified color and returns a tensor indicating the presence of the color for each object.
        Args:
            obj_var (Variable): The variable representing the object.
            color_name (str): The name of the color to check for in the objects.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether each object has the specified color.
        """
        anim_data = self.scene()
        objects = anim_data["info"]["objects"]
        res = []
        for object in objects:
            if ("fill" in object and color_name in object["fill"]) or ("stroke" in object and color_name in object["stroke"]):
                res.append(1)
            else:
                res.append(0)
        
        if self.print_verbose:
            print("---in color")
            
            if 1 in res:
                print(f"There exists an object with color {color_name}")
            else:
                print(f"No object with color {color_name}")
        
        res_tensor = TensorValue(BOOL, [obj_var.name], torch.tensor(res), quantized=False)
        return res_tensor
    
    
    def shape(self, obj_var, shape_name):
        """
        Determines if there exists an object with the specified shape in the scene.
        Args:
            obj_var (Variable): The variable representing the object.
            shape_name (str): The name of the shape to check for.
        Returns:
            TensorValue: A tensor containing boolean values indicating the presence of the specified shape for each object.
        """
        anim_data = self.scene()
        objects = anim_data["info"]["objects"]
        res = []
        for object in objects:
            if object["shape"] == shape_name:
                res.append(1)
            else:
                res.append(0)
        
        if self.print_verbose:
            print("---in shape")
            if 1 in res:
                print(f"There exists an object with shape {shape_name}")
            else:
                print(f"No object with shape {shape_name}")

        res_tensor = TensorValue(BOOL, [obj_var.name], torch.tensor(res), quantized=False)
        return res_tensor
    
    
    def id(self, obj_var, id_name):
        """
        Determines if there exists an object with the specified id in the scene.
        Args:
            obj_var (Variable): The variable representing the object.
            id_name (str): The id of the object to check for.
        Returns:
            TensorValue: A tensor containing boolean values indicating the presence of the specified id for each object.
        """
        anim_data = self.scene()
        objects = anim_data["info"]["objects"]
        res = []
        for object in objects:
            if object["id"] == id_name:
                res.append(1)
            else:
                res.append(0)
        
        if self.print_verbose:
            print("---in id")
            if 1 in res:
                print(f"There exists an object with id {id_name}")
            else:
                print(f"No object with id {id_name}")

        res_tensor = TensorValue(BOOL, [obj_var.name], torch.tensor(res), quantized=False)
        return res_tensor
    
    
    ## Motion
    def type(self, motion_var, motion_type_str):
        """
        Determines the frames during which a specified type of motion occurs for each object in the scene.
        Args:
            motion_var (Variable): The variable representing the motion.
            motion_type_str (str): The string representation of the motion type. The types are "translate", "rotate", and "scale".
        Returns:
            TensorValue: A tensor containing boolean values indicating the presence of the specified motion type 
                         for each object over the frames.

        Motion Vocabulary:
            Motion type: "translate"
                Example verbs:
                    "translate", "shift", "displace", "slide", "move", "relocate", "transfer", "transport", 
                    "convey", "glide", "reposition", "dislocate", "drift", "propel", "push", "migrate", "traverse", "travel", "advance"
                Example verbs with upward direction:
                    "lift", "elevate", "raise", "heighten", "ascend"
                Example verbs with downward direction:
                    "lower", "drop", "fall", "descend", "sink"

            Motion type: "rotate"
                Example verbs:
                    "rotate", "turn", "tilt", "revolve", "pivot", "circumvolve", "gyrate", "birl", "twirl", "whirl", "swirl"
                Example verbs with center:
                    "spin"

            Motion type: "scale"
                Example verbs:
                    "scale", "resize"
                Example verbs with upward direction:
                    "grow", "enlarge", "expand", "dilate", "inflate", "amplify", "magnify"
                Example verbs with upward direction and single axis:
                    "stretch", "extend", "widen", "broaden", "elongate"
                Example verbs with downward direction:
                    "shrink", "contract", "compress", "taper", "diminish", "reduce"
                Example verbs with downward direction and single axis:
                    "narrow", "constrict"
        """
        anim_data, objects, steps = self.get_scene_info()
        motion_type_enum = parse_motion_type_expr(motion_type_str)
        
        res = []
        # generalize to include all objects
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]
            for i in range(steps):
                transform_types = element['transformTypes'][i] # can be more than one transformation type
                transform_types_enum = parse_motion_type_string(transform_types)
                
                type_intersection = motion_type_enum & transform_types_enum
                if type_intersection != TransformType.NONE:
                    frames[i] = True
            
            res.append(frames)
            
        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in type")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion of type {motion_type_str} by {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion of type {motion_type_str} by {res_object_id} takes place")
        
        self.variable_info[motion_var.name]["type"] = motion_type_str
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)


    def direction(self, motion_var, target_direction):
        """
        Determines the frames in which a specified motion variable moves in a target direction for all objects in the scene.
        Args:
            motion_var (Variable): The motion variable to check.
            target_direction (str or list): The target direction to check for. Can be a string for rotation ("clockwise" or "counterclockwise") or a 2D vector for translation and scaling directions. For translation, use [1.0, 0.0] for rightward, [-1.0, 0.0] for leftward, [0.0, 1.0] for upward, and [0.0, -1.0] for downward. For scaling, use 1.0 for increase and -1.0 for decrease, and 0.0 if the direction along a certain axis is not specified.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether the motion variable moves in the target direction for each object over the frames.
        Examples:
            translate upward: direction(m_1, [0.0, 1.0])
            translate to the left: direction(m_1, [-1.0, 0.0])
            rotate clockwise: direction(m_1, "clockwise")
            scale down along the x-axis: direction(m_1, [-1.0, 0.0]) ## Do not use values other than -1.0, 0.0, 1.0 for scaling directions
            scale up (uniformly): direction(m_1, [1.0, 1.0]) ## Do not use values other than -1.0, 0.0, 1.0 for scaling directions
            NOTE: Pay attention that, for scaling, the direction is a 2D vector, where the first element is the x-axis direction and the second element is the y-axis direction.
            If the direction along only one axis is specified, the other axis should be 0.0.
        """
        anim_data, objects, steps = self.get_scene_info()
        
        assert self.variable_info[motion_var.name]
        motion_type_enum = parse_motion_type_expr(self.variable_info[motion_var.name]["type"])
        
        ## generalize to include all objects
        res = []
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]            
            for i in range(steps): 
                transform_types_enum = parse_motion_type_string(element['transformTypes'][i])
                type_intersection = motion_type_enum & transform_types_enum
                if type_intersection != TransformType.NONE:
                    list_is_transform_in_dir = []
                    for curr_type_intersection in type_intersection:
                        motion_type = transform_string_to_type[curr_type_intersection.name.lower()]
                        transform_direction = element['{}_directions'.format(transform_type_to_string[motion_type])][i]
                        match motion_type:
                            case "T":
                                if None in transform_direction:
                                    is_transform_in_dir = False
                                else:    
                                    is_transform_in_dir = self.is_vector_in_direction(transform_direction, target_direction)
                                
                            case "R":
                                if transform_direction == None:
                                    is_transform_in_dir = False
                                else:
                                    direction_0 = 1 if target_direction == "clockwise" else -1
                                    is_transform_in_dir = np.sign(transform_direction) == np.sign(direction_0)
                                
                            case "S":
                                if None in transform_direction:
                                    is_transform_in_dir = False
                                else:
                                    is_tranform_x_in_dir = np.sign(transform_direction[0]) == np.sign(target_direction[0])
                                    if np.sign(target_direction[0]) == 0: ## 0 means there is no constraint on direction
                                        is_tranform_x_in_dir = True
                                    is_tranform_y_in_dir = np.sign(transform_direction[1]) == np.sign(target_direction[1])
                                    if np.sign(target_direction[1]) == 0:
                                        is_tranform_y_in_dir = True
                                    is_transform_in_dir = is_tranform_x_in_dir and is_tranform_y_in_dir
                        list_is_transform_in_dir.append(is_transform_in_dir)
                        
                    any_is_transform_in_dir = np.any(list_is_transform_in_dir)
                    frames[i] = any_is_transform_in_dir
            res.append(frames)

        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in direction")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion with direction {target_direction} by {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion with direction {target_direction} by {res_object_id} takes place")
        
        self.variable_info[motion_var.name]["direction"] = target_direction
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)


    def magnitude(self, motion_var, target_magnitude):
        """
        Analyzes the magnitude of a specified motion variable over a series of animation frames and determines 
        if it matches the target magnitude within a specified tolerance.
        Args:
            motion_var (Variable): The motion variable to analyze.
            target_magnitude (float or list of floats): The target magnitude to compare against. 
                If the motion type is "S" (scale), this should be a list of two floats representing 
                the target scale factors for x and y axes. For scale, if a direction along a certain axis is
                not specified, the target magnitude should be 0.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether the motion with the specified 
            magnitude occurs for each object over the animation frames.
        """
        anim_data, objects, steps = self.get_scene_info()
        
        assert self.variable_info[motion_var.name]
        motion_type_enum = parse_motion_type_expr(self.variable_info[motion_var.name]["type"])
        
        res = []
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]
            tweens = element['tweens']
            
            for tween in tweens:
                start_frame = math.floor(tween['start'])
                end_frame = math.ceil(tween['end'])
                end_frame_index = min(end_frame + 1, steps + 1)
                for i in range(start_frame, end_frame_index):
                    transform_types_enum = parse_motion_type_string(element['transformTypes'][i])
                    type_intersection = motion_type_enum & transform_types_enum
                    if type_intersection != TransformType.NONE:
                        for curr_type_intersection in type_intersection:
                            motion_type = transform_string_to_type[curr_type_intersection.name.lower()]
                            match motion_type:
                                case "T":
                                    start_x = element["translateX_acc"][start_frame]
                                    start_y = element["translateY_acc"][start_frame]
                                    start_mag = math.sqrt(start_x ** 2 + start_y ** 2)
                                    curr_x = element["translateX_acc"][i]
                                    curr_y = element["translateY_acc"][i]
                                    curr_mag = math.sqrt(curr_x ** 2 + curr_y ** 2)
                                    diff = abs(curr_mag - start_mag)
                                    if np.isclose(diff, target_magnitude, atol=0.1):
                                        for i in range(start_frame, i):
                                            frames[i] = True
                                    
                                case "S": 
                                    assert len(target_magnitude) == 2
                                    start_x = element["scaleX_acc"][start_frame]
                                    start_y = element["scaleY_acc"][start_frame]
                                    curr_x = element["scaleX_acc"][i]
                                    curr_y = element["scaleY_acc"][i]
                                    diff_x = curr_x / start_x
                                    diff_y = curr_y / start_y
                                    
                                    is_close_x = np.isclose(diff_x, target_magnitude[0], atol=0.1)
                                    if target_magnitude[0] == 0:
                                        is_close_x = True
                                    is_close_y = np.isclose(diff_y, target_magnitude[1], atol=0.1)
                                    if target_magnitude[1] == 0:
                                        is_close_y = True
                                    
                                    if is_close_x and is_close_y:
                                        for i in range(start_frame, i):
                                            frames[i] = True
                                
                                case "R":
                                    start_mag = element["rotate_acc"][start_frame]
                                    curr_mag = element["rotate_acc"][i]
                                    diff = abs(curr_mag - start_mag)
                                    if np.isclose(diff, target_magnitude, atol=0.1):
                                        for i in range(start_frame, min(i+1, len(frames))):
                                            frames[i] = True
            res.append(frames)
        
        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in magnitude")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion with magnitude {target_magnitude} by {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion with magnitude {target_magnitude} by {res_object_id} takes place")
        
        self.variable_info[motion_var.name]["magnitude"] = target_magnitude
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)
    
    
    def origin(self, motion_var, target_origin):
        """
        Determines the frames during which objects in the scene have a specific origin.
        Args:
            motion_var (Variable): The motion variable associated with the scene.
            target_origin (tuple): The target origin coordinates to check for each object. % sign is used to indicate relative origin.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether each object 
                         has the target origin at each frame.
        
        Examples:
            Note: always output numbers with one decimal points, even if they are integers.
            Rotate around the point (400, 200): origin(m_1, [400.0, 200.0])
            Scale around the object's own center: origin(m_1, ["50%", "50%"])
            Scale around the object's top left corner: origin(m_1, ["0%", "0%"])
        """
        anim_data, objects, steps = self.get_scene_info()
        
        res = []
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]
            for i in range(steps):
                ## if input is absolute origin
                target_world = target_origin
                
                ## if input is relative origin
                if "%" in str(target_origin[0]):
                    ## Get percentages
                    percent_x = float(str(target_origin[0]).strip("%")) / 100.0
                    percent_y = float(str(target_origin[1]).strip("%")) / 100.0
                    
                    ## Get transformed AABB corners that are already computed
                    transformed_AABB = element['transformedPts'][i]
                    top_left = transformed_AABB[0]  # top-left
                    top_right = transformed_AABB[1]  # top-right
                    bottom_right = transformed_AABB[2]  # bottom-right
                    bottom_left = transformed_AABB[3]  # bottom-left
                    
                    ## Compute target point using bilinear interpolation
                    ## First interpolate along top and bottom edges
                    top_point = [
                        top_left[0] * (1 - percent_x) + top_right[0] * percent_x,
                        top_left[1] * (1 - percent_x) + top_right[1] * percent_x
                    ]
                    bottom_point = [
                        bottom_left[0] * (1 - percent_x) + bottom_right[0] * percent_x,
                        bottom_left[1] * (1 - percent_x) + bottom_right[1] * percent_x
                    ]
                    
                    ## Then interpolate between top and bottom points
                    target_world = [
                        top_point[0] * (1 - percent_y) + bottom_point[0] * percent_y,
                        top_point[1] * (1 - percent_y) + bottom_point[1] * percent_y
                    ]
                
                ## Compare with world origin
                frames[i] = np.isclose(element['wldOrigin'][i][0], target_world[0], atol=0.1) and np.isclose(element['wldOrigin'][i][1], target_world[1], atol=0.1)
                
            res.append(frames)
            
        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in origin")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion with origin {target_origin} by {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion with origin {target_origin} by {res_object_id} takes place")
        
        self.variable_info[motion_var.name]["origin"] = target_origin
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)
    
    
    def post(self, motion_var, spatial_concept):
        """
        Processes the motion data and checks if, at the end of the duration of "motion_var", 
        the spatial relationship between two objects expressed in "spatial_concept" has been satisfied.
        Args:
            motion_var (Variable): The motion variable to check.
            spatial_concept (TensorValue): A tensor containing boolean values indicating whether two objects maintained a certain spatial relationship at each frame of the animation.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether spatial_concept has been satisfied at the end of motion_var.
        """
        
        res_frames = spatial_concept.tensor.tolist()
        
        anim_data, objects, steps = self.get_scene_info()
        
        res = []
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]
            tweens = element['tweens']
            
            for tween in tweens:
                start_frame = math.floor(tween['start'])
                end_frame = math.ceil(tween['end'])
                
                if res_frames[min(end_frame, len(res_frames)-1)]:
                    for i in range(start_frame, end_frame):
                        frames[i] = True
            
            res.append(frames)
        
        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in post")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion by {res_object_id} reaches post state at {ranges[0][-1]}")
                else:
                    print(f"No motion by {res_object_id} reaches post state")
                    
        self.variable_info[motion_var.name]["post"] = spatial_concept
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)
        
    
    def duration(self, motion_var, target_duration):
        """
        Determines the frames during which a motion of a specified duration occurs for each object in the scene.
        Args:
            motion_var (Variable): The motion variable to check.
            target_duration (float): The target duration to match for the motion.
        Returns:
            TensorValue: A tensor containing boolean values indicating the frames during which the motion occurs for each object.
        """
        anim_data, objects, steps = self.get_scene_info()
        
        assert self.variable_info[motion_var.name]
        motion_type_enum = parse_motion_type_expr(self.variable_info[motion_var.name]["type"])
        
        res = []
        for object in objects:
            frames = [False for i in range(steps)]
            ## Pad to include all objects
            if object["id"] not in anim_data:
                res.append(frames)
                continue
            
            element = anim_data[object["id"]]
            tweens = element['tweens']
            
            for tween in tweens:
                for motion_type in motion_type_enum:
                    if motion_type.name.lower() == tween['type']:
                        duration = tween['duration']
                        if np.isclose(duration, target_duration):
                            start_frame = math.floor(tween['start'])
                            end_frame = math.ceil(tween['end'])
                            end_frame_index = min(end_frame + 1, steps)
                            for i in range(start_frame, end_frame_index):
                                frames[i] = True
            res.append(frames)
        
        any_res = np.array(res)
        
        ## message
        if self.print_verbose:
            print("---in duration")
            for i in range(len(any_res)):
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion with duration {target_duration} by {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion with duration {target_duration} by {res_object_id} takes place")
        
        self.variable_info[motion_var.name]["duration"] = target_duration
        return TensorValue(BOOL, [motion_var.name], torch.tensor(any_res), quantized=False)
    
    
    def agent(self, motion_var, obj_var):
        """
        Determines if the motion is performed by the agents specified in the object variable.
        Args:
            motion_var (Variable): The motion variable containing the name of the motion.
            obj_var (Variable): The object variable representing the object(s) involved.
        Returns:
            TensorValue: A tensor value indicating the presence of motion for each object over the time steps.
        """
        anim_data, objects, steps = self.get_scene_info()
        
        # handle iota
        if isinstance(obj_var, TensorValue):
            batch_variables = [motion_var.name]
            
            iota = obj_var.tensor
            iota_objects = [objects[i] for i in range(len(objects)) if iota[i] == 1]
            self.variable_info[motion_var.name]["agent"] = obj_var

            if len(iota_objects) == 0:
                res = torch.zeros(len(objects) + 1, steps)
                print("No valid agent is present")
                return TensorValue(BOOL, batch_variables, res, quantized=False)
            
        # handle variable
        else:
            batch_variables = [motion_var.name]
        
        res = []
        for i, object in enumerate(objects):
            frames = [False for i in range(steps)]
            if iota[i] != 1:
                res.append(frames)
                continue
            
            obj_id = object["id"]
            if obj_id in anim_data:
                element = anim_data[obj_id]
                for i in range(steps):
                    transform_types = element['transformTypes'][i]
                    if len(transform_types) > 0:
                        frames[i] = True

            res.append(frames)

        any_res = np.array(res)

        ## message
        if self.print_verbose:
            print("---in agent")
            for i in range(len(any_res)-1):
                if iota[i] != 1:
                    continue
                res_frames = any_res[i]
                res_object_id = objects[i]["id"]
                ranges, is_there = frames_to_range(res_frames)
                if is_there:
                    print(f"A motion with agent {res_object_id} takes place over frames {ranges}")
                else:
                    print(f"No motion with agent {res_object_id} takes place")
        
        return TensorValue(BOOL, batch_variables, torch.tensor(any_res), quantized=False)
    
    
    ## Timing
    def precedes_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("precedes_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if a_range[1]+1 < b_range[0]: # +1 because they cannot be right next to each other
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def precedes(self, motion_var_a, motion_var_b):
        return self.precedes_inner(motion_var_a, motion_var_b, "precedes", True)
    
    def preceded_by(self, motion_var_a, motion_var_b):
        return self.precedes_inner(motion_var_b, motion_var_a, "is preceded by", False)
    
    
    def meets_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("meets_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if np.isclose(a_range[1], b_range[0], atol=5):
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def meets(self, motion_var_a, motion_var_b):
        return self.meets_inner(motion_var_a, motion_var_b, "meets", True)
    
    def met_by(self, motion_var_a, motion_var_b):
        return self.meets_inner(motion_var_b, motion_var_a, "is met by", False)
    
    
    def overlaps_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("overlaps_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if a_range[0] < b_range[0] and b_range[0] < a_range[1] and a_range[1] < b_range[1]:
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def overlaps(self, motion_var_a, motion_var_b):
        """
        Determines if two motion tensors overlap.
        Args:
            motion_var_a (TensorValue): The first motion tensor.
            motion_var_b (TensorValue): The second motion tensor.
        Returns:
            TensorValue: A tensor containing a boolean value indicating whether the two motion tensors overlap.
        """
        return self.overlaps_inner(motion_var_a, motion_var_b, "overlaps", True)
    
    def overlapped_by(self, motion_var_a, motion_var_b):
        return self.overlaps_inner(motion_var_b, motion_var_a, "is overlapped by", False)
    
    
    def finished_by_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("overlaps_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if a_range[0] < b_range[0] and np.isclose(a_range[1], b_range[1], atol=5):
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def finished_by(self, motion_var_a, motion_var_b):
        return self.finished_by_inner(motion_var_a, motion_var_b, "is finished by", True)
    
    def finishes(self, motion_var_a, motion_var_b):
        return self.finished_by_inner(motion_var_b, motion_var_a, "finishes", False)
    
    
    def contains_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("contains_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if a_range[0] < b_range[0] and a_range[1] > b_range[1]:
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def contains(self, motion_var_a, motion_var_b):
        return self.contains_inner(motion_var_a, motion_var_b, "contains", True)
    
    def during(self, motion_var_a, motion_var_b):
        return self.contains_inner(motion_var_b, motion_var_a, "during", False)
    
    
    def starts_inner(self, motion_var_a, motion_var_b, func_name, is_forward):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("starts_inner requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
            
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if np.isclose(a_range[0], b_range[0], atol=5) and a_range[1] < b_range[1]:
                    res = True
                    break
        
        if self.print_verbose:
            if not is_forward:
                a_name, b_name = swap(a_name, b_name)
            if res:
                print(f"[{a_name} {func_name} {b_name}] holds")
            else:
                print(f"[{a_name} {func_name} {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    def starts(self, motion_var_a, motion_var_b):
        return self.starts_inner(motion_var_a, motion_var_b, "starts", True)
    
    def started_by(self, motion_var_a, motion_var_b):
        return self.starts_inner(motion_var_b, motion_var_a, "is started by", False)
    
    
    def equals(self, motion_var_a, motion_var_b):
        if (not isinstance(motion_var_a, TensorValue)) or (not isinstance(motion_var_b, TensorValue)):
            raise ValueError("equals requires two motion tensors")        
        a_ranges, is_there_a = frames_to_range(motion_var_a.tensor.tolist())
        b_ranges, is_there_b = frames_to_range(motion_var_b.tensor.tolist())
        a_name, b_name = motion_var_a.batch_variables, motion_var_b.batch_variables
        if self.print_verbose:
            if (not is_there_a) or (not is_there_b):
                if not is_there_a:
                    print(f"Motion {a_name} does not satisfy all requirements")
                if not is_there_b:
                    print(f"Motion {b_name}n does not satisfy all requirements")
                return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([False]), quantized=False)
        
        res = False
        for a_range in a_ranges:
            for b_range in b_ranges:
                if np.isclose(a_range[0], b_range[0], atol=5) and np.isclose(a_range[1], b_range[1], atol=5):
                    res = True
                    break
        
        if self.print_verbose:
            if res:
                print(f"[{a_name} equals {b_name}] holds")
            else:
                print(f"[{a_name} equals {b_name}] does not hold")
        return TensorValue(BOOL, motion_var_a.batch_variables + motion_var_b.batch_variables, torch.tensor([res]), quantized=False)
    
    
    def t_aggregate(self, concept, motion_var_a, motion_var_b):
        allen_preds = timing_to_temporal[concept]
        res_tensors = []
        for pred in allen_preds:
            for i in range(motion_var_a.tensor.shape[0]):
                for j in range(motion_var_b.tensor.shape[0]):
                    motion_var_a_row = TensorValue(BOOL, [motion_var_a.batch_variables], motion_var_a.tensor[i], quantized=False)
                    motion_var_b_row = TensorValue(BOOL, [motion_var_b.batch_variables], motion_var_b.tensor[j], quantized=False)
                    pred_tensor = eval("self." + pred)(motion_var_a_row, motion_var_b_row).tensor
                    res_tensors.append(pred_tensor)
                
        res = np.any(np.array(res_tensors), axis=0)
        if self.print_verbose:
            print(res)
        return TensorValue(BOOL, [motion_var_a.batch_variables, motion_var_b.batch_variables], torch.tensor(np.array([res])), quantized=False)
    
    
    def t_before(self, motion_var_a, motion_var_b):
        """
        Determines if the motion represented by `motion_var_a` happens before the motion represented by `motion_var_b`.
        Args:
            motion_var_a: The variable that represents he first motion tensor.
            motion_var_b: The variable that represents he second motion tensor.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `motion_var_a` happens before `motion_var_b`.
        """
        return self.t_aggregate("t_before", motion_var_a, motion_var_b)

    
    def t_while(self, motion_var_a, motion_var_b):
        """
        Determines if the motion represented by `motion_var_a` overlaps in time with the motion represented by `motion_var_b`.
        Args:
            motion_var_a: The variable that represents he first motion tensor.
            motion_var_b: The variable that represents he second motion tensor.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `motion_var_a` overlaps with `motion_var_b`.
        """
        return self.t_aggregate("t_while", motion_var_a, motion_var_b)
    
    
    def t_after(self, motion_var_a, motion_var_b):
        """
        Determines if the motion represented by `motion_var_a` happens after the motion represented by `motion_var_b`.
        Args:
            motion_var_a: The variable that represents he first motion tensor.
            motion_var_b: The variable that represents he second motion tensor.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `motion_var_a` happens after by `motion_var_b`.
        """
        return self.t_aggregate("t_after", motion_var_a, motion_var_b)

    
    
    ## Spatial
    def s_precedes_inner(self, obj_var_a, obj_var_b, dimension = str, message = str):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_precedes_inner requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]            
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            comp_operator = operator.lt if dimension == "x" else operator.gt
            if comp_operator(end_a, start_b):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)
    
    def s_precedes(self, obj_var_a, obj_var_b, dimension):
        return self.s_precedes_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} precedes {obj_var_b.batch_variables}")

    def s_preceded_by(self, obj_var_a, obj_var_b, dimension):
        return self.s_precedes_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} is preceded by {obj_var_b.batch_variables}")
    
    
    def s_meets_inner(self, obj_var_a, obj_var_b, dimension = str, message = str):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_meets requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]            
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            if np.isclose(end_a, start_b, atol=5):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)
    
    def s_meets(self, obj_var_a, obj_var_b, dimension):
        return self.s_meets_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} meets {obj_var_b.batch_variables}")

    def s_met_by(self, obj_var_a, obj_var_b, dimension):
        return self.s_meets_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} is met by {obj_var_b.batch_variables}")
        
    
    def s_overlaps_inner(self, obj_var_a, obj_var_b, dimension, message):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_overlaps requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            comp_operator = operator.lt if dimension == "x" else operator.gt
            if comp_operator(start_a, start_b) and comp_operator(start_b, end_a) and comp_operator(end_a, end_b):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)
    
    def s_overlaps(self, obj_var_a, obj_var_b, dimension):
        return self.s_overlaps_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} overlaps {obj_var_b.batch_variables}")
    
    def s_overlapped_by(self, obj_var_a, obj_var_b, dimension):
        return self.s_overlaps_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} is overlapped by {obj_var_b.batch_variables}")
    
    
    def s_finished_by_inner(self, obj_var_a, obj_var_b, dimension, message):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_finished_by requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            comp_operator = operator.lt if dimension == "x" else operator.gt
            if comp_operator(start_a, start_b) and np.isclose(end_a, end_b, atol=5):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)
    
    def s_finished_by(self, obj_var_a, obj_var_b, dimension):
        return self.s_finished_by_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} is finished by {obj_var_b.batch_variables}")
    
    def s_finishes(self, obj_var_a, obj_var_b, dimension):
        return self.s_finished_by_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} finishes {obj_var_b.batch_variables}")
    
    
    def s_contains_inner(self, obj_var_a, obj_var_b, dimension, message):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_contains requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            comp_operator = operator.lt if dimension == "x" else operator.gt
            comp_reverse_operator = operator.gt if dimension == "x" else operator.lt
            if comp_operator(start_a, start_b) and comp_reverse_operator(end_a, end_b):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)

    def s_contains(self, obj_var_a, obj_var_b, dimension):
        return self.s_contains_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} contains {obj_var_b.batch_variables}")
    
    def s_during(self, obj_var_a, obj_var_b, dimension):
        return self.s_contains_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} is during {obj_var_b.batch_variables}")
    
    
    def s_starts_inner(self, obj_var_a, obj_var_b, dimension, message):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_starts requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            comp_operator = operator.lt if dimension == "x" else operator.gt
            if np.isclose(start_a, start_b, atol=5) and comp_operator(end_a, end_b):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{message} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)
    
    def s_starts(self, obj_var_a, obj_var_b, dimension):
        return self.s_starts_inner(obj_var_a, obj_var_b, dimension, f"{obj_var_a.batch_variables} starts {obj_var_b.batch_variables}")

    def s_started_by(self, obj_var_a, obj_var_b, dimension):
        return self.s_starts_inner(obj_var_b, obj_var_a, dimension, f"{obj_var_a.batch_variables} is started by {obj_var_b.batch_variables}")
    
    
    def s_equals(self, obj_var_a, obj_var_b, dimension):
        if (not True in obj_var_a.tensor) or (not True in obj_var_b.tensor):
            raise ValueError("s_equals requires two object tensors with at least one True value")
        anim_data, objects, steps = self.get_scene_info()
        element_a = anim_data[objects[obj_var_a.tensor.tolist().index(1)]["id"]]
        element_b = anim_data[objects[obj_var_b.tensor.tolist().index(1)]["id"]]
        res_frames = [False for i in range(steps)]
        for i in range(steps):
            AABB_a = element_a['AABB'][i]
            AABB_b = element_b['AABB'][i]
            start_a, end_a, start_b, end_b = self.AABB_by_dimension(AABB_a, AABB_b, dimension)
            if np.isclose(start_a, start_b, atol=5) and np.isclose(end_a, end_b, atol=5):
                res_frames[i] = True
        ranges, is_there = frames_to_range(res_frames)
        if self.print_verbose:
            print(f"{obj_var_a.batch_variables} equals {obj_var_b.batch_variables} along {dimension} over frames {ranges}")
        return TensorValue(BOOL, [obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res_frames), quantized=False)

    
    def s_aggregate(self, concept, obj_var_a, obj_var_b):
        spatial_pred_pairs = region_to_spatial[concept]
        res_tensors = []
        for pred_pair in spatial_pred_pairs:
            x_pred = eval("self." + pred_pair[0])(obj_var_a, obj_var_b, "x").tensor
            y_pred = eval("self." + pred_pair[1])(obj_var_a, obj_var_b, "y").tensor
            res_tensors.append(np.all([x_pred, y_pred], axis=0))
        res = np.any(np.array(res_tensors), axis=0)
        if self.print_verbose:
            print("in s_aggregate")
            print(concept)
            ranges, is_there = frames_to_range(res)
            print(ranges)
        return TensorValue(BOOL, [concept, obj_var_a.batch_variables, obj_var_b.batch_variables], torch.tensor(res), quantized=False)
    
    
    def s_top(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located on the top of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the top of `obj_var_b`.
        """
        return self.s_aggregate("top", obj_var_a, obj_var_b)
    
    
    def s_bottom(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located on the bottom of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the bottom of `obj_var_b`.
        """
        return self.s_aggregate("bottom", obj_var_a, obj_var_b)
    
    
    def s_left(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located on the left of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the left of `obj_var_b`.
        """
        return self.s_aggregate("left", obj_var_a, obj_var_b)
    
    
    def s_right(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located on the right of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the right of `obj_var_b`.
        """
        return self.s_aggregate("right", obj_var_a, obj_var_b)
    
    
    def s_top_left(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located to the top left of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the top left of `obj_var_b`.
        """
        return self.s_aggregate("top_left", obj_var_a, obj_var_b)
    
    
    def s_top_right(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located to the top right of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the top right of `obj_var_b`.
        """
        return self.s_aggregate("top_right", obj_var_a, obj_var_b)
    
    
    def s_bottom_left(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located to the bottom left of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the bottom left of `obj_var_b`.
        """
        return self.s_aggregate("bottom_left", obj_var_a, obj_var_b)
    
    
    def s_bottom_right(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` is located to the bottom right of object B `obj_var_b` in space.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the bottom right of `obj_var_b`.
        """
        return self.s_aggregate("bottom_right", obj_var_a, obj_var_b)
    
    
    def s_intersect(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` intersects with object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "intersect" and "overlap", etc, or asks that object A should be moved to where object B is.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` intersects with `obj_var_b`.
        """
        return self.s_aggregate("intersecting", obj_var_a, obj_var_b)
    
    
    def s_border(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` borders with (but does not intersect with) object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "adjacent", "touch".
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with `obj_var_b`.
        """
        return self.s_aggregate("bordering", obj_var_a, obj_var_b)
    
    
    def s_top_border(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the top side of object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the top side of B.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the top side of `obj_var_b`.
        """
        return self.s_aggregate("top_bordering", obj_var_a, obj_var_b)
    
    
    def s_bottom_border(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the bottom side of object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the bottom side of B. Note that "to the bottom side of B" does not express bordering, so the general s_bottom() predicate should be used instead.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the bottom side of `obj_var_b`.
        """
        return self.s_aggregate("bottom_bordering", obj_var_a, obj_var_b)
    
    
    def s_left_border(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the left side of object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the left side of B.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the left side of `obj_var_b`.
        """
        return self.s_aggregate("left_bordering", obj_var_a, obj_var_b)
    
    
    def s_right_border(self, obj_var_a, obj_var_b):
        """
        Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the right side of object B `obj_var_b` in space.
        Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the right side of B.
        Args:
            obj_var_a: The variable that represents object a.
            obj_var_b: The variable that represents object b.
        Returns:
            TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the right side of `obj_var_b`.
        """
        return self.s_aggregate("right_bordering", obj_var_a, obj_var_b)
    
    
    ## Functions
    def get_pos(self, obj_var):
        """
        Get the initial position (frame 0) of an object specified by the object variable. Default to the center position of the object.

        Args:
            obj_var (Variable): The object variable representing the object.

        Returns:
            list: A list containing the x and y coordinates of the object.
        """
        anim_data, objects, steps = self.get_scene_info()
        element = anim_data[objects[obj_var.tensor.tolist().index(1)]["id"]]
        
        AABB = element['AABB'][0]
        return [AABB["left"] + AABB["width"] / 2, AABB["top"] + AABB["height"] / 2]
   
   
    def get_size(self, obj_var):
        """
        Retrieves the initial size (frame 0) of the object specified by the object variable.

        Args:
            obj_var (Variable): The object variable representing the object.

        Returns:
            list: A list containing the initial width and height of the object.
        """
        anim_data, objects, steps = self.get_scene_info()
        return objects[obj_var.tensor.tolist().index(1)]["size"]
    
    
    def get_color(self, obj_var):
        """
        Retrieves the color of the object specified by the object variable.

        Args:
            obj_var (Variable): The object variable representing the object.

        Returns:
            str: The color of the object.
        """
        anim_data, objects, steps = self.get_scene_info()
        return objects[obj_var.tensor.tolist().index(1)]["fill"]
    
    
    def get_shape(self, obj_var):
        """
        Retrieves the shape of the object specified by the object variable.

        Args:
            obj_var (Variable): The object variable representing the object.

        Returns:
            str: The shape of the object.
        """
        anim_data, objects, steps = self.get_scene_info()
        return objects[obj_var.tensor.tolist().index(1)]["shape"]
    
    
    def get_id(self, obj_var):
        """
        Retrieves the ID of the object specified by the object variable.

        Args:
            obj_var (Variable): The object variable representing the object.

        Returns:
            str: The ID of the object.
        """
        anim_data, objects, steps = self.get_scene_info()
        return objects[obj_var.tensor.tolist().index(1)]["id"]
    
    
    def get_motion_attribute(self, motion_var, attr):
        """
        Retrieves the value of the attribute of the motion variable specified by attr.

        Args:
            motion_var (Variable): The motion variable representing the motion.
            attr (str): The attribute to retrieve.

        Returns:
            Any: The value of the attribute.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "type" in var_info:
            return var_info["type"]
        return None
    
    
    def get_agent(self, motion_var):
        """
        Retrieves the agent of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            str: The agent of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "agent" in var_info:
            return var_info["agent"]
        return None
    
    
    def get_direction(self, motion_var):
        """
        Retrieves the direction of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            str: The direction of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "direction" in var_info:
            return var_info["direction"]
        return None
    
    
    def get_magnitude(self, motion_var):
        """
        Retrieves the magnitude of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            float: The magnitude of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "magnitude" in var_info:
            return var_info["magnitude"]
        return None
    
    
    def get_origin(self, motion_var):
        """
        Retrieves the origin of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            list: The origin of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "origin" in var_info:
            return var_info["origin"]
        return None
    
    
    def get_duration(self, motion_var):
        """
        Retrieves the duration of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            float: The duration of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "duration" in var_info:
            return var_info["duration"]
        return None
    
    
    def get_post(self, motion_var):
        """
        Retrieves the post condition of the motion variable.

        Args:
            motion_var (Variable): The motion variable representing the motion.

        Returns:
            list: The post condition of the motion.
        """
        var_info = self.persistent_variable_info[motion_var.batch_variables]
        if "post" in var_info:
            return var_info["post"]
        return None
    
    
    @property
    def training(self):
        return self.grounding.training

    def _count(self, x: TensorValue) -> TensorValue:
        if self.training:
            return torch.sigmoid(x.tensor).sum(dim=-1)
        else:
            return (x.tensor > 0).sum(dim=-1).float()
        
    @contextlib.contextmanager
    def record_execution_trace(self):
        self._record_execution_trace = True
        self._execution_trace = list()
        yield ExecutionTraceGetter(self._execution_trace)
        self._record_execution_trace = False
        self._execution_trace = None
    
    def execute_multiple_expressions(self, expr: Tuple[E.Expression, ...], grounding: Optional[Any] = None, expr_list: Optional[List[str]] = None) -> Tuple[TensorValueExecutorReturnType, ...]:
        assert(len(expr) == len(expr_list))

        results = list()
        results_message = []
        
        grounding = grounding if grounding is not None else self._grounding
        with self.with_grounding(grounding):
            for i, e in enumerate(expr):                
                if i > 0:
                    ## print("")
                    results_message.append("")
                    
                res = None
                if isinstance(e, FOLProgramAssignmentExpression):
                    res = self._execute(e.value)
                    assert len(res.batch_variables) == 1, f"Expected a single batch variable, got {res.batch_variables}"
                    res.batch_variables = (e.variable.name)
                    self.variable_stack[e.variable.name] = res
                    results.append(None)
                    
                    assert len(self.variable_info) == 1, f"Expected a variable info stack of size 1, got {len(self.variable_info)}"
                    self.persistent_variable_info[e.variable.name] = self.variable_info.popitem()[1]
                    
                else:
                    res = self._execute(e)
                    results.append(res)
                    
                
                statement_result = True in res.tensor if hasattr(res, "tensor") else res
                header_message = f"{expr_list[i]}: {statement_result}"
                ## print(header_message)
                results_message.append(header_message)
                results_message.append(tree_to_message(self.execution_tree, res))
                self.execution_tree = Tree()
                
            self.variable_stack.clear()
            return "\n".join(results_message)


    def _execute(self, expr: E.Expression, parent_node: Node = None) -> TensorValue:
        rv = self._execute_inner(expr, parent_node)
        
        if self._record_execution_trace:
            self._execution_trace.append((expr, rv))
        return rv

    def _execute_inner(self, expr: E.Expression, parent_node: Node = None) -> TensorValue:
        if isinstance(expr, E.BoolExpression):
            if expr.bool_op is E.BoolOpType.AND:
                curr_node = self.execution_tree.create_node(expr.bool_op, expr, parent=parent_node, data={"type": expr})
                args = [self._execute(arg, parent_node=curr_node) for arg in expr.arguments]
                expanded_args = args
                result = torch.tensor(np.all([a.tensor for a in expanded_args], axis=0))
                and_res = TensorValue(expanded_args[0].dtype, expanded_args[0].batch_variables, result, quantized=False)
                return and_res
                
            elif expr.bool_op is E.BoolOpType.OR:
                curr_node = self.execution_tree.create_node(expr.bool_op, expr, parent=parent_node, data={"type": expr})
                args = [self._execute(arg, parent_node=curr_node) for arg in expr.arguments]
                expanded_args = args
                result = torch.stack([a.tensor for a in args], dim=-1).amax(dim=-1)
                or_res = TensorValue(expanded_args[0].dtype, expanded_args[0].batch_variables, result, quantized=False)
                return or_res
            
            elif expr.bool_op is E.BoolOpType.NOT:
                curr_node = self.execution_tree.create_node(expr.bool_op, expr, parent=parent_node, data={"type": expr})
                args = [self._execute(arg, parent_node=curr_node) for arg in expr.arguments]
                assert len(args) == 1
                result = args[0].tensor
                result = np.logical_not(result)
                return TensorValue(args[0].dtype, args[0].batch_variables, result, quantized=False)
        
        ## executes functions defined in the DSL
        elif isinstance(expr, E.FunctionApplicationExpression):
            if expr.function.name in self.function_implementations:
                func = self.function_implementations[expr.function.name]                
                curr_node = self.execution_tree.create_node(expr.function.name, expr, parent=parent_node, data={"type": expr})
                args = [self._execute(arg, parent_node=curr_node) for arg in expr.arguments]
                func_res = func(*args)
                curr_node.data["res"] = func_res
                return func_res

        elif isinstance(expr, E.VariableExpression):
            assert expr.variable.name in self.variable_stack
            if isinstance(parent_node.data["type"], E.FunctionApplicationExpression):
                parent_node.data.setdefault("VariableExpression", []).append(expr.variable.name)
            return self.variable_stack[expr.variable.name]

        elif isinstance(expr, E.ConstantExpression):
            # e.g., true, false, str, etc
            if isinstance(parent_node.data["type"], E.FunctionApplicationExpression):
                parent_node.data.setdefault("ConstantExpression", []).append(expr.constant.value)
            return expr.constant

        elif isinstance(expr, E.QuantificationExpression):
            # e.g., for all, exists
            assert expr.variable.name not in self.variable_stack
            assert expr.variable.name not in self.variable_info
            self.variable_stack[expr.variable.name] = expr.variable
            self.variable_info[expr.variable.name] = dict()

            try:
                curr_node = self.execution_tree.create_node(expr.quantification_op, expr, parent=parent_node, data={"type": expr, "var": expr.variable})
                value = self._execute(expr.expression, parent_node=curr_node)
                variable_index = value.batch_variables.index(expr.variable.name)
                if expr.quantification_op is E.QuantificationOpType.FORALL:
                    return TensorValue(
                        value.dtype,
                        value.batch_variables[:variable_index] + value.batch_variables[variable_index + 1:],
                        value.tensor.amin(variable_index),
                        quantized=False
                    )
                
                elif expr.quantification_op is E.QuantificationOpType.EXISTS:
                    return TensorValue(
                        value.dtype,
                        value.batch_variables[:variable_index] + value.batch_variables[variable_index + 1:],
                        value.tensor.amax(variable_index),
                        quantized=False
                    )
                    
                else:
                    raise ValueError(f'Unknown quantification op {expr.quantification_op}.')
            finally:
                del self.variable_stack[expr.variable.name]
                del self.variable_info[expr.variable.name] ## del here for exists and forall

        elif isinstance(expr, E.GeneralizedQuantificationExpression):
            ## e.g., iota, 
            if expr.quantification_op == 'iota' or expr.quantification_op == 'all':
                assert expr.variable.name not in self.variable_stack
                assert expr.variable.name not in self.variable_info
                self.variable_stack[expr.variable.name] = expr.variable
                self.variable_info[expr.variable.name] = dict()

                try:
                    curr_node = self.execution_tree.create_node(expr.quantification_op, expr, parent=parent_node, data={"type": expr, "var": expr.variable})
                    value = self._execute(expr.expression, parent_node=curr_node)
                    assert expr.variable.name in value.batch_variables, f'Variable {expr.variable.name} is not in {value.batch_variables}.'

                    ## For object and iota, keep only the first instance of true in the value.tensor, and mask the rest to false
                    if expr.variable.dtype.typename == "Object" and expr.quantification_op == "iota":
                        variable_index = value.batch_variables.index(expr.variable.name)
                        first_true = torch.zeros_like(value.tensor)
                        # Only set to 1 if there is at least one True value
                        if torch.any(value.tensor):
                            first_true_indices = torch.argmax(value.tensor.float(), dim=variable_index)
                            idx = [slice(None)] * value.tensor.ndim
                            idx[variable_index] = first_true_indices
                            first_true[tuple(idx)] = value.tensor[tuple(idx)]
                        value.tensor = first_true
                    
                    
                    ## For motion and iota, keep only the first consecutive range of true values, and mask the rest to false
                    ## if there's no true value, return all false
                    if expr.variable.dtype.typename == "Motion" and expr.quantification_op == "iota":
                        variable_index = value.batch_variables.index(expr.variable.name)
                        consecutive_true = torch.zeros_like(value.tensor)
                        
                        for row_idx in range(value.tensor.shape[0]):
                            if torch.any(value.tensor[row_idx]):
                                row_values = value.tensor[row_idx].tolist()
                                try:
                                    start_idx = row_values.index(True)
                                    end_idx = start_idx
                                    while end_idx < len(row_values) and row_values[end_idx]:
                                        end_idx += 1
                                    consecutive_true[row_idx][start_idx:end_idx] = True
                                    
                                except ValueError:
                                    pass
                                
                        value.tensor = consecutive_true
                        
                    return TensorValue(
                        expr.return_type,
                        value.batch_variables,
                        value.tensor, # just return true false
                        quantized=False
                    )
                finally:
                    del self.variable_stack[expr.variable.name]
    
        else:
            raise ValueError(f'Unknown expression type {type(expr)}.')


def add_index_to_groupby(grouped):
    grouped_with_index = {}
    index = 0
    for k, g in grouped:
        key = k
        list_g = list(g)
        start_index = index
        end_index = index + len(list_g) - 1
        
        grouped_with_index.setdefault(key, []).append([start_index, end_index])
        index = end_index + 1
        
    return grouped_with_index


def frames_to_range(frames):
    frames_group_index_range = add_index_to_groupby(it.groupby(frames))
    is_there = True in frames_group_index_range.keys()
    res_range = frames_group_index_range[True] if is_there else []
    return res_range, is_there


def print_stack(stack):
    for key, value in stack.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, TensorValue):
                    print(f"{key}: {k} - {v.batch_variables}")
                else:
                    print(f"{key}: {k} - {v}")
            if len(value.items()) == 0:
                print(f"{key}:")
        else:
            print(f"{key}: {value}")


def tree_to_message(execution_tree, res):
    res_message = []
    for node in execution_tree.all_nodes():
        if "res" in node.data and hasattr(node.data["res"], "tensor"):
            result = True in node.data["res"].tensor
            msg = f"{node.tag}: {result}"
            ## print(msg)
            res_message.append(msg)
    return '\n'.join(res_message)