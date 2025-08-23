"""
Core data classes for motion natural language generation.
"""
import json
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from mover.nlg.utils import map_dict_type, realize_cont_to_str


@dataclass
class Object:
    """Visual object with shape, fill, stroke, and size properties."""
    shape: Optional[str] = None
    fill: Optional[str] = None
    stroke: Optional[str] = None
    size: Optional[Tuple[float, float]] = None
    
    def __str__(self) -> str:
        """String representation of the object."""
        parts = []
        if self.fill:
            parts.append(self.fill)
        if self.shape:
            parts.append(self.shape)
        return ' '.join(parts) if parts else 'object'


@dataclass
class Motion:
    """Motion with type, agent, direction, magnitude, origin, duration, and post-conditions."""
    type: Optional[str] = None
    agent: Optional[List[Object]] = None
    magnitude: Optional[Union[Tuple[float, float], float]] = None
    direction: Optional[Union[Tuple[float, float], float]] = None
    origin: Optional[Union[Tuple, str]] = None
    duration: Optional[float] = None
    post: Optional[str] = None
    post_reference: Optional[List[Object]] = None
    
    def __str__(self) -> str:
        """String representation of the motion."""
        parts = []
        if self.type:
            parts.append(self.type)
        if self.agent:
            agent_str = ', '.join(str(obj) for obj in self.agent)
            parts.append(f"agent: {agent_str}")
        return ' '.join(parts) if parts else 'motion'
    

predicate_to_type = {
    "object": {"predicate": ["color", "shape"]},
    "motion": {"predicate": ["type", "direction", "magnitude", "origin", "post", "duration", "agent"]},
    "timing": {"predicate": ["precedes", "overlaps", "t_before", "t_while", "t_after"]},
    "function": {"predicate": ["center_concept_name", "center_Object"]},
    "spatial": {"predicate": ["s_top", "s_right", "s_left", "s_bottom", "s_border", "s_intersect", "s_top_left", "s_top_right", "s_bottom_left", "s_bottom_right", "s_right_border", "s_left_border", "s_top_border", "s_bottom_border"]},
}


class Predicate:
    def __init__(self, name: str, vars: list, other_vars: list, args: list):
        self.name = name
        self.vars = vars
        self.other_vars = other_vars
        self.args = args
        self.type = map_dict_type(self.name, predicate_to_type)
        

    def map_dict_type(word: str, dictionary: dict | list):
        for type, words_by_pos in dictionary.items():
            for pos, words in words_by_pos.items():
                rendered_words = realize_cont_to_str(words)
                if word in rendered_words:
                    return type
        return None


    def __str__(self):
        return "Predicate: {}({})".format(self.name, ", ".join([str(var) for var in self.vars] + self.other_vars + self.args))

    def __repr__(self):
        return self.__str__()
    

    def render(self, lambda_var=None):
        rendered_vars = lambda_var if lambda_var is not None else self.vars
        rendered_args = []
        for arg in self.args:
            if isinstance(arg, str):
                rendered_args.append('"{}"'.format(arg))
            elif isinstance(arg, dict):
                rendered_args.append("{}".format(arg["str"]))
            # elif isinstance(arg, DSLFunction):
            #     rendered_args.append(arg.render())
            elif isinstance(arg, list) and isinstance(arg[0], str):
                rendered_args.append(json.dumps(arg))
            else:
                rendered_args.append(str(arg))
                
        arguements = ", ".join(rendered_vars + self.other_vars + rendered_args)
        return "{}({})".format(self.name, arguements)