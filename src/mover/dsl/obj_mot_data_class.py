from dataclasses import dataclass
from typing import Tuple
    

@dataclass
class Scene(object):
    anim_data: dict
    
@dataclass
class Object(object):
    shape: str = None
    fill: str = None
    stroke: str = None
    size: Tuple[float, float] = None

@dataclass
class Motion(object):
    type: str = None
    agent: list[Object] = None
    magnitude: Tuple[float, float] | float = None ## same as direction for scale
    magnitude_reference: list[Object] = None
    direction: Tuple[float, float] | float = None
    direction_reference: list[Object] = None
    origin: Tuple | str = None ## absolute or relative
    origin_reference: list[Object] = None
    duration: float = None
    duration_reference: list[Object] = None
    post: str = None
    post_reference: list[Object] = None