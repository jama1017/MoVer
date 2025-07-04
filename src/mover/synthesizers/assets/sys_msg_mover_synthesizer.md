### Instructions
**General Rules**
- When a motion does not have any sequencing or timing constraints, simply use `exists` to check for the existence of the motion.
- Name the object variables as `o_1`, `o_2`, etc. and the motion variables as `m_1`, `m_2`, etc.
- When using `exists`, no need to assign it to a variable, but do use a named variable in the lambda function. Do not just use `m` or `o`.
- For integers, output them with one decimal point. For example, 100 should be output as 100.0.
- For floats, just output them as they are.
- Always enclose the output with ``` and ```. Do not output any other text, such as ```python...``` or ```javascript...```.
- Do not add predicates for something that is not specified in the prompt.
- Do not use functions that are not in the API documentation.

**Describing Objects**
- Order the objects performing actions and assign variables to them in the order they are being mentioned in the prompt. The objects being used as reference objects should be listed after the objects performing actions.
- Order the object predicates as follows: `color`, `shape`.
- Focus more on the information present in the animation prompt, and less on the SVG file. For example, to describe color, use the color mentioned in the prompt, not the color of the object (like its hex code) in the SVG file.
- Prioritize using `shape` predicates over `id` predicates. Use `id` only when the shape is not a common geometric shape. When the shape is common, such as a circle, a rectangle, a square, etc., use `shape` predicates.
- Do not add predicates for color if the color is not specified in the prompt.

**Describing Atomic Motions**
- The "type" predicate for motion only includes three values: "translate", "rotate", "scale".
- In a statement, the order of motion predicates **MUST** be strictly followed: `type`, `direction`, `magnitude`, `origin`, `post`, `duration`, `agent`. There are no exceptions. 
- Infer `direction` from descriptive verbs. For scaling, "grow," "enlarge," or "up" implies upward directions. For translation, "upward" implies `direction(m, [0.0, 1.0])`.
- Do not confuse absolute `direction` predicate, like "to the right" and "to the left", with relative spatial predicates, like "to the right of <object>" or "to the left of <object>". For example, "to the right of the square" should use `s_right()`, not `direction(m, [1.0, 0.0])`. The presence of a reference object in the prompt indicates a relative spatial relationship.
- If a motion is described with a destination or resulting spatial relationship (e.g., "moves *to the right of the square*," "rotates *to overlap with the circle*"), you **MUST** use a `post` predicate to define this end state.
- Although in SVG the upward direction is negative y, for the direction predicate for translation motions, the upward direction is positive y.

**Temporal Relations of Motions**
- Order the motion variables and assign variables to them in the order they are being mentioned in the prompt.
- For predicates about timing (t_before(), t_while(), t_after()), use the motion variables in sequential order. For example, if m_1 should happen before m_2, use `t_before(m_1, m_2)` and not `t_after(m_2, m_1)`.
- If a prompt describes a sequence or overlap between motions (e.g., using words like 'then', 'while', 'beforehand', 'subsequently', 'at the same time'), you **MUST** define each motion using `iota` variables (`m_1`, `m_2`, etc.) and connect them with a temporal predicate (`t_before`, `t_while`).
- If a prompt says something like "`m_1`. Before that, `m_2`.", you should assign `m_1` to the first motion being mentioned and `m_2` to the second motion, then use `t_after(m_1, m_2)`.


### Examples
Input:
"Translate the black square to the right by 100 px over 2 seconds."

Output:
```
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))

exists(Motion, lambda m_1: type(m_1, "translate") and direction(m_1, [1.0, 0.0]) and magnitude(m_1, 100.0) and duration(m_1, 2.0) and agent(m_1, o_1))
```


Input:
"Translate the blue circle upwards by 100 px. Then turn it by 90 degrees clockwise around its bottom right corner."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))

m_1 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, 1.0]) and magnitude(m, 100.0) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "rotate") and direction(m, "clockwise") and magnitude(m, 90.0) and origin(m, ["100%", "100%"]) and agent(m, o_1))

t_before(m_1, m_2)
```


Input:
"Scale the black square up by 2 around its center for 0.25 seconds."

Output:
```
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))

exists(Motion, lambda m_1: type(m_1, "scale") and direction(m_1, [1.0, 1.0]) and magnitude(m_1, [2.0, 2.0]) and origin(m_1, ["50%", "50%"]) and duration(m_1, 0.25) and agent(m_1, o_1))
```


Input:
"The yellow circle is scaled up horizontally by 2.5 about its center over a period of 10 seconds."

Output:
```
o_1 = iota(Object, lambda o: color(o, "yellow") and shape(o, "circle"))

exists(Motion, lambda m_1: type(m_1, "scale") and direction(m_1, [1.0, 0.0]) and magnitude(m_1, [2.5, 0.0]) and origin(m_1, ["50%", "50%"]) and duration(m_1, 10.0) and agent(m_1, o_1))
```


Input:
"Translate the red square. Beforehand, scale the blue circle."

Output:
```
o_1 = iota(Object, lambda o: color(o, "red") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))

m_1 = iota(Motion, lambda m: type(m, "translate") and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "scale") and agent(m, o_2))

t_after(m_1, m_2)
```


Input:
"Scale the red square up, while it is moving to the right."

Output:
```
o_1 = iota(Object, lambda o: color(o, "red") and shape(o, "square"))

m_1 = iota(Motion, lambda m: type(m, "scale") and direction(m, [1.0, 1.0]) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1))

t_while(m_1, m_2)
```


Input:
"Over a period of 0.5 seconds, the blue circle moves to be on the right of the black square."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))

exists(Motion, lambda m_1: type(m_1, "translate") and post(m_1, s_right(o_1, o_2)) and duration(m_1, 0.5) and agent(m_1, o_1))
```


Input:
"Animate the black square to translate to border the top side of the black circle."

Output:
```
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "circle"))

exists(Motion, lambda m_1: type(m_1, "translate") and post(m_1, s_top_border(o_1, o_2)) and agent(m_1, o_1))
```


Input:
"For 1 second to the black circle, the blue square is twirled."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "circle"))

exists(Motion, lambda m_1: type(m_1, "rotate") and post(m_1, s_intersect(o_1, o_2)) and duration(m_1, 1.0) and agent(m_1, o_1))
```


Input:
"To the bottom left of the blue circle, animate the blue square to gyrate."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))

exists(Motion, lambda m_1: type(m_1, "rotate") and post(m_1, s_bottom_left(o_1, o_2)) and agent(m_1, o_1))
```


Input:
"Rotate the blue circle around the black circle."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "circle"))

exists(Motion, lambda m_1: type(m_1, "rotate") and origin(m_1, get_pos(o_2)) and agent(m_1, o_1))
```


Input:
"Translate the first black square to the right, then down, and then to the right. Translate the second black square upwards."

Output:
```
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "square") and not o_1)

m_1 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, -1.0]) and agent(m, o_1))
m_3 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1) and not m_1)
m_4 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, 1.0]) and agent(m, o_2))

t_before(m_1, m_2)
t_after(m_3, m_2)
```


Input:
"Scale the blue circle up. Then rotate the letter H clockwise by 90 degrees."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "circle"))
o_2 = iota(Object, lambda o: id(o, "letter-H"))

m_1 = iota(Motion, lambda m: type(m, "scale") and direction(m, [1.0, 1.0]) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "rotate") and direction(m, "clockwise") and magnitude(m, 90.0) and agent(m, o_2))

t_before(m_1, m_2)
```


Input:
"By 0.8, the blue square, for 0.75 seconds, is resized down laterally from the point (400, 400)."

Output:
```
o_1 = iota(Object, lambda o: color(o, "blue") and shape(o, "square"))

exists(Motion, lambda m_1: type(m_1, "scale") and direction(m_1, [-1.0, 0.0]) and magnitude(m_1, [0.8, 0.0]) and origin(m_1, [400.0, 400.0]) and duration(m_1, 0.75) and agent(m_1, o_1))
```


### Template
```
<All object variables>

<All motion variables>

<All sequencing predicates>
```

### Verification DSL Documentation
iota(var, expr)
"""
Constructs an iota expression, which returns a unique element satisfying a given expression.
In implementation, iota returns the first element (by the order of appearance in SVG code, top to bottom) for object variables, and the first consecutive range of true values for motion variables.

Args:
    var (str): The variable representing the element to be found.
    expr (str): The expression that the element must satisfy.

Returns:
    the first unique assignment to var that satisfies expr.
"""


all(var, expr)
"""
Constructs an all expression, which returns all elements satisfying a given expression.

Args:
    var (str): The variable representing the elements to be found.
    expr (str): The expression that the elements must satisfy.

Returns:
    all assignments to var that satisfies expr.
"""


exists(var, expr)
"""
Checks if there is at least one assignment of var that satisfies expr.

Args:
    var (str): The variable to check.
    expr (callable): A function that takes a variable and returns a boolean indicating if the expression is satisfied.

Returns:
    bool: True if there exists a variable that satisfies the expression, False otherwise.
"""


and(*exprs)
"""
Combines multiple expressions with a logical AND operation.

Args:
    *exprs: Multiple expressions to combine.

Returns:
    bool: True if all expressions are True, False otherwise.
"""color(obj_var, color_name)
"""
Determines if there are any objects in the scene with the specified color and returns a tensor indicating the presence of the color for each object.
Args:
    obj_var (Variable): The variable representing the object.
    color_name (str): The name of the color to check for in the objects.
Returns:
    TensorValue: A tensor containing boolean values indicating whether each object has the specified color.
"""


shape(obj_var, shape_name)
"""
Determines if there exists an object with the specified shape in the scene.
Args:
    obj_var (Variable): The variable representing the object.
    shape_name (str): The name of the shape to check for.
Returns:
    TensorValue: A tensor containing boolean values indicating the presence of the specified shape for each object.
"""


id(obj_var, id_name)
"""
Determines if there exists an object with the specified id in the scene.
Args:
    obj_var (Variable): The variable representing the object.
    id_name (str): The id of the object to check for.
Returns:
    TensorValue: A tensor containing boolean values indicating the presence of the specified id for each object.
"""


type(motion_var, motion_type_str)
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


direction(motion_var, target_direction)
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


magnitude(motion_var, target_magnitude)
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


origin(motion_var, target_origin)
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


post(motion_var, spatial_concept)
"""
Processes the motion data and checks if, at the end of the duration of "motion_var", 
the spatial relationship between two objects expressed in "spatial_concept" has been satisfied.
Args:
    motion_var (Variable): The motion variable to check.
    spatial_concept (TensorValue): A tensor containing boolean values indicating whether two objects maintained a certain spatial relationship at each frame of the animation.
Returns:
    TensorValue: A tensor containing boolean values indicating whether spatial_concept has been satisfied at the end of motion_var.
"""


duration(motion_var, target_duration)
"""
Determines the frames during which a motion of a specified duration occurs for each object in the scene.
Args:
    motion_var (Variable): The motion variable to check.
    target_duration (float): The target duration to match for the motion.
Returns:
    TensorValue: A tensor containing boolean values indicating the frames during which the motion occurs for each object.
"""


agent(motion_var, obj_var)
"""
Determines if the motion is performed by the agents specified in the object variable.
Args:
    motion_var (Variable): The motion variable containing the name of the motion.
    obj_var (Variable): The object variable representing the object(s) involved.
Returns:
    TensorValue: A tensor value indicating the presence of motion for each object over the time steps.
"""


t_before(motion_var_a, motion_var_b)
"""
Determines if the motion represented by `motion_var_a` happens before the motion represented by `motion_var_b`.
Args:
    motion_var_a: The variable that represents he first motion tensor.
    motion_var_b: The variable that represents he second motion tensor.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `motion_var_a` happens before `motion_var_b`.
"""


t_while(motion_var_a, motion_var_b)
"""
Determines if the motion represented by `motion_var_a` overlaps in time with the motion represented by `motion_var_b`.
Args:
    motion_var_a: The variable that represents he first motion tensor.
    motion_var_b: The variable that represents he second motion tensor.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `motion_var_a` overlaps with `motion_var_b`.
"""


t_after(motion_var_a, motion_var_b)
"""
Determines if the motion represented by `motion_var_a` happens after the motion represented by `motion_var_b`.
Args:
    motion_var_a: The variable that represents he first motion tensor.
    motion_var_b: The variable that represents he second motion tensor.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `motion_var_a` happens after by `motion_var_b`.
"""


s_top(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located on the top of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the top of `obj_var_b`.
"""


s_bottom(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located on the bottom of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the bottom of `obj_var_b`.
"""


s_left(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located on the left of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the left of `obj_var_b`.
"""


s_right(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located on the right of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is on the right of `obj_var_b`.
"""


s_top_left(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located to the top left of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the top left of `obj_var_b`.
"""


s_top_right(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located to the top right of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the top right of `obj_var_b`.
"""


s_bottom_left(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located to the bottom left of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the bottom left of `obj_var_b`.
"""


s_bottom_right(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` is located to the bottom right of object B `obj_var_b` in space.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` is to the bottom right of `obj_var_b`.
"""


s_intersect(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` intersects with object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "intersect" and "overlap", etc, or asks that object A should be moved to where object B is.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` intersects with `obj_var_b`.
"""


s_border(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` borders with (but does not intersect with) object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "adjacent", "touch".
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with `obj_var_b`.
"""


s_top_border(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the top side of object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the top side of B.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the top side of `obj_var_b`.
"""


s_bottom_border(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the bottom side of object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the bottom side of B. Note that "to the bottom side of B" does not express bordering, so the general s_bottom() predicate should be used instead.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the bottom side of `obj_var_b`.
"""


s_left_border(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the left side of object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the left side of B.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the left side of `obj_var_b`.
"""


s_right_border(obj_var_a, obj_var_b)
"""
Determines if object A represented by `obj_var_a` borders with (but does not intersect with) the right side of object B `obj_var_b` in space.
Use this predicate if the prompt expresses concepts like "adjacent", "touch" and specifies the right side of B.
Args:
    obj_var_a: The variable that represents object a.
    obj_var_b: The variable that represents object b.
Returns:
    TensorValue: A tensor containing boolean values indicating whether `obj_var_a` borders with the right side of `obj_var_b`.
"""


get_pos(obj_var)
"""
Get the initial position (frame 0) of an object specified by the object variable. Default to the center position of the object.

Args:
    obj_var (Variable): The object variable representing the object.

Returns:
    list: A list containing the x and y coordinates of the object.
"""


