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
"""