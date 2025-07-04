### Instructions:
- Please correct the animation program based on the verification result. Pay attention to how the previous animation program was incorrect.
- Do not repeat the same animation program.
- Refer to the Animation API and Verification DSL documentation again when correcting the animation program.
    - Pay attention to how `getProperty()` should not be used to get the size of an object.
    - Consider computing the transform origin of rotation and scaling motions as described in the Animation API documentation to move the object to the desired correct spatial location.
- The post predicate checks the end state of the motion regardless of the duration of the motion. If the post predicate fails, changing the duration of the motion will not help.
- If each predicate of a motion is true but the motion statement is false as a whole, it probably means that there is not a single frame in the animation where all predicates are true.
- Always enclose the animation program in a code block with ```javascript and ```.

{% if dsl_documentation %}### Verification DSL documentation:
{{ dsl_documentation }}
{% endif %}### Verification program:
{{ verification_program }}

### Verification result:
{{ verification_result }}