"""
MoVer shared configuration constants.
"""

# Animation library configuration
# Maps library names to their template and output file names
LIBRARY_CONFIG = {
    "default": {
        "template": "template_animation.md",
        "output": "sys_msg_animation_synthesizer.md"
    },
    "gsap": {
        "template": "template_animation_allow_gsap.md",
        "output": "sys_msg_animation_synthesizer_with_implementation.md"
    }
}
