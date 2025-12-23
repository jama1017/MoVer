from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_parser, SupportedLanguage
from pathlib import Path
import argparse


def extract_text(source_code: str, node) -> str:
    """Extract text from source code using node's start and end byte positions."""
    return source_code[node.start_byte:node.end_byte].decode('utf-8')


def has_previous_comment(node, source_code: str) -> tuple[bool, str]:
    """Check if there's a JSDoc comment before the node."""
    prev_sibling = node.prev_sibling
    while prev_sibling:
        if prev_sibling.type == 'comment' and extract_text(source_code, prev_sibling).startswith('/*'):
            return True, extract_text(source_code, prev_sibling)
        prev_sibling = prev_sibling.prev_sibling
    return False, ""


def parse_api_code(api_file_path: str, include_implementation: bool = False) -> str:
    """
    Parse JavaScript API code to extract function declarations and their JSDoc comments.
    Returns the formatted string containing comments and function declarations.
    If include_implementation is True, returns full function implementation including body.
    """
    # Read the API file
    with open(api_file_path, 'rb') as f:
        source_code = f.read()
    
    # Get parser and parse
    parser = get_parser("javascript")
    tree = parser.parse(source_code)
    
    # Functions to exclude
    excluded_functions = {
        "getAABB",
        "computeTranslationForAlignTo",
        "computeTranslationForAdjacentTo"
    }
    
    functions = []
    
    # Find all function declarations
    for node in tree.root_node.children:
        if node.type != 'function_declaration':
            continue
            
        # Get function name
        name_node = next((child for child in node.children if child.type == 'identifier'), None)
        if not name_node:
            continue
            
        func_name = extract_text(source_code, name_node)
        if func_name in excluded_functions:
            continue
        
        # Check for JSDoc comment
        has_comment, comment = has_previous_comment(node, source_code)
        if not has_comment:
            continue
            
        # Extract full function including body
        func_text = extract_text(source_code, node)
        if not include_implementation:
            # Get function declaration without body
            # Find the start of the function body
            body_start = None
            for child in node.children:
                if child.type == 'statement_block':
                    body_start = child.start_byte
                    break
            
            if body_start is None:
                continue
                
            # Extract everything from function start to body start
            func_text = source_code[node.start_byte:body_start].decode('utf-8')
        
        functions.append({
            'comment': comment,
            'declaration': func_text
        })
    
    # Combine comments and declarations
    return ''.join(f"{func['comment']}\n{func['declaration']}\n\n" for func in functions)


def load_sys_msg(sys_msg_path: str, api_code: str) -> str:
    with open(sys_msg_path, 'r') as f:
        sys_msg = f.read()
    return sys_msg.replace("{{api-code}}", api_code)


def compose(library: str = "default"):
    ## Set up paths
    root_dir = Path(__file__).parent.parent.parent
    api_path = root_dir / "mover" / "converter" / "assets" / "api.js"
    
    ## Library-specific configuration
    library_config = {
        "default": {
            "template": "template_animation.md",
            "output": "sys_msg_animation_synthesizer.md"
        },
        "gsap": {
            "template": "template_animation_allow_gsap.md",
            "output": "sys_msg_animation_synthesizer_with_implementation.md"
        }
    }
    
    if library not in library_config:
        raise ValueError(f"Unsupported animation library: {library}. Supported: {list(library_config.keys())}")
    
    config = library_config[library]
    sys_msg_path = root_dir / "mover" / "composers" / "assets" / config["template"]
    
    ## Parse API code and combine with system message
    include_implementation = (library != "default")
    api_code = parse_api_code(str(api_path), include_implementation=include_implementation)
    sys_msg = load_sys_msg(str(sys_msg_path), api_code)
    
    ## Save the sys_msg to a file
    output_path = root_dir / "mover" / "synthesizers" / "assets" / config["output"]
    with open(output_path, 'w') as f:
        f.write(sys_msg)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose animation synthesizer system message")
    parser.add_argument("-l", "--library", type=str, default="default", choices=["default", "gsap"], help="Animation library to support (default: default)")
    args = parser.parse_args()
    compose(library=args.library)
    
    