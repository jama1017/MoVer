from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_parser, SupportedLanguage
from pathlib import Path

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

def parse_api_code(api_file_path: str) -> str:
    """
    Parse JavaScript API code to extract function declarations and their JSDoc comments.
    Returns the formatted string containing comments and function declarations.
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
        decl = source_code[node.start_byte:body_start].decode('utf-8')
        
        functions.append({
            'comment': comment,
            'declaration': decl
        })
    
    # Combine comments and declarations
    return ''.join(f"{func['comment']}\n{func['declaration']}\n\n" for func in functions)

def load_sys_msg(sys_msg_path: str, api_code: str) -> str:
    with open(sys_msg_path, 'r') as f:
        sys_msg = f.read()
    return sys_msg.replace("{{api-code}}", api_code)

def compose():
    # Set up paths
    root_dir = Path(__file__).parent.parent.parent
    api_path = root_dir / "mover" / "converter" / "assets" / "api.js"
    sys_msg_path = root_dir / "mover" / "composers" / "assets" / "template_animation.md"
    
    # Parse API code and combine with system message
    api_code = parse_api_code(str(api_path))
    sys_msg = load_sys_msg(str(sys_msg_path), api_code)
    
    # save the sys_msg to a file
    with open(root_dir / "mover" / "synthesizers" / "assets" / "sys_msg_animation_synthesizer.md", 'w') as f:
        f.write(sys_msg)
    
    
if __name__ == "__main__":
    compose()
    
    