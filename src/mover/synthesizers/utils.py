def extract_code_block(message: str, start_str: str, end_str: str) -> str:
    """
    Extract code block from message between start_str and end_str.
    
    Args:
        message: The message containing the code block
        start_str: The starting delimiter
        end_str: The ending delimiter
        
    Returns:
        str: The extracted code block content
        
    Raises:
        ValueError: If code block delimiters are not found
    """
    try:
        start_index = message.index(start_str)
        end_index = message.index(end_str, start_index + len(start_str))
    except ValueError:
        raise ValueError(f"Code block not found with delimiters '{start_str}' and '{end_str}'")
    
    res = message[start_index + len(start_str):end_index]
    res = res.lstrip('\n').rstrip('\n')
    return res


def get_svg_code(svg_file_path: str) -> str:
    """Read SVG file content."""
    with open(svg_file_path, 'r') as f:
        svg_code = f.read()
    return svg_code 