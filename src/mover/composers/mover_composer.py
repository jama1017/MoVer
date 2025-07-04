import ast
from pathlib import Path


def compose():
    current_dir = Path(__file__).parent
    filename = current_dir.parent / 'dsl' / 'fol_executor.py'
    output_filename = current_dir.parent / 'synthesizers' / 'assets' / 'sys_msg_mover_synthesizer.md'
    
    file_lines = []
    with open(filename) as file:
        file_lines = file.readlines()
        file.seek(0)
        parsed_ast = ast.parse(file.read())

    functions = []
    docstrings = []

    allowed_function_names = ['color', 'shape', 'id', 'type', 'direction', 'magnitude', 'origin', 'post', 'duration', 'agent', 't_before', 't_after', 't_while', 's_top', 's_bottom', 's_left', 's_right', 's_top_left', 's_top_right', 's_bottom_left', 's_bottom_right', 's_intersect', 's_border', 's_top_border', 's_bottom_border', 's_left_border', 's_right_border','exists', 'iota', 'all', 'get_pos']

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if isinstance(node, ast.FunctionDef) and node.name in allowed_function_names:            
                ## remove self from the arguments
                function_declaration = file_lines[node.lineno - 1].strip(" ")
                function_declaration = function_declaration.replace('self, ', '').replace(':', '').replace('def ', '')
                
                functions.append(function_declaration)
                docstrings.append(ast.get_docstring(node, clean=True))
                return node

    visitor = FunctionVisitor()
    modified_ast = visitor.visit(parsed_ast)

    ## system message template
    template_path = current_dir / 'assets' / 'template_mover.md'

    ## add template_mover.md to the output file
    with open(output_filename, 'w') as file:
        with open(template_path) as doc_file:
            file.write(doc_file.read())
        
        ## add the functions and docstrings to the output file
        function_num_counter = 0
        for i in range(len(functions)):
            file.write(functions[i])
            file.write('"""\n')
            file.write(docstrings[i])
            file.write('\n"""\n')
            
            if i < len(functions):
                file.write('\n\n')

if __name__ == '__main__':
    compose() 