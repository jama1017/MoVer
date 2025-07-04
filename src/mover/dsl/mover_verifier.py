import json
import argparse
from pathlib import Path
from jinja2 import Template
from mover.dsl.obj_mot_data_class import *
from mover.dsl.fol_domain import init_domain
from mover.dsl.fol_parser import FOLParser
from mover.dsl.fol_executor import FOLExecutor
from mover.dsl.utils import clean_up_program_string


class MoverVerifier:
    """
    A verifier for MoVer programs that can execute verification logic on animation data.
    """
    
    ## Path to default correction template file
    _correction_template_filepath = 'correction_msg_template.md'
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the MoverVerifier.
        
        Args:
            verbose: Whether to print verbose output during parsing
        """
        self.domain = init_domain()
        
        ## Initialize parser with configuration
        self.parser = FOLParser(self.domain, inplace_definition=True, inplace_polymorphic_function=True)
        self.parser.print_verbose = verbose
        
        ## Initialize executor
        self.executor = FOLExecutor(self.domain, self.parser)
        
        ## Load default correction template
        default_path = Path(__file__).parent / 'assets' / self._correction_template_filepath
        with open(default_path, "r") as f:
            template_content = f.read()
            self._correction_template = Template(template_content)


    def get_correction_msg_template(self) -> Template:
        """Get the current correction message template."""
        return self._correction_template


    def set_correction_msg_template(self, template_file_path: str = None) -> None:
        """Set custom correction message template.
        
        Args:
            template_file_path: Path to the correction template file.
                              If not provided, uses the default file from assets directory.
        """
        file_path = template_file_path if template_file_path is not None else Path(__file__).parent / 'assets' / self._correction_template_filepath
        
        with open(file_path, "r") as f:
            template_content = f.read()
            self._correction_template = Template(template_content)


    def verify(self, program_file: str, animation_file: str) -> str:
        """
        Verify a program against animation data from files.
        
        Args:
            program_file: Path to file containing the verification program
            animation_file: Path to file containing animation data (JSON)
            
        Returns:
            Verification result from executor
        """
        ## Load program from file
        with open(program_file, 'r') as f:
            program_str = f.read()
        
        ## Load animation data from file
        with open(animation_file, 'r') as f:
            anim_data = json.load(f)

        scene = Scene(anim_data)
        
        program = self.parser.parse_multiple_expressions(program_str)
        program_statements_list = clean_up_program_string(program_str)
        
        res = self.executor.execute_multiple_expressions(program, scene, expr_list=program_statements_list)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a verification program on a given animation data.")
    parser.add_argument("program_file", type=str, help="The file containing the program to execute.")
    parser.add_argument("animation_data", type=str, help="The animation data associated with the prompt.")
    args = parser.parse_args()
    
    verifier = MoverVerifier()
    res = verifier.verify(args.program_file, args.animation_data)    
    
    
    
    