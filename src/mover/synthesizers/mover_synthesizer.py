from pathlib import Path
from typing import List, Dict, Optional, Tuple
from mover.synthesizers.base_synthesizer import BaseSynthesizer
from mover.synthesizers.utils import extract_code_block, get_svg_code

class MoverSynthesizer(BaseSynthesizer):
    ## System message file path for mover synthesizer
    _sys_msg_filepath = 'sys_msg_mover_synthesizer.md'

    def get_dsl_documentation(self) -> str:
        """Get DSL documentation."""
        ## check if the system message contains the API documentation
        header = "### Verification DSL Documentation"
        if header not in self.read_sys_msg():
            raise ValueError(f"System message does not contain the proper header {header}.")
        
        return self.read_sys_msg().split(header)[1].lstrip('\n')


    def generate(self, chat_history: List[Dict[str, str]], program_file_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Generate a new message from LLM.
        
        Args:
            chat_history: List of chat messages with role and content, but chat_history is not being updated in this function
            program_file_path: Path to save the generated program
            
        Returns:
            tuple: (response, error_msg)
        """
        ## get next message from LLM
        response = self.llm_client.create(chat_history)
        
        ## extract code block
        try:
            mover_code = extract_code_block(response, '```', '```')
        except Exception as e:
            error_msg = f"Error extracting code block: {e}"
            return response, error_msg
            
        ## write program to file if program_file_path is specified
        if program_file_path is not None:
            output_path = Path(program_file_path)
            if not output_path.parent.is_dir():
                raise ValueError(f"Invalid directory: {output_path.parent}")
            
            with open(output_path, 'w') as f:
                f.write(mover_code)
            
        return response, None
