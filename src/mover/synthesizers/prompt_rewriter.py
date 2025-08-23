import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from mover.synthesizers.base_synthesizer import BaseSynthesizer
from mover.synthesizers.utils import extract_code_block, get_svg_code
import jinja2

class PromptRewriter(BaseSynthesizer):
    ## System message file path for prompt rewriter
    _sys_msg_filepath = 'sys_msg_prompt_rewriter.md'
    
    
    def compose_initial_user_prompt(self, animation_prompt: str, number_of_paraphrases: int) -> Dict[str, str]:
        """
        Override the base class method to not include SVG code in the user prompt.
        """
        initial_prompt_template = jinja2.Template("Motion description to paraphrase:\n{{ animation_prompt }}\n\nNumber of paraphrases to generate: {{ number_of_paraphrases }}")
        rendered_prompt = initial_prompt_template.render(animation_prompt=animation_prompt, number_of_paraphrases=number_of_paraphrases)
        usr_msg = {"role": "user", "content": rendered_prompt}
        return usr_msg


    def generate(self, chat_history: List[Dict[str, str]], json_file_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Generate a new message from LLM.
        
        Args:
            chat_history: List of chat messages with role and content, but chat_history is not being updated in this function
            json_file_path: Path to save the generated JSON object
            
        Returns:
            tuple: (response, error_msg)
        """
        ## get next message from LLM
        response = self.llm_client.create(chat_history)
        
        ## try parsing the response as a JSON object
        try:
            json_code = extract_code_block(response, '```json', '```')
            response_json = json.loads(json_code)
        except Exception as e:
            error_msg = f"Error parsing response as a JSON object: {e}"
            return response, error_msg

        ## write response_json to file if program_file_path is specified
        if json_file_path is not None:
            output_path = Path(json_file_path)
            if not output_path.parent.is_dir():
                raise ValueError(f"Invalid directory: {output_path.parent}")
            
            with open(output_path, 'w') as f:
                json.dump(response_json, f, indent=4)
            
        return response, None
