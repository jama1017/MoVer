from pathlib import Path
import jinja2
from typing import List, Dict, Any, Optional, Tuple
from mover.synthesizers.base_synthesizer import BaseSynthesizer
from mover.synthesizers.utils import extract_code_block, get_svg_code


class AnimationSynthesizer(BaseSynthesizer):
    ## System message file path for animation synthesizer
    _sys_msg_filepath = 'sys_msg_animation_synthesizer.md'
    ## HTML template file path for animation synthesizer
    _html_template_filepath = Path(__file__).parent / 'assets' / 'template.html'

    def __init__(self, model_name: str = "gpt-4.1", provider: str = "openai", num_ctx: int = 128000, params: Dict[str, Any] = {}):
        """
        Initialize the Animation Synthesizer.
        
        Args:
            model_name: Name of the model to use (default: "gpt-4o")
            provider: LLM provider to use (default: "openai")
            num_ctx: Max context length. Only used for the locally-hosted models (default: 128000)
            params: Additional parameters to pass to the LLM client (default: {})
        """
        super().__init__(model_name=model_name, provider=provider, num_ctx=num_ctx, params=params)
        
        ## Load HTML template
        with open(self._html_template_filepath, 'r') as f:
            self.html_template = f.read()
            
            
    def set_html_template(self, html_template_path: str) -> None:
        """
        Set the HTML template for the animation synthesizer.
        
        Args:
            html_template_path: Path to HTML template file
        """
        self._html_template_filepath = Path(html_template_path)
        with open(self._html_template_filepath, 'r') as f:
            self.html_template = f.read()


    def generate(self, chat_history: List[Dict[str, str]], svg_file_path: Optional[str] = None, 
                html_output_file_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Generate a new message from LLM and extract JavaScript code from it.
        
        Args:
            chat_history: List of chat messages with role and content, but chat_history is not being updated in this function
            svg_file_path: Path to SVG file for HTML generation
            html_output_file_path: Path to save HTML file (optional)
            
        Returns:
            tuple: (response, error_msg)
        """
        ## get next message from LLM
        response = self.llm_client.create(chat_history)
        
        ## extract JavaScript code
        try:
            javascript_code = extract_code_block(response, '```javascript', '```')
        except Exception as e:
            error_msg = f"Error extracting code block: {e}"
            return response, error_msg
        
        ## write javascript code to html if output_dir is specified
        if html_output_file_path is not None and svg_file_path is not None:
            svg_code = get_svg_code(svg_file_path)
            html_code = self.html_template.replace("{{svg-code}}", svg_code)
            html_code = html_code.replace("let placeholder = 0;", javascript_code)
            
            output_path = Path(html_output_file_path)
            if not output_path.parent.is_dir():
                raise ValueError(f"Invalid directory: {output_path.parent}")
            
            with open(output_path, "w") as f:
                f.write(html_code)
        
        return response, None