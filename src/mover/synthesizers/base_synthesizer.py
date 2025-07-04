import jinja2
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from mover.synthesizers.llm_client import LLMClient
from mover.synthesizers.utils import get_svg_code


class BaseSynthesizer(ABC):
    ## Name of the system message file to use
    _sys_msg_filepath: str = None

    def __init__(self, model_name: str = "gpt-4.1", provider: str = "openai", num_ctx: int = 128000, params: Dict[str, Any] = {}):
        """
        Initialize the Base Synthesizer.
        
        Args:
            model_name: Name of the model to use (default: "gpt-4o")
            provider: LLM provider to use (default: "openai")
            num_ctx: Max context length. Only used for the locally-hosted models (default: 128000)
            params: Additional parameters to pass to the LLM client (default: {})
            
        Raises:
            ValueError: If _sys_msg_filepath is not set by subclass
        """
        if self._sys_msg_filepath is None:
            raise ValueError("_sys_msg_filepath must be set by subclass")
            
        self.llm_client = LLMClient(model_name=model_name, provider=provider, num_ctx=num_ctx, params=params)


    def read_sys_msg(self, sys_msg_file_path: str = None) -> str:
        """
        Read system message from file.
        """
        file_path = Path(sys_msg_file_path) if sys_msg_file_path is not None else Path(__file__).parent / 'assets' / self._sys_msg_filepath
        
        with open(file_path, 'r') as f:
            sys_msg_content = f.read()
            
        return sys_msg_content


    def compose_sys_msg(self, sys_msg_file_path: str = None) -> Dict[str, str]:
        """
        Build system message for LLM.
        
        Args:
            sys_msg_file_path: Optional custom path to system message file.
                             If not provided, uses the class's _sys_msg_filepath from assets directory.
        
        Returns:
            Dict[str, str]: System message in the format {"role": "system", "content": content}
        """
        sys_msg_content = self.read_sys_msg(sys_msg_file_path)
        return {"role": "system", "content": sys_msg_content}


    def compose_initial_user_prompt(self, animation_prompt: str, svg_file_path: str) -> Dict[str, str]:
        """
        Build initial user prompt with SVG and animation description.
        
        Args:
            animation_prompt: The prompt describing the desired animation/behavior
            svg_file_path: Path to the SVG file to be animated/processed
            
        Returns:
            Dict[str, str]: User message in the format {"role": "user", "content": content}
        """
        svg_code = get_svg_code(svg_file_path)
        initial_prompt_template = jinja2.Template("svg code:\n{{ svg_code }}\n\nprompt:\n{{ animation_prompt }}")
        rendered_prompt = initial_prompt_template.render(svg_code=svg_code, animation_prompt=animation_prompt)
        usr_msg = {"role": "user", "content": rendered_prompt}
        return usr_msg


    @abstractmethod
    def generate(self, chat_history: List[Dict[str, str]], *args, **kwargs):
        """
        Generate a new message from LLM.
        
        Args:
            chat_history: List of chat messages with role and content
            
        Returns:
            The generated response from the LLM
        """
        pass 