import os
from typing import List, Dict, Any
from openai import OpenAI
from groq import Groq

## Optional imports with individual handling
_OPTIONAL_IMPORTS = {}

## Ollama
try:
    from ollama import chat, ChatResponse
    _OPTIONAL_IMPORTS['ollama'] = True
except ImportError:
    _OPTIONAL_IMPORTS['ollama'] = False
    chat = None
    ChatResponse = None

## Google Auth
try:
    from google.auth import default
    import google.auth.transport.requests
    _OPTIONAL_IMPORTS['google_auth'] = True
except ImportError:
    _OPTIONAL_IMPORTS['google_auth'] = False
    default = None


def _check_optional_import(package_name: str, provider: str) -> None:
    """Check if an optional package is available and raise informative error if not."""
    if not _OPTIONAL_IMPORTS.get(package_name, False):
        raise ImportError(
            f"Package '{package_name}' is required for provider '{provider}' but is not installed. "
            f"Please install it with: pip install {package_name}"
        )


def get_available_providers() -> List[str]:
    """Get a list of available providers based on installed dependencies."""
    providers = ['openai', 'gemini', 'groq', 'vllm-serve']  # Always available
    
    if _OPTIONAL_IMPORTS.get('ollama', False):
        providers.append('ollama')
    
    if _OPTIONAL_IMPORTS.get('google_auth', False):
        providers.append('vertex')
        
    return providers


class LLMClient:
    """A unified client for interacting with various LLM providers (OpenAI, Gemini, Groq, Ollama, vLLM-serve)."""
    
    def __init__(self, model_name: str, provider: str, num_ctx: int = 128000, params: Dict[str, Any] = {}):
        """
        Initialize the LLM client. Make sure the api key is set in the environment variables.
        
        Args:
            model_name: Name of the model to use (e.g., 'gpt-4', 'mixtral-8x7b', 'llama2')
        """
        self.model_name = model_name
        self.provider = provider
        self.num_ctx = num_ctx
        self.params = params
        
        if self.provider == 'vertex':
            self.credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self.credentials.refresh(google.auth.transport.requests.Request())
            
        self.client = self._initialize_client()
        
            
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on the provider."""
        
        ## OpenAI
        if self.provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            return OpenAI(api_key=api_key)
        
        ## Google GEMINI
        elif self.provider == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found")
            return OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
        ## Google Vertex
        elif self.provider == 'vertex':
            _check_optional_import('google_auth', 'vertex')
            REGION = os.getenv("MOVER_REGION")
            PROJECT_ID = os.getenv("MOVER_PROJECT")
            ENDPOINT_ID = os.getenv("MOVER_ENDPOINT")

            if not REGION or not PROJECT_ID or not ENDPOINT_ID:
                raise ValueError("Vertex environment variables not found")
            return OpenAI(
                base_url=f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}",
                api_key=self.credentials.token,
            )
                    
        ## Groq
        elif self.provider == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("Groq API key not found")
            return Groq(api_key=api_key)
        
        ## Ollama
        elif self.provider == 'ollama':
            _check_optional_import('ollama', 'ollama')
            return chat
        
        ## vLLM-serve
        elif self.provider == 'vllm-serve':
            serve_port = 8000
            if 'vllm_serve_port' in self.params:
                serve_port = self.params['vllm_serve_port']
                self.params.pop('vllm_serve_port')
            return OpenAI(
                base_url=f"http://localhost:{serve_port}/v1",
            ) 
        
        else:
            raise ValueError(f"Provider {self.provider} not found")
        
        
    def create(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a request to the LLM and get the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The model's response text
        """
        if self.provider in ['openai', 'gemini', 'groq', 'vertex']:
            if self.provider == 'vertex':
                _check_optional_import('google_auth', 'vertex')
                if not self.credentials.valid:
                    self.credentials.refresh(google.auth.transport.requests.Request())
                    self.client.api_key = self.credentials.token
                    
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params if self.params else {}
            )
            return response.choices[0].message.content
        
        ## ollama
        elif self.provider == 'ollama':
            _check_optional_import('ollama', 'ollama')
            response: ChatResponse = self.client(
                model=self.model_name, 
                messages=messages,
                options={
                    "num_ctx": self.num_ctx, # maybe around 5 rounds for gemma?
                },
                **self.params if self.params else {}
            )
            return response['message']['content']
        
        ## vLLM-serve
        elif self.provider == 'vllm-serve':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params if self.params else {}
            )
            return response.choices[0].message.content
        
        else:
            raise ValueError(f"Provider {self.provider} not found")