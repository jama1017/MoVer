import os
from typing import List, Dict, Any

from mover._optional import import_optional_module, is_module_available


def get_available_providers() -> List[str]:
    """Get a list of available providers based on installed dependencies."""
    providers = []
    if is_module_available("openai"):
        providers.extend(["openai", "gemini", "vllm-serve"])
    if is_module_available("groq"):
        providers.append("groq")
    if is_module_available("ollama"):
        providers.append('ollama')
    if is_module_available("openai") and is_module_available("google.auth"):
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
        self.credentials = None
        self._google_auth_requests = None
        self.client = self._initialize_client()
        
            
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on the provider."""
        
        ## OpenAI
        if self.provider == 'openai':
            openai = import_optional_module(
                "openai",
                distribution="openai",
                extra="openai",
                feature="The OpenAI provider",
            )
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            return openai.OpenAI(api_key=api_key)
        
        ## Google GEMINI
        elif self.provider == 'gemini':
            openai = import_optional_module(
                "openai",
                distribution="openai",
                extra="openai",
                feature="The Gemini provider",
            )
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found")
            return openai.OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
        ## Google Vertex
        elif self.provider == 'vertex':
            openai = import_optional_module(
                "openai",
                distribution="openai",
                extra="vertex",
                feature="The Vertex provider",
            )
            google_auth = import_optional_module(
                "google.auth",
                distribution="google-auth",
                extra="vertex",
                feature="The Vertex provider",
            )
            self._google_auth_requests = import_optional_module(
                "google.auth.transport.requests",
                distribution="google-auth",
                extra="vertex",
                feature="The Vertex provider",
            )
            self.credentials, _ = google_auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.credentials.refresh(self._google_auth_requests.Request())
            REGION = os.getenv("MOVER_REGION")
            PROJECT_ID = os.getenv("MOVER_PROJECT")
            ENDPOINT_ID = os.getenv("MOVER_ENDPOINT")

            if not REGION or not PROJECT_ID or not ENDPOINT_ID:
                raise ValueError("Vertex environment variables not found")
            return openai.OpenAI(
                base_url=f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}",
                api_key=self.credentials.token,
            )
                    
        ## Groq
        elif self.provider == 'groq':
            groq = import_optional_module(
                "groq",
                distribution="groq",
                extra="groq",
                feature="The Groq provider",
            )
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("Groq API key not found")
            return groq.Groq(api_key=api_key)
        
        ## Ollama
        elif self.provider == 'ollama':
            ollama = import_optional_module(
                "ollama",
                distribution="ollama",
                extra="ollama",
                feature="The Ollama provider",
            )
            return ollama.chat
        
        ## vLLM-serve
        elif self.provider == 'vllm-serve':
            openai = import_optional_module(
                "openai",
                distribution="openai",
                extra="openai",
                feature="The vLLM-compatible provider",
            )
            serve_port = 8000
            if 'vllm_serve_port' in self.params:
                serve_port = self.params['vllm_serve_port']
                self.params.pop('vllm_serve_port')
            return openai.OpenAI(
                base_url=f"http://localhost:{serve_port}/v1",
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
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
                if not self.credentials.valid:
                    self.credentials.refresh(self._google_auth_requests.Request())
                    self.client.api_key = self.credentials.token
                    
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params if self.params else {}
            )
            return response.choices[0].message.content
        
        ## ollama
        elif self.provider == 'ollama':
            response = self.client(
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