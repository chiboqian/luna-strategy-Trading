"""
Vertex AI Client for GenAI SDK
Supports authentication via API key or service account JSON key file.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config/AI.yaml - try current directory first, then parent
        current_dir = Path(__file__).parent
        config_path = current_dir / "config" / "AI.yaml"
        if not config_path.exists():
            config_path = current_dir.parent / "config" / "AI.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class VertexAIClient:
    """Client for interacting with Vertex AI using the GenAI SDK."""
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None, config_section: str = "default", **override_params):
        """Create client from configuration file.
        
        Args:
            config_path: Path to config file. If None, uses default path.
            config_section: Section of config to use ('default' uses root config, or specify example name)
            **override_params: Parameters to override from config
            
        Returns:
            VertexAIClient instance
        """
        # Load environment variables - try current directory first, then parent, then grandparent (workspace root)
        current_dir = Path(__file__).parent
        env_path = current_dir / ".env"
        if not env_path.exists():
            env_path = current_dir.parent / ".env"
        if not env_path.exists():
            env_path = current_dir.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path if env_path.exists() else None)
        # Load config
        config = load_config(config_path)
        # Get config section
        if config_section != "default" and "examples" in config:
            section_config = config.get("examples", {}).get(config_section, {})
        else:
            section_config = {}
        
        # Merge configurations (section overrides root, override_params overrides both)
        model_config = config.get("model", {})
        gen_config = config.get("generation", {})

        # Handle thinking config
        thinking_config = None
        if model_config.get("enable_thinking_config", False):
            thinking_config = section_config.get("thinking_config") or config.get("thinking_config")

        # Handle http_options from model section
        http_options = model_config.get("http_options") or section_config.get("http_options") or config.get("http_options")

        # Handle tools config
        tools_config = section_config.get("tools") or config.get("tools")

        # Load system instruction from file if path provided
        system_instruction = section_config.get("system_instruction") or config.get("system_instruction")
        system_instruction_path = section_config.get("system_instruction_path") or config.get("system_instruction_path")
        if not system_instruction and system_instruction_path:
            try:
                with open(system_instruction_path, "r") as f:
                    system_instruction = f.read().strip()
            except Exception as e:
                import sys
                # Only print warning if not in JSON mode (we can't easily check args here, so we print to stderr)
                print(f"Warning: failed to read system_instruction_path '{system_instruction_path}': {e}", file=sys.stderr)

        params = {
            "model_name": section_config.get("model_name") or model_config.get("name", "gemini-2.0-flash-exp"),
            "location": model_config.get("location", "us-central1"),
            "system_instruction": system_instruction,
            "temperature": section_config.get("temperature") or gen_config.get("temperature", 1.0),
            "top_p": section_config.get("top_p") or gen_config.get("top_p", 0.95),
            "max_output_tokens": section_config.get("max_output_tokens") or gen_config.get("max_output_tokens", 8192),
            "safety_settings": config.get("safety_settings", []),
            "thinking_config": thinking_config,
            "http_options": http_options,
            "tools_config": tools_config,
        }
        
        # Add authentication from environment
        params["api_key"] = os.getenv("GOOGLE_API_KEY")
        params["service_account_file"] = os.getenv("SERVICE_ACCOUNT_FILE") or os.getenv("SERVICE_ACCOUNT_JSON")
        params["project_id"] = os.getenv("GCP_PROJECT_ID")
        
        # Apply overrides
        params.update(override_params)
        
        return cls(**params)
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        service_account_file: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        system_instruction: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        thinking_config: Optional[Dict[str, Any]] = None,
        http_options: Optional[Dict[str, Any]] = None,
        tools_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Vertex AI client.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (alternative to service account)
            service_account_file: Path to service account JSON key file
            project_id: GCP project ID (required for service account auth)
            location: GCP location/region
            system_instruction: System instruction for the model
            temperature: Controls randomness (0.0 to 2.0)
            top_p: Controls diversity via nucleus sampling
            max_output_tokens: Maximum number of tokens to generate
            safety_settings: Safety settings configuration
            thinking_config: Thinking configuration for the model (e.g. {"include_thoughts": True, "thinking_level": "high"})
            http_options: HTTP options for the client (e.g. {"api_version": "v1beta1"})
            tools_config: Configuration for tools (e.g. {"google_search": True})
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.system_instruction = system_instruction
        
        # Configuration parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.safety_settings = safety_settings or []
        self.thinking_config = thinking_config
        self.http_options = http_options
        self.tools_config = tools_config
        
        # Authenticate
        self._authenticate(api_key, service_account_file)
        
        # Initialize the client
        self._initialize_client()
    
    def _authenticate(self, api_key: Optional[str], service_account_file: Optional[str]):
        """Handle authentication via service account by default, fallback to API key."""
        if service_account_file:
            # Authentication via service account (default)
            if not self.project_id:
                raise ValueError("project_id is required when using service account authentication")

            creds = None
            # Check if it's a file path or JSON string
            if os.path.exists(service_account_file):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                # print(f"Authenticated with service account from file: {service_account_file}")
            else:
                # It's a JSON string, parse it and create in-memory credentials
                try:
                    service_account_info = json.loads(service_account_file)
                    creds = Credentials.from_service_account_info(
                        service_account_info,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    # print("Authenticated with service account from JSON string (in-memory)")
                except json.JSONDecodeError:
                    raise ValueError("Service account parameter is neither a valid file path nor JSON string")

            # Build client kwargs
            client_kwargs = {
                "vertexai": True,
                "project": self.project_id,
                "location": self.location
            }
            if creds:
                client_kwargs["credentials"] = creds

            if self.http_options:
                client_kwargs["http_options"] = self.http_options

            self.client = genai.Client(**client_kwargs)
        elif api_key:
            # Fallback: Authentication via API key (Gemini API)
            client_kwargs = {"vertexai": True, "api_key": api_key}
            if self.http_options:
                client_kwargs["http_options"] = self.http_options
            self.client = genai.Client(**client_kwargs)
            # print("Authenticated with API key")
        else:
            raise ValueError("Either service_account_file or api_key must be provided")
    def _initialize_client(self):
        """Initialize the model with configuration."""
        # Build generation config
        config_dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }
        
        if self.thinking_config:
            config_dict["thinking_config"] = self.thinking_config
        
        # Build tools if provided
        if self.tools_config and self.tools_config.get("google_search"):
            config_dict["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        self.generation_config = types.GenerateContentConfig(**config_dict)
        
        # Build safety settings if provided
        # Build safety settings if provided
        if self.safety_settings:
            # Convert dict safety settings to SafetySetting objects
            converted_settings = []
            for setting in self.safety_settings:
                if isinstance(setting, dict):
                    converted_settings.append(
                        types.SafetySetting(
                            category=setting.get("category"),
                            threshold=setting.get("threshold")
                        )
                    )
                else:
                    converted_settings.append(setting)
            self.generation_config.safety_settings = converted_settings
        
        # Add system instruction if provided
        if self.system_instruction:
            self.generation_config.system_instruction = self.system_instruction
        
        # print(f"Initialized model: {self.model_name}")
    
    def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            model_names = [model.name for model in models]
            return model_names
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate_content(
        self,
        prompt: str,
        override_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate content using the model.
        
        Args:
            prompt: The prompt/question to send to the model
            override_config: Optional config to override default settings for this request
            
        Returns:
            Generated text response
        """
        # Use default config or override
        config = self.generation_config
        if override_config:
            config_dict = {
                "temperature": override_config.get("temperature", self.temperature),
                "top_p": override_config.get("top_p", self.top_p),
                "max_output_tokens": override_config.get("max_output_tokens", self.max_output_tokens),
            }
            

            
            config = types.GenerateContentConfig(**config_dict)
            
            if "safety_settings" in override_config:
                # Convert dict safety settings to SafetySetting objects
                settings = override_config["safety_settings"]
                converted_settings = []
                for setting in settings:
                    if isinstance(setting, dict):
                        converted_settings.append(
                            types.SafetySetting(
                                category=setting.get("category"),
                                threshold=setting.get("threshold")
                            )
                        )
                    else:
                        converted_settings.append(setting)
                config.safety_settings = converted_settings
            elif self.safety_settings:
                config.safety_settings = self.generation_config.safety_settings
            
            if "system_instruction" in override_config:
                config.system_instruction = override_config["system_instruction"]
            elif self.system_instruction:
                config.system_instruction = self.system_instruction
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response.text or ""
    
    def generate_content_stream(
        self,
        prompt: str,
        override_config: Optional[Dict[str, Any]] = None
    ):
        """
        Generate content using streaming.
        
        Args:
            prompt: The prompt/question to send to the model
            override_config: Optional config to override default settings for this request
            
        Yields:
            Chunks of generated text
        """
        # Use default config or override
        config = self.generation_config
        if override_config:
            config_dict = {
                "temperature": override_config.get("temperature", self.temperature),
                "top_p": override_config.get("top_p", self.top_p),
                "max_output_tokens": override_config.get("max_output_tokens", self.max_output_tokens),
            }

            
            config = types.GenerateContentConfig(**config_dict)
            
            if "safety_settings" in override_config:
                # Convert dict safety settings to SafetySetting objects
                settings = override_config["safety_settings"]
                converted_settings = []
                for setting in settings:
                    if isinstance(setting, dict):
                        converted_settings.append(
                            types.SafetySetting(
                                category=setting.get("category"),
                                threshold=setting.get("threshold")
                            )
                        )
                    else:
                        converted_settings.append(setting)
                config.safety_settings = converted_settings
            elif self.safety_settings:
                config.safety_settings = self.generation_config.safety_settings
            
            if "system_instruction" in override_config:
                config.system_instruction = override_config["system_instruction"]
            elif self.system_instruction:
                config.system_instruction = self.system_instruction
        
        # Generate content with streaming
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        ):
            if chunk.text:
                yield chunk.text


def main():
    """Example usage of the VertexAIClient."""
    
    # Example 0: List available models
    print("=== Example 0: List Available Models ===")
    try:
        client = VertexAIClient.from_config(config_section="service_account_example")
        models = client.list_models()
        print(f"Available models ({len(models)}):")
        for model in models[:10]:  # Show first 10 models
            print(f"  - {model}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        print()
    except Exception as e:
        print(f"Error listing models: {e}\n")
    # Example 1: Using config file with API key authentication
    print("=== Example 1: API Key Authentication (from config) ===")
    try:
        client = VertexAIClient.from_config(config_section="api_key_example")
        
        response = client.generate_content("What are the key factors to consider in algorithmic trading?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error with API key auth: {e}\n")
    
    # Example 2: Using config file with service account authentication
    print("=== Example 2: Service Account Authentication (from config) ===")
    try:
        client = VertexAIClient.from_config(config_section="service_account_example")
        
        response = client.generate_content("Explain the concept of market volatility.")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error with service account auth: {e}\n")
    
    # Example 3: Streaming response with config
    print("=== Example 3: Streaming Response (from config) ===")
    try:
        client = VertexAIClient.from_config(config_section="streaming_example")
        
        print("Streaming response:")
        for chunk in client.generate_content_stream("List 3 popular trading strategies in brief."):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error with streaming: {e}\n")
    

if __name__ == "__main__":
    main()
