# Vertex AI Client

A Python client for interacting with Vertex AI using the GenAI SDK.

## Features

- **Dual Authentication**: Supports both API key and service account JSON key file authentication
- **YAML Configuration**: Centralized configuration management via YAML files
- **Configurable Model Parameters**: 
  - Temperature
  - Top-p
  - Max output tokens
  - Safety settings
  - Thinking config
- **System Instructions**: Set custom system instructions for the model
- **Streaming Support**: Generate content with streaming responses

## Installation

1. Activate your virtual environment:
```bash
source ../venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### YAML Configuration File

Edit `../config/config.yaml` to set default values:

```yaml
# Model Configuration
model:
  name: "gemini-2.0-flash-exp"
  location: "us-central1"
  
# Generation Configuration
generation:
  temperature: 1.0
  top_p: 0.95
  max_output_tokens: 8192
  
# System Instructions
system_instruction: "You are a helpful AI assistant specializing in trading and finance."

# Safety Settings
safety_settings:
  - category: "HARM_CATEGORY_HATE_SPEECH"
    threshold: "BLOCK_MEDIUM_AND_ABOVE"
```

### Environment Variables (.env file)

Create a `.env` file with your credentials:

```bash
# API Key (for API key authentication)
GOOGLE_API_KEY=your-api-key

# GCP Project ID
GCP_PROJECT_ID=your-project-id

# Service Account (choose one method)
# Method 1: File path
SERVICE_ACCOUNT_FILE=path/to/service-account.json

# Method 2: Inline JSON
SERVICE_ACCOUNT_JSON={"type": "service_account", ...}
```

## Usage

### Using Configuration File (Recommended)

```python
from vertex_ai_client import VertexAIClient

# Use default configuration
client = VertexAIClient.from_config()

# Use specific example configuration
client = VertexAIClient.from_config(config_section="api_key_example")

# Override specific parameters
client = VertexAIClient.from_config(
    temperature=0.3,
    max_output_tokens=500
)

response = client.generate_content("What is algorithmic trading?")
print(response)
```

### Manual Configuration

```python
from vertex_ai_client import VertexAIClient
import os
from dotenv import load_dotenv

load_dotenv()

client = VertexAIClient(
    model_name="gemini-2.0-flash-exp",
    api_key=os.getenv("GOOGLE_API_KEY"),
    system_instruction="You are a helpful AI assistant.",
    temperature=0.7,
    top_p=0.9,
    max_output_tokens=1024
)

response = client.generate_content("What is algorithmic trading?")
print(response)
```

### Authentication with API Key

```python
# Using config file (credentials from .env)
client = VertexAIClient.from_config(config_section="api_key_example")

response = client.generate_content("What is algorithmic trading?")
print(response)
```

### Authentication with Service Account

```python
# Using config file (credentials from .env)
client = VertexAIClient.from_config(config_section="service_account_example")

response = client.generate_content("Explain market volatility.")
print(response)
```

### Streaming Responses

```python
client = VertexAIClient.from_config(config_section="streaming_example")

for chunk in client.generate_content_stream("Tell me about trading strategies."):
    print(chunk, end="", flush=True)
```

## Configuration Parameters

### Model Configuration
- `model.name`: Name of the Vertex AI model
- `model.location`: GCP region (default: "us-central1")

### Generation Configuration
- `generation.temperature`: Controls randomness (0.0 to 2.0, default: 1.0)
- `generation.top_p`: Controls diversity (0.0 to 1.0, default: 0.95)
- `generation.max_output_tokens`: Maximum tokens to generate (default: 8192)

### Other Settings
- `system_instruction`: System instruction for the model
- `safety_settings`: List of safety setting configurations
- `thinking_config`: Configuration for model thinking behavior

## Example Configurations

The `config.yaml` includes several example configurations:
- `api_key_example`: Optimized for API key authentication
- `service_account_example`: Optimized for service account authentication
- `streaming_example`: Optimized for streaming responses

## Running Examples

```bash
python vertex_ai_client.py
```

This will run all examples demonstrating different authentication methods and features.

## Utilities

### Research Session Manager

`util/research.py` creates a timestamped directory under `data/` with JSON logs, executes commands from `config/commands.yaml`, and logs outputs.

Files:
- `logs.jsonl`: command execution history
- `stocks.jsonl`: stock recommendations
- `orders.jsonl`: order tracking

Usage:
```bash
python util/research.py
python util/research.py --dry-run
python util/research.py --verbose
python util/research.py --config config/commands.yaml
```

## API Server

A standalone FastAPI server is available to expose the Vertex AI capabilities over HTTP.

### Running the API

```bash
./run_ai_api.sh
```

The server runs on port **8001** by default.

### Authentication

The API uses the same authentication mechanism as the main Trading API. You must provide the following headers:

- `X-API-Key`: Your `TRADING_API_KEY` from `.env`
- `X-API-Secret`: Your `TRADING_API_SECRET` from `.env`

### Endpoints

#### `POST /generate`

Generate content using Vertex AI.

**Request Body:**

```json
{
  "prompt": "Explain quantum computing",
  "config_section": "default",
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.7,
  "thinking": true,
  "google_search": false
}
```

**Example cURL:**

```bash
curl -X POST "http://localhost:8001/generate" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -H "X-API-Secret: YOUR_SECRET" \
     -d '{
           "prompt": "Analyze the current market trends for tech stocks",
           "thinking": true
         }'
```

