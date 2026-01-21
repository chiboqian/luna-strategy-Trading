# Vertex AI CLI Usage Guide

A command-line interface for interacting with Google's Vertex AI Gemini models.

## Quick Start

```bash
# Show help
python AI/vertex_ai_cli.py --help

# Interactive chat
python AI/vertex_ai_cli.py chat

# Generate content
python AI/vertex_ai_cli.py generate "What is algorithmic trading?"

# List available models
python AI/vertex_ai_cli.py list-models
```

## Commands

### `chat` - Interactive Chat Mode

Start an interactive conversation with the AI model.

```bash
python AI/vertex_ai_cli.py chat
```

**Features:**
- Persistent command history (saved to `~/.vertex_ai_chat_history`)
- Bash-like line editing:
  - `↑/↓` - Navigate through previous commands
  - `Ctrl+A` - Jump to start of line
  - `Ctrl+E` - Jump to end of line
  - `Ctrl+K` - Clear from cursor to end
  - `Ctrl+U` - Clear entire line
  - `Ctrl+R` - Reverse search history
- Type `exit`, `quit`, or `q` to end the session

**Example:**
```bash
python AI/vertex_ai_cli.py chat --temperature 0.7 --verbose
```

### `generate` - Generate Content from Prompt

Generate a response to a single prompt.

```bash
python AI/vertex_ai_cli.py generate "Your prompt here"
```

**Streaming (default):**
By default, responses stream in real-time:
```bash
python AI/vertex_ai_cli.py generate "Explain trading strategies"
```

**Non-streaming:**
Use `--no-stream` to wait for the complete response:
```bash
python AI/vertex_ai_cli.py generate "Explain trading strategies" --no-stream
```

### `list-models` - List Available Models

Display all available Vertex AI models.

```bash
python AI/vertex_ai_cli.py list-models
```

## Configuration Options

### Model Selection

**`--model MODEL_NAME`**  
Override the model specified in config.

```bash
python AI/vertex_ai_cli.py generate "Hello" --model gemini-2.0-flash-exp
```

### Config File

**`--config PATH`**  
Path to configuration file (default: `config/config.yaml`).

```bash
python AI/vertex_ai_cli.py chat --config /path/to/custom/config.yaml
```

**`--config-section SECTION`**  
Config section to use (default: "default").

```bash
python AI/vertex_ai_cli.py chat --config-section service_account_example
```

### Generation Parameters

**`--temperature FLOAT`**  
Controls randomness (0.0 to 2.0). Lower = more deterministic.

```bash
python AI/vertex_ai_cli.py generate "Analyze AAPL" --temperature 0.3
```

**`--top-p FLOAT`**  
Controls diversity via nucleus sampling (0.0 to 1.0).

```bash
python AI/vertex_ai_cli.py generate "Market analysis" --top-p 0.9
```

**`--max-tokens INT`**  
Maximum number of tokens to generate.

```bash
python AI/vertex_ai_cli.py generate "Detailed report" --max-tokens 4096
```

### System Instructions

Override the system instruction for the model.

**`--system-instruction TEXT`**  
Provide instruction as a string:

```bash
python AI/vertex_ai_cli.py generate "TSLA" --system-instruction "You are a financial analyst focused on electric vehicles."
```

**`--system-instruction-file PATH`**  
Load instruction from a file:

```bash
python AI/vertex_ai_cli.py generate "OKLO" --system-instruction-file config/SI/stock_review.txt
```

### Deep Research & Tools

**`--thinking`**  
Enable thinking model capabilities (Deep Research). This enables the model to "think" before responding, useful for complex reasoning tasks.

```bash
python AI/vertex_ai_cli.py generate "Analyze the impact of interest rates on tech stocks" --thinking
```

**`--google-search`**  
Enable Google Search grounding. This allows the model to search the web for up-to-date information.

```bash
python AI/vertex_ai_cli.py generate "What is the current price of AAPL?" --google-search
```

### Output Control

**`--no-stream`**  
Disable streaming (wait for complete response).

```bash
python AI/vertex_ai_cli.py generate "Long analysis" --no-stream
```

**`-v, --verbose`**  
Print resolved configuration and environment variables.

```bash
python AI/vertex_ai_cli.py chat --verbose
```

## Complete Examples

### Example 1: Stock Analysis with Custom Instructions

```bash
python AI/vertex_ai_cli.py generate "Analyze NVDA stock performance" \
  --system-instruction-file config/SI/stock_review.txt \
  --temperature 0.5 \
  --max-tokens 2048
```

### Example 2: Interactive Trading Chat with Specific Model

```bash
python AI/vertex_ai_cli.py chat \
  --model gemini-2.5-flash \
  --config-section service_account_example \
  --temperature 0.7 \
  --verbose
```

### Example 3: Quick Query with High Creativity

```bash
python AI/vertex_ai_cli.py generate "What are emerging trends in fintech?" \
  --temperature 1.5 \
  --top-p 0.95
```

### Example 4: Deterministic Financial Analysis

```bash
python AI/vertex_ai_cli.py generate "Calculate portfolio risk metrics" \
  --temperature 0.1 \
  --no-stream \
  --system-instruction "You are a quantitative analyst. Provide precise calculations."
```

## Environment Setup

The CLI automatically loads environment variables from `.env` files:
- First checks `AI/.env`
- Falls back to project root `.env`

**Required environment variables:**

For service account authentication:
```bash
SERVICE_ACCOUNT_FILE=/path/to/service-account.json
GCP_PROJECT_ID=your-project-id
```

For API key authentication:
```bash
GOOGLE_API_KEY=your-api-key
```

## Configuration File

The default configuration is loaded from `config/config.yaml`:

```yaml
model:
  name: "gemini-2.5-flash-lite"
  location: "us-central1"

generation:
  temperature: 1.0
  top_p: 0.95
  max_output_tokens: 8192

system_instruction_path: "config/SI/system_instruction.txt"
```

## Tips & Best Practices

1. **Use verbose mode** to debug configuration issues:
   ```bash
   python AI/vertex_ai_cli.py chat --verbose
   ```

2. **Lower temperature** for factual, consistent responses:
   ```bash
   python AI/vertex_ai_cli.py generate "Explain P/E ratio" --temperature 0.3
   ```

3. **Higher temperature** for creative, diverse responses:
   ```bash
   python AI/vertex_ai_cli.py generate "Brainstorm investment strategies" --temperature 1.5
   ```

4. **Custom system instructions** for specialized tasks:
   ```bash
   python AI/vertex_ai_cli.py generate "AAPL" --system-instruction-file config/SI/trader.txt
   ```

5. **Use `--no-stream`** when you need the full response before processing:
   ```bash
   python AI/vertex_ai_cli.py generate "JSON data" --no-stream > output.json
   ```

## Troubleshooting

**"Either service_account_file or api_key must be provided"**
- Ensure your `.env` file contains either `SERVICE_ACCOUNT_FILE` or `GOOGLE_API_KEY`

**"Failed to read system_instruction_path"**
- Check that the file path in `config.yaml` is correct
- Use absolute paths or paths relative to the project root

**Authentication errors**
- Verify service account has Vertex AI permissions
- Check that `GCP_PROJECT_ID` is set correctly
- Run with `--verbose` to see which auth method is being used

**No response or hanging**
- Try `--no-stream` to see if it's a streaming issue
- Check network connectivity
- Verify model name is valid with `list-models` command

## Advanced Usage

### Piping and Redirection

```bash
# Save response to file
python AI/vertex_ai_cli.py generate "Market summary" --no-stream > report.txt

# Read prompt from file
cat prompt.txt | xargs -I {} python AI/vertex_ai_cli.py generate "{}"

# Chain with other commands
python AI/vertex_ai_cli.py generate "Latest tech stocks" | grep -i "nvidia"
```

### Scripting

```bash
#!/bin/bash
# Analyze multiple stocks
for stock in AAPL MSFT GOOGL; do
  echo "=== Analyzing $stock ==="
  python AI/vertex_ai_cli.py generate "Analyze $stock" \
    --system-instruction-file config/SI/stock_review.txt \
    --temperature 0.5
  echo ""
done
```

## Command Summary

| Command | Description |
|---------|-------------|
| `chat` | Interactive chat with history |
| `generate` | Generate response to prompt |
| `list-models` | Show available models |

| Option | Description |
|--------|-------------|
| `--model` | Override model name |
| `--config` | Custom config file path |
| `--config-section` | Config section to use |
| `--temperature` | Randomness (0.0-2.0) |
| `--top-p` | Nucleus sampling (0.0-1.0) |
| `--max-tokens` | Max output tokens |
| `--system-instruction` | System instruction text |
| `--system-instruction-file` | System instruction file |
| `--thinking` | Enable thinking (Deep Research) |
| `--google-search` | Enable Google Search |
| `--no-stream` | Disable streaming |
| `-v, --verbose` | Show configuration |
