#!/usr/bin/env python3
"""
Command-line interface for Vertex AI Client.
"""

import sys
import os
import argparse
import readline
import json
import re
from pathlib import Path

# Add parent directory to sys.path to allow importing from AI package
sys.path.append(str(Path(__file__).parent.parent / "AI"))
from vertex_ai_client import VertexAIClient


def setup_readline():
    """Configure readline for bash-like history and line editing."""
    # Enable history
    history_file = Path.home() / ".vertex_ai_chat_history"
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    
    # Set history length
    readline.set_history_length(1000)
    
    # Enable tab completion (basic)
    readline.parse_and_bind("tab: complete")
    
    # Enable emacs-style keybindings (Ctrl+A, Ctrl+E, etc.)
    readline.parse_and_bind("set editing-mode emacs")
    
    return history_file


def chat_interactive(client: VertexAIClient):
    """Start an interactive chat session with bash-like history and editing."""
    print("\n🤖 Interactive Chat Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("Use ↑/↓ arrows to navigate history, Ctrl+A/E for line start/end")
    print("Ctrl+R for reverse search, Ctrl+K to clear line\n")
    
    history_file = setup_readline()
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            response = client.generate_content(prompt)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
        finally:
            # Save history on exit
            try:
                readline.write_history_file(history_file)
            except:
                pass


def chat_stream(client: VertexAIClient, prompt: str):
    """Generate a streaming response."""
    for chunk in client.generate_content_stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")


def list_models_cmd(client: VertexAIClient):
    """List all available models."""
    print("\n📋 Available Models:")
    models = client.list_models()
    if models:
        for i, model in enumerate(models, 1):
            print(f"{i:3d}. {model}")
        print(f"\nTotal: {len(models)} models")
    else:
        print("No models found or error occurred.")


def generate_content_cmd(client: VertexAIClient, prompt: str, stream: bool = False, json_output: bool = False):
    """Generate content from a prompt."""
    if stream and not json_output:
        chat_stream(client, prompt)
    else:
        response = client.generate_content(prompt)
        if json_output:
            # Try to extract JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                try:
                    json_content = json.loads(json_match.group(1))
                    # Log is everything except the JSON block
                    log_content = response.replace(json_match.group(0), "").strip()
                    output = {
                        "log": log_content,
                        "list": json_content
                    }
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails but block exists
                    output = {"log": response, "list": None, "error": "Failed to parse JSON block"}
            else:
                # No JSON block found, treat whole response as log
                output = {"log": response, "list": None}
            
            # Ensure we only print the JSON output
            print(json.dumps(output))
        else:
            print(response)


def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Client CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat mode
  python vertex_ai_cli.py chat
  
  # Generate content from prompt
  python vertex_ai_cli.py generate "What is algorithmic trading?"
  
    # Generate with streaming (default)
    python vertex_ai_cli.py generate "Explain trading strategies"

    # Generate without streaming
    python vertex_ai_cli.py generate "Explain trading strategies" --no-stream
  
  # List available models
  python vertex_ai_cli.py list-models
  
  # Use specific config section
  python vertex_ai_cli.py chat --config-section api_key_example
        """
    )
    
    parser.add_argument(
        "command",
        choices=["chat", "generate", "list-models"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt for generate command"
    )
    
    parser.add_argument(
        "--config",
        help="Path to config file (default: config/AI.yaml)"
    )
    
    parser.add_argument(
        "--config-section",
        default="default",
        help="Config section to use (default: default)"
    )
    
    # Streaming is enabled by default; provide flag to disable
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming; return full response at once"
    )
    
    parser.add_argument(
        "--model",
        help="Override model name"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Override top-p (nucleus sampling)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override max output tokens"
    )

    # Verbose output
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print resolved configuration and environment variables"
    )

    # Override system instruction
    parser.add_argument(
        "--system-instruction",
        help="Override system instruction text"
    )
    parser.add_argument(
        "--system-instruction-file",
        help="Path to a file whose contents override system instruction"
    )

    # Deep Research / Thinking / Tools
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking model capabilities (Deep Research)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking model capabilities"
    )
    parser.add_argument(
        "--google-search",
        action="store_true",
        help="Enable Google Search grounding"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    # Print help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate command requirements
    if args.command == "generate" and not args.prompt:
        parser.error("generate command requires a prompt")
    
    # Build override parameters
    overrides = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.top_p is not None:
        overrides["top_p"] = args.top_p
    if args.max_tokens:
        overrides["max_output_tokens"] = args.max_tokens
    
    # System instruction overrides (file takes precedence)
    if args.system_instruction_file:
        try:
            with open(args.system_instruction_file, "r") as f:
                overrides["system_instruction"] = f.read().strip()
        except Exception as e:
            print(f"\n❌ Failed to read system instruction file '{args.system_instruction_file}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.system_instruction:
        overrides["system_instruction"] = args.system_instruction
    
    # Handle Deep Research / Thinking / Tools overrides
    if args.thinking:
        overrides["thinking_config"] = {"include_thoughts": True, "thinking_level": "high"}
        # Thinking models often require a specific model name, but we'll let the user/config decide or default
        # If the user didn't specify a model, we might want to suggest one, but for now we trust the config/args.
    elif args.no_thinking:
        overrides["thinking_config"] = None
    
    if args.google_search:
        overrides["tools_config"] = {"google_search": True}
    
    try:
        # Initialize client
        if not args.json:
            print(f"🔧 Initializing client (config section: {args.config_section})...")
        client = VertexAIClient.from_config(
            config_path=args.config,
            config_section=args.config_section,
            **overrides
        )
        if not args.json:
            print("✅ Client initialized successfully\n")

        # Verbose: print resolved configuration
        if args.verbose and not args.json:
            def yes_no(val):
                return "yes" if val else "no"
            print("🔎 Resolved configuration:")
            try:
                print(f"  model_name: {client.model_name}")
                print(f"  location: {client.location}")
                print(f"  project_id: {client.project_id}")
                sys_instr_preview = (client.system_instruction or "").strip()
                if sys_instr_preview:
                    preview = (sys_instr_preview[:80] + ("…" if len(sys_instr_preview) > 80 else ""))
                else:
                    preview = "<none>"
                print(f"  system_instruction: {preview}")
                print(f"  temperature: {client.temperature}")
                print(f"  top_p: {client.top_p}")
                print(f"  max_output_tokens: {client.max_output_tokens}")
                print(f"  safety_settings: {len(client.safety_settings)} entries")
                # Thinking config
                print(f"  thinking_config: {client.thinking_config if getattr(client, 'thinking_config', None) else 'none'}")
                # Tools config
                print(f"  tools_config: {client.tools_config if getattr(client, 'tools_config', None) else 'none'}")
                # Auth info
                api_key_set = 'GOOGLE_API_KEY' in (client.__dict__.get('_env_keys', []) or []) or bool("GOOGLE_API_KEY" in dict(**{}))
                sa_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                print(f"  auth: service_account={'yes' if sa_env else 'no'}, api_key={yes_no(bool(os.getenv('GOOGLE_API_KEY')))}")
                if sa_env:
                    print(f"  GOOGLE_APPLICATION_CREDENTIALS: {sa_env}")
            except Exception as e:
                print(f"  (verbose) failed to print config: {e}")
        
        # Execute command
        if args.command == "chat":
            chat_interactive(client)
        elif args.command == "generate":
            # Default to streaming unless --no-stream is set
            stream = not args.no_stream
            generate_content_cmd(client, args.prompt, stream, args.json)
        elif args.command == "list-models":
            list_models_cmd(client)
            
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
