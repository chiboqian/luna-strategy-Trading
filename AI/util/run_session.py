#!/usr/bin/env python3
"""
Session Orchestrator Script
Starts a session, logs to Cloudflare D1, calls Vertex AI, and saves outputs to R2.
"""

import sys
import os
import argparse
import uuid
import datetime
import json
import requests
import boto3
import yaml
import subprocess
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Add parent directory to sys.path to allow importing from AI package
sys.path.append(str(Path(__file__).parent.parent / "AI"))
from vertex_ai_client import VertexAIClient


class CloudflareLogger:
    """Logs session status to Cloudflare D1."""
    
    def __init__(self, account_id: str, database_id: str, api_token: str):
        self.account_id = account_id
        self.database_id = database_id
        self.api_token = api_token
        self.api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def _execute_sql(self, sql: str, params: list = None, suppress_error: bool = False):
        """Execute a SQL query against D1."""
        payload = {
            "sql": sql,
            "params": params or []
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("success"):
                if not suppress_error:
                    print(f"❌ D1 Error: {data.get('errors')}", file=sys.stderr)
            return data
        except Exception as e:
            if not suppress_error:
                print(f"❌ Failed to log to D1: {e}", file=sys.stderr)
            return None

    def create_table_if_not_exists(self):
        """Creates the sessions table if it doesn't exist."""
        sql_sessions = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            consolidated_list_path TEXT
        );
        """
        self._execute_sql(sql_sessions)
        # Ensure column exists for existing tables
        self._execute_sql("ALTER TABLE sessions ADD COLUMN consolidated_list_path TEXT", suppress_error=True)

        sql_analysis = """
        CREATE TABLE IF NOT EXISTS symbol_analysis (
            session_id TEXT,
            symbol TEXT,
            analysis_rating TEXT,
            normalized_rating TEXT,
            conviction_level TEXT,
            run_index INTEGER,
            raw_json TEXT,
            created_at TEXT,
            PRIMARY KEY (session_id, symbol, run_index)
        );
        """
        self._execute_sql(sql_analysis)
        self._execute_sql("ALTER TABLE symbol_analysis ADD COLUMN normalized_rating TEXT", suppress_error=True)

    def _normalize_rating(self, rating: str) -> str:
        if not rating:
            return "Unknown"
        r = rating.lower().strip()
        if r in ["buy", "strong buy", "outperform", "overweight", "strong_buy"]:
            return "Buy"
        elif r in ["hold", "neutral", "market perform"]:
            return "Hold"
        elif r in ["sell", "strong sell", "underperform", "underweight", "strong_sell"]:
            return "Sell"
        return r.title()

    def log_analysis_result(self, session_id: str, result: dict):
        """Logs a single analysis result to D1."""
        now = datetime.datetime.utcnow().isoformat()
        original_rating = result.get("analysis_rating", "")
        normalized_rating = self._normalize_rating(original_rating)
        
        sql = """
        INSERT OR REPLACE INTO symbol_analysis 
        (session_id, symbol, analysis_rating, normalized_rating, conviction_level, run_index, raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self._execute_sql(sql, [
                session_id,
                result.get("symbol"),
                original_rating,
                normalized_rating,
                str(result.get("conviction_level")), # Ensure string if it's a number
                result.get("run_index"),
                json.dumps(result.get("raw_json")),
                now
            ])
        except Exception as e:
            print(f"⚠️ Failed to log analysis for {result.get('symbol')}: {e}")

    def update_consolidated_list(self, session_id: str, path: str):
        """Updates the consolidated list path for a session."""
        now = datetime.datetime.utcnow().isoformat()
        sql = """
        UPDATE sessions 
        SET consolidated_list_path = ?, updated_at = ?
        WHERE session_id = ?
        """
        self._execute_sql(sql, [path, now, session_id])
        print(f"📝 Session {session_id} updated with list path: {path}")

    def log_session(self, session_id: str, status: str):
        """Upserts session status."""
        now = datetime.datetime.utcnow().isoformat()
        
        # Use simple Upsert (requires SQLite 3.24+)
        # This preserves other columns like consolidated_list_path that are not in the insert list
        sql = """
        INSERT INTO sessions (session_id, status, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            status = excluded.status,
            updated_at = excluded.updated_at
        """
        self._execute_sql(sql, [session_id, status, now, now])
        print(f"📝 Session {session_id} status logged: {status}")


class R2Storage:
    """Handles object storage in Cloudflare R2."""

    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto" # R2 requires a region, but auto usually works or 'us-east-1'
        )

    def save_artifact(self, session_id: str, index: int, prompt: str, content: str):
        """Saves the output to R2."""
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        # Naming convention: {date}/{session_id}/{index}_output.json
        object_key = f"{date_str}/{session_id}/{index:03d}_output.json"
        
        data = {
            "session_id": session_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "prompt": prompt,
            "content": content
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=json.dumps(data, indent=2),
                ContentType="application/json"
            )
            print(f"💾 Saved output to R2: s3://{self.bucket_name}/{object_key}")
        except Exception as e:
            print(f"❌ Failed to save to R2: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run an AI Session with logging.")
    parser.add_argument("--session-id", help="Existing session ID (optional)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts", nargs="+", help="List of prompts to execute")
    group.add_argument("--command-config", help="Path to command config file (yaml)")
    parser.add_argument("--config", help="Path to AI config file")
    
    # Env var overrides can be handy, but assume Env vars are set for credentials
    args = parser.parse_args()

    # Load environment variables from .env files
    # Try loading from AI directory (../.env) and Workspace Root (../../.env)
    script_dir = Path(__file__).parent
    
    # Load from AI directory
    ai_env = script_dir.parent / ".env"
    if ai_env.exists():
        load_dotenv(dotenv_path=ai_env)
        
    # Load from Workspace Root
    root_env = script_dir.parent.parent / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env)

    # Load Credentials from Environment
    cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf_api_token = os.environ.get("CLOUDFLARE_API_TOKEN") # For D1
    d1_database_id = os.environ.get("CLOUDFLARE_D1_DATABASE_ID")
    
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket_name = os.environ.get("R2_BUCKET_NAME")

    # missing_creds = []
    # if not cf_account_id: missing_creds.append("CLOUDFLARE_ACCOUNT_ID")
    # ... (Add checks if needed, but for now assuming user will provide them)
    
    # Initialize components
    if args.session_id:
        session_id = args.session_id
    else:
        # Generate session ID with timestamp and short UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
        
    print(f"🚀 Starting Session: {session_id}")

    # Setup Logging & Storage
    # If credentials are poor, we might skip logging but let's assume they are needed as per requirement
    logger = None
    if cf_account_id and cf_api_token and d1_database_id:
        logger = CloudflareLogger(cf_account_id, d1_database_id, cf_api_token)
        logger.create_table_if_not_exists()
        logger.log_session(session_id, "START")
    else:
        print("⚠️  Missing Cloudflare D1 credentials. Skipping SQL logging.")

    storage = None
    if cf_account_id and r2_access_key and r2_secret_key and r2_bucket_name:
        storage = R2Storage(cf_account_id, r2_access_key, r2_secret_key, r2_bucket_name)
    else:
        print("⚠️  Missing R2 credentials. Skipping Object Storage.")

    # Initialize AI Client or Load Commands
    try:
        if args.command_config:
            # Mode 1: Execute Commands from YAML Config
            print(f"📖 Reading command config from {args.command_config}")
            with open(args.command_config, 'r') as f:
                cmd_data = yaml.safe_load(f)
            
            commands = cmd_data.get('commands', [])
            if not commands:
                print("⚠️  No commands found in config file.")
                sys.exit(0)
                
            # Update status
            if logger:
                logger.log_session(session_id, "IN_PROGRESS")
            
            combined_tickers = set()

            for i, cmd_info in enumerate(commands, 1):
                cmd_str = cmd_info.get('command')
                name = cmd_info.get('name', f"Command {i}")
                desc = cmd_info.get('description', "")
                
                print(f"\nArguments {i}/{len(commands)}: {name}")
                if desc:
                    print(f"  Description: {desc}")
                print(f"  Command: {cmd_str}")

                # Execute shell command
                try:
                    # Run subprocess with captured output
                    # shell=True to handle arguments/pipes.
                    result = subprocess.run(
                        cmd_str, 
                        shell=True, 
                        check=True, 
                        text=True, 
                        capture_output=True
                    )
                    
                    response = result.stdout
                    print("✅ Command executed successfully.")
                    
                    # Process Output for Tickers
                    try:
                        output_data = json.loads(response)
                        if isinstance(output_data, dict):
                            # Handle standard output format {"log": "...", "list": [...]}
                            items_list = output_data.get("list")
                            if items_list and isinstance(items_list, list):
                                for item in items_list:
                                    if isinstance(item, dict):
                                        ticker = item.get("ticker") or item.get("ticket")
                                        if ticker:
                                            combined_tickers.add(ticker)
                    except json.JSONDecodeError:
                        pass
                    
                    if response.strip():
                        print(f"Output preview: {response[:200]}..." if len(response) > 200 else f"Output: {response}")
                    
                    # Log stderr if any, just in case
                    if result.stderr:
                        print(f"Stderr: {result.stderr}")

                    # Save to R2 (using command name/prompt as key info if possible, or just the command string)
                    if storage:
                        storage.save_artifact(session_id, i, f"{name}: {cmd_str}", response)

                except subprocess.CalledProcessError as e:
                    print(f"❌ Command failed with return code {e.returncode}")
                    print(f"Stderr: {e.stderr}")
                    if logger:
                        logger.log_session(session_id, f"ERROR: Command {i} failed")
                    raise e
                    
            # Log the final unique list of symbols
            final_symbols = list(combined_tickers)
            print("\n📊 Consolidated Symbols List:")
            print(json.dumps({"symbols": final_symbols}, indent=2))
            
            # Save consolidated list to R2
            if storage:
                storage.save_artifact(session_id, 999, "Consolidated Symbols", json.dumps({"symbols": final_symbols}))
                
                # Also save as consolidated-list inside the session directory for easier access
                try:
                    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                    list_key = f"{date_str}/{session_id}/consolidated-list.json"
                    storage.s3_client.put_object(
                        Bucket=storage.bucket_name,
                        Key=list_key,
                        Body=json.dumps({"symbols": final_symbols}, indent=2),
                        ContentType="application/json"
                    )
                    print(f"💾 Saved consolidated list to R2: s3://{storage.bucket_name}/{list_key}")
                    
                    if logger:
                        logger.update_consolidated_list(session_id, list_key)
                        
                except Exception as e:
                    print(f"❌ Failed to save consolidated list to R2: {e}", file=sys.stderr)

            # Deep Dive Step
            if final_symbols:
                print("\n🌊 Starting Deep Dive Analysis...")
                if logger:
                    logger.log_session(session_id, "DEEP_DIVE_IN_PROGRESS")
                
                try:
                    deep_dive_script = script_dir / "deep_dive.py"
                    if deep_dive_script.exists():
                        cmd_args = [sys.executable, str(deep_dive_script), "--symbols"] + final_symbols + ["--json-output"]
                        
                        # Execute deep dive
                        # We stream stderr to console for progress, but capture stdout for JSON data
                        result = subprocess.run(
                            cmd_args,
                            check=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=None  # inherit stderr so it prints to console
                        )
                        
                        deep_dive_output = result.stdout
                        print("✅ Deep Dive completed.")
                        
                        # Parse and log Deep Dive Results to D1
                        if logger:
                            try:
                                analysis_results = json.loads(deep_dive_output)
                                if isinstance(analysis_results, list):
                                    print(f"📝 Logging {len(analysis_results)} analysis results to D1...")
                                    for res in analysis_results:
                                        if isinstance(res, dict):
                                            logger.log_analysis_result(session_id, res)
                            except json.JSONDecodeError:
                                print("⚠️ Could not parse deep dive output for D1 logging.")
                        
                        # Save deep dive artifact to R2
                        if storage:
                            storage.save_artifact(session_id, 1000, "Deep Dive Analysis", deep_dive_output)
                            
                            # Also save as readable file
                            try:
                                date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                                dd_key = f"{date_str}/{session_id}/deep-dive-results.json"
                                storage.s3_client.put_object(
                                    Bucket=storage.bucket_name,
                                    Key=dd_key,
                                    Body=deep_dive_output,
                                    ContentType="application/json"
                                )
                                print(f"💾 Saved deep dive results to R2: s3://{storage.bucket_name}/{dd_key}")
                            except Exception as e:
                                print(f"❌ Failed to save deep dive file to R2: {e}", file=sys.stderr)
                                
                    else:
                        print(f"⚠️ Deep dive script not found at {deep_dive_script}")
                        
                except subprocess.CalledProcessError as e:
                    print(f"❌ Deep Dive failed with return code {e.returncode} (See above for details)", file=sys.stderr)
                    if logger:
                        logger.log_session(session_id, "DEEP_DIVE_ERROR")
                except Exception as e:
                    print(f"❌ Error running Deep Dive: {e}", file=sys.stderr)

        elif args.prompts:
            # Mode 2: Execute Prompts using Python Client
            client = VertexAIClient.from_config(config_path=args.config)
            
            # Update status
            if logger:
                logger.log_session(session_id, "IN_PROGRESS")

            for i, prompt in enumerate(args.prompts, 1):
                print(f"\nArguments {i}/{len(args.prompts)}: {prompt[:50]}...")
                
                # Call Vertex AI
                try:
                    response = client.generate_content(prompt)
                    print(f"✅ Generated response:\n{response}\n")
                    
                    # Save to R2
                    if storage:
                        storage.save_artifact(session_id, i, prompt, response)
                        
                except Exception as e:
                    print(f"❌ Error during generation: {e}")
                    raise e

        # Success loop finished
        if logger:
            logger.log_session(session_id, "COMPLETED")
        print(f"\n✨ Session {session_id} completed successfully.")
        
        # Output Final JSON with session ID
        print(json.dumps({"session_id": session_id, "status": "COMPLETED"}))

    except Exception as e:
        print(f"\n💥 Session failed: {e}")
        if logger:
            logger.log_session(session_id, f"ERROR: {str(e)}")
        # Output Final JSON for failure as well
        print(json.dumps({"session_id": session_id, "status": "ERROR", "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
