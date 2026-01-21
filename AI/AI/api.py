from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import os
import json
import uuid
import datetime
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
# We load from the parent directory's .env
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
# Try loading from workspace root as well
load_dotenv(BASE_DIR / ".env.cloudflare")

# Security Configuration
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_secret_header = APIKeyHeader(name="X-API-Secret", auto_error=False)

async def verify_credentials(
    api_key: str = Security(api_key_header),
    api_secret: str = Security(api_secret_header)
):
    expected_key = os.getenv("TRADING_API_KEY")
    expected_secret = os.getenv("TRADING_API_SECRET")
    
    if not expected_key or not expected_secret:
        # If not configured, block access to be safe
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security not configured"
        )
        
    if api_key != expected_key or api_secret != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid credentials"
        )
    return True

app = FastAPI(title="Vertex AI API", dependencies=[Depends(verify_credentials)])

class RunSessionRequest(BaseModel):
    command_config: str = "config/commands.yaml"
    session_id: Optional[str] = None
    config: Optional[str] = None

def get_d1_status(session_id: str):
    """Query Cloudflare D1 for session status."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        return {"error": "Cloudflare credentials not configured in API"}

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    sql = "SELECT * FROM sessions WHERE session_id = ? LIMIT 1"
    payload = {
        "sql": sql,
        "params": [session_id]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success") and data.get("result"):
            results = data["result"][0]["results"]
            if results:
                return results[0]
            else:
                return {"status": "NOT_FOUND"}
        else:
            return {"error": f"D1 Query failed: {data.get('errors')}"}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/run-session")
def run_session(req: RunSessionRequest):
    script = BASE_DIR / "util" / "run_session.py"
    
    if not script.exists():
         raise HTTPException(status_code=500, detail=f"Script not found: {script}")
    
    # Generate session ID if not provided
    if req.session_id:
        session_id = req.session_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
    
    args = ["--command-config", req.command_config, "--session-id", session_id]
    
    if req.config:
        args.extend(["--config", req.config])
        
    cmd = ["python3", str(script)] + args
    
    # Prepare environment variables
    env = os.environ.copy()
    
    try:
        # Run in background using Popen
        subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.DEVNULL, # Optionally redirect to a log file
            stderr=subprocess.DEVNULL
        )
        
        return {
            "message": "Session started in background",
            "session_id": session_id,
            "status_endpoint": f"/session-status/{session_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session-status/{session_id}")
def get_session_status(session_id: str):
    status_info = get_d1_status(session_id)
    
    if "error" in status_info:
        raise HTTPException(status_code=500, detail=status_info["error"])
        
    if status_info.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if session is still running
    # Known running states from run_session.py
    running_states = ["START", "IN_PROGRESS", "DEEP_DIVE_IN_PROGRESS"]
    current_status = status_info.get("status")
    
    if current_status in running_states:
        # Return 425 Too Early if process is not yet completed/terminal
        return JSONResponse(status_code=425, content=status_info)
        
    return status_info

@app.get("/")
def root():
    return {
        "message": "Vertex AI API is running",
        "endpoints": [
            "/run-session",
            "/session-status/{session_id}"
        ]
    }
