from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, model_validator
from typing import Optional, List, Any
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
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

app = FastAPI(title="AI Trading API", dependencies=[Depends(verify_credentials)])

BASE_DIR = Path(__file__).parent.parent

class BaseRequest(BaseModel):
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_base_url: Optional[str] = None

class ExecuteBuysRequest(BaseRequest):
    dollars: Optional[float] = None
    market: bool = False
    mid_price: bool = False
    dry_run: bool = False
    verbose: bool = False
    recommendations: Optional[List[Any]] = None

class ExecuteSellsRequest(BaseRequest):
    dollars: Optional[float] = None
    market: bool = False
    mid_price: bool = False
    use_bid: bool = False
    price_offset: float = 0.0
    dry_run: bool = False
    verbose: bool = False
    recommendations: Optional[List[Any]] = None

class ExecuteTradesRequest(BaseRequest):
    dollars: Optional[float] = None
    market: bool = False
    mid_price: bool = False
    use_bid: bool = False
    price_offset: float = 0.0
    dry_run: bool = False
    verbose: bool = False
    recommendations: Optional[List[Any]] = None
    session_id: Optional[str] = None

class CloseOldPositionsRequest(BaseRequest):
    days: Optional[int] = None
    dry_run: bool = False
    use_d1: bool = True

class SyntheticLongRequest(BaseRequest):
    symbol: str
    quantity: Optional[int] = None
    amount: Optional[float] = None
    days: Optional[int] = 30
    window: Optional[int] = 35
    protection_pct: Optional[float] = 5.0
    dry_run: bool = False

class GetRecommendationsRequest(BaseRequest):
    min_score: Optional[float] = None
    limit: Optional[int] = None

class RecommendationItem(BaseModel):
    action: str
    ticker: str

    @model_validator(mode='before')
    @classmethod
    def map_ticket_to_ticker(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'ticket' in data and 'ticker' not in data:
                data['ticker'] = data['ticket']
        return data

class LogData(BaseModel):
    log: str
    list: List[RecommendationItem]

class LogInputItem(BaseModel):
    data: LogData
    class Config:
        extra = "ignore"

class ExecuteSingleBuyRequest(BaseRequest):
    symbol: str
    dollars: Optional[float] = None
    market: bool = False
    mid_price: bool = False
    verbose: bool = False
    recommendation: Optional[str] = None

class ExecuteSingleSellRequest(BaseRequest):
    symbol: str
    dollars: Optional[float] = None
    market: bool = False
    mid_price: bool = False
    use_bid: bool = False
    price_offset: float = 0.0
    verbose: bool = False
    recommendation: Optional[str] = None

class MonitorOrdersRequest(BaseRequest):
    order_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    interval: Optional[int] = None
    max_retries: Optional[int] = None
    quiet: bool = True

class MarketStatusRequest(BaseRequest):
    min_minutes: Optional[int] = None

class AccountSummaryRequest(BaseRequest):
    pass

def get_alpaca_env(req: BaseRequest) -> dict:
    env_vars = {}
    if req.alpaca_api_key:
        env_vars["ALPACA_API_KEY"] = req.alpaca_api_key
    if req.alpaca_api_secret:
        env_vars["ALPACA_API_SECRET"] = req.alpaca_api_secret
    if req.alpaca_base_url:
        env_vars["ALPACA_BASE_URL"] = req.alpaca_base_url
    return env_vars

def run_script(script_path: Path, args: list, env_vars: dict = None):
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"Script not found: {script_path}")
    
    cmd = ["python3", str(script_path)] + args
    
    # Prepare environment variables
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute/buys")
def execute_buys(req: ExecuteBuysRequest):
    script = BASE_DIR / "util" / "execute_buys.py"
    args = []
    if req.dollars:
        args.extend(["--dollars", str(req.dollars)])
    if req.market:
        args.append("--market")
    if req.mid_price:
        args.append("--mid-price")
    if req.dry_run:
        args.append("--dry-run")
    if req.verbose:
        args.append("--verbose")
    if req.recommendations:
        args.extend(["--json-payload", json.dumps(req.recommendations)])
    
    args.append("--json")
    
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/monitor")
def monitor_orders(req: MonitorOrdersRequest):
    # Validate that either order_ids or session_id is provided
    if not req.order_ids and not req.session_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'order_ids' or 'session_id' must be provided"
        )
    
    script = BASE_DIR / "util" / "monitor_orders.py"
    args = []
    
    # Use session_id if provided, otherwise use order_ids
    if req.session_id:
        args.extend(["--session-id", req.session_id])
    else:
        args.extend(["--orders", json.dumps(req.order_ids)])
    
    args.append("--json")
    
    if req.interval:
        args.extend(["--interval", str(req.interval)])
    if req.max_retries:
        args.extend(["--max-retries", str(req.max_retries)])
    if req.quiet:
        args.append("--quiet")
        
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/execute/sells")
def execute_sells(req: ExecuteSellsRequest):
    script = BASE_DIR / "util" / "execute_sells.py"
    args = []
    if req.dollars:
        args.extend(["--dollars", str(req.dollars)])
    if req.market:
        args.append("--market")
    if req.mid_price:
        args.append("--mid-price")
    if req.use_bid:
        args.append("--use-bid")
    if req.price_offset != 0.0:
        args.extend(["--price-offset", str(req.price_offset)])
    if req.dry_run:
        args.append("--dry-run")
    if req.verbose:
        args.append("--verbose")
    if req.recommendations:
        args.extend(["--json-payload", json.dumps(req.recommendations)])
    
    args.append("--json")
    
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/execute/all")
def execute_all(req: ExecuteTradesRequest):
    script = BASE_DIR / "util" / "execute_session.py"
    args = []
    
    # Session ID
    if req.session_id:
        args.extend(["--session-id", req.session_id])
        
    # Standard flags
    if req.dollars:
        args.extend(["--dollars", str(req.dollars)])
    if req.market:
        args.append("--market")
    if req.mid_price:
        args.append("--mid-price")
    if req.dry_run:
        args.append("--dry-run")
    if req.verbose:
        args.append("--verbose")
        
    # Sell specific
    if req.use_bid:
        args.append("--use-bid")
    if req.price_offset != 0.0:
        args.extend(["--price-offset", str(req.price_offset)])
        
    # Recommendations override
    if req.recommendations:
        args.extend(["--recommendations-json", json.dumps(req.recommendations)])
    
    # Always output JSON for API consumption
    args.append("--json")
    
    env_vars = get_alpaca_env(req)

    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing fails but success is False, propagate error
    if not result["success"]:
        error_msg = result.get("stderr") or result.get("error") or "Unknown execution error"
        raise HTTPException(status_code=500, detail=error_msg)
        
    return result

@app.post("/positions/close-old")
def close_old_positions(req: CloseOldPositionsRequest):
    script = BASE_DIR / "util" / "close_old_positions.py"
    args = ["--json"]
    if req.days:
        args.extend(["--days", str(req.days)])
    if req.dry_run:
        args.append("--dry-run")
    if req.use_d1:
        args.append("--use-d1")
    
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/execute/synthetic_long")
def execute_synthetic_long(req: SyntheticLongRequest):
    script = BASE_DIR / "util" / "synthetic_long.py"
    args = [req.symbol]
    
    if req.quantity:
        args.extend(["--quantity", str(req.quantity)])
    if req.amount:
        args.extend(["--amount", str(req.amount)])
    
    if req.days:
        args.extend(["--days", str(req.days)])
    if req.window:
        args.extend(["--window", str(req.window)])
    if req.protection_pct:
        args.extend(["--protection-pct", str(req.protection_pct)])
    
    if req.dry_run:
        args.append("--dry-run")
        
    args.append("--json")
    
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/recommendations/buys")
def get_buy_recommendations(req: GetRecommendationsRequest):
    script = BASE_DIR / "util" / "get_buy_list.py"
    args = []
    if req.min_score is not None:
        args.extend(["--min-score", str(req.min_score)])
    if req.limit is not None:
        args.extend(["--limit", str(req.limit)])
        
    env_vars = get_alpaca_env(req)
    return run_script(script, args, env_vars)

@app.post("/recommendations/sells")
def get_sell_recommendations(req: GetRecommendationsRequest):
    script = BASE_DIR / "util" / "get_sell_list.py"
    args = []
    if req.min_score is not None:
        args.extend(["--min-score", str(req.min_score)])
    if req.limit is not None:
        args.extend(["--limit", str(req.limit)])
        
    env_vars = get_alpaca_env(req)
    return run_script(script, args, env_vars)

@app.post("/orders/buy")
def execute_single_buy(req: ExecuteSingleBuyRequest):
    script = BASE_DIR / "util" / "execute_single_buy.py"
    args = [req.symbol]
    if req.dollars:
        args.extend(["--dollars", str(req.dollars)])
    if req.market:
        args.append("--market")
    if req.mid_price:
        args.append("--mid-price")
    if req.verbose:
        args.append("--verbose")
    if req.recommendation:
        args.extend(["--recommendation", req.recommendation])
        
    env_vars = get_alpaca_env(req)
    return run_script(script, args, env_vars)

@app.post("/orders/sell")
def execute_single_sell(req: ExecuteSingleSellRequest):
    script = BASE_DIR / "util" / "execute_single_sell.py"
    args = [req.symbol]
    if req.dollars:
        args.extend(["--dollars", str(req.dollars)])
    if req.market:
        args.append("--market")
    if req.mid_price:
        args.append("--mid-price")
    if req.use_bid:
        args.append("--use-bid")
    if req.price_offset != 0.0:
        args.extend(["--price-offset", str(req.price_offset)])
    if req.verbose:
        args.append("--verbose")
    if req.recommendation:
        args.extend(["--recommendation", req.recommendation])
        
    env_vars = get_alpaca_env(req)
    return run_script(script, args, env_vars)

@app.post("/market/status")
def check_market_status(req: MarketStatusRequest):
    script = BASE_DIR / "Trading" / "market_open.py"
    args = ["--json"]
    if req.min_minutes is not None:
        args.extend(["--min-minutes", str(req.min_minutes)])
        
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    # Try to parse JSON output for cleaner response
    if result["success"]:
        try:
            result["data"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.post("/account/summary")
def get_account_summary(req: AccountSummaryRequest):
    script = BASE_DIR / "util" / "account_summary.py"
    args = ["--json"]
    
    env_vars = get_alpaca_env(req)
    result = run_script(script, args, env_vars)
    
    if result["stdout"]:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
            
    return result

@app.get("/trading/keys")
@app.get("/trading/keys/{label}")
def get_api_keys(label: Optional[str] = None):
    # Reload env vars to ensure we have the latest values
    load_dotenv(override=True)
    
    keys_config = {}
    config_loaded = False
    
    # Try ALPACA_KEYS environment variable first
    alpaca_keys_env = os.getenv("ALPACA_KEYS") or os.getenv("APLACA_KEYS")
    if alpaca_keys_env:
        try:
            keys_config = json.loads(alpaca_keys_env)
            config_loaded = True
        except json.JSONDecodeError:
            pass

    # Fallback to file if env var not present or invalid
    config_path = BASE_DIR / ".secrets" / "alpaca_api_keys.json"
    if not config_loaded:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    keys_config = json.load(f)
                config_loaded = True
            except json.JSONDecodeError:
                pass
    
    if label is None:
        return keys_config

    # If label not found in config, try to construct from env vars
    if label not in keys_config:
        env_key = os.getenv("ALPACA_API_KEY")
        env_secret = os.getenv("ALPACA_API_SECRET")
        
        if env_key and env_secret:
            keys_config[label] = {
                "ALPACA_API_KEY": env_key,
                "ALPACA_API_SECRET": env_secret,
                "API_KEY": env_key,
                "API_SECRET": env_secret,
                "SECRET_KEY": env_secret
            }
        elif not config_loaded:
            # If no config loaded and no env vars, raise the original 500
            raise HTTPException(
                status_code=500, 
                detail=f"Keys configuration not found. Checked env var ALPACA_KEYS and file {config_path}"
            )
        else:
            # Config loaded but label missing and no env vars
            raise HTTPException(status_code=404, detail=f"Label '{label}' not found in configuration")
        
    keys = keys_config[label]
    
    # Overwrite with environment variables if they exist
    env_key = os.getenv("ALPACA_API_KEY")
    env_secret = os.getenv("ALPACA_API_SECRET")
    
    if env_key:
        keys["ALPACA_API_KEY"] = env_key
        if "API_KEY" in keys:
            keys["API_KEY"] = env_key
            
    if env_secret:
        keys["ALPACA_API_SECRET"] = env_secret
        if "API_SECRET" in keys:
            keys["API_SECRET"] = env_secret
        if "SECRET_KEY" in keys:
            keys["SECRET_KEY"] = env_secret
            
    return keys

@app.get("/")
def root():
    return {
        "message": "AI Trading API is running",
        "endpoints": [
            "/execute/buys",
            "/execute/sells",
            "/execute/synthetic_long",
            "/positions/close-old",
            "/recommendations/buys",
            "/recommendations/sells",
            "/orders/buy",
            "/orders/sell",
            "/market/status",
            "/account/summary",
            "/trading/keys/{label}"
        ]
    }
