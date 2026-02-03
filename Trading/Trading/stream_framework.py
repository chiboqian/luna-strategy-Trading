#!/usr/bin/env python3
import os
import sys
import yaml
import asyncio
import logging
import subprocess
import time
import math
import json
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from logging_config import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StreamFramework")

# Check for alpaca-py
try:
    from alpaca.data.live.stock import StockDataStream
    from alpaca.data.live.option import OptionDataStream
    from alpaca.data.models import Quote, Bar
    from alpaca.data.enums import DataFeed
except ImportError as e:
    logger.error(f"Failed to import alpaca-py: {e}")
    logger.error("alpaca-py is required. Please install it: pip install alpaca-py")
    if "pytz" in str(e):
        logger.error("Missing dependency 'pytz'. Please install it: pip install pytz")
    sys.exit(1)

class StreamFramework:
    def __init__(self, config_path: str, log_dir: Optional[str] = None, log_file: Optional[str] = None, data_feed: Optional[str] = None):
        load_dotenv()
        self.config_path = Path(config_path)
        self.config = self._load_config(str(self.config_path))
        self.last_mtime = self._get_config_mtime()
        
        setup_logging(log_dir, log_file, self.config, default_dir='trading_logs/streaming', default_file='stream_framework.log')
        
        self._setup_data_recording()
        
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment variables.")

        # Determine data feed
        # Priority: CLI arg > Config file > Default 'iex'
        data_feed_str = data_feed
        if not data_feed_str:
            data_feed_str = self.config.get('data_feed', 'iex')
        data_feed_str = data_feed_str.lower()
        
        if data_feed_str == 'sip':
            self.data_feed = DataFeed.SIP
        elif data_feed_str == 'otc':
            self.data_feed = DataFeed.OTC
        else:
            self.data_feed = DataFeed.IEX
        logger.info(f"Initializing StockDataStream with feed: {data_feed_str}")

        # Initialize streams immediately so they are ready for dynamic subscriptions
        self.stock_stream = StockDataStream(self.api_key, self.secret_key, feed=self.data_feed)
        self.option_stream = OptionDataStream(self.api_key, self.secret_key)
        
        # Rules storage: symbol -> list of rules
        self.stock_rules = {}
        self.option_rules = {}
        
        # Bar history: symbol -> deque of close prices
        self.bar_history = {}
        history_dir = Path("trading_logs/streaming")
        history_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.history_file = history_dir / f"bar_history_{today_str}.json"
        # Cooldown tracking: rule_name -> last_triggered_timestamp
        self.last_triggered = {}
        # Track rules that are one-off and have been triggered
        self.triggered_rules = set()
        
        self._load_history()
        self._load_daily_data()
        
        self.schedule_active = None
        self._apply_config_rules()
        self._check_schedule()

    def _setup_data_recording(self):
        self.data_dir = Path("trading_logs/market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.stock_quote_file = self.data_dir / "stock_quotes.csv"
        self.option_quote_file = self.data_dir / "option_quotes.csv"
        self.stock_bar_file = self.data_dir / "stock_bars.csv"
        
        # Open files in append mode with line buffering
        self.stock_quote_fh = open(self.stock_quote_file, 'a', buffering=1)
        self.option_quote_fh = open(self.option_quote_file, 'a', buffering=1)
        self.stock_bar_fh = open(self.stock_bar_file, 'a', buffering=1)
        
        # Write headers if files are empty
        if self.stock_quote_file.stat().st_size == 0:
            self.stock_quote_fh.write("timestamp,symbol,bid_price,bid_size,ask_price,ask_size\n")
        if self.option_quote_file.stat().st_size == 0:
            self.option_quote_fh.write("timestamp,symbol,bid_price,bid_size,ask_price,ask_size\n")
        if self.stock_bar_file.stat().st_size == 0:
            self.stock_bar_fh.write("timestamp,symbol,open,high,low,close,volume\n")

    def _record_quote(self, data: Quote, fh):
        try:
            fh.write(f"{data.timestamp},{data.symbol},{data.bid_price},{data.bid_size},{data.ask_price},{data.ask_size}\n")
        except Exception as e:
            logger.error(f"Failed to record quote: {e}")

    def _record_bar(self, data: Bar, fh):
        try:
            fh.write(f"{data.timestamp},{data.symbol},{data.open},{data.high},{data.low},{data.close},{data.volume}\n")
        except Exception as e:
            logger.error(f"Failed to record bar: {e}")

    def _load_history(self):
        file_to_load = self.history_file
        
        if not file_to_load.exists():
            history_dir = self.history_file.parent
            # Try to find the most recent dated history file
            files = list(history_dir.glob("bar_history_*.json"))
            valid_files = [f for f in files if len(f.name) >= 22] # Basic validation
            
            if valid_files:
                valid_files.sort(reverse=True)
                file_to_load = valid_files[0]
                logger.info(f"Today's history not found. Loading most recent: {file_to_load.name}")
            elif (history_dir / "bar_history.json").exists():
                file_to_load = history_dir / "bar_history.json"
                logger.info(f"Loading legacy history file: {file_to_load.name}")

        if file_to_load.exists():
            try:
                with open(file_to_load, 'r') as f:
                    data = json.load(f)
                    for symbol, prices in data.items():
                        # Validate data format (handle migration from list of floats to list of [close, volume])
                        if prices and isinstance(prices[0], (int, float)):
                            # Old format: convert to (price, 0) or discard. Discarding to ensure clean state.
                            logger.warning(f"Old history format detected for {symbol}. Resetting history.")
                            self.bar_history[symbol] = deque(maxlen=300)
                        else:
                            self.bar_history[symbol] = deque(prices, maxlen=300)
                logger.info(f"Loaded bar history for {len(self.bar_history)} symbols from {file_to_load.name}.")
            except Exception as e:
                logger.error(f"Failed to load history: {e}")

    def _load_daily_data(self):
        """Loads existing bar data for the current day from CSV."""
        if not hasattr(self, 'stock_bar_file') or not self.stock_bar_file.exists():
            return

        # Avoid duplicating data if history file was already saved today
        if self.history_file.exists():
            mtime = self.history_file.stat().st_mtime
            mdate = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d")
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            if mdate == today_str:
                logger.info("History file updated today. Skipping CSV load to avoid duplicates.")
                return

        logger.info(f"Checking for existing daily data in {self.stock_bar_file}...")
        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            count = 0
            
            with open(self.stock_bar_file, 'r') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 7: continue
                    
                    # timestamp,symbol,open,high,low,close,volume
                    ts_str, symbol, o, h, l, c, v = parts
                    
                    if today_str in ts_str:
                        if symbol not in self.bar_history:
                            self.bar_history[symbol] = deque(maxlen=300)
                        
                        try:
                            close_val = float(c)
                            vol_val = float(v)
                            self.bar_history[symbol].append((close_val, vol_val))
                            count += 1
                        except ValueError:
                            continue
            if count > 0:
                logger.info(f"Loaded {count} bars from today's session.")
        except Exception as e:
            logger.error(f"Failed to load daily data: {e}")

    def _save_history(self):
        try:
            data = {sym: list(hist) for sym, hist in self.bar_history.items()}
            with open(self.history_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved bar history to {self.history_file}.")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _get_config_mtime(self) -> float:
        """Get the latest modification time for config file or directory."""
        if self.config_path.is_dir():
            mtimes = [f.stat().st_mtime for f in self.config_path.glob("*.yaml")]
            mtimes.append(self.config_path.stat().st_mtime)
            return max(mtimes)
        elif self.config_path.exists():
            return self.config_path.stat().st_mtime
        return 0.0

    def _load_config(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        if p.is_dir():
            combined_rules = []
            for f in sorted(p.glob("*.yaml")):
                try:
                    with open(f, 'r') as file:
                        data = yaml.safe_load(file)
                        if not data: continue
                        
                        if isinstance(data, dict):
                            # Case 1: File contains 'rules' list and potentially other settings
                            if 'rules' in data:
                                if isinstance(data['rules'], list):
                                    combined_rules.extend(data['rules'])
                                # Merge other top-level keys (e.g. logging) into final config?
                                # The current structure returns {'rules': ...}. 
                                # We need to return a merged dict.
                                pass 
                            # Case 2: File is a single rule (heuristic)
                            elif 'symbol' in data and 'asset_class' in data:
                                combined_rules.append(data)
                        elif isinstance(data, list):
                            combined_rules.extend(data)
                except Exception as e:
                    logger.error(f"Error loading config file {f}: {e}")
            
            # Re-scan to merge global settings properly
            final_config = {'rules': combined_rules}
            for f in sorted(p.glob("*.yaml")):
                try:
                    with open(f, 'r') as file:
                        data = yaml.safe_load(file)
                        if isinstance(data, dict) and 'rules' not in data and 'symbol' not in data:
                            # Assume it's a settings file (e.g. logging: ...)
                            final_config.update(data)
                except: pass
            return final_config
        else:
            with open(p, 'r') as f:
                return yaml.safe_load(f)

    def _check_schedule(self) -> bool:
        """Checks if current time is within configured schedule windows."""
        schedule = self.config.get('schedule')
        
        # If no schedule config, default to Always Active
        if not schedule or not schedule.get('enabled', False):
            if self.schedule_active is None:
                logger.info("No schedule configured or enabled. Running 24/7.")
            return True
            
        tz_name = schedule.get('timezone', 'America/New_York')
        try:
            tz = ZoneInfo(tz_name)
        except Exception as e:
            logger.error(f"Invalid timezone {tz_name}: {e}")
            return True
            
        now = datetime.now(tz)
        
        # Default to active unless restricted
        is_active = True
        
        # 1. Check Weekends
        if not schedule.get('run_on_weekends', False):
            if now.weekday() >= 5: # 5=Saturday, 6=Sunday
                is_active = False
        
        # 2. Check Time Windows (only if currently active day)
        if is_active:
            windows = schedule.get('windows', [])
            if windows:
                # If windows are defined, we must be inside one of them
                in_window = False
                current_time_str = now.strftime("%H:%M")
                for window in windows:
                    # Format "09:30-16:00"
                    try:
                        start_str, end_str = window.split('-')
                        if start_str.strip() <= current_time_str <= end_str.strip():
                            in_window = True
                            break
                    except ValueError:
                        logger.warning(f"Invalid window format: {window}")
                
                if not in_window:
                    is_active = False

        # Log state changes
        if is_active != self.schedule_active:
            self.schedule_active = is_active
            status = "ACTIVE" if is_active else "PAUSED (Outside Schedule)"
            logger.info(f"📅 Schedule Update: Framework is now {status} ({now.strftime('%H:%M')} {tz_name})")
            
        return is_active

    def _apply_config_rules(self):
        """Parses config and updates subscriptions."""
        rules = self.config.get('rules', [])
        stock_symbols = set()
        stock_bar_symbols = set()
        option_symbols = set()

        # Reset rules mapping (but keep streams running)
        self.stock_rules = {}
        self.option_rules = {}

        for rule in rules:
            # Default to stock if not specified
            asset_class = rule.get('asset_class', 'stock').lower()
            trigger = rule.get('trigger', 'quote') # 'quote' or 'bar'
            symbol = rule.get('symbol')
            
            if not symbol:
                continue

            if asset_class == 'stock':
                if trigger == 'bar':
                    stock_bar_symbols.add(symbol)
                else:
                    stock_symbols.add(symbol)
                
                if symbol not in self.stock_rules:
                    self.stock_rules[symbol] = []
                self.stock_rules[symbol].append(rule)
            elif asset_class == 'option':
                option_symbols.add(symbol)
                if symbol not in self.option_rules:
                    self.option_rules[symbol] = []
                self.option_rules[symbol].append(rule)

        # Update Subscriptions
        if stock_symbols:
            self.stock_stream.subscribe_quotes(self._handle_stock_quote, *stock_symbols)
            logger.info(f"Subscribed to STOCK quotes: {stock_symbols}")
            
        if stock_bar_symbols:
            self.stock_stream.subscribe_bars(self._handle_stock_bar, *stock_bar_symbols)
            logger.info(f"Subscribed to STOCK bars: {stock_bar_symbols}")

        if option_symbols:
            self.option_stream.subscribe_quotes(self._handle_option_quote, *option_symbols)
            logger.info(f"Subscribed to OPTION quotes: {option_symbols}")

    async def _handle_stock_quote(self, data: Quote):
        if not self._check_schedule():
            return
        self._record_quote(data, self.stock_quote_fh)
        await self._evaluate_rules(data, self.stock_rules.get(data.symbol, []), data_type='quote')

    async def _handle_stock_bar(self, data: Bar):
        if not self._check_schedule():
            return
        self._record_bar(data, self.stock_bar_fh)
        symbol = data.symbol
        if symbol not in self.bar_history:
            self.bar_history[symbol] = deque(maxlen=300) # Keep last 300 bars (enough for SMA 200)
        
        self.bar_history[symbol].append((data.close, data.volume))
        await self._evaluate_rules(data, self.stock_rules.get(symbol, []), data_type='bar')

    async def _handle_option_quote(self, data: Quote):
        if not self._check_schedule():
            return
        self._record_quote(data, self.option_quote_fh)
        await self._evaluate_rules(data, self.option_rules.get(data.symbol, []), data_type='quote')

    async def _evaluate_rules(self, data: Any, rules: List[Dict], data_type: str = 'quote'):
        for rule in rules:
            # Only evaluate rules meant for this data type
            if rule.get('trigger', 'quote') != data_type:
                continue
            
            # Check if rule is one-off and already triggered
            rule_name = rule.get('name', 'unnamed_rule')
            if rule.get('one_off', False) and rule_name in self.triggered_rules:
                continue
            
            # Support multiple conditions (AND logic)
            conditions = rule.get('conditions')
            if conditions is None:
                # Fallback to single condition
                single_cond = rule.get('condition')
                conditions = [single_cond] if single_cond else []
            
            if not conditions:
                continue

            if all(self._check_condition(data, cond, data_type) for cond in conditions):
                await self._trigger_action(rule, data)

    def _get_field_value(self, data: Any, field: str, data_type: str) -> Optional[float]:
        if data_type == 'bar':
            # Handle calculated fields for bars
            if field.startswith('sma_'):
                # Format: sma_20
                try:
                    period = int(field.split('_')[1])
                    return self._calculate_sma(data.symbol, period)
                except (IndexError, ValueError):
                    logger.error(f"Invalid SMA field format: {field}")
                    return None
            elif field.startswith('vwap_'):
                # Format: vwap_20
                try:
                    period = int(field.split('_')[1])
                    return self._calculate_vwap(data.symbol, period)
                except (IndexError, ValueError):
                    return None
            elif field.startswith('stddev_'):
                # Format: stddev_20
                try:
                    period = int(field.split('_')[1])
                    return self._calculate_stddev(data.symbol, period)
                except (IndexError, ValueError):
                    return None
            elif field.startswith('z_score_vwap_'):
                # Format: z_score_vwap_20 (Z-Score of Price vs VWAP)
                try:
                    period = int(field.split('_')[3])
                    return self._calculate_z_score_vwap(data.symbol, period)
                except (IndexError, ValueError):
                    return None
            elif field.startswith('z_score_sma_'):
                # Format: z_score_sma_20 (Z-Score of Price vs SMA)
                try:
                    period = int(field.split('_')[3])
                    return self._calculate_z_score_sma(data.symbol, period)
                except (IndexError, ValueError):
                    return None
            elif field.startswith('rsi_'):
                # Format: rsi_14
                try:
                    period = int(field.split('_')[1])
                    return self._calculate_rsi(data.symbol, period)
                except (IndexError, ValueError):
                    return None
            else:
                return getattr(data, field, None)
        else:
            # Quote data
            return getattr(data, field, None)

    def _check_condition(self, data: Any, condition: Dict, data_type: str) -> bool:
        field = condition.get('field')
        operator = condition.get('operator')
        raw_value = condition.get('value')

        current_value = self._get_field_value(data, field, data_type)
        
        # Resolve target value (can be float or another field)
        target_value = None
        try:
            target_value = float(raw_value)
        except (ValueError, TypeError):
            if isinstance(raw_value, str):
                target_value = self._get_field_value(data, raw_value, data_type)
        
        if current_value is None or target_value is None:
            return False

        if operator == '>':
            return current_value > target_value
        elif operator == '>=':
            return current_value >= target_value
        elif operator == '<':
            return current_value < target_value
        elif operator == '<=':
            return current_value <= target_value
        elif operator == '==':
            return current_value == target_value
        elif operator == 'abs<':
            return abs(current_value) < target_value
        elif operator == 'abs>':
            return abs(current_value) > target_value
        
        return False

    def _calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        history = self.bar_history.get(symbol)
        if not history or len(history) < period:
            return None
        # History items are [close, volume]
        closes = [x[0] for x in list(history)[-period:]]
        return sum(closes) / period

    def _calculate_vwap(self, symbol: str, period: int) -> Optional[float]:
        history = self.bar_history.get(symbol)
        if not history or len(history) < period:
            return None
        subset = list(history)[-period:]
        total_pv = sum(x[0] * x[1] for x in subset)
        total_v = sum(x[1] for x in subset)
        return total_pv / total_v if total_v > 0 else None

    def _calculate_stddev(self, symbol: str, period: int) -> Optional[float]:
        history = self.bar_history.get(symbol)
        if not history or len(history) < period:
            return None
        closes = [x[0] for x in list(history)[-period:]]
        mean = sum(closes) / period
        variance = sum((x - mean) ** 2 for x in closes) / period
        return math.sqrt(variance)

    def _calculate_z_score_vwap(self, symbol: str, period: int) -> Optional[float]:
        # Z = (Price - VWAP) / StdDev
        # Using the most recent close price
        history = self.bar_history.get(symbol)
        if not history or len(history) < period:
            return None
        
        current_price = history[-1][0]
        vwap = self._calculate_vwap(symbol, period)
        stddev = self._calculate_stddev(symbol, period)
        
        if vwap is None or stddev is None or stddev == 0:
            return None
            
        return (current_price - vwap) / stddev

    def _calculate_z_score_sma(self, symbol: str, period: int) -> Optional[float]:
        # Z = (Price - SMA) / StdDev
        # Using the most recent close price
        history = self.bar_history.get(symbol)
        if not history or len(history) < period:
            return None
        
        current_price = history[-1][0]
        sma = self._calculate_sma(symbol, period)
        stddev = self._calculate_stddev(symbol, period)
        
        if sma is None or stddev is None or stddev == 0:
            return None
            
        return (current_price - sma) / stddev

    def _calculate_rsi(self, symbol: str, period: int) -> Optional[float]:
        history = self.bar_history.get(symbol)
        if not history or len(history) < period + 1:
            return None
        
        # Extract closing prices
        closes = [x[0] for x in history]
        
        # Calculate deltas
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        
        if len(deltas) < period:
            return None

        # First 'period' deltas for initial average
        seed_deltas = deltas[:period]
        avg_gain = sum(d for d in seed_deltas if d > 0) / period
        avg_loss = sum(abs(d) for d in seed_deltas if d < 0) / period
        
        # Smooth for the rest of the data to get current RSI
        for d in deltas[period:]:
            gain = d if d > 0 else 0.0
            loss = abs(d) if d < 0 else 0.0
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    async def _trigger_action(self, rule: Dict, data: Any):
        rule_name = rule.get('name', 'unnamed_rule')
        cooldown = rule.get('cooldown', 0)
        
        if rule.get('one_off', False) and rule_name in self.triggered_rules:
            return
            
        now = time.time()

        # Check cooldown
        if rule_name in self.last_triggered:
            if now - self.last_triggered[rule_name] < cooldown:
                return

        # Prepare value string safely
        val_str = ""
        if 'condition' in rule and isinstance(rule['condition'], dict):
             field = rule['condition'].get('field')
             if field:
                 val_str = f" (Value: {getattr(data, field, 'N/A')})"

        print(f"Trigger Name: {rule_name}")
        logger.info(f"⚡ RULE TRIGGERED: {rule_name} for {data.symbol}{val_str}")
        self.last_triggered[rule_name] = now
        
        if rule.get('one_off', False):
            self.triggered_rules.add(rule_name)
            logger.info(f"   Rule '{rule_name}' is set to one-off. Removing from active triggers.")

        action = rule.get('action', {})
        
        # Support both 'command' (single string) and 'commands' (list of strings)
        command_templates = []
        if 'commands' in action and isinstance(action['commands'], list):
            command_templates.extend(action['commands'])
        elif 'command' in action:
            command_templates.append(action['command'])
        
        if command_templates:
            # Prepare context for variable substitution in command
            context = {
                'symbol': data.symbol,
                'ask_price': getattr(data, 'ask_price', 0),
                'bid_price': getattr(data, 'bid_price', 0),
                'close': getattr(data, 'close', 0),
                'timestamp': str(data.timestamp),
                'python': sys.executable  # Allow using the current python interpreter
            }
            
            for tmpl in command_templates:
                try:
                    # Substitute variables like {symbol} or {ask_price}
                    command = tmpl.format(**context)
                    print(f"Command: {command}")
                    logger.info(f"   🚀 Executing: {command}")
                    
                    # Execute non-blocking
                    subprocess.Popen(command, shell=True)
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to execute action for {rule_name}: {e}")

    async def _watch_config_changes(self):
        """Watches for changes in the config file and reloads rules dynamically."""
        while True:
            await asyncio.sleep(5)
            try:
                current_mtime = self._get_config_mtime()
                if current_mtime > self.last_mtime:
                    logger.info("Config file changed. Reloading rules...")
                    self.last_mtime = current_mtime
                    self.config = self._load_config(str(self.config_path))
                    self._apply_config_rules()
            except Exception as e:
                logger.error(f"Error watching config file: {e}")

    async def _safe_stream_runner(self, stream, name):
        """Runs a stream and logs any errors that cause it to exit."""
        try:
            await stream._run_forever()
        except Exception as e:
            logger.error(f"Stream {name} crashed: {e}", exc_info=True)
            raise e

    async def run(self):
        tasks = []
        loop = asyncio.get_running_loop()
        
        # Always run streams (they handle empty subscriptions gracefully)
        self.stock_stream._loop = loop
        tasks.append(self._safe_stream_runner(self.stock_stream, "StockStream"))
        
        self.option_stream._loop = loop
        tasks.append(self._safe_stream_runner(self.option_stream, "OptionStream"))
        
        # Add config watcher
        tasks.append(self._watch_config_changes())
        
        if not tasks:
            logger.warning("No streams configured. Please check your config file.")
            return

        logger.info("Starting streams... Press Ctrl+C to stop.")
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            # Handle subscription errors gracefully
            if "insufficient subscription" in str(e):
                logger.error("❌ CRITICAL ERROR: Insufficient subscription for the requested data feed.")
                logger.error("   You are likely trying to use 'sip' (paid) without a subscription.")
                logger.error("   -> Please switch to 'iex' (free) in your config or CLI arguments.")
                return
            if "connection limit exceeded" in str(e):
                logger.error("❌ CRITICAL ERROR: Connection limit exceeded.")
                logger.error("   Alpaca limits concurrent connections. You may have another script running or a zombie connection.")
                logger.error("   Waiting 45 seconds to allow connections to reset before exiting...")
                await asyncio.sleep(45)
                return
            logger.error(f"Unexpected error in run loop: {e}", exc_info=True)
            raise e
        finally:
            self._save_history()
            self._close_data_recording()
            if self.stock_stream:
                await self.stock_stream.close()
            if self.option_stream:
                await self.option_stream.close()

    def _close_data_recording(self):
        for fh_name in ['stock_quote_fh', 'option_quote_fh', 'stock_bar_fh']:
            fh = getattr(self, fh_name, None)
            if fh:
                fh.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpaca Streaming Framework")
    parser.add_argument("--config", default="Trading/config/streaming_rules.yaml", help="Path to rules config file or directory")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/streaming)")
    parser.add_argument("--log-file", help="Log file name (default: stream_framework.log)")
    parser.add_argument("--feed", help="Data feed for stocks (iex or sip)")
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try finding it relative to project root if run from elsewhere
        project_root = Path(__file__).resolve().parent.parent.parent
        alt_path = project_root / args.config
        if alt_path.exists():
            config_path = alt_path
        else:
            # Fallback to absolute path if provided
            if not config_path.is_absolute():
                 config_path = Path.cwd() / args.config
            
            if not config_path.exists():
                 print(f"Error: Config file not found at {config_path}")
                 sys.exit(1)

    framework = StreamFramework(str(config_path), log_dir=args.log_dir, log_file=args.log_file, data_feed=args.feed)
    
    try:
        asyncio.run(framework.run())
    except KeyboardInterrupt:
        logger.info("Stopping streams...")