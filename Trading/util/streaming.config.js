const path = require('path');

module.exports = {
  apps : [{
    name: "streaming",
    script: "./Trading/stream_framework.py",
    args: "--config ./config/streaming_rules/",
    // USE ABSOLUTE PATHS to avoid ambiguity
    interpreter: "/home/paper1/luna-strategy-Trading/Trading/.venv/bin/python", 
    cwd: "/home/paper1/luna-strategy-Trading/Trading", 
    autorestart: true,
    watch: true,
    // Ignore logs to prevent infinite restart loops
    ignore_watch: ["trading_logs", "logs", "*.log", "__pycache__", "*.csv", "data"],
    cron_restart: "0 9 * * *",
    env: {
      PYTHONUNBUFFERED: "1", // Ensure logs appear immediately

    }
  }]
};
