
module.exports = {
  apps : [{
    name: "streaming",
    script: "./Trading/stream_framework.py",
    args: "--config ./config/streaming_rules/",
    interpreter: "./.venv/bin/python", // Points to your venv
    cwd: "~/paper1/luna-strategy-Trading/Trading", // Ensure this path is correct
    autorestart: true,
    watch: true,
    ignore_watch: ["trading_logs", "logs", "*.log", "__pycache__", "*.csv"],
    cron_restart: "0 9 * * *",
  }]
};