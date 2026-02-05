
module.exports = {
  apps : [{
    name: "streaming",
    script: "./Trading/stream_framework.py --config ./config/streaming_rules/",
    interpreter: "./.venv/bin/python", // Points to your venv
    cwd: "/home/paper1/luna-strategy-Trading/Trading", // Ensure this path is correct
    autorestart: true,
    watch: false,
    cron_restart: "0 9 * * *",
  }]
};