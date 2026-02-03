
module.exports = {
  apps : [{
    name: "alpaca-stream-framework",
    script: "./Trading/stream_framework.py",
    interpreter: "./.venv/bin/python", // Points to your venv
    cwd: "/home/paper1/luna-strategy-Trading/Trading", // Ensure this path is correct
    autorestart: true,
    watch: true,
    cron_restart: "0 9 * * *",
  }]
};