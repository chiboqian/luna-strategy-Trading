pm2 install pm2-logrotate
# 1. Limit file size to 100MB (Rotates when it hits this size)
pm2 set pm2-logrotate:max_size 100M

# 2. Keep only the last 10 log files (Deletes older ones)
pm2 set pm2-logrotate:retain 10

# 3. Compress rotated logs (Saves huge amount of space, turns text to .gz)
pm2 set pm2-logrotate:compress true

pm2 restart all
pm2 conf pm2-logrotate


#pm2 flush to clear logs