#!/bin/bash

# Load environment variables from .env if it exists
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# Check if CLOUDFLARE_API_TOKEN is set
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
  echo "Error: CLOUDFLARE_API_TOKEN is not set."
  echo "Please create a .env file with CLOUDFLARE_API_TOKEN=your_token"
  exit 1
fi

# Build the image (using pipe to avoid snap fs issues)
echo "Building Caddy image..."
cat Dockerfile | docker build -t caddy-cloudflare -

# Stop existing container if running
docker stop caddy 2>/dev/null || true
docker rm caddy 2>/dev/null || true

# Run the container
echo "Starting Caddy..."
docker run -d \
  --name caddy \
  --restart unless-stopped \
  -p 80:80 \
  -p 443:443 \
  -e CLOUDFLARE_API_TOKEN="$CLOUDFLARE_API_TOKEN" \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/Caddyfile:/etc/caddy/Caddyfile" \
  -v caddy_data:/data \
  caddy-cloudflare

echo "Caddy is running."
docker logs caddy
