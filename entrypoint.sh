#!/bin/bash
set -e

# Export API key from secret file if it exists
if [ -f "/run/secrets/api_key" ]; then
    export X_API_KEY=$(cat /run/secrets/api_key)
fi

# Execute the main command
exec "$@"