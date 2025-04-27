#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -m # Enable Job Control (less critical now, but good practice)

echo "Starting Application Container..."

# Trap SIGTERM and SIGINT to allow graceful shutdown
# This helps the shell script exit cleanly; Uvicorn handles its own signals.
trap 'echo "Received stop signal, exiting entrypoint..."; exit 0' SIGTERM SIGINT EXIT

# Start Main Application in the foreground
# Cloud Run sets the PORT environment variable, default to 8080 if not set.
APP_PORT=${PORT:-8080}
echo "Starting Main Bot API on port $APP_PORT..."

# Use exec to replace the shell process; ensures signals (like SIGTERM from Cloud Run)
# are passed directly to the uvicorn process.
# Uvicorn will load the app, triggering the @app.on_event("startup") in main.py
exec uvicorn main:app --host 0.0.0.0 --port $APP_PORT

# The script effectively ends here when uvicorn takes over via exec.