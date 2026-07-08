#!/bin/bash

PORT=8001
PROJECT_DIR="/ematrix_calcsalary/pyapi/devfacerec_devwebapp"
VENV_PATH="/devfacerec_devwebapp/env/bin/activate"
LOG_FILE="$PROJECT_DIR/uvicorn.log"

# Check if port is in use
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "Port $PORT is already running."
else
    echo "Port $PORT not running. Starting server..."

    cd "$PROJECT_DIR" || exit 1

    # Activate virtual environment
    source "$VENV_PATH"

    # Start uvicorn in background
    nohup uvicorn app:app \
        --host 127.0.0.1 \
        --port 8001 \
        --workers 1 \
        --timeout-keep-alive 180 \
        > "$LOG_FILE" 2>&1 &

    echo "Uvicorn started on port $PORT"
fi
# */3 * * * * sh /devfacerec_devwebapp/check_and_start_uvicorn.sh

# uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1 --timeout-keep-alive 180 "/devfacerec_devwebapp/uvicorn.log" 2>&1 &