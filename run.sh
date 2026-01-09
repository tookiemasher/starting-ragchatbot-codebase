#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."

# Start Ollama if not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2  # Give Ollama time to start
else
    echo "Ollama is already running"
fi

# Change to backend directory and start the server
cd backend && mamba run uvicorn app:app --reload --port 8000