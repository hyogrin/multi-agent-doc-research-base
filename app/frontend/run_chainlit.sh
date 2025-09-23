#!/bin/bash

# Doc ResearchChat - Chainlit Version
# Run the Chainlit application

echo "Starting Doc ResearchChat (Chainlit Version)..."

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | xargs)
fi

# Set default values if not provided
export API_URL=${API_URL:-"http://localhost:8000/plan_search"}

echo "API_URL: $API_URL"

# Run the Chainlit application
chainlit run src/app_chainlit.py --host 127.0.0.1 --port 7860
