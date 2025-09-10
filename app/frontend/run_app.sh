#!/bin/bash

# Test script for the Plan Search Chat applications

echo "=== Plan Search Chat Test Script ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

echo "✅ Python3 is available"

# Check if uv is available  
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✅ uv is available"

# Check if dependencies are installed
echo "📦 Checking dependencies..."

if uv pip list | grep -q "chainlit"; then
    echo "✅ Chainlit is installed"
else
    echo "❌ Chainlit is not installed. Installing dependencies..."
    uv sync
fi

# Check if environment variables are set
echo "🔧 Checking environment variables..."

if [ -f .env ]; then
    echo "✅ .env file found"
    source .env
else
    echo "⚠️  .env file not found. Using default values."
    export API_URL="http://localhost:8000/plan_search"
fi

echo "API_URL: $API_URL"

# Test connection to backend
echo ""
echo "🔗 Testing backend connection..."

if curl -s --connect-timeout 5 "$API_URL" > /dev/null 2>&1; then
    echo "✅ Backend is reachable at $API_URL"
else
    echo "⚠️  Backend is not reachable at $API_URL"
    echo "   Make sure the backend server is running"
fi

# Option to run the application
echo ""
echo "🚀 Ready to run applications!"
echo ""
echo "Choose an option:"
echo "1) Run Chainlit version (Recommended)"
echo "2) Run Gradio version (Original)"
echo "3) Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Starting Chainlit version..."
        ./run_chainlit.sh
        ;;
    2)
        echo "Starting Gradio version..."
        python src/app.py
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
