#!/bin/bash

# Databricks Use Case Delivery Agent - Startup Script

echo "🚀 Starting Databricks Use Case Delivery Agent..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

echo "📦 Installing dependencies..."

# Install backend dependencies
echo "Installing backend dependencies..."
cd ui-backend
npm install

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd ../ui-frontend
npm install

echo "✅ Dependencies installed successfully!"

echo ""
echo "🎯 To start the application:"
echo ""
echo "1. Set up your Databricks token:"
echo "   export DATABRICKS_TOKEN=your_token_here"
echo "   (Get your token from: https://adb-984752964297111.11.azuredatabricks.net/settings/access-tokens)"
echo ""
echo "2. Start the backend server:"
echo "   cd ui-backend && npm start"
echo ""
echo "3. In a new terminal, start the frontend:"
echo "   cd ui-frontend && npm start"
echo ""
echo "4. Open your browser to http://localhost:3000"
echo ""
echo "🔧 Backend will run on http://localhost:5000"
echo "🎨 Frontend will run on http://localhost:3000"
echo "🤖 Chat will connect to your Databricks agent!"
echo ""
echo "Happy coding! 🎉"
