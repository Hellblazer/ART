#!/bin/bash

# REST API Server for ART Text Generation
# Starts the HTTP server on port 8080

echo "=================================="
echo "ART Text Generation REST API Server"
echo "=================================="

cd /Users/hal.hildebrand/git/ART/text-generation

# Check if port 8080 is already in use
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "ERROR: Port 8080 is already in use!"
    echo "Please stop the existing service or choose a different port."
    exit 1
fi

echo "Starting REST API server on port 8080..."
echo ""
echo "API Endpoints:"
echo "  POST /api/generate      - Generate text"
echo "  POST /api/train         - Train model"
echo "  POST /api/metrics       - Calculate metrics"
echo "  GET  /api/config        - Get configuration"
echo "  POST /api/config        - Update configuration"
echo "  GET  /api/health        - Health check"
echo "  GET  /api/stats         - Usage statistics"
echo "  GET  /api/docs          - API documentation"
echo "  POST /api/model/save    - Save model"
echo "  POST /api/model/load    - Load model"
echo "  POST /api/model/reset   - Reset model"
echo ""
echo "Documentation: http://localhost:8080/api/docs"
echo "Health Check:  http://localhost:8080/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Compile and run the server
mvn compile exec:java -Dexec.mainClass="com.art.textgen.api.RestAPIServer"
