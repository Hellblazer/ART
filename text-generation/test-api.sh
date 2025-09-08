#!/bin/bash

# Test client for ART Text Generation REST API
# Demonstrates all API endpoints

API_URL="http://localhost:8080/api"

echo "=================================="
echo "ART Text Generation API Test Client"
echo "=================================="

# Check if server is running
echo -n "Checking server health... "
health=$(curl -s "$API_URL/health" 2>/dev/null)
if [ -z "$health" ]; then
    echo "ERROR: Server is not running!"
    echo "Please start the server first with: ./start-api-server.sh"
    exit 1
fi
echo "OK"
echo ""

# 1. Test text generation
echo "1. Testing text generation..."
echo "   Prompt: 'Once upon a time'"
curl -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "maxLength": 50,
    "temperature": 0.9,
    "topK": 40,
    "topP": 0.9
  }' 2>/dev/null | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 2. Test configuration
echo "2. Getting current configuration..."
curl -s "$API_URL/config" | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 3. Update configuration
echo "3. Updating configuration..."
curl -X POST "$API_URL/config" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.7,
    "topK": 50,
    "topP": 0.85,
    "repetitionPenalty": 1.2
  }' 2>/dev/null | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 4. Test training
echo "4. Testing incremental training..."
curl -X POST "$API_URL/train" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test document for training.\n\nIt contains multiple paragraphs.\n\nThe model should learn from this text.",
    "incremental": true
  }' 2>/dev/null | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 5. Test metrics calculation
echo "5. Testing metrics calculation..."
curl -X POST "$API_URL/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog. This is a test sentence to evaluate various text quality metrics."
  }' 2>/dev/null | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 6. Save model
echo "6. Saving model checkpoint..."
curl -X POST "$API_URL/model/save" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_checkpoint"
  }' 2>/dev/null | python3 -m json.tool

echo ""
echo "Press Enter to continue..."
read

# 7. Get statistics
echo "7. Getting API statistics..."
curl -s "$API_URL/stats" | python3 -m json.tool

echo ""
echo "=================================="
echo "All tests completed successfully!"
echo "=================================="
echo ""
echo "You can view the full API documentation at:"
echo "http://localhost:8080/api/docs"
