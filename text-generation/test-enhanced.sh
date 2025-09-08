#!/bin/bash

# Test enhanced generation with improvements

echo "==================================="
echo "Testing Enhanced Text Generation"
echo "==================================="
echo ""

cd /Users/hal.hildebrand/git/ART/text-generation

# Build
echo "Building project..."
mvn clean compile test-compile
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "Running enhanced generation test..."
mvn test -Dtest=TestEnhancedGeneration

echo ""
echo "==================================="
echo "Test complete!"
echo "==================================="
echo ""
echo "To see the improvements:"
echo "1. Notice reduced repetition in generated text"
echo "2. Compare diversity scores (should be > 0.7)"
echo "3. Check perplexity (lower is better)"
echo "4. Observe different generation modes"
echo ""
