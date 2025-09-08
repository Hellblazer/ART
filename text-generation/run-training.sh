#!/bin/bash

# Full Training Run with Integrated Pipeline
# This script runs the complete training with all new components

echo "========================================="
echo "ART Text Generation - Full Training Run"
echo "========================================="
echo ""

cd /Users/hal.hildebrand/git/ART/text-generation

# Check current corpus size
echo "Current corpus size:"
du -sh training-corpus/
echo ""

# Build the project
echo "Building project..."
mvn clean compile
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Run the integrated pipeline
echo "Starting integrated training pipeline..."
echo "This will:"
echo "  - Use IncrementalTrainer (no catastrophic forgetting)"
echo "  - Calculate metrics in real-time"
echo "  - Show training dashboard"
echo "  - Save checkpoints automatically"
echo ""

# Create directories for outputs
mkdir -p checkpoints
mkdir -p reports
mkdir -p statistics
mkdir -p experiments

# Run with limited epochs for testing
mvn exec:java -Dexec.mainClass="com.art.textgen.integration.IntegratedPipeline" \
    -Dexec.args="--epochs 2 --batch-size 10"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Check outputs in:"
echo "  - checkpoints/ - Saved model states"
echo "  - reports/ - Training reports"
echo "  - statistics/ - Epoch statistics"
echo ""
