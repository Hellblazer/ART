#!/bin/bash

# Enhanced Corpus Download and Training Script
# This script downloads a comprehensive corpus and trains the ART text generation model

echo "========================================="
echo "ART Text Generation - Corpus Expansion"
echo "========================================="
echo ""

cd /Users/hal.hildebrand/git/ART/text-generation

# Step 1: Build the project with new dependencies
echo "Step 1: Building project with enhanced downloader..."
mvn clean compile
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Please fix compilation errors."
    exit 1
fi
echo "✅ Build successful"
echo ""

# Step 2: Download comprehensive corpus
echo "Step 2: Downloading comprehensive corpus..."
echo "This will download:"
echo "  - Project Gutenberg books"
echo "  - Wikipedia articles"
echo "  - ArXiv abstracts"
echo "  - Generated training data"
echo ""
echo "⚠️  This may take 10-20 minutes depending on your connection"
echo ""

mvn exec:java -Dexec.mainClass="com.art.textgen.training.EnhancedCorpusDownloader"
if [ $? -ne 0 ]; then
    echo "⚠️  Corpus download had some errors, but continuing..."
fi
echo ""

# Step 3: Check corpus statistics
echo "Step 3: Checking corpus statistics..."
if [ -f "training-corpus/CORPUS_REPORT.md" ]; then
    echo "Corpus report generated:"
    head -20 training-corpus/CORPUS_REPORT.md
else
    echo "⚠️  No corpus report found"
fi
echo ""

# Step 4: Train the model
echo "Step 4: Training the model on new corpus..."
echo "Choose training mode:"
echo "  1. Quick test (sample corpus only)"
echo "  2. Full training (downloaded corpus)"
echo "  3. Skip training"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Running quick test with sample corpus..."
        java -cp target/classes com.art.textgen.TestRun
        ;;
    2)
        echo "Running full training on downloaded corpus..."
        mvn exec:java -Dexec.mainClass="com.art.textgen.ARTTextGenerationApp"
        ;;
    3)
        echo "Skipping training."
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

echo ""
echo "========================================="
echo "Process Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review corpus in training-corpus/"
echo "2. Run the main application to test generation"
echo "3. Tune parameters for better quality"
echo ""
echo "To run the application:"
echo "  mvn exec:java -Dexec.mainClass=\"com.art.textgen.ARTTextGenerationApp\""
echo ""
