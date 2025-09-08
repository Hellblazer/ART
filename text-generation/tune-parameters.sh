#!/bin/bash

# Parameter Tuning Script for ART Text Generation
# Finds optimal generation parameters

echo "=================================="
echo "ART Text Generation Parameter Tuning"
echo "=================================="

cd /Users/hal.hildebrand/git/ART/text-generation

echo "Choose tuning mode:"
echo "1. Quick Tune (81 combinations, ~5 minutes)"
echo "2. Full Tune (500 combinations, ~30 minutes)"
echo -n "Enter choice (1 or 2): "
read choice

if [ "$choice" = "1" ]; then
    echo "Running quick parameter tuning..."
    mvn compile exec:java -Dexec.mainClass="com.art.textgen.tuning.ParameterTuner" -Dexec.args="--quick"
else
    echo "Running full parameter tuning..."
    mvn compile exec:java -Dexec.mainClass="com.art.textgen.tuning.ParameterTuner"
fi

echo ""
echo "Parameter tuning complete!"
echo "Best parameters saved to: best_parameters.txt"
echo ""
echo "To apply these parameters, update your configuration or use the REST API."
