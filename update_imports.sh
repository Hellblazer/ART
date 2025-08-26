#!/bin/bash

# Script to update all import statements after package reorganization
cd /Users/hal.hildebrand/git/ART

echo "Updating algorithm imports..."
# Algorithm classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.FuzzyART;/import com.hellblazer.art.core.algorithms.FuzzyART;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.HypersphereART;/import com.hellblazer.art.core.algorithms.HypersphereART;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.GaussianART;/import com.hellblazer.art.core.algorithms.GaussianART;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.EllipsoidART;/import com.hellblazer.art.core.algorithms.EllipsoidART;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.BayesianART;/import com.hellblazer.art.core.algorithms.BayesianART;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ART2;/import com.hellblazer.art.core.algorithms.ART2;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTA;/import com.hellblazer.art.core.algorithms.ARTA;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTE;/import com.hellblazer.art.core.algorithms.ARTE;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTSTAR;/import com.hellblazer.art.core.algorithms.ARTSTAR;/g' {} \;

echo "Updating parameter imports..."
# Parameter classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.FuzzyParameters;/import com.hellblazer.art.core.parameters.FuzzyParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.HypersphereParameters;/import com.hellblazer.art.core.parameters.HypersphereParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.GaussianParameters;/import com.hellblazer.art.core.parameters.GaussianParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.EllipsoidParameters;/import com.hellblazer.art.core.parameters.EllipsoidParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.BayesianParameters;/import com.hellblazer.art.core.parameters.BayesianParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ART2Parameters;/import com.hellblazer.art.core.parameters.ART2Parameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTAParameters;/import com.hellblazer.art.core.parameters.ARTAParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTEParameters;/import com.hellblazer.art.core.parameters.ARTEParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTSTARParameters;/import com.hellblazer.art.core.parameters.ARTSTARParameters;/g' {} \;

echo "Updating weight imports..."
# Weight classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.FuzzyWeight;/import com.hellblazer.art.core.weights.FuzzyWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.HypersphereWeight;/import com.hellblazer.art.core.weights.HypersphereWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.GaussianWeight;/import com.hellblazer.art.core.weights.GaussianWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.EllipsoidWeight;/import com.hellblazer.art.core.weights.EllipsoidWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.BayesianWeight;/import com.hellblazer.art.core.weights.BayesianWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ART2Weight;/import com.hellblazer.art.core.weights.ART2Weight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTAWeight;/import com.hellblazer.art.core.weights.ARTAWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTEWeight;/import com.hellblazer.art.core.weights.ARTEWeight;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTSTARWeight;/import com.hellblazer.art.core.weights.ARTSTARWeight;/g' {} \;

echo "Updating ARTMAP imports..."
# ARTMAP classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTMAP;/import com.hellblazer.art.core.artmap.ARTMAP;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.SimpleARTMAP;/import com.hellblazer.art.core.artmap.SimpleARTMAP;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.DeepARTMAP;/import com.hellblazer.art.core.artmap.DeepARTMAP;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTMAPParameters;/import com.hellblazer.art.core.artmap.ARTMAPParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.DeepARTMAPParameters;/import com.hellblazer.art.core.artmap.DeepARTMAPParameters;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ARTMAPResult;/import com.hellblazer.art.core.artmap.ARTMAPResult;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.DeepARTMAPResult;/import com.hellblazer.art.core.artmap.DeepARTMAPResult;/g' {} \;

echo "Updating result imports..."
# Result classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.ActivationResult;/import com.hellblazer.art.core.results.ActivationResult;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.BayesianActivationResult;/import com.hellblazer.art.core.results.BayesianActivationResult;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.EllipsoidActivationResult;/import com.hellblazer.art.core.results.EllipsoidActivationResult;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.CategoryResult;/import com.hellblazer.art.core.results.CategoryResult;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.MatchResult;/import com.hellblazer.art.core.results.MatchResult;/g' {} \;

echo "Updating utility imports..."
# Utility classes
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.Matrix;/import com.hellblazer.art.core.utils.Matrix;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.SimpleVector;/import com.hellblazer.art.core.utils.SimpleVector;/g' {} \;
find . -name "*.java" -exec sed -i '' 's/import com\.hellblazer\.art\.core\.DataBounds;/import com.hellblazer.art.core.utils.DataBounds;/g' {} \;

echo "Import updates completed!"