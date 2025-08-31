/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of the Java ART Library project.
 * 
 * The Java ART Library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The Java ART Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with the Java ART Library. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core.hierarchical;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.algorithms.*;
import com.hellblazer.art.core.parameters.*;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.*;

/**
 * SMART (Self-Monitoring Adaptive Resonance Theory) for Hierarchical Clustering.
 * 
 * This implementation follows the algorithm described in:
 * Bartfai, G. (1994). Hierarchical clustering with ART neural networks.
 * In Proc. IEEE International Conference on Neural Networks (ICNN)
 * (pp. 940–944). volume 2. doi:10.1109/ICNN.1994.374307.
 * 
 * SMART hierarchically clusters data in a divisive fashion by using a set of
 * vigilance values that monotonically increase in their restrictiveness.
 * All layers receive the same input data but with different vigilance thresholds.
 * 
 * @author Hal Hildebrand
 */
public class SMART {
    
    /**
     * Types of ART modules supported by SMART.
     */
    public enum ARTType {
        FUZZY_ART,
        BAYESIAN_ART,
        GAUSSIAN_ART,
        HYPERSPHERE_ART,
        TOPO_ART
    }
    
    private final List<BaseART> modules;
    private final double[] vigilanceValues;
    private final ARTType artType;
    private final Object baseParameters;
    // Improved hierarchical mapping with proper key structure
    private final Map<HierarchicalKey, int[]> hierarchicalMappings;
    private final List<int[]> categoryPaths;
    
    /**
     * Create a new SMART instance with FuzzyART modules.
     * 
     * @param vigilanceValues Array of vigilance values (must be monotonically increasing)
     * @param alpha Choice parameter for FuzzyART
     * @param beta Learning rate for FuzzyART
     */
    public static SMART createWithFuzzyART(double[] vigilanceValues, double alpha, double beta) {
        validateVigilanceValues(vigilanceValues, false);
        var modules = new ArrayList<BaseART>();
        
        for (double vigilance : vigilanceValues) {
            var params = new FuzzyParameters(vigilance, alpha, beta);
            modules.add(new FuzzyART());
        }
        
        return new SMART(modules, vigilanceValues, ARTType.FUZZY_ART, 
                        new FuzzyParameters(vigilanceValues[0], alpha, beta));
    }
    
    /**
     * Create a new SMART instance with BayesianART modules.
     * 
     * @param vigilanceValues Array of vigilance values (must be monotonically decreasing for Bayesian)
     * @param inputDimension The dimensionality of input patterns
     * @param noiseVariance The noise variance parameter
     * @param priorPrecision The prior precision parameter
     * @param maxCategories Maximum number of categories per layer
     */
    public static SMART createWithBayesianART(double[] vigilanceValues, int inputDimension, 
                                              double noiseVariance, double priorPrecision, 
                                              int maxCategories) {
        validateVigilanceValues(vigilanceValues, true);
        var modules = new ArrayList<BaseART>();
        
        // Create proper BayesianART parameters
        var priorMean = new double[inputDimension];
        var priorCov = com.hellblazer.art.core.utils.Matrix.eye(inputDimension);
        
        for (double vigilance : vigilanceValues) {
            var params = new BayesianParameters(vigilance, priorMean, priorCov, 0.1, 1.0, 100);
            modules.add(new BayesianART(params));
        }
        
        return new SMART(modules, vigilanceValues, ARTType.BAYESIAN_ART,
                        new BayesianParameters(vigilanceValues[0], priorMean, priorCov, 0.1, 1.0, 100));
    }
    
    /**
     * Create a new SMART instance with GaussianART modules.
     * 
     * @param vigilanceValues Array of vigilance values (must be monotonically increasing)
     * @param inputDimension The dimensionality of input patterns
     * @param sigma Standard deviation parameter for GaussianART (applied to all dimensions)
     */
    public static SMART createWithGaussianART(double[] vigilanceValues, int inputDimension, double sigma) {
        validateVigilanceValues(vigilanceValues, false);
        var modules = new ArrayList<BaseART>();
        
        // Create sigma array with same value for all dimensions
        var sigmaArray = new double[inputDimension];
        java.util.Arrays.fill(sigmaArray, sigma);
        
        for (double vigilance : vigilanceValues) {
            var params = new GaussianParameters(vigilance, sigmaArray);
            modules.add(new GaussianART());
        }
        
        return new SMART(modules, vigilanceValues, ARTType.GAUSSIAN_ART,
                        new GaussianParameters(vigilanceValues[0], sigmaArray));
    }
    
    /**
     * Private constructor for SMART.
     */
    private SMART(List<BaseART> modules, double[] vigilanceValues, 
                  ARTType artType, Object baseParameters) {
        this.modules = modules;
        this.vigilanceValues = Arrays.copyOf(vigilanceValues, vigilanceValues.length);
        this.artType = artType;
        this.baseParameters = baseParameters;
        this.hierarchicalMappings = new HashMap<>();
        this.categoryPaths = new ArrayList<>();
    }
    
    /**
     * Fit the SMART model to the data.
     * 
     * @param patterns The dataset to fit the model on
     * @param maxIterations The number of iterations to run
     * @return Cluster labels for each pattern
     */
    public int[] fit(List<Pattern> patterns, int maxIterations) {
        var labels = new int[patterns.size()];
        
        for (int iter = 0; iter < maxIterations; iter++) {
            for (int i = 0; i < patterns.size(); i++) {
                labels[i] = stepFit(patterns.get(i));
            }
        }
        
        return labels;
    }
    
    /**
     * Partial fit on a single pattern.
     * 
     * @param pattern The pattern to learn
     * @return The cluster label at the highest level
     */
    public int stepFit(Pattern pattern) {
        var categoryPath = new int[modules.size()];
        
        // Process through each layer with increasing vigilance
        for (int i = 0; i < modules.size(); i++) {
            var module = modules.get(i);
            var params = createParametersForLevel(i);
            
            // Learn at this level
            var result = module.stepFit(pattern, params);
            
            // Extract category from result
            int category = extractCategory(result);
            categoryPath[i] = category;
            
            // Track hierarchical mappings with proper key
            if (i > 0) {
                var key = new HierarchicalKey(i, category);
                final int previousCategory = categoryPath[i-1];
                hierarchicalMappings.computeIfAbsent(key, k -> new int[]{previousCategory});
            }
        }
        
        // Store the complete path
        categoryPaths.add(categoryPath);
        
        // Return the top-level (most specific) category
        return categoryPath[modules.size() - 1];
    }
    
    /**
     * Predict cluster labels for patterns.
     * 
     * @param patterns The patterns to predict
     * @return Cluster labels at the highest level
     */
    public int[] predict(List<Pattern> patterns) {
        return patterns.stream()
            .mapToInt(this::predictSingle)
            .toArray();
    }
    
    /**
     * Predict a single pattern.
     */
    private int predictSingle(Pattern pattern) {
        var categoryPath = new int[modules.size()];
        
        // Predict through each layer
        for (int i = 0; i < modules.size(); i++) {
            var module = modules.get(i);
            var params = createParametersForLevel(i);
            
            // Get best matching category without learning
            var weights = module.getCategories();
            if (weights.isEmpty()) {
                categoryPath[i] = -1;
                continue;
            }
            
            // Find best match using proper activation from module
            double bestActivation = -1;
            int bestCategory = -1;
            
            for (int j = 0; j < weights.size(); j++) {
                double activation = module.getActivationValue(pattern, j, params);
                if (activation > bestActivation) {
                    bestActivation = activation;
                    bestCategory = j;
                }
            }
            
            categoryPath[i] = bestCategory;
        }
        
        return categoryPath[modules.size() - 1];
    }
    
    /**
     * Get the number of clusters at each level.
     */
    public int[] getClusterCounts() {
        return modules.stream()
            .mapToInt(m -> m.getCategoryCount())
            .toArray();
    }
    
    /**
     * Get the vigilance values.
     */
    public double[] getVigilanceValues() {
        return Arrays.copyOf(vigilanceValues, vigilanceValues.length);
    }
    
    /**
     * Get the number of hierarchical levels.
     */
    public int getNumLevels() {
        return modules.size();
    }
    
    /**
     * Get the ART type used in this SMART instance.
     */
    public ARTType getArtType() {
        return artType;
    }
    
    /**
     * Get hierarchical path from root to a specific category.
     * 
     * @param patternIndex Index of the pattern
     * @return Path of categories from each layer, or null if pattern not found
     */
    public int[] getHierarchicalPath(int patternIndex) {
        if (patternIndex < 0 || patternIndex >= categoryPaths.size()) {
            return null;
        }
        return Arrays.copyOf(categoryPaths.get(patternIndex), modules.size());
    }
    
    /**
     * Get statistics about the hierarchical structure.
     */
    public HierarchicalStats getStatistics() {
        var stats = new HierarchicalStats();
        stats.numLevels = modules.size();
        stats.clusterCounts = getClusterCounts();
        stats.vigilanceValues = Arrays.copyOf(vigilanceValues, vigilanceValues.length);
        
        // Calculate average branching factor
        double totalBranching = 0;
        int validLevels = 0;
        for (int i = 1; i < modules.size(); i++) {
            if (stats.clusterCounts[i - 1] > 0) {
                totalBranching += (double) stats.clusterCounts[i] / stats.clusterCounts[i - 1];
                validLevels++;
            }
        }
        stats.avgBranchingFactor = validLevels > 0 ? totalBranching / validLevels : 0;
        
        return stats;
    }
    
    /**
     * Create parameters for a specific level.
     */
    private Object createParametersForLevel(int level) {
        double vigilance = vigilanceValues[level];
        
        return switch (artType) {
            case FUZZY_ART -> {
                var base = (FuzzyParameters) baseParameters;
                yield new FuzzyParameters(vigilance, base.alpha(), base.beta());
            }
            case BAYESIAN_ART -> {
                var base = (BayesianParameters) baseParameters;
                yield new BayesianParameters(vigilance, base.priorMean(), base.priorCovariance(), 
                                            base.noiseVariance(), base.priorPrecision(), base.maxCategories());
            }
            case GAUSSIAN_ART -> {
                var base = (GaussianParameters) baseParameters;
                yield new GaussianParameters(vigilance, base.sigmaInit());
            }
            case HYPERSPHERE_ART -> {
                var base = (HypersphereParameters) baseParameters;
                yield new HypersphereParameters(vigilance, base.defaultRadius(), base.adaptiveRadius());
            }
            case TOPO_ART -> {
                var base = (TopoARTParameters) baseParameters;
                yield new TopoARTParameters(base.inputDimension(), vigilance, 
                                           base.learningRateSecond(), base.phi(), base.tau(), base.alpha());
            }
        };
    }
    
    /**
     * Extract category index from activation result.
     */
    private int extractCategory(ActivationResult result) {
        if (result instanceof ActivationResult.Success success) {
            return success.categoryIndex();
        } else if (result instanceof ActivationResult.NoMatch) {
            // New category was created - return the latest index
            return modules.get(0).getCategoryCount() - 1;
        }
        return -1;
    }
    
    /**
     * Validate that vigilance values are monotonic in the correct direction.
     */
    private static void validateVigilanceValues(double[] values, boolean shouldDecrease) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Vigilance values cannot be null or empty");
        }
        
        for (int i = 1; i < values.length; i++) {
            if (shouldDecrease) {
                if (values[i] >= values[i - 1]) {
                    throw new IllegalArgumentException(
                        "Vigilance values must be monotonically decreasing for Bayesian-type ART"
                    );
                }
            } else {
                if (values[i] <= values[i - 1]) {
                    throw new IllegalArgumentException(
                        "Vigilance values must be monotonically increasing"
                    );
                }
            }
        }
    }
    
    /**
     * Key for hierarchical mappings between levels.
     */
    private record HierarchicalKey(int level, int category) {
        public HierarchicalKey {
            if (level < 0) {
                throw new IllegalArgumentException("Level must be non-negative");
            }
            if (category < 0) {
                throw new IllegalArgumentException("Category must be non-negative");
            }
        }
    }
    
    /**
     * Statistics about the hierarchical structure.
     */
    public static class HierarchicalStats {
        public int numLevels;
        public int[] clusterCounts;
        public double[] vigilanceValues;
        public double avgBranchingFactor;
        
        @Override
        public String toString() {
            return String.format(
                "SMART Hierarchical Stats:\n" +
                "  Levels: %d\n" +
                "  Cluster counts: %s\n" +
                "  Vigilance values: %s\n" +
                "  Avg branching factor: %.2f",
                numLevels,
                Arrays.toString(clusterCounts),
                Arrays.toString(vigilanceValues),
                avgBranchingFactor
            );
        }
    }
}