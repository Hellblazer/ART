/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance.algorithms;

import java.util.Arrays;

/**
 * Parameters for VectorizedARTE (Enhanced ART) algorithm.
 * 
 * ART-E introduces several enhancements over standard ART with adaptive learning:
 * - Adaptive learning rates that adjust based on pattern familiarity
 * - Feature importance weighting for better discrimination with SIMD optimization
 * - Dynamic topology adjustment capabilities
 * - Performance optimization through selective learning
 * - Enhanced vigilance control with context sensitivity
 * - Vectorized operations for improved performance
 */
public class VectorizedARTEParameters {
    
    private final double vigilance;
    private final double alpha;
    private final double baseLearningRate;
    private final double adaptiveLearningFactor;
    private final boolean featureWeightingEnabled;
    private final double[] featureWeights;
    private final double topologyAdjustmentRate;
    private final double performanceThreshold;
    private final double contextSensitivity;
    private final double maxLearningRate;
    private final double minLearningRate;
    private final int performanceWindowSize;
    private final double convergenceThreshold;
    private final VectorizedParameters baseParameters;
    private final boolean enablePerformanceOptimization;
    private final double familiarityDecayRate;
    private final int inputDimension;
    
    public VectorizedARTEParameters(
            double vigilance,
            double alpha,
            double baseLearningRate,
            double adaptiveLearningFactor,
            boolean featureWeightingEnabled,
            double[] featureWeights,
            double topologyAdjustmentRate,
            double performanceThreshold,
            double contextSensitivity,
            double maxLearningRate,
            double minLearningRate,
            int performanceWindowSize,
            double convergenceThreshold,
            VectorizedParameters baseParameters,
            boolean enablePerformanceOptimization,
            double familiarityDecayRate,
            int inputDimension) {
        
        validateInputs(vigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                       featureWeights, topologyAdjustmentRate, performanceThreshold,
                       contextSensitivity, maxLearningRate, minLearningRate,
                       performanceWindowSize, convergenceThreshold, familiarityDecayRate,
                       inputDimension);
        
        this.vigilance = vigilance;
        this.alpha = alpha;
        this.baseLearningRate = baseLearningRate;
        this.adaptiveLearningFactor = adaptiveLearningFactor;
        this.featureWeightingEnabled = featureWeightingEnabled;
        this.featureWeights = featureWeights != null ? Arrays.copyOf(featureWeights, featureWeights.length) : new double[0];
        this.topologyAdjustmentRate = topologyAdjustmentRate;
        this.performanceThreshold = performanceThreshold;
        this.contextSensitivity = contextSensitivity;
        this.maxLearningRate = maxLearningRate;
        this.minLearningRate = minLearningRate;
        this.performanceWindowSize = performanceWindowSize;
        this.convergenceThreshold = convergenceThreshold;
        this.baseParameters = baseParameters;
        this.enablePerformanceOptimization = enablePerformanceOptimization;
        this.familiarityDecayRate = familiarityDecayRate;
        this.inputDimension = inputDimension;
    }
    
    public static VectorizedARTEParameters defaults() {
        return forDimension(100); // Default 100-dimensional input
    }
    
    public static VectorizedARTEParameters forDimension(int inputDimension) {
        var defaultBaseParams = VectorizedParameters.createDefault();
        
        // Initialize uniform feature weights
        var defaultFeatureWeights = new double[inputDimension];
        double uniformWeight = 1.0 / inputDimension;
        Arrays.fill(defaultFeatureWeights, uniformWeight);
        
        return new VectorizedARTEParameters(
            0.75,    // vigilance
            0.001,   // alpha
            0.8,     // baseLearningRate
            0.5,     // adaptiveLearningFactor
            true,    // featureWeightingEnabled
            defaultFeatureWeights,
            0.1,     // topologyAdjustmentRate
            0.6,     // performanceThreshold
            0.3,     // contextSensitivity
            1.0,     // maxLearningRate
            0.01,    // minLearningRate
            10,      // performanceWindowSize
            0.001,   // convergenceThreshold
            defaultBaseParams,
            true,    // enablePerformanceOptimization
            0.01,    // familiarityDecayRate
            inputDimension
        );
    }
    
    private static void validateInputs(double vigilance, double alpha, double baseLearningRate,
                                       double adaptiveLearningFactor, double[] featureWeights,
                                       double topologyAdjustmentRate, double performanceThreshold,
                                       double contextSensitivity, double maxLearningRate,
                                       double minLearningRate, int performanceWindowSize,
                                       double convergenceThreshold, double familiarityDecayRate,
                                       int inputDimension) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        
        if (baseLearningRate < 0.0 || baseLearningRate > 1.0) {
            throw new IllegalArgumentException("Base learning rate must be in [0, 1], got: " + baseLearningRate);
        }
        
        if (adaptiveLearningFactor < 0.0 || adaptiveLearningFactor > 1.0) {
            throw new IllegalArgumentException("Adaptive learning factor must be in [0, 1], got: " + adaptiveLearningFactor);
        }
        
        if (topologyAdjustmentRate < 0.0 || topologyAdjustmentRate > 1.0) {
            throw new IllegalArgumentException("Topology adjustment rate must be in [0, 1], got: " + topologyAdjustmentRate);
        }
        
        if (performanceThreshold < 0.0 || performanceThreshold > 1.0) {
            throw new IllegalArgumentException("Performance threshold must be in [0, 1], got: " + performanceThreshold);
        }
        
        if (contextSensitivity < 0.0 || contextSensitivity > 1.0) {
            throw new IllegalArgumentException("Context sensitivity must be in [0, 1], got: " + contextSensitivity);
        }
        
        if (maxLearningRate < minLearningRate) {
            throw new IllegalArgumentException("Max learning rate must be >= min learning rate, got max=" + maxLearningRate + ", min=" + minLearningRate);
        }
        
        if (performanceWindowSize < 1) {
            throw new IllegalArgumentException("Performance window size must be >= 1, got: " + performanceWindowSize);
        }
        
        if (convergenceThreshold < 0.0) {
            throw new IllegalArgumentException("Convergence threshold must be non-negative, got: " + convergenceThreshold);
        }
        
        if (familiarityDecayRate < 0.0 || familiarityDecayRate > 1.0) {
            throw new IllegalArgumentException("Familiarity decay rate must be in [0, 1], got: " + familiarityDecayRate);
        }
        
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive, got: " + inputDimension);
        }
        
        // Validate feature weights if provided
        if (featureWeights != null) {
            for (int i = 0; i < featureWeights.length; i++) {
                if (featureWeights[i] < 0.0) {
                    throw new IllegalArgumentException("Feature weight " + i + " must be non-negative, got: " + featureWeights[i]);
                }
            }
        }
    }
    
    // Getters
    
    public double vigilanceThreshold() {
        return vigilance;
    }
    
    public double getVigilance() {
        return vigilance;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getBaseLearningRate() {
        return baseLearningRate;
    }
    
    public double getAdaptiveLearningFactor() {
        return adaptiveLearningFactor;
    }
    
    public boolean isFeatureWeightingEnabled() {
        return featureWeightingEnabled;
    }
    
    public double[] getFeatureWeights() {
        return Arrays.copyOf(featureWeights, featureWeights.length);
    }
    
    public double getTopologyAdjustmentRate() {
        return topologyAdjustmentRate;
    }
    
    public double getPerformanceThreshold() {
        return performanceThreshold;
    }
    
    public double getContextSensitivity() {
        return contextSensitivity;
    }
    
    public double getMaxLearningRate() {
        return maxLearningRate;
    }
    
    public double getMinLearningRate() {
        return minLearningRate;
    }
    
    public int getPerformanceWindowSize() {
        return performanceWindowSize;
    }
    
    public double getConvergenceThreshold() {
        return convergenceThreshold;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    public boolean isPerformanceOptimizationEnabled() {
        return enablePerformanceOptimization;
    }
    
    public double getFamiliarityDecayRate() {
        return familiarityDecayRate;
    }
    
    public int getInputDimension() {
        return inputDimension;
    }
    
    /**
     * Calculate adaptive learning rate based on familiarity.
     */
    public double calculateAdaptiveLearningRate(double familiarity) {
        double adaptedRate = baseLearningRate * (1.0 - familiarity * adaptiveLearningFactor);
        return Math.max(minLearningRate, Math.min(maxLearningRate, adaptedRate));
    }
    
    /**
     * Calculate context-adjusted vigilance.
     */
    public double calculateContextVigilance(double contextValue) {
        return vigilance * (1.0 + contextValue * contextSensitivity);
    }
    
    /**
     * Check if the given dimension matches the expected input dimension.
     */
    public boolean isValidDimension(int dimension) {
        return dimension == inputDimension;
    }
    
    /**
     * Apply familiarity decay to a score.
     */
    public double applyFamiliarityDecay(double currentFamiliarity) {
        return currentFamiliarity * (1.0 - familiarityDecayRate);
    }
    
    /**
     * Check if performance meets threshold.
     */
    public boolean meetsPerformanceThreshold(double performance) {
        return performance >= performanceThreshold;
    }
    
    /**
     * Create a builder for more complex parameter configurations.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.75;
        private double alpha = 0.001;
        private double baseLearningRate = 0.8;
        private double adaptiveLearningFactor = 0.5;
        private boolean featureWeightingEnabled = true;
        private double[] featureWeights;
        private double topologyAdjustmentRate = 0.1;
        private double performanceThreshold = 0.6;
        private double contextSensitivity = 0.3;
        private double maxLearningRate = 1.0;
        private double minLearningRate = 0.01;
        private int performanceWindowSize = 10;
        private double convergenceThreshold = 0.001;
        private VectorizedParameters baseParameters;
        private boolean enablePerformanceOptimization = true;
        private double familiarityDecayRate = 0.01;
        private int inputDimension = 100;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        public Builder baseLearningRate(double baseLearningRate) {
            this.baseLearningRate = baseLearningRate;
            return this;
        }
        
        public Builder adaptiveLearningFactor(double adaptiveLearningFactor) {
            this.adaptiveLearningFactor = adaptiveLearningFactor;
            return this;
        }
        
        public Builder featureWeightingEnabled(boolean featureWeightingEnabled) {
            this.featureWeightingEnabled = featureWeightingEnabled;
            return this;
        }
        
        public Builder featureWeights(double[] featureWeights) {
            this.featureWeights = featureWeights != null ? Arrays.copyOf(featureWeights, featureWeights.length) : null;
            return this;
        }
        
        public Builder topologyAdjustmentRate(double topologyAdjustmentRate) {
            this.topologyAdjustmentRate = topologyAdjustmentRate;
            return this;
        }
        
        public Builder performanceThreshold(double performanceThreshold) {
            this.performanceThreshold = performanceThreshold;
            return this;
        }
        
        public Builder contextSensitivity(double contextSensitivity) {
            this.contextSensitivity = contextSensitivity;
            return this;
        }
        
        public Builder maxLearningRate(double maxLearningRate) {
            this.maxLearningRate = maxLearningRate;
            return this;
        }
        
        public Builder minLearningRate(double minLearningRate) {
            this.minLearningRate = minLearningRate;
            return this;
        }
        
        public Builder performanceWindowSize(int performanceWindowSize) {
            this.performanceWindowSize = performanceWindowSize;
            return this;
        }
        
        public Builder convergenceThreshold(double convergenceThreshold) {
            this.convergenceThreshold = convergenceThreshold;
            return this;
        }
        
        public Builder baseParameters(VectorizedParameters baseParameters) {
            this.baseParameters = baseParameters;
            return this;
        }
        
        public Builder enablePerformanceOptimization(boolean enablePerformanceOptimization) {
            this.enablePerformanceOptimization = enablePerformanceOptimization;
            return this;
        }
        
        public Builder familiarityDecayRate(double familiarityDecayRate) {
            this.familiarityDecayRate = familiarityDecayRate;
            return this;
        }
        
        public Builder inputDimension(int inputDimension) {
            this.inputDimension = inputDimension;
            return this;
        }
        
        public VectorizedARTEParameters build() {
            if (baseParameters == null) {
                baseParameters = VectorizedParameters.createDefault();
            }
            
            if (featureWeights == null) {
                featureWeights = new double[inputDimension];
                double uniformWeight = 1.0 / inputDimension;
                Arrays.fill(featureWeights, uniformWeight);
            }
            
            return new VectorizedARTEParameters(
                vigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                featureWeightingEnabled, featureWeights, topologyAdjustmentRate,
                performanceThreshold, contextSensitivity, maxLearningRate,
                minLearningRate, performanceWindowSize, convergenceThreshold,
                baseParameters, enablePerformanceOptimization, familiarityDecayRate,
                inputDimension
            );
        }
    }
    
    /**
     * Create default VectorizedARTE parameters.
     */
    public VectorizedARTEParameters() {
        this(0.9, 0.001, 0.5, 0.1, true, null, 0.01, 0.7, 0.1, 1.0, 0.01, 100, 
             1e-6, VectorizedParameters.createDefault(), true, 0.99, 100);
    }
    
    /**
     * Check if topology adjustment is enabled.
     */
    public boolean isTopologyAdjustmentEnabled() {
        return topologyAdjustmentRate > 0;
    }
    
    /**
     * Get topology adjustment probability.
     */
    public double getTopologyAdjustmentProbability() {
        return topologyAdjustmentRate;
    }
    
    /**
     * Convert to core ARTEParameters.
     */
    public com.hellblazer.art.core.parameters.ARTEParameters toParameters() {
        return new com.hellblazer.art.core.parameters.ARTEParameters(
            vigilance,
            alpha,
            baseLearningRate,
            adaptiveLearningFactor,
            featureWeightingEnabled,
            featureWeights,
            topologyAdjustmentRate,
            performanceThreshold,
            contextSensitivity,
            maxLearningRate,
            minLearningRate,
            performanceWindowSize,
            convergenceThreshold
        );
    }
}