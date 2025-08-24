package com.hellblazer.art.core;

import java.util.Objects;

/**
 * Parameters for ART-E (Enhanced ART) with adaptive learning features.
 * 
 * ART-E introduces several enhancements over standard ART:
 * - Adaptive learning rates that adjust based on pattern familiarity
 * - Feature importance weighting for better discrimination
 * - Dynamic topology adjustment capabilities
 * - Performance optimization through selective learning
 * - Enhanced vigilance control with context sensitivity
 * 
 * Key parameters:
 * - baseLearningRate: Initial learning rate (beta equivalent)
 * - adaptiveLearningFactor: Controls how much learning rate adapts
 * - featureWeightingEnabled: Whether to use dynamic feature weighting
 * - topologyAdjustmentRate: Rate of network topology changes
 * - performanceThreshold: Minimum performance for category retention
 * - contextSensitivity: How much context influences vigilance
 */
public record ARTEParameters(
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
    double convergenceThreshold
) {
    
    /**
     * Create ARTEParameters with validation.
     */
    public ARTEParameters {
        Objects.requireNonNull(featureWeights, "Feature weights cannot be null");
        
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be >= 0, got: " + alpha);
        }
        if (baseLearningRate < 0.0 || baseLearningRate > 1.0) {
            throw new IllegalArgumentException("Base learning rate must be in [0,1], got: " + baseLearningRate);
        }
        if (adaptiveLearningFactor < 0.0 || adaptiveLearningFactor > 1.0) {
            throw new IllegalArgumentException("Adaptive learning factor must be in [0,1], got: " + adaptiveLearningFactor);
        }
        if (topologyAdjustmentRate < 0.0 || topologyAdjustmentRate > 1.0) {
            throw new IllegalArgumentException("Topology adjustment rate must be in [0,1], got: " + topologyAdjustmentRate);
        }
        if (performanceThreshold < 0.0 || performanceThreshold > 1.0) {
            throw new IllegalArgumentException("Performance threshold must be in [0,1], got: " + performanceThreshold);
        }
        if (contextSensitivity < 0.0 || contextSensitivity > 1.0) {
            throw new IllegalArgumentException("Context sensitivity must be in [0,1], got: " + contextSensitivity);
        }
        if (maxLearningRate < minLearningRate) {
            throw new IllegalArgumentException("Max learning rate (" + maxLearningRate + 
                                             ") must be >= min learning rate (" + minLearningRate + ")");
        }
        if (performanceWindowSize < 1) {
            throw new IllegalArgumentException("Performance window size must be >= 1, got: " + performanceWindowSize);
        }
        if (convergenceThreshold < 0.0) {
            throw new IllegalArgumentException("Convergence threshold must be >= 0, got: " + convergenceThreshold);
        }
        
        // Ensure feature weights are normalized and valid
        if (featureWeightingEnabled && featureWeights.length == 0) {
            throw new IllegalArgumentException("Feature weights cannot be empty when feature weighting is enabled");
        }
        
        for (int i = 0; i < featureWeights.length; i++) {
            if (featureWeights[i] < 0.0) {
                throw new IllegalArgumentException("Feature weight " + i + " must be >= 0, got: " + featureWeights[i]);
            }
        }
    }
    
    /**
     * Create default ART-E parameters for specified input dimension.
     */
    public static ARTEParameters createDefault(int inputDimension) {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be > 0, got: " + inputDimension);
        }
        
        // Initialize uniform feature weights
        var defaultFeatureWeights = new double[inputDimension];
        double uniformWeight = 1.0 / inputDimension;
        for (int i = 0; i < inputDimension; i++) {
            defaultFeatureWeights[i] = uniformWeight;
        }
        
        return new ARTEParameters(
            0.75,                    // vigilance
            0.001,                   // alpha  
            0.1,                     // baseLearningRate
            0.2,                     // adaptiveLearningFactor
            true,                    // featureWeightingEnabled
            defaultFeatureWeights,   // featureWeights
            0.05,                    // topologyAdjustmentRate
            0.3,                     // performanceThreshold
            0.1,                     // contextSensitivity
            0.8,                     // maxLearningRate
            0.01,                    // minLearningRate
            10,                      // performanceWindowSize
            0.001                    // convergenceThreshold
        );
    }
    
    /**
     * Builder for creating ARTEParameters with fluent interface.
     */
    public static final class Builder {
        private double vigilance = 0.75;
        private double alpha = 0.001;
        private double baseLearningRate = 0.1;
        private double adaptiveLearningFactor = 0.2;
        private boolean featureWeightingEnabled = true;
        private double[] featureWeights = new double[0];
        private double topologyAdjustmentRate = 0.05;
        private double performanceThreshold = 0.3;
        private double contextSensitivity = 0.1;
        private double maxLearningRate = 0.8;
        private double minLearningRate = 0.01;
        private int performanceWindowSize = 10;
        private double convergenceThreshold = 0.001;
        
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
            this.featureWeights = Objects.requireNonNull(featureWeights, "Feature weights cannot be null");
            return this;
        }
        
        public Builder uniformFeatureWeights(int dimension) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Dimension must be > 0");
            }
            this.featureWeights = new double[dimension];
            double uniformWeight = 1.0 / dimension;
            for (int i = 0; i < dimension; i++) {
                this.featureWeights[i] = uniformWeight;
            }
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
        
        public ARTEParameters build() {
            return new ARTEParameters(vigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                                     featureWeightingEnabled, featureWeights.clone(),
                                     topologyAdjustmentRate, performanceThreshold, contextSensitivity,
                                     maxLearningRate, minLearningRate, performanceWindowSize,
                                     convergenceThreshold);
        }
    }
    
    /**
     * Create a new Builder instance.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Get effective vigilance adjusted for context.
     */
    public double getEffectiveVigilance(double contextFactor) {
        var contextAdjustment = contextSensitivity * (contextFactor - 0.5) * 2.0; // Range: [-contextSensitivity, +contextSensitivity]
        return Math.max(0.0, Math.min(1.0, vigilance + contextAdjustment));
    }
    
    /**
     * Calculate adaptive learning rate based on pattern familiarity.
     */
    public double getAdaptiveLearningRate(double familiarityScore) {
        // Higher familiarity -> lower learning rate (don't overwrite well-learned patterns)
        // Lower familiarity -> higher learning rate (learn new patterns quickly)
        var adaptiveComponent = adaptiveLearningFactor * (1.0 - familiarityScore);
        var effectiveLearningRate = baseLearningRate + adaptiveComponent;
        return Math.max(minLearningRate, Math.min(maxLearningRate, effectiveLearningRate));
    }
    
    /**
     * Get feature weight for specified dimension.
     */
    public double getFeatureWeight(int dimension) {
        if (!featureWeightingEnabled || dimension < 0 || dimension >= featureWeights.length) {
            return 1.0; // Default weight when disabled or out of bounds
        }
        return featureWeights[dimension];
    }
    
    /**
     * Check if topology adjustment should be applied based on rate.
     */
    public boolean shouldAdjustTopology(double randomValue) {
        return randomValue < topologyAdjustmentRate;
    }
    
    /**
     * Create updated parameters with new feature weights.
     */
    public ARTEParameters withFeatureWeights(double[] newFeatureWeights) {
        Objects.requireNonNull(newFeatureWeights, "New feature weights cannot be null");
        return new ARTEParameters(vigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                                 featureWeightingEnabled, newFeatureWeights.clone(),
                                 topologyAdjustmentRate, performanceThreshold, contextSensitivity,
                                 maxLearningRate, minLearningRate, performanceWindowSize,
                                 convergenceThreshold);
    }
    
    /**
     * Create updated parameters with new vigilance.
     */
    public ARTEParameters withVigilance(double newVigilance) {
        return new ARTEParameters(newVigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                                 featureWeightingEnabled, featureWeights,
                                 topologyAdjustmentRate, performanceThreshold, contextSensitivity,
                                 maxLearningRate, minLearningRate, performanceWindowSize,
                                 convergenceThreshold);
    }
    
    @Override
    public String toString() {
        return String.format("ARTEParameters{vig=%.3f, α=%.3f, baseLR=%.3f, adaptLR=%.3f, " +
                           "featWt=%s, topAdj=%.3f, perfThr=%.3f, ctxSens=%.3f}",
                           vigilance, alpha, baseLearningRate, adaptiveLearningFactor,
                           featureWeightingEnabled, topologyAdjustmentRate, 
                           performanceThreshold, contextSensitivity);
    }
}