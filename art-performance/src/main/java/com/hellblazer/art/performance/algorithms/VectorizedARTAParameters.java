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

/**
 * Parameters for VectorizedARTA (Attentional ART) algorithm.
 * 
 * ART-A extends traditional ART with attention mechanisms that dynamically
 * weight input features based on their discriminative power for each category.
 * This vectorized version provides enhanced performance through SIMD operations
 * and optimized attention weight computations.
 * 
 * Key features:
 * - Attention-weighted activation computation
 * - Dynamic attention weight learning with SIMD optimization
 * - Attention-based vigilance testing
 * - Feature importance analysis through attention weights
 */
public class VectorizedARTAParameters {
    
    private final double vigilance;
    private final double alpha;
    private final double beta;
    private final double attentionLearningRate;
    private final double attentionVigilance;
    private final double minAttentionWeight;
    private final VectorizedParameters baseParameters;
    private final boolean enableAdaptiveAttention;
    private final double attentionDecayRate;
    private final double maxAttentionWeight;
    private final int inputDimension;
    private final boolean enableAttentionRegularization;
    private final double attentionRegularizationFactor;
    
    public VectorizedARTAParameters(
            double vigilance,
            double alpha,
            double beta,
            double attentionLearningRate,
            double attentionVigilance,
            double minAttentionWeight,
            VectorizedParameters baseParameters,
            boolean enableAdaptiveAttention,
            double attentionDecayRate,
            double maxAttentionWeight,
            int inputDimension,
            boolean enableAttentionRegularization,
            double attentionRegularizationFactor) {
        
        validateInputs(vigilance, alpha, beta, attentionLearningRate, attentionVigilance,
                       minAttentionWeight, attentionDecayRate, maxAttentionWeight,
                       inputDimension, attentionRegularizationFactor);
        
        this.vigilance = vigilance;
        this.alpha = alpha;
        this.beta = beta;
        this.attentionLearningRate = attentionLearningRate;
        this.attentionVigilance = attentionVigilance;
        this.minAttentionWeight = minAttentionWeight;
        this.baseParameters = baseParameters;
        this.enableAdaptiveAttention = enableAdaptiveAttention;
        this.attentionDecayRate = attentionDecayRate;
        this.maxAttentionWeight = maxAttentionWeight;
        this.inputDimension = inputDimension;
        this.enableAttentionRegularization = enableAttentionRegularization;
        this.attentionRegularizationFactor = attentionRegularizationFactor;
    }
    
    public static VectorizedARTAParameters defaults() {
        var defaultBaseParams = VectorizedParameters.createDefault();
        
        return new VectorizedARTAParameters(
            0.75,    // vigilance
            0.001,   // alpha (choice parameter)
            0.8,     // beta (learning rate)
            0.1,     // attentionLearningRate
            0.8,     // attentionVigilance
            0.01,    // minAttentionWeight
            defaultBaseParams,
            true,    // enableAdaptiveAttention
            0.001,   // attentionDecayRate
            10.0,    // maxAttentionWeight
            100,     // inputDimension (default)
            true,    // enableAttentionRegularization
            0.001    // attentionRegularizationFactor
        );
    }
    
    public static VectorizedARTAParameters forDimension(int inputDimension) {
        var baseParams = VectorizedParameters.createDefault();
        
        return new VectorizedARTAParameters(
            0.75,    // vigilance
            0.001,   // alpha
            0.8,     // beta
            0.1,     // attentionLearningRate
            0.8,     // attentionVigilance
            0.01,    // minAttentionWeight
            baseParams,
            true,    // enableAdaptiveAttention
            0.001,   // attentionDecayRate
            10.0,    // maxAttentionWeight
            inputDimension, // inputDimension
            true,    // enableAttentionRegularization
            0.001    // attentionRegularizationFactor
        );
    }
    
    private static void validateInputs(double vigilance, double alpha, double beta,
                                       double attentionLearningRate, double attentionVigilance,
                                       double minAttentionWeight, double attentionDecayRate,
                                       double maxAttentionWeight, int inputDimension,
                                       double attentionRegularizationFactor) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in [0, 1], got: " + beta);
        }
        
        if (attentionLearningRate < 0.0 || attentionLearningRate > 1.0) {
            throw new IllegalArgumentException("Attention learning rate must be in [0, 1], got: " + attentionLearningRate);
        }
        
        if (attentionVigilance < 0.0 || attentionVigilance > 1.0) {
            throw new IllegalArgumentException("Attention vigilance must be in [0, 1], got: " + attentionVigilance);
        }
        
        if (minAttentionWeight < 0.0 || minAttentionWeight > 1.0) {
            throw new IllegalArgumentException("Min attention weight must be in [0, 1], got: " + minAttentionWeight);
        }
        
        if (attentionDecayRate < 0.0 || attentionDecayRate > 1.0) {
            throw new IllegalArgumentException("Attention decay rate must be in [0, 1], got: " + attentionDecayRate);
        }
        
        if (maxAttentionWeight <= minAttentionWeight) {
            throw new IllegalArgumentException("Max attention weight must be > min attention weight, got max=" + maxAttentionWeight + ", min=" + minAttentionWeight);
        }
        
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive, got: " + inputDimension);
        }
        
        if (attentionRegularizationFactor < 0.0) {
            throw new IllegalArgumentException("Attention regularization factor must be non-negative, got: " + attentionRegularizationFactor);
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
    
    public double getBeta() {
        return beta;
    }
    
    public double getAttentionLearningRate() {
        return attentionLearningRate;
    }
    
    public double getAttentionVigilance() {
        return attentionVigilance;
    }
    
    public double getMinAttentionWeight() {
        return minAttentionWeight;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    public boolean isAdaptiveAttentionEnabled() {
        return enableAdaptiveAttention;
    }
    
    public double getAttentionDecayRate() {
        return attentionDecayRate;
    }
    
    public double getMaxAttentionWeight() {
        return maxAttentionWeight;
    }
    
    public int getInputDimension() {
        return inputDimension;
    }
    
    public boolean isAttentionRegularizationEnabled() {
        return enableAttentionRegularization;
    }
    
    public double getAttentionRegularizationFactor() {
        return attentionRegularizationFactor;
    }
    
    /**
     * Clamp attention weight to valid range.
     */
    public double clampAttentionWeight(double weight) {
        return Math.max(minAttentionWeight, Math.min(maxAttentionWeight, weight));
    }
    
    /**
     * Check if the given dimension matches the expected input dimension.
     */
    public boolean isValidDimension(int dimension) {
        return dimension == inputDimension;
    }
    
    /**
     * Calculate the total number of attention weights needed.
     */
    public int getAttentionWeightCount() {
        return inputDimension;
    }
    
    /**
     * Get the total number of parameters (category weights + attention weights).
     */
    public int getTotalParameterCount() {
        return inputDimension * 2; // Category weights + attention weights
    }
    
    /**
     * Apply attention decay to a weight.
     */
    public double applyAttentionDecay(double currentWeight) {
        if (!enableAdaptiveAttention) {
            return currentWeight;
        }
        return currentWeight * (1.0 - attentionDecayRate);
    }
    
    /**
     * Apply attention regularization.
     */
    public double applyAttentionRegularization(double gradient) {
        if (!enableAttentionRegularization) {
            return gradient;
        }
        return gradient - attentionRegularizationFactor * gradient;
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
        private double beta = 0.8;
        private double attentionLearningRate = 0.1;
        private double attentionVigilance = 0.8;
        private double minAttentionWeight = 0.01;
        private VectorizedParameters baseParameters;
        private boolean enableAdaptiveAttention = true;
        private double attentionDecayRate = 0.001;
        private double maxAttentionWeight = 10.0;
        private int inputDimension = 100;
        private boolean enableAttentionRegularization = true;
        private double attentionRegularizationFactor = 0.001;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        public Builder beta(double beta) {
            this.beta = beta;
            return this;
        }
        
        public Builder attentionLearningRate(double attentionLearningRate) {
            this.attentionLearningRate = attentionLearningRate;
            return this;
        }
        
        public Builder attentionVigilance(double attentionVigilance) {
            this.attentionVigilance = attentionVigilance;
            return this;
        }
        
        public Builder minAttentionWeight(double minAttentionWeight) {
            this.minAttentionWeight = minAttentionWeight;
            return this;
        }
        
        public Builder baseParameters(VectorizedParameters baseParameters) {
            this.baseParameters = baseParameters;
            return this;
        }
        
        public Builder enableAdaptiveAttention(boolean enableAdaptiveAttention) {
            this.enableAdaptiveAttention = enableAdaptiveAttention;
            return this;
        }
        
        public Builder attentionDecayRate(double attentionDecayRate) {
            this.attentionDecayRate = attentionDecayRate;
            return this;
        }
        
        public Builder maxAttentionWeight(double maxAttentionWeight) {
            this.maxAttentionWeight = maxAttentionWeight;
            return this;
        }
        
        public Builder inputDimension(int inputDimension) {
            this.inputDimension = inputDimension;
            return this;
        }
        
        public Builder enableAttentionRegularization(boolean enableAttentionRegularization) {
            this.enableAttentionRegularization = enableAttentionRegularization;
            return this;
        }
        
        public Builder attentionRegularizationFactor(double attentionRegularizationFactor) {
            this.attentionRegularizationFactor = attentionRegularizationFactor;
            return this;
        }
        
        public VectorizedARTAParameters build() {
            if (baseParameters == null) {
                baseParameters = VectorizedParameters.createDefault();
            }
            
            return new VectorizedARTAParameters(
                vigilance, alpha, beta, attentionLearningRate, attentionVigilance,
                minAttentionWeight, baseParameters, enableAdaptiveAttention,
                attentionDecayRate, maxAttentionWeight, inputDimension,
                enableAttentionRegularization, attentionRegularizationFactor
            );
        }
    }
}