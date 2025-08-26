package com.hellblazer.art.core.parameters;

import java.util.Objects;

/**
 * Immutable parameters for FuzzyART algorithm.
 * 
 * @param vigilance the vigilance parameter (ρ) in range [0, 1]
 * @param alpha the choice parameter (α) for activation function, α ≥ 0
 * @param beta the learning rate (β) in range [0, 1]
 */
public record FuzzyParameters(double vigilance, double alpha, double beta) {
    
    /**
     * Constructor with validation.
     */
    public FuzzyParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in range [0, 1], got: " + beta);
        }
    }
    
    /**
     * Create FuzzyParameters with specified values.
     * @param vigilance the vigilance parameter ρ ∈ [0, 1]
     * @param alpha the choice parameter α ≥ 0
     * @param beta the learning rate β ∈ [0, 1]
     * @return new FuzzyParameters instance
     */
    public static FuzzyParameters of(double vigilance, double alpha, double beta) {
        return new FuzzyParameters(vigilance, alpha, beta);
    }
    
    /**
     * Create FuzzyParameters with default values.
     * Default: vigilance=0.5, alpha=0.0, beta=1.0
     * @return default FuzzyParameters
     */
    public static FuzzyParameters defaults() {
        return new FuzzyParameters(0.5, 0.0, 1.0);
    }
    
    /**
     * Create a new FuzzyParameters with different vigilance value.
     * @param newVigilance the new vigilance value
     * @return new FuzzyParameters instance
     */
    public FuzzyParameters withVigilance(double newVigilance) {
        return new FuzzyParameters(newVigilance, alpha, beta);
    }
    
    /**
     * Create a new FuzzyParameters with different alpha value.
     * @param newAlpha the new alpha value
     * @return new FuzzyParameters instance
     */
    public FuzzyParameters withAlpha(double newAlpha) {
        return new FuzzyParameters(vigilance, newAlpha, beta);
    }
    
    /**
     * Create a new FuzzyParameters with different beta value.
     * @param newBeta the new beta value
     * @return new FuzzyParameters instance
     */
    public FuzzyParameters withBeta(double newBeta) {
        return new FuzzyParameters(vigilance, alpha, newBeta);
    }
    
    /**
     * Create a builder for FuzzyParameters.
     * @return new FuzzyParametersBuilder
     */
    public static FuzzyParametersBuilder builder() {
        return new FuzzyParametersBuilder();
    }
    
    /**
     * Builder class for FuzzyParameters.
     */
    public static class FuzzyParametersBuilder {
        private double vigilance = 0.5;
        private double alpha = 0.0;
        private double beta = 1.0;
        
        /**
         * Set the vigilance parameter.
         * @param vigilance the vigilance value ρ ∈ [0, 1]
         * @return this builder
         */
        public FuzzyParametersBuilder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        /**
         * Set the choice parameter (alpha).
         * @param alpha the choice parameter α ≥ 0
         * @return this builder
         */
        public FuzzyParametersBuilder choiceParameter(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Set the learning rate (beta).
         * @param beta the learning rate β ∈ [0, 1]
         * @return this builder
         */
        public FuzzyParametersBuilder learningRate(double beta) {
            this.beta = beta;
            return this;
        }
        
        /**
         * Build the FuzzyParameters instance.
         * @return new FuzzyParameters with specified values
         */
        public FuzzyParameters build() {
            return new FuzzyParameters(vigilance, alpha, beta);
        }
    }
    
    @Override
    public String toString() {
        return String.format("FuzzyParameters{ρ=%.3f, α=%.3f, β=%.3f}", 
                           vigilance, alpha, beta);
    }
}