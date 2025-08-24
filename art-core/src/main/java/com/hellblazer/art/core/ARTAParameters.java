package com.hellblazer.art.core;

/**
 * Immutable parameters for ART-A (Attentional ART) algorithm.
 * ART-A extends basic ART with attention weighting mechanisms that
 * dynamically focus on the most discriminative input features.
 * 
 * @param vigilance the vigilance parameter (ρ) in range [0, 1]
 * @param alpha the choice parameter (α) for activation function, α ≥ 0
 * @param beta the learning rate (β) in range [0, 1]
 * @param attentionLearningRate the attention weight learning rate (γ) in range [0, 1]
 * @param attentionVigilance the attention-based vigilance parameter (ρa) in range [0, 1]
 * @param minAttentionWeight minimum allowed attention weight to prevent complete suppression
 */
public record ARTAParameters(
    double vigilance,
    double alpha, 
    double beta,
    double attentionLearningRate,
    double attentionVigilance,
    double minAttentionWeight
) {
    
    /**
     * Constructor with validation.
     */
    public ARTAParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in range [0, 1], got: " + beta);
        }
        if (attentionLearningRate < 0.0 || attentionLearningRate > 1.0) {
            throw new IllegalArgumentException("Attention learning rate must be in range [0, 1], got: " + attentionLearningRate);
        }
        if (attentionVigilance < 0.0 || attentionVigilance > 1.0) {
            throw new IllegalArgumentException("Attention vigilance must be in range [0, 1], got: " + attentionVigilance);
        }
        if (minAttentionWeight < 0.0 || minAttentionWeight > 1.0) {
            throw new IllegalArgumentException("Min attention weight must be in range [0, 1], got: " + minAttentionWeight);
        }
        if (minAttentionWeight >= 1.0 / Math.max(1, Integer.MAX_VALUE)) { // Reasonable lower bound
            // This is just a sanity check - in practice minAttentionWeight should be small like 0.01
        }
    }
    
    /**
     * Create ARTAParameters with specified values.
     * @param vigilance the vigilance parameter ρ ∈ [0, 1]
     * @param alpha the choice parameter α ≥ 0
     * @param beta the learning rate β ∈ [0, 1]
     * @param attentionLearningRate the attention learning rate γ ∈ [0, 1]
     * @param attentionVigilance the attention vigilance ρa ∈ [0, 1] 
     * @param minAttentionWeight the minimum attention weight ∈ [0, 1]
     * @return new ARTAParameters instance
     */
    public static ARTAParameters of(double vigilance, double alpha, double beta,
                                   double attentionLearningRate, double attentionVigilance, 
                                   double minAttentionWeight) {
        return new ARTAParameters(vigilance, alpha, beta, attentionLearningRate, 
                                 attentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create ARTAParameters with default values.
     * Default: vigilance=0.7, alpha=0.0, beta=1.0, attentionLearningRate=0.1,
     *          attentionVigilance=0.8, minAttentionWeight=0.01
     * @return default ARTAParameters
     */
    public static ARTAParameters defaults() {
        return new ARTAParameters(0.7, 0.0, 1.0, 0.1, 0.8, 0.01);
    }
    
    /**
     * Create a new ARTAParameters with different vigilance value.
     * @param newVigilance the new vigilance value
     * @return new ARTAParameters instance
     */
    public ARTAParameters withVigilance(double newVigilance) {
        return new ARTAParameters(newVigilance, alpha, beta, attentionLearningRate,
                                 attentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create a new ARTAParameters with different alpha value.
     * @param newAlpha the new alpha value
     * @return new ARTAParameters instance
     */
    public ARTAParameters withAlpha(double newAlpha) {
        return new ARTAParameters(vigilance, newAlpha, beta, attentionLearningRate,
                                 attentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create a new ARTAParameters with different beta value.
     * @param newBeta the new beta value
     * @return new ARTAParameters instance
     */
    public ARTAParameters withBeta(double newBeta) {
        return new ARTAParameters(vigilance, alpha, newBeta, attentionLearningRate,
                                 attentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create a new ARTAParameters with different attention learning rate.
     * @param newAttentionLearningRate the new attention learning rate
     * @return new ARTAParameters instance
     */
    public ARTAParameters withAttentionLearningRate(double newAttentionLearningRate) {
        return new ARTAParameters(vigilance, alpha, beta, newAttentionLearningRate,
                                 attentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create a new ARTAParameters with different attention vigilance.
     * @param newAttentionVigilance the new attention vigilance value
     * @return new ARTAParameters instance
     */
    public ARTAParameters withAttentionVigilance(double newAttentionVigilance) {
        return new ARTAParameters(vigilance, alpha, beta, attentionLearningRate,
                                 newAttentionVigilance, minAttentionWeight);
    }
    
    /**
     * Create a new ARTAParameters with different minimum attention weight.
     * @param newMinAttentionWeight the new minimum attention weight
     * @return new ARTAParameters instance
     */
    public ARTAParameters withMinAttentionWeight(double newMinAttentionWeight) {
        return new ARTAParameters(vigilance, alpha, beta, attentionLearningRate,
                                 attentionVigilance, newMinAttentionWeight);
    }
    
    /**
     * Create a builder for ARTAParameters.
     * @return new ARTAParametersBuilder
     */
    public static ARTAParametersBuilder builder() {
        return new ARTAParametersBuilder();
    }
    
    /**
     * Builder class for ARTAParameters.
     */
    public static class ARTAParametersBuilder {
        private double vigilance = 0.7;
        private double alpha = 0.0;
        private double beta = 1.0;
        private double attentionLearningRate = 0.1;
        private double attentionVigilance = 0.8;
        private double minAttentionWeight = 0.01;
        
        /**
         * Set the vigilance parameter.
         * @param vigilance the vigilance value ρ ∈ [0, 1]
         * @return this builder
         */
        public ARTAParametersBuilder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        /**
         * Set the choice parameter (alpha).
         * @param alpha the choice parameter α ≥ 0
         * @return this builder
         */
        public ARTAParametersBuilder choiceParameter(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Set the learning rate (beta).
         * @param beta the learning rate β ∈ [0, 1]
         * @return this builder
         */
        public ARTAParametersBuilder learningRate(double beta) {
            this.beta = beta;
            return this;
        }
        
        /**
         * Set the attention learning rate (gamma).
         * @param attentionLearningRate the attention learning rate γ ∈ [0, 1]
         * @return this builder
         */
        public ARTAParametersBuilder attentionLearningRate(double attentionLearningRate) {
            this.attentionLearningRate = attentionLearningRate;
            return this;
        }
        
        /**
         * Set the attention vigilance parameter.
         * @param attentionVigilance the attention vigilance ρa ∈ [0, 1]
         * @return this builder
         */
        public ARTAParametersBuilder attentionVigilance(double attentionVigilance) {
            this.attentionVigilance = attentionVigilance;
            return this;
        }
        
        /**
         * Set the minimum attention weight.
         * @param minAttentionWeight the minimum attention weight ∈ [0, 1]
         * @return this builder
         */
        public ARTAParametersBuilder minAttentionWeight(double minAttentionWeight) {
            this.minAttentionWeight = minAttentionWeight;
            return this;
        }
        
        /**
         * Build the ARTAParameters instance.
         * @return new ARTAParameters with specified values
         */
        public ARTAParameters build() {
            return new ARTAParameters(vigilance, alpha, beta, attentionLearningRate,
                                     attentionVigilance, minAttentionWeight);
        }
    }
    
    @Override
    public String toString() {
        return String.format("ARTAParameters{ρ=%.3f, α=%.3f, β=%.3f, γ=%.3f, ρa=%.3f, min_att=%.3f}", 
                           vigilance, alpha, beta, attentionLearningRate, 
                           attentionVigilance, minAttentionWeight);
    }
}