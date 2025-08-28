package com.hellblazer.art.core.parameters;

import java.util.Objects;

/**
 * Immutable parameters for TopoART algorithm.
 * 
 * @param inputDimension the dimension of input vectors (d)
 * @param vigilanceA the vigilance parameter for component A (ρₐ ∈ [0, 1])
 * @param learningRateSecond the learning rate for second-best neurons (βₛᵦₘ ∈ [0, 1])
 * @param phi the permanence threshold (φ > 0)
 * @param tau the cleanup cycle period (τ > 0)
 * @param alpha the choice parameter (α ≥ 0)
 */
public record TopoARTParameters(int inputDimension, double vigilanceA, 
                               double learningRateSecond, int phi, int tau, double alpha) {
    
    /**
     * Constructor with validation.
     */
    public TopoARTParameters {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive, got: " + inputDimension);
        }
        if (vigilanceA < 0.0 || vigilanceA > 1.0) {
            throw new IllegalArgumentException("Vigilance A must be in [0, 1], got: " + vigilanceA);
        }
        if (learningRateSecond < 0.0 || learningRateSecond > 1.0) {
            throw new IllegalArgumentException("Learning rate second must be in [0, 1], got: " + learningRateSecond);
        }
        if (phi <= 0) {
            throw new IllegalArgumentException("Phi must be positive, got: " + phi);
        }
        if (tau <= 0) {
            throw new IllegalArgumentException("Tau must be positive, got: " + tau);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
    }
    
    /**
     * Calculate the vigilance parameter for component B.
     * Component B vigilance: ρₑ = (ρₐ + 1) / 2
     * 
     * @return the vigilance parameter for component B
     */
    public double vigilanceB() {
        return 0.5 * (vigilanceA + 1.0);
    }
    
    /**
     * Get the complement-coded dimension (2 * input dimension).
     * 
     * @return 2 * inputDimension
     */
    public int complementCodedDimension() {
        return 2 * inputDimension;
    }
    
    /**
     * Create TopoARTParameters with specified values.
     * 
     * @param inputDimension the input dimension
     * @param vigilanceA the vigilance for component A
     * @param learningRateSecond the learning rate for second-best neurons
     * @param phi the permanence threshold
     * @param tau the cleanup cycle period
     * @param alpha the choice parameter
     * @return new TopoARTParameters instance
     */
    public static TopoARTParameters of(int inputDimension, double vigilanceA, 
                                      double learningRateSecond, int phi, int tau, double alpha) {
        return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, phi, tau, alpha);
    }
    
    /**
     * Create TopoARTParameters with default alpha value.
     * 
     * @param inputDimension the input dimension
     * @param vigilanceA the vigilance for component A
     * @param learningRateSecond the learning rate for second-best neurons
     * @param phi the permanence threshold
     * @param tau the cleanup cycle period
     * @return new TopoARTParameters instance with alpha = 0.001
     */
    public static TopoARTParameters of(int inputDimension, double vigilanceA, 
                                      double learningRateSecond, int phi, int tau) {
        return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, phi, tau, 0.001);
    }
    
    /**
     * Create default TopoARTParameters for specified input dimension.
     * Defaults: vigilanceA=0.9, learningRateSecond=0.6, phi=5, tau=100, alpha=0.001
     * 
     * @param inputDimension the input dimension
     * @return default TopoARTParameters
     */
    public static TopoARTParameters defaults(int inputDimension) {
        return new TopoARTParameters(inputDimension, 0.9, 0.6, 5, 100, 0.001);
    }
    
    /**
     * Create a new TopoARTParameters with different vigilance A value.
     * 
     * @param newVigilanceA the new vigilance A value
     * @return new TopoARTParameters instance
     */
    public TopoARTParameters withVigilanceA(double newVigilanceA) {
        return new TopoARTParameters(inputDimension, newVigilanceA, learningRateSecond, phi, tau, alpha);
    }
    
    /**
     * Create a new TopoARTParameters with different learning rate.
     * 
     * @param newLearningRateSecond the new learning rate for second-best
     * @return new TopoARTParameters instance
     */
    public TopoARTParameters withLearningRateSecond(double newLearningRateSecond) {
        return new TopoARTParameters(inputDimension, vigilanceA, newLearningRateSecond, phi, tau, alpha);
    }
    
    /**
     * Create a new TopoARTParameters with different phi value.
     * 
     * @param newPhi the new permanence threshold
     * @return new TopoARTParameters instance
     */
    public TopoARTParameters withPhi(int newPhi) {
        return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, newPhi, tau, alpha);
    }
    
    /**
     * Create a new TopoARTParameters with different tau value.
     * 
     * @param newTau the new cleanup cycle period
     * @return new TopoARTParameters instance
     */
    public TopoARTParameters withTau(int newTau) {
        return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, phi, newTau, alpha);
    }
    
    /**
     * Create a new TopoARTParameters with different alpha value.
     * 
     * @param newAlpha the new choice parameter
     * @return new TopoARTParameters instance
     */
    public TopoARTParameters withAlpha(double newAlpha) {
        return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, phi, tau, newAlpha);
    }
    
    /**
     * Create a builder for TopoARTParameters.
     * 
     * @return new TopoARTParametersBuilder
     */
    public static TopoARTParametersBuilder builder() {
        return new TopoARTParametersBuilder();
    }
    
    /**
     * Builder class for TopoARTParameters.
     */
    public static class TopoARTParametersBuilder {
        private int inputDimension = -1; // Force explicit setting
        private double vigilanceA = 0.9;
        private double learningRateSecond = 0.6;
        private int phi = 5;
        private int tau = 100;
        private double alpha = 0.001;
        
        /**
         * Set the input dimension.
         * 
         * @param inputDimension the input dimension (must be > 0)
         * @return this builder
         */
        public TopoARTParametersBuilder inputDimension(int inputDimension) {
            this.inputDimension = inputDimension;
            return this;
        }
        
        /**
         * Set the vigilance parameter for component A.
         * 
         * @param vigilanceA the vigilance parameter ρₐ ∈ [0, 1]
         * @return this builder
         */
        public TopoARTParametersBuilder vigilanceA(double vigilanceA) {
            this.vigilanceA = vigilanceA;
            return this;
        }
        
        /**
         * Set the learning rate for second-best neurons.
         * 
         * @param learningRateSecond the learning rate βₛᵦₘ ∈ [0, 1]
         * @return this builder
         */
        public TopoARTParametersBuilder learningRateSecond(double learningRateSecond) {
            this.learningRateSecond = learningRateSecond;
            return this;
        }
        
        /**
         * Set the permanence threshold.
         * 
         * @param phi the permanence threshold φ > 0
         * @return this builder
         */
        public TopoARTParametersBuilder phi(int phi) {
            this.phi = phi;
            return this;
        }
        
        /**
         * Set the cleanup cycle period.
         * 
         * @param tau the cleanup cycle period τ > 0
         * @return this builder
         */
        public TopoARTParametersBuilder tau(int tau) {
            this.tau = tau;
            return this;
        }
        
        /**
         * Set the choice parameter.
         * 
         * @param alpha the choice parameter α ≥ 0
         * @return this builder
         */
        public TopoARTParametersBuilder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Build the TopoARTParameters instance.
         * 
         * @return new TopoARTParameters with specified values
         * @throws IllegalArgumentException if inputDimension was not set
         */
        public TopoARTParameters build() {
            if (inputDimension <= 0) {
                throw new IllegalArgumentException("Input dimension must be set and positive");
            }
            return new TopoARTParameters(inputDimension, vigilanceA, learningRateSecond, phi, tau, alpha);
        }
    }
    
    @Override
    public String toString() {
        return String.format("TopoARTParameters{d=%d, ρₐ=%.3f, ρₑ=%.3f, βₛᵦₘ=%.3f, φ=%d, τ=%d, α=%.3f}", 
                           inputDimension, vigilanceA, vigilanceB(), learningRateSecond, phi, tau, alpha);
    }
}