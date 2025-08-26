package com.hellblazer.art.core.parameters;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable parameters for GaussianART algorithm.
 * 
 * @param vigilance the vigilance parameter (ρ) in range [0, 1]
 * @param sigmaInit initial sigma values for Gaussian clusters (must be positive)
 */
public record GaussianParameters(double vigilance, double[] sigmaInit) {
    
    /**
     * Constructor with validation.
     */
    public GaussianParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        
        Objects.requireNonNull(sigmaInit, "SigmaInit cannot be null");
        if (sigmaInit.length == 0) {
            throw new IllegalArgumentException("SigmaInit cannot be empty");
        }
        
        // Validate that all sigma values are positive
        for (int i = 0; i < sigmaInit.length; i++) {
            if (sigmaInit[i] <= 0.0) {
                throw new IllegalArgumentException("All sigma values must be positive, got " + 
                    sigmaInit[i] + " at index " + i);
            }
        }
        
        // Copy array to ensure immutability
        sigmaInit = Arrays.copyOf(sigmaInit, sigmaInit.length);
    }
    
    /**
     * Create GaussianParameters with specified values.
     * @param vigilance the vigilance parameter ρ ∈ [0, 1]
     * @param sigmaInit initial sigma values (must be positive)
     * @return new GaussianParameters instance
     */
    public static GaussianParameters of(double vigilance, double[] sigmaInit) {
        return new GaussianParameters(vigilance, sigmaInit);
    }
    
    /**
     * Create GaussianParameters with default vigilance and specified dimension.
     * All sigma values are initialized to 1.0.
     * 
     * @param dimension the number of dimensions
     * @return new GaussianParameters with default values
     */
    public static GaussianParameters withDimension(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive, got: " + dimension);
        }
        
        var sigma = new double[dimension];
        Arrays.fill(sigma, 1.0);
        return new GaussianParameters(0.5, sigma);
    }
    
    /**
     * Create a new GaussianParameters with different vigilance value.
     * @param newVigilance the new vigilance value
     * @return new GaussianParameters instance
     */
    public GaussianParameters withVigilance(double newVigilance) {
        return new GaussianParameters(newVigilance, sigmaInit);
    }
    
    /**
     * Create a new GaussianParameters with different sigma initialization.
     * @param newSigmaInit the new sigma initialization values
     * @return new GaussianParameters instance
     */
    public GaussianParameters withSigmaInit(double[] newSigmaInit) {
        return new GaussianParameters(vigilance, newSigmaInit);
    }
    
    /**
     * Get the dimension of the sigma initialization.
     * @return the number of dimensions
     */
    public int dimension() {
        return sigmaInit.length;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof GaussianParameters other)) return false;
        return Double.compare(vigilance, other.vigilance) == 0 && 
               Arrays.equals(sigmaInit, other.sigmaInit);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(vigilance, Arrays.hashCode(sigmaInit));
    }
    
    @Override
    public String toString() {
        return String.format("GaussianParameters{ρ=%.3f, σ_init=%s}", 
                           vigilance, Arrays.toString(sigmaInit));
    }
}