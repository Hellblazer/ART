package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * GaussianWeight represents a weight vector for GaussianART using statistical parameters.
 * Maintains mean, covariance (sigma), and sample count for Gaussian cluster representation.
 * Updates use incremental mean and variance calculations.
 */
public record GaussianWeight(double[] mean, double[] sigma, double[] invSigma, 
                           double sqrtDetSigma, long sampleCount) implements WeightVector {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public GaussianWeight {
        Objects.requireNonNull(mean, "Mean cannot be null");
        Objects.requireNonNull(sigma, "Sigma cannot be null");
        Objects.requireNonNull(invSigma, "InvSigma cannot be null");
        
        if (mean.length == 0) {
            throw new IllegalArgumentException("Mean cannot be empty");
        }
        if (mean.length != sigma.length) {
            throw new IllegalArgumentException("Mean and sigma dimensions must match: " + 
                mean.length + " vs " + sigma.length);
        }
        if (mean.length != invSigma.length) {
            throw new IllegalArgumentException("Mean and invSigma dimensions must match: " + 
                mean.length + " vs " + invSigma.length);
        }
        if (sampleCount <= 0) {
            throw new IllegalArgumentException("Sample count must be positive, got: " + sampleCount);
        }
        if (sqrtDetSigma <= 0) {
            throw new IllegalArgumentException("Square root determinant of sigma must be positive, got: " + sqrtDetSigma);
        }
        
        // Validate sigma values are positive
        for (int i = 0; i < sigma.length; i++) {
            if (sigma[i] <= 0.0) {
                throw new IllegalArgumentException("All sigma values must be positive, got " + 
                    sigma[i] + " at index " + i);
            }
        }
        
        // Copy arrays to ensure immutability
        mean = Arrays.copyOf(mean, mean.length);
        sigma = Arrays.copyOf(sigma, sigma.length);
        invSigma = Arrays.copyOf(invSigma, invSigma.length);
    }
    
    /**
     * Create a GaussianWeight with specified parameters.
     * Computes inverse sigma and determinant automatically.
     * 
     * @param mean the mean vector
     * @param sigma the diagonal covariance vector (all values must be positive)
     * @param sampleCount the number of samples this weight represents
     * @return new GaussianWeight instance
     */
    public static GaussianWeight of(double[] mean, double[] sigma, long sampleCount) {
        Objects.requireNonNull(mean, "Mean cannot be null");
        Objects.requireNonNull(sigma, "Sigma cannot be null");
        
        if (mean.length != sigma.length) {
            throw new IllegalArgumentException("Mean and sigma dimensions must match: " + 
                mean.length + " vs " + sigma.length);
        }
        
        // Compute inverse sigma (diagonal matrix)
        var invSigma = new double[sigma.length];
        double detSigma = 1.0;
        
        for (int i = 0; i < sigma.length; i++) {
            if (sigma[i] <= 0.0) {
                throw new IllegalArgumentException("All sigma values must be positive, got " + 
                    sigma[i] + " at index " + i);
            }
            invSigma[i] = 1.0 / sigma[i];
            detSigma *= sigma[i];
        }
        
        var sqrtDetSigma = Math.sqrt(detSigma);
        
        return new GaussianWeight(mean, sigma, invSigma, sqrtDetSigma, sampleCount);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= mean.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for vector of size " + mean.length);
        }
        return mean[index];
    }
    
    @Override
    public int dimension() {
        return mean.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : mean) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    /**
     * Update this GaussianWeight with a new sample using incremental statistics.
     * Updates both mean and variance using online algorithms.
     * 
     * @param input the new sample vector
     * @param parameters GaussianParameters (not used in current implementation)
     * @return new updated GaussianWeight
     */
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof GaussianParameters)) {
            throw new IllegalArgumentException("Parameters must be GaussianParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != mean.length) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match weight dimension " + mean.length);
        }
        
        var newSampleCount = sampleCount + 1;
        var newMean = new double[mean.length];
        var newSigma = new double[sigma.length];
        
        // Incremental mean update: μ_new = μ_old + (x - μ_old) / n_new
        for (int i = 0; i < mean.length; i++) {
            var delta = input.get(i) - mean[i];
            newMean[i] = mean[i] + delta / newSampleCount;
        }
        
        // Incremental variance update for online algorithm
        // For simplicity, we use a basic incremental approach
        // σ²_new = σ²_old + (x - μ_old)(x - μ_new) / n_new
        for (int i = 0; i < sigma.length; i++) {
            var delta1 = input.get(i) - mean[i];
            var delta2 = input.get(i) - newMean[i];
            
            // Update variance using incremental formula
            var oldVariance = sigma[i] * sigma[i];
            var newVariance = (oldVariance * sampleCount + delta1 * delta2) / newSampleCount;
            newSigma[i] = Math.sqrt(Math.max(newVariance, 1e-10)); // Ensure positive variance
        }
        
        return GaussianWeight.of(newMean, newSigma, newSampleCount);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof GaussianWeight other)) return false;
        return sampleCount == other.sampleCount &&
               Double.compare(sqrtDetSigma, other.sqrtDetSigma) == 0 &&
               Arrays.equals(mean, other.mean) &&
               Arrays.equals(sigma, other.sigma) &&
               Arrays.equals(invSigma, other.invSigma);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(mean), Arrays.hashCode(sigma), 
                          Arrays.hashCode(invSigma), sqrtDetSigma, sampleCount);
    }
    
    @Override
    public String toString() {
        return "GaussianWeight{mean=" + Arrays.toString(mean) + 
               ", sigma=" + Arrays.toString(sigma) +
               ", samples=" + sampleCount + "}";
    }
}