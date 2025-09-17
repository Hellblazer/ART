package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;

import java.util.Objects;

/**
 * Immutable parameters for VectorizedSimpleARTMAP supervised learning algorithm.
 * Extends SimpleARTMAPParameters with vectorization-specific settings and type-safe parameter handling.
 * 
 * @param mapFieldVigilance the map field vigilance parameter (ρ_map) in range [0, 1]
 * @param epsilon small positive value for vigilance adjustment during match tracking
 * @param enableMatchTracking whether to enable match tracking algorithm
 * @param maxSearchAttempts maximum number of vigilance search attempts
 * @param artAParams type-safe parameters for the ART module (VectorizedParameters)
 */
public record VectorizedSimpleARTMAPParameters(
    double mapFieldVigilance,
    double epsilon,
    boolean enableMatchTracking,
    int maxSearchAttempts,
    VectorizedParameters artAParams
) {
    
    /**
     * Constructor with validation.
     */
    public VectorizedSimpleARTMAPParameters {
        if (mapFieldVigilance < 0.0 || mapFieldVigilance > 1.0) {
            throw new IllegalArgumentException("Map field vigilance must be in range [0, 1], got: " + mapFieldVigilance);
        }
        if (epsilon <= 0.0) {
            throw new IllegalArgumentException("Epsilon must be positive, got: " + epsilon);
        }
        if (epsilon > 0.1) {
            throw new IllegalArgumentException("Epsilon should be small (typically < 0.1), got: " + epsilon);
        }
        if (maxSearchAttempts <= 0) {
            throw new IllegalArgumentException("Max search attempts must be positive, got: " + maxSearchAttempts);
        }
        
        Objects.requireNonNull(artAParams, "ART parameters cannot be null");
        
        // Check for NaN and infinite values
        if (Double.isNaN(mapFieldVigilance) || Double.isNaN(epsilon) ||
            Double.isInfinite(mapFieldVigilance) || Double.isInfinite(epsilon)) {
            throw new IllegalArgumentException("Parameters cannot be NaN or infinite");
        }
    }
    
    /**
     * Convert to base SimpleARTMAPParameters for backward compatibility.
     * @return equivalent SimpleARTMAPParameters
     */
    public SimpleARTMAPParameters toSimpleARTMAPParameters() {
        return SimpleARTMAPParameters.of(mapFieldVigilance, epsilon);
    }
    
    /**
     * Create VectorizedSimpleARTMAPParameters with default values.
     * Default: mapFieldVigilance=0.95, epsilon=0.001, matchTracking=true, maxAttempts=10
     * @return default VectorizedSimpleARTMAPParameters
     */
    public static VectorizedSimpleARTMAPParameters defaults() {
        var defaultArtParams = VectorizedParameters.createDefault().withVigilance(0.7);
        
        return new VectorizedSimpleARTMAPParameters(
            0.95,   // mapFieldVigilance
            0.001,  // epsilon
            true,   // enableMatchTracking
            10,     // maxSearchAttempts
            defaultArtParams
        );
    }
    
    /**
     * Create a new VectorizedSimpleARTMAPParameters with different map field vigilance.
     * @param newMapFieldVigilance the new map field vigilance value
     * @return new VectorizedSimpleARTMAPParameters instance
     */
    public VectorizedSimpleARTMAPParameters withMapFieldVigilance(double newMapFieldVigilance) {
        return new VectorizedSimpleARTMAPParameters(
            newMapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts, artAParams
        );
    }
    
    /**
     * Create a new VectorizedSimpleARTMAPParameters with different epsilon.
     * @param newEpsilon the new epsilon value
     * @return new VectorizedSimpleARTMAPParameters instance
     */
    public VectorizedSimpleARTMAPParameters withEpsilon(double newEpsilon) {
        return new VectorizedSimpleARTMAPParameters(
            mapFieldVigilance, newEpsilon, enableMatchTracking, maxSearchAttempts, artAParams
        );
    }
    
    /**
     * Create a new VectorizedSimpleARTMAPParameters with different match tracking setting.
     * @param newEnableMatchTracking the new match tracking setting
     * @return new VectorizedSimpleARTMAPParameters instance
     */
    public VectorizedSimpleARTMAPParameters withEnableMatchTracking(boolean newEnableMatchTracking) {
        return new VectorizedSimpleARTMAPParameters(
            mapFieldVigilance, epsilon, newEnableMatchTracking, maxSearchAttempts, artAParams
        );
    }
    
    /**
     * Create a new VectorizedSimpleARTMAPParameters with different max search attempts.
     * @param newMaxSearchAttempts the new max search attempts value
     * @return new VectorizedSimpleARTMAPParameters instance
     */
    public VectorizedSimpleARTMAPParameters withMaxSearchAttempts(int newMaxSearchAttempts) {
        return new VectorizedSimpleARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, newMaxSearchAttempts, artAParams
        );
    }
    
    /**
     * Create a new VectorizedSimpleARTMAPParameters with different ART parameters.
     * @param newArtAParams the new ART parameters
     * @return new VectorizedSimpleARTMAPParameters instance
     */
    public VectorizedSimpleARTMAPParameters withArtAParams(VectorizedParameters newArtAParams) {
        return new VectorizedSimpleARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts, newArtAParams
        );
    }
    
    /**
     * Calculate adjusted vigilance for match tracking.
     * Returns the original vigilance plus epsilon to force search for new category.
     * 
     * @param currentVigilance the current vigilance value
     * @return adjusted vigilance value
     */
    public double adjustVigilance(double currentVigilance) {
        return Math.min(1.0, currentVigilance + epsilon);
    }
    
    /**
     * Check if the map field accepts a given match value.
     * 
     * @param matchValue the match value to test
     * @return true if matchValue >= mapFieldVigilance
     */
    public boolean acceptsMapping(double matchValue) {
        return matchValue >= mapFieldVigilance;
    }
    
    /**
     * Create a builder for VectorizedSimpleARTMAPParameters.
     * @return new VectorizedSimpleARTMAPParametersBuilder
     */
    public static VectorizedSimpleARTMAPParametersBuilder builder() {
        return new VectorizedSimpleARTMAPParametersBuilder();
    }
    
    /**
     * Builder class for VectorizedSimpleARTMAPParameters.
     */
    public static class VectorizedSimpleARTMAPParametersBuilder {
        private double mapFieldVigilance = 0.95;
        private double epsilon = 0.001;
        private boolean enableMatchTracking = true;
        private int maxSearchAttempts = 10;
        private VectorizedParameters artAParams;
        
        /**
         * Set the map field vigilance parameter.
         * @param mapFieldVigilance the map field vigilance ρ_map ∈ [0, 1]
         * @return this builder
         */
        public VectorizedSimpleARTMAPParametersBuilder mapFieldVigilance(double mapFieldVigilance) {
            this.mapFieldVigilance = mapFieldVigilance;
            return this;
        }
        
        /**
         * Set the epsilon parameter.
         * @param epsilon the epsilon value > 0
         * @return this builder
         */
        public VectorizedSimpleARTMAPParametersBuilder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }
        
        /**
         * Set the match tracking enabled flag.
         * @param enableMatchTracking whether to enable match tracking
         * @return this builder
         */
        public VectorizedSimpleARTMAPParametersBuilder enableMatchTracking(boolean enableMatchTracking) {
            this.enableMatchTracking = enableMatchTracking;
            return this;
        }
        
        /**
         * Set the maximum search attempts parameter.
         * @param maxSearchAttempts the maximum search attempts (> 0)
         * @return this builder
         */
        public VectorizedSimpleARTMAPParametersBuilder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        /**
         * Set the ART parameters.
         * @param artAParams the ART parameters
         * @return this builder
         */
        public VectorizedSimpleARTMAPParametersBuilder artAParams(VectorizedParameters artAParams) {
            this.artAParams = artAParams;
            return this;
        }
        
        /**
         * Build the VectorizedSimpleARTMAPParameters instance.
         * Uses default VectorizedParameters if not specified.
         * @return new VectorizedSimpleARTMAPParameters with specified values
         */
        public VectorizedSimpleARTMAPParameters build() {
            // Set defaults if not provided
            if (artAParams == null) {
                artAParams = VectorizedParameters.createDefault().withVigilance(0.7);
            }
            
            return new VectorizedSimpleARTMAPParameters(
                mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts, artAParams
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedSimpleARTMAPParameters{ρ_map=%.3f, ε=%.6f, matchTracking=%b, maxAttempts=%d}",
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts
        );
    }
}