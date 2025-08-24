package com.hellblazer.art.core;

/**
 * Immutable parameters for ARTMAP supervised learning algorithm.
 * ARTMAP uses two ART modules (ARTa and ARTb) connected by a map field.
 * 
 * @param mapVigilance the map field vigilance parameter (ρab) in range [0, 1]
 * @param baselineVigilance the baseline vigilance for ARTa reset operations
 */
public record ARTMAPParameters(double mapVigilance, double baselineVigilance) {
    
    /**
     * Constructor with validation.
     */
    public ARTMAPParameters {
        if (mapVigilance < 0.0 || mapVigilance > 1.0) {
            throw new IllegalArgumentException("Map vigilance must be in range [0, 1], got: " + mapVigilance);
        }
        if (baselineVigilance < 0.0 || baselineVigilance > 1.0) {
            throw new IllegalArgumentException("Baseline vigilance must be in range [0, 1], got: " + baselineVigilance);
        }
    }
    
    /**
     * Create ARTMAPParameters with specified values.
     * @param mapVigilance the map field vigilance ρab ∈ [0, 1]
     * @param baselineVigilance the baseline vigilance for resets ∈ [0, 1]
     * @return new ARTMAPParameters instance
     */
    public static ARTMAPParameters of(double mapVigilance, double baselineVigilance) {
        return new ARTMAPParameters(mapVigilance, baselineVigilance);
    }
    
    /**
     * Create ARTMAPParameters with default values.
     * Default: mapVigilance=0.9, baselineVigilance=0.0
     * @return default ARTMAPParameters
     */
    public static ARTMAPParameters defaults() {
        return new ARTMAPParameters(0.9, 0.0);
    }
    
    /**
     * Create a new ARTMAPParameters with different map vigilance.
     * @param newMapVigilance the new map vigilance value
     * @return new ARTMAPParameters instance
     */
    public ARTMAPParameters withMapVigilance(double newMapVigilance) {
        return new ARTMAPParameters(newMapVigilance, baselineVigilance);
    }
    
    /**
     * Create a new ARTMAPParameters with different baseline vigilance.
     * @param newBaselineVigilance the new baseline vigilance value
     * @return new ARTMAPParameters instance
     */
    public ARTMAPParameters withBaselineVigilance(double newBaselineVigilance) {
        return new ARTMAPParameters(mapVigilance, newBaselineVigilance);
    }
    
    /**
     * Create a builder for ARTMAPParameters.
     * @return new ARTMAPParametersBuilder
     */
    public static ARTMAPParametersBuilder builder() {
        return new ARTMAPParametersBuilder();
    }
    
    /**
     * Builder class for ARTMAPParameters.
     */
    public static class ARTMAPParametersBuilder {
        private double mapVigilance = 0.9;
        private double baselineVigilance = 0.0;
        
        /**
         * Set the map field vigilance parameter.
         * @param mapVigilance the map vigilance ρab ∈ [0, 1]
         * @return this builder
         */
        public ARTMAPParametersBuilder mapVigilance(double mapVigilance) {
            this.mapVigilance = mapVigilance;
            return this;
        }
        
        /**
         * Set the baseline vigilance parameter.
         * @param baselineVigilance the baseline vigilance ∈ [0, 1]
         * @return this builder
         */
        public ARTMAPParametersBuilder baselineVigilance(double baselineVigilance) {
            this.baselineVigilance = baselineVigilance;
            return this;
        }
        
        /**
         * Build the ARTMAPParameters instance.
         * @return new ARTMAPParameters with specified values
         */
        public ARTMAPParameters build() {
            return new ARTMAPParameters(mapVigilance, baselineVigilance);
        }
    }
    
    @Override
    public String toString() {
        return String.format("ARTMAPParameters{ρab=%.3f, baseline=%.3f}", 
                           mapVigilance, baselineVigilance);
    }
}