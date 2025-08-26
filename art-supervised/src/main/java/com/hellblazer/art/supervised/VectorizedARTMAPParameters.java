package com.hellblazer.art.supervised;

import com.hellblazer.art.algorithms.VectorizedParameters;
import com.hellblazer.art.core.artmap.ARTMAPParameters;

import java.util.Objects;

/**
 * Immutable parameters for VectorizedARTMAP supervised learning algorithm.
 * Extends ARTMAPParameters with vectorization-specific settings and type-safe parameter handling.
 * 
 * @param mapVigilance the map field vigilance parameter (ρab) in range [0, 1]
 * @param baselineVigilance the baseline vigilance for ARTa reset operations
 * @param vigilanceIncrement the increment for vigilance search during match tracking
 * @param maxVigilance the maximum vigilance to prevent infinite search
 * @param enableMatchTracking whether to enable match tracking algorithm
 * @param enableParallelSearch whether to enable parallel vigilance search
 * @param maxSearchAttempts maximum number of vigilance search attempts
 * @param artAParams type-safe parameters for ARTa (VectorizedParameters)
 * @param artBParams type-safe parameters for ARTb (VectorizedParameters)
 */
public record VectorizedARTMAPParameters(
    double mapVigilance,
    double baselineVigilance,
    double vigilanceIncrement,
    double maxVigilance,
    boolean enableMatchTracking,
    boolean enableParallelSearch,
    int maxSearchAttempts,
    VectorizedParameters artAParams,
    VectorizedParameters artBParams
) {
    
    /**
     * Constructor with validation.
     */
    public VectorizedARTMAPParameters {
        if (mapVigilance < 0.0 || mapVigilance > 1.0) {
            throw new IllegalArgumentException("Map vigilance must be in range [0, 1], got: " + mapVigilance);
        }
        if (baselineVigilance < 0.0 || baselineVigilance > 1.0) {
            throw new IllegalArgumentException("Baseline vigilance must be in range [0, 1], got: " + baselineVigilance);
        }
        if (vigilanceIncrement <= 0.0 || vigilanceIncrement > 1.0) {
            throw new IllegalArgumentException("Vigilance increment must be in range (0, 1], got: " + vigilanceIncrement);
        }
        if (maxVigilance < 0.0 || maxVigilance > 1.0) {
            throw new IllegalArgumentException("Max vigilance must be in range [0, 1], got: " + maxVigilance);
        }
        if (maxVigilance < mapVigilance) {
            throw new IllegalArgumentException("Max vigilance (" + maxVigilance + 
                ") must be >= map vigilance (" + mapVigilance + ")");
        }
        if (maxSearchAttempts <= 0) {
            throw new IllegalArgumentException("Max search attempts must be positive, got: " + maxSearchAttempts);
        }
        
        Objects.requireNonNull(artAParams, "ARTa parameters cannot be null");
        Objects.requireNonNull(artBParams, "ARTb parameters cannot be null");
    }
    
    /**
     * Convert to base ARTMAPParameters for backward compatibility.
     * @return equivalent ARTMAPParameters
     */
    public ARTMAPParameters toARTMAPParameters() {
        return ARTMAPParameters.of(mapVigilance, baselineVigilance);
    }
    
    /**
     * Create VectorizedARTMAPParameters with default values.
     * @return default VectorizedARTMAPParameters
     */
    public static VectorizedARTMAPParameters defaults() {
        var defaultArtAParams = VectorizedParameters.createDefault().withVigilance(0.7);
        var defaultArtBParams = VectorizedParameters.createDefault().withVigilance(0.8);
            
        return new VectorizedARTMAPParameters(
            0.9,    // mapVigilance
            0.0,    // baselineVigilance
            0.05,   // vigilanceIncrement
            0.95,   // maxVigilance
            true,   // enableMatchTracking
            false,  // enableParallelSearch
            10,     // maxSearchAttempts
            defaultArtAParams,
            defaultArtBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different map vigilance.
     * @param newMapVigilance the new map vigilance value
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withMapVigilance(double newMapVigilance) {
        return new VectorizedARTMAPParameters(
            newMapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different baseline vigilance.
     * @param newBaselineVigilance the new baseline vigilance value
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withBaselineVigilance(double newBaselineVigilance) {
        return new VectorizedARTMAPParameters(
            mapVigilance, newBaselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different vigilance increment.
     * @param newVigilanceIncrement the new vigilance increment value
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withVigilanceIncrement(double newVigilanceIncrement) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, newVigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different max vigilance.
     * @param newMaxVigilance the new max vigilance value
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withMaxVigilance(double newMaxVigilance) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, newMaxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different match tracking setting.
     * @param newEnableMatchTracking the new match tracking setting
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withEnableMatchTracking(boolean newEnableMatchTracking) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            newEnableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different parallel search setting.
     * @param newEnableParallelSearch the new parallel search setting
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withEnableParallelSearch(boolean newEnableParallelSearch) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, newEnableParallelSearch, maxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different max search attempts.
     * @param newMaxSearchAttempts the new max search attempts value
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withMaxSearchAttempts(int newMaxSearchAttempts) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, newMaxSearchAttempts,
            artAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different ARTa parameters.
     * @param newArtAParams the new ARTa parameters
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withArtAParams(VectorizedParameters newArtAParams) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            newArtAParams, artBParams
        );
    }
    
    /**
     * Create a new VectorizedARTMAPParameters with different ARTb parameters.
     * @param newArtBParams the new ARTb parameters
     * @return new VectorizedARTMAPParameters instance
     */
    public VectorizedARTMAPParameters withArtBParams(VectorizedParameters newArtBParams) {
        return new VectorizedARTMAPParameters(
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts,
            artAParams, newArtBParams
        );
    }
    
    /**
     * Create a builder for VectorizedARTMAPParameters.
     * @return new VectorizedARTMAPParametersBuilder
     */
    public static VectorizedARTMAPParametersBuilder builder() {
        return new VectorizedARTMAPParametersBuilder();
    }
    
    /**
     * Builder class for VectorizedARTMAPParameters.
     */
    public static class VectorizedARTMAPParametersBuilder {
        private double mapVigilance = 0.9;
        private double baselineVigilance = 0.0;
        private double vigilanceIncrement = 0.05;
        private double maxVigilance = 0.95;
        private boolean enableMatchTracking = true;
        private boolean enableParallelSearch = false;
        private int maxSearchAttempts = 10;
        private VectorizedParameters artAParams;
        private VectorizedParameters artBParams;
        
        /**
         * Set the map field vigilance parameter.
         * @param mapVigilance the map vigilance ρab ∈ [0, 1]
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder mapVigilance(double mapVigilance) {
            this.mapVigilance = mapVigilance;
            return this;
        }
        
        /**
         * Set the baseline vigilance parameter.
         * @param baselineVigilance the baseline vigilance ∈ [0, 1]
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder baselineVigilance(double baselineVigilance) {
            this.baselineVigilance = baselineVigilance;
            return this;
        }
        
        /**
         * Set the vigilance increment parameter.
         * @param vigilanceIncrement the vigilance increment ∈ (0, 1]
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder vigilanceIncrement(double vigilanceIncrement) {
            this.vigilanceIncrement = vigilanceIncrement;
            return this;
        }
        
        /**
         * Set the maximum vigilance parameter.
         * @param maxVigilance the maximum vigilance ∈ [0, 1]
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder maxVigilance(double maxVigilance) {
            this.maxVigilance = maxVigilance;
            return this;
        }
        
        /**
         * Set the match tracking enabled flag.
         * @param enableMatchTracking whether to enable match tracking
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder enableMatchTracking(boolean enableMatchTracking) {
            this.enableMatchTracking = enableMatchTracking;
            return this;
        }
        
        /**
         * Set the parallel search enabled flag.
         * @param enableParallelSearch whether to enable parallel search
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder enableParallelSearch(boolean enableParallelSearch) {
            this.enableParallelSearch = enableParallelSearch;
            return this;
        }
        
        /**
         * Set the maximum search attempts parameter.
         * @param maxSearchAttempts the maximum search attempts (> 0)
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        /**
         * Set the ARTa parameters.
         * @param artAParams the ARTa parameters
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder artAParams(VectorizedParameters artAParams) {
            this.artAParams = artAParams;
            return this;
        }
        
        /**
         * Set the ARTb parameters.
         * @param artBParams the ARTb parameters
         * @return this builder
         */
        public VectorizedARTMAPParametersBuilder artBParams(VectorizedParameters artBParams) {
            this.artBParams = artBParams;
            return this;
        }
        
        /**
         * Build the VectorizedARTMAPParameters instance.
         * Uses default VectorizedParameters if not specified.
         * @return new VectorizedARTMAPParameters with specified values
         */
        public VectorizedARTMAPParameters build() {
            // Set defaults if not provided
            if (artAParams == null) {
                artAParams = VectorizedParameters.createDefault().withVigilance(0.7);
            }
            if (artBParams == null) {
                artBParams = VectorizedParameters.createDefault().withVigilance(0.8);
            }
            
            return new VectorizedARTMAPParameters(
                mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
                enableMatchTracking, enableParallelSearch, maxSearchAttempts,
                artAParams, artBParams
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedARTMAPParameters{ρab=%.3f, baseline=%.3f, increment=%.3f, " +
            "maxVigilance=%.3f, matchTracking=%b, parallelSearch=%b, maxAttempts=%d}",
            mapVigilance, baselineVigilance, vigilanceIncrement, maxVigilance,
            enableMatchTracking, enableParallelSearch, maxSearchAttempts
        );
    }
}