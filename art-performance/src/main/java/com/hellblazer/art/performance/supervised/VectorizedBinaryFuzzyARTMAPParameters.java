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
package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;

import java.util.Objects;

/**
 * Immutable parameters for VectorizedBinaryFuzzyARTMAP supervised learning algorithm.
 * Extends SimpleARTMAP parameters with BinaryFuzzyART-specific settings for binary pattern processing.
 * 
 * @param mapFieldVigilance the map field vigilance parameter (ρ_map) in range [0, 1]
 * @param epsilon small positive value for vigilance adjustment during match tracking
 * @param enableMatchTracking whether to enable match tracking algorithm
 * @param maxSearchAttempts maximum number of vigilance search attempts
 * @param binaryThreshold threshold for binary pattern detection (0.0 to 1.0)
 * @param enableComplementCoding whether to enable automatic complement coding
 * @param binaryFuzzyParams vectorized parameters for the underlying BinaryFuzzyART module
 */
public record VectorizedBinaryFuzzyARTMAPParameters(
    double mapFieldVigilance,
    double epsilon,
    boolean enableMatchTracking,
    int maxSearchAttempts,
    double binaryThreshold,
    boolean enableComplementCoding,
    VectorizedParameters binaryFuzzyParams
) {
    
    /**
     * Constructor with validation.
     */
    public VectorizedBinaryFuzzyARTMAPParameters {
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
        if (binaryThreshold < 0.0 || binaryThreshold > 1.0) {
            throw new IllegalArgumentException("Binary threshold must be in range [0, 1], got: " + binaryThreshold);
        }
        
        Objects.requireNonNull(binaryFuzzyParams, "Binary fuzzy parameters cannot be null");
        
        // Check for NaN and infinite values
        if (Double.isNaN(mapFieldVigilance) || Double.isNaN(epsilon) || Double.isNaN(binaryThreshold) ||
            Double.isInfinite(mapFieldVigilance) || Double.isInfinite(epsilon) || Double.isInfinite(binaryThreshold)) {
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
     * Create VectorizedBinaryFuzzyARTMAPParameters with default values.
     * Default: mapFieldVigilance=0.95, epsilon=0.001, matchTracking=true, maxAttempts=10,
     *          binaryThreshold=0.95, complementCoding=true
     * @return default VectorizedBinaryFuzzyARTMAPParameters
     */
    public static VectorizedBinaryFuzzyARTMAPParameters defaults() {
        var defaultBinaryFuzzyParams = VectorizedParameters.createDefault();
        
        return new VectorizedBinaryFuzzyARTMAPParameters(
            0.95,   // mapFieldVigilance
            0.001,  // epsilon
            true,   // enableMatchTracking
            10,     // maxSearchAttempts
            0.95,   // binaryThreshold - 95% binary values to trigger optimization
            true,   // enableComplementCoding
            defaultBinaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different map field vigilance.
     * @param newMapFieldVigilance the new map field vigilance value
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withMapFieldVigilance(double newMapFieldVigilance) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            newMapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts, 
            binaryThreshold, enableComplementCoding, binaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different epsilon.
     * @param newEpsilon the new epsilon value
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withEpsilon(double newEpsilon) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            mapFieldVigilance, newEpsilon, enableMatchTracking, maxSearchAttempts,
            binaryThreshold, enableComplementCoding, binaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different match tracking setting.
     * @param newEnableMatchTracking the new match tracking setting
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withEnableMatchTracking(boolean newEnableMatchTracking) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            mapFieldVigilance, epsilon, newEnableMatchTracking, maxSearchAttempts,
            binaryThreshold, enableComplementCoding, binaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different binary threshold.
     * @param newBinaryThreshold the new binary threshold value
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withBinaryThreshold(double newBinaryThreshold) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            newBinaryThreshold, enableComplementCoding, binaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different complement coding setting.
     * @param newEnableComplementCoding the new complement coding setting
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withEnableComplementCoding(boolean newEnableComplementCoding) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            binaryThreshold, newEnableComplementCoding, binaryFuzzyParams
        );
    }
    
    /**
     * Create a new VectorizedBinaryFuzzyARTMAPParameters with different binary fuzzy parameters.
     * @param newBinaryFuzzyParams the new binary fuzzy parameters
     * @return new VectorizedBinaryFuzzyARTMAPParameters instance
     */
    public VectorizedBinaryFuzzyARTMAPParameters withBinaryFuzzyParams(VectorizedParameters newBinaryFuzzyParams) {
        return new VectorizedBinaryFuzzyARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            binaryThreshold, enableComplementCoding, newBinaryFuzzyParams
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
     * Check if a pattern is considered binary based on the threshold.
     * 
     * @param pattern the input pattern to check
     * @return true if pattern has >= binaryThreshold proportion of binary values
     */
    public boolean isBinaryPattern(double[] pattern) {
        if (pattern == null || pattern.length == 0) {
            return false;
        }
        
        int binaryCount = 0;
        for (double value : pattern) {
            if (value == 0.0 || value == 1.0) {
                binaryCount++;
            }
        }
        
        return (double) binaryCount / pattern.length >= binaryThreshold;
    }
    
    /**
     * Apply complement coding to a binary pattern if enabled.
     * 
     * @param pattern the input pattern
     * @return complement-coded pattern if enabled, otherwise original pattern
     */
    public double[] applyComplementCoding(double[] pattern) {
        if (!enableComplementCoding) {
            return pattern.clone();
        }
        
        int originalDim = pattern.length;
        double[] complementCoded = new double[originalDim * 2];
        
        // First half: original data (ensure binary if threshold met)
        for (int i = 0; i < originalDim; i++) {
            if (isBinaryPattern(pattern)) {
                complementCoded[i] = (pattern[i] > 0.5) ? 1.0 : 0.0;
            } else {
                complementCoded[i] = pattern[i];
            }
        }
        
        // Second half: complement
        for (int i = 0; i < originalDim; i++) {
            complementCoded[originalDim + i] = 1.0 - complementCoded[i];
        }
        
        return complementCoded;
    }
    
    /**
     * Create a builder for VectorizedBinaryFuzzyARTMAPParameters.
     * @return new VectorizedBinaryFuzzyARTMAPParametersBuilder
     */
    public static VectorizedBinaryFuzzyARTMAPParametersBuilder builder() {
        return new VectorizedBinaryFuzzyARTMAPParametersBuilder();
    }
    
    /**
     * Builder class for VectorizedBinaryFuzzyARTMAPParameters.
     */
    public static class VectorizedBinaryFuzzyARTMAPParametersBuilder {
        private double mapFieldVigilance = 0.95;
        private double epsilon = 0.001;
        private boolean enableMatchTracking = true;
        private int maxSearchAttempts = 10;
        private double binaryThreshold = 0.95;
        private boolean enableComplementCoding = true;
        private VectorizedParameters binaryFuzzyParams;
        
        /**
         * Set the map field vigilance parameter.
         * @param mapFieldVigilance the map field vigilance ρ_map ∈ [0, 1]
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder mapFieldVigilance(double mapFieldVigilance) {
            this.mapFieldVigilance = mapFieldVigilance;
            return this;
        }
        
        /**
         * Set the epsilon parameter.
         * @param epsilon the epsilon value > 0
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }
        
        /**
         * Set the match tracking enabled flag.
         * @param enableMatchTracking whether to enable match tracking
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder enableMatchTracking(boolean enableMatchTracking) {
            this.enableMatchTracking = enableMatchTracking;
            return this;
        }
        
        /**
         * Set the maximum search attempts parameter.
         * @param maxSearchAttempts the maximum search attempts (> 0)
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        /**
         * Set the binary threshold parameter.
         * @param binaryThreshold the binary threshold (0.0 to 1.0)
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder binaryThreshold(double binaryThreshold) {
            this.binaryThreshold = binaryThreshold;
            return this;
        }
        
        /**
         * Set the complement coding enabled flag.
         * @param enableComplementCoding whether to enable complement coding
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder enableComplementCoding(boolean enableComplementCoding) {
            this.enableComplementCoding = enableComplementCoding;
            return this;
        }
        
        /**
         * Set the binary fuzzy parameters.
         * @param binaryFuzzyParams the binary fuzzy parameters
         * @return this builder
         */
        public VectorizedBinaryFuzzyARTMAPParametersBuilder binaryFuzzyParams(VectorizedParameters binaryFuzzyParams) {
            this.binaryFuzzyParams = binaryFuzzyParams;
            return this;
        }
        
        /**
         * Build the VectorizedBinaryFuzzyARTMAPParameters instance.
         * Uses default VectorizedParameters if not specified.
         * @return new VectorizedBinaryFuzzyARTMAPParameters with specified values
         */
        public VectorizedBinaryFuzzyARTMAPParameters build() {
            // Set defaults if not provided
            if (binaryFuzzyParams == null) {
                binaryFuzzyParams = VectorizedParameters.createDefault();
            }
            
            return new VectorizedBinaryFuzzyARTMAPParameters(
                mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
                binaryThreshold, enableComplementCoding, binaryFuzzyParams
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedBinaryFuzzyARTMAPParameters{ρ_map=%.3f, ε=%.6f, matchTracking=%b, binaryThreshold=%.3f, complement=%b}",
            mapFieldVigilance, epsilon, enableMatchTracking, binaryThreshold, enableComplementCoding
        );
    }
}