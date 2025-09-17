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

import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;
import com.hellblazer.art.core.parameters.HypersphereParameters;

import java.util.Objects;

/**
 * Immutable parameters for VectorizedHypersphereARTMAP supervised learning algorithm.
 * Extends SimpleARTMAP parameters with HypersphereART-specific settings for spherical clustering.
 * 
 * @param mapFieldVigilance the map field vigilance parameter (ρ_map) in range [0, 1]
 * @param epsilon small positive value for vigilance adjustment during match tracking
 * @param enableMatchTracking whether to enable match tracking algorithm
 * @param maxSearchAttempts maximum number of vigilance search attempts
 * @param defaultRadius default radius for new hypersphere clusters (r_hat)
 * @param adaptiveRadius whether to use adaptive radius adjustment
 * @param hypersphereParams vectorized parameters for the underlying HypersphereART module
 */
public record VectorizedHypersphereARTMAPParameters(
    double mapFieldVigilance,
    double epsilon,
    boolean enableMatchTracking,
    int maxSearchAttempts,
    double defaultRadius,
    boolean adaptiveRadius,
    VectorizedHypersphereParameters hypersphereParams
) {
    
    /**
     * Constructor with validation.
     */
    public VectorizedHypersphereARTMAPParameters {
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
        if (defaultRadius <= 0.0) {
            throw new IllegalArgumentException("Default radius must be positive, got: " + defaultRadius);
        }
        
        Objects.requireNonNull(hypersphereParams, "Hypersphere parameters cannot be null");
        
        // Check for NaN and infinite values
        if (Double.isNaN(mapFieldVigilance) || Double.isNaN(epsilon) || Double.isNaN(defaultRadius) ||
            Double.isInfinite(mapFieldVigilance) || Double.isInfinite(epsilon) || Double.isInfinite(defaultRadius)) {
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
     * Convert to HypersphereParameters for the underlying ART module.
     * @return equivalent HypersphereParameters
     */
    public HypersphereParameters toHypersphereParameters() {
        return new HypersphereParameters(hypersphereParams.vigilance(), defaultRadius, adaptiveRadius);
    }
    
    /**
     * Create VectorizedHypersphereARTMAPParameters with default values.
     * Default: mapFieldVigilance=0.95, epsilon=0.001, matchTracking=true, maxAttempts=10,
     *          defaultRadius=0.5, adaptiveRadius=true
     * @return default VectorizedHypersphereARTMAPParameters
     */
    public static VectorizedHypersphereARTMAPParameters defaults() {
        var defaultHypersphereParams = VectorizedHypersphereParameters.builder()
            .inputDimensions(100) // default input dimension
            .build();
        
        return new VectorizedHypersphereARTMAPParameters(
            0.95,   // mapFieldVigilance
            0.001,  // epsilon
            true,   // enableMatchTracking
            10,     // maxSearchAttempts
            0.5,    // defaultRadius
            true,   // adaptiveRadius
            defaultHypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different map field vigilance.
     * @param newMapFieldVigilance the new map field vigilance value
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withMapFieldVigilance(double newMapFieldVigilance) {
        return new VectorizedHypersphereARTMAPParameters(
            newMapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts, 
            defaultRadius, adaptiveRadius, hypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different epsilon.
     * @param newEpsilon the new epsilon value
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withEpsilon(double newEpsilon) {
        return new VectorizedHypersphereARTMAPParameters(
            mapFieldVigilance, newEpsilon, enableMatchTracking, maxSearchAttempts,
            defaultRadius, adaptiveRadius, hypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different match tracking setting.
     * @param newEnableMatchTracking the new match tracking setting
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withEnableMatchTracking(boolean newEnableMatchTracking) {
        return new VectorizedHypersphereARTMAPParameters(
            mapFieldVigilance, epsilon, newEnableMatchTracking, maxSearchAttempts,
            defaultRadius, adaptiveRadius, hypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different default radius.
     * @param newDefaultRadius the new default radius value
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withDefaultRadius(double newDefaultRadius) {
        return new VectorizedHypersphereARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            newDefaultRadius, adaptiveRadius, hypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different adaptive radius setting.
     * @param newAdaptiveRadius the new adaptive radius setting
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withAdaptiveRadius(boolean newAdaptiveRadius) {
        return new VectorizedHypersphereARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            defaultRadius, newAdaptiveRadius, hypersphereParams
        );
    }
    
    /**
     * Create a new VectorizedHypersphereARTMAPParameters with different hypersphere parameters.
     * @param newHypersphereParams the new hypersphere parameters
     * @return new VectorizedHypersphereARTMAPParameters instance
     */
    public VectorizedHypersphereARTMAPParameters withHypersphereParams(VectorizedHypersphereParameters newHypersphereParams) {
        return new VectorizedHypersphereARTMAPParameters(
            mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
            defaultRadius, adaptiveRadius, newHypersphereParams
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
     * Calculate the effective radius for a hypersphere cluster.
     * Returns the minimum of defaultRadius and current vigilance-based radius.
     * 
     * @param vigilanceRadius the radius implied by current vigilance
     * @return effective radius to use
     */
    public double effectiveRadius(double vigilanceRadius) {
        return adaptiveRadius ? Math.min(defaultRadius, vigilanceRadius) : defaultRadius;
    }
    
    /**
     * Create a builder for VectorizedHypersphereARTMAPParameters.
     * @return new VectorizedHypersphereARTMAPParametersBuilder
     */
    public static VectorizedHypersphereARTMAPParametersBuilder builder() {
        return new VectorizedHypersphereARTMAPParametersBuilder();
    }
    
    /**
     * Builder class for VectorizedHypersphereARTMAPParameters.
     */
    public static class VectorizedHypersphereARTMAPParametersBuilder {
        private double mapFieldVigilance = 0.95;
        private double epsilon = 0.001;
        private boolean enableMatchTracking = true;
        private int maxSearchAttempts = 10;
        private double defaultRadius = 0.5;
        private boolean adaptiveRadius = true;
        private VectorizedHypersphereParameters hypersphereParams;
        
        /**
         * Set the map field vigilance parameter.
         * @param mapFieldVigilance the map field vigilance ρ_map ∈ [0, 1]
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder mapFieldVigilance(double mapFieldVigilance) {
            this.mapFieldVigilance = mapFieldVigilance;
            return this;
        }
        
        /**
         * Set the epsilon parameter.
         * @param epsilon the epsilon value > 0
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }
        
        /**
         * Set the match tracking enabled flag.
         * @param enableMatchTracking whether to enable match tracking
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder enableMatchTracking(boolean enableMatchTracking) {
            this.enableMatchTracking = enableMatchTracking;
            return this;
        }
        
        /**
         * Set the maximum search attempts parameter.
         * @param maxSearchAttempts the maximum search attempts (> 0)
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        /**
         * Set the default radius parameter.
         * @param defaultRadius the default radius (> 0)
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder defaultRadius(double defaultRadius) {
            this.defaultRadius = defaultRadius;
            return this;
        }
        
        /**
         * Set the adaptive radius flag.
         * @param adaptiveRadius whether to use adaptive radius
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder adaptiveRadius(boolean adaptiveRadius) {
            this.adaptiveRadius = adaptiveRadius;
            return this;
        }
        
        /**
         * Set the hypersphere parameters.
         * @param hypersphereParams the hypersphere parameters
         * @return this builder
         */
        public VectorizedHypersphereARTMAPParametersBuilder hypersphereParams(VectorizedHypersphereParameters hypersphereParams) {
            this.hypersphereParams = hypersphereParams;
            return this;
        }
        
        /**
         * Build the VectorizedHypersphereARTMAPParameters instance.
         * Uses default VectorizedParameters if not specified.
         * @return new VectorizedHypersphereARTMAPParameters with specified values
         */
        public VectorizedHypersphereARTMAPParameters build() {
            // Set defaults if not provided
            if (hypersphereParams == null) {
                hypersphereParams = VectorizedHypersphereParameters.builder()
                    .inputDimensions(100) // default input dimension
                    .build();
            }
            
            return new VectorizedHypersphereARTMAPParameters(
                mapFieldVigilance, epsilon, enableMatchTracking, maxSearchAttempts,
                defaultRadius, adaptiveRadius, hypersphereParams
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedHypersphereARTMAPParameters{ρ_map=%.3f, ε=%.6f, matchTracking=%b, r_hat=%.3f, adaptive=%b}",
            mapFieldVigilance, epsilon, enableMatchTracking, defaultRadius, adaptiveRadius
        );
    }
}