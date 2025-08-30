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
package com.hellblazer.art.core.parameters;

/**
 * Parameters for SimpleARTMAP algorithm.
 * 
 * SimpleARTMAP is a simplified version of ARTMAP for classification that uses:
 * - A single ART module for clustering input patterns
 * - A map field that maintains many-to-one mappings from clusters to labels
 * - Match tracking to handle label conflicts
 * 
 * @param mapFieldVigilance vigilance parameter for map field validation (ρ_map) in range [0, 1]
 * @param epsilon small positive value for vigilance adjustment during match tracking
 */
public record SimpleARTMAPParameters(
    double mapFieldVigilance,
    double epsilon
) {
    
    /**
     * Constructor with validation.
     */
    public SimpleARTMAPParameters {
        // Validate map field vigilance
        if (mapFieldVigilance < 0.0 || mapFieldVigilance > 1.0) {
            throw new IllegalArgumentException(
                "Map field vigilance must be in range [0, 1], got: " + mapFieldVigilance);
        }
        
        // Validate epsilon
        if (epsilon <= 0.0) {
            throw new IllegalArgumentException(
                "Epsilon must be positive, got: " + epsilon);
        }
        
        if (epsilon > 0.1) {
            throw new IllegalArgumentException(
                "Epsilon should be small (typically < 0.1), got: " + epsilon);
        }
        
        // Check for NaN values
        if (Double.isNaN(mapFieldVigilance) || Double.isNaN(epsilon)) {
            throw new IllegalArgumentException("Parameters cannot be NaN");
        }
        
        // Check for infinite values
        if (Double.isInfinite(mapFieldVigilance) || Double.isInfinite(epsilon)) {
            throw new IllegalArgumentException("Parameters cannot be infinite");
        }
    }
    
    /**
     * Create SimpleARTMAPParameters with specified values.
     * 
     * @param mapFieldVigilance map field vigilance ρ_map ∈ [0, 1]
     * @param epsilon vigilance adjustment epsilon > 0
     * @return new SimpleARTMAPParameters instance
     */
    public static SimpleARTMAPParameters of(double mapFieldVigilance, double epsilon) {
        return new SimpleARTMAPParameters(mapFieldVigilance, epsilon);
    }
    
    /**
     * Create SimpleARTMAPParameters with default values.
     * Default: mapFieldVigilance=0.95, epsilon=0.001
     * 
     * @return default SimpleARTMAPParameters
     */
    public static SimpleARTMAPParameters defaults() {
        return new SimpleARTMAPParameters(0.95, 0.001);
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
     * Create a new SimpleARTMAPParameters with different map field vigilance.
     * 
     * @param newMapFieldVigilance the new map field vigilance value
     * @return new SimpleARTMAPParameters instance
     */
    public SimpleARTMAPParameters withMapFieldVigilance(double newMapFieldVigilance) {
        return new SimpleARTMAPParameters(newMapFieldVigilance, epsilon);
    }
    
    /**
     * Create a new SimpleARTMAPParameters with different epsilon.
     * 
     * @param newEpsilon the new epsilon value
     * @return new SimpleARTMAPParameters instance
     */
    public SimpleARTMAPParameters withEpsilon(double newEpsilon) {
        return new SimpleARTMAPParameters(mapFieldVigilance, newEpsilon);
    }
    
    @Override
    public String toString() {
        return String.format(
            "SimpleARTMAPParameters{ρ_map=%.3f, ε=%.6f}", 
            mapFieldVigilance, epsilon);
    }
}