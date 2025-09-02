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
package com.hellblazer.art.core.visualization;

import java.util.Optional;

/**
 * Interface for ART algorithms that support visualization capabilities.
 * 
 * This interface is designed for zero overhead when visualization is not used:
 * - Default methods return empty/false values (JIT optimizes away)
 * - No data is collected unless explicitly enabled
 * - No performance impact on core algorithm execution
 * 
 * The interface allows ART algorithms to optionally provide visualization data
 * without any performance penalty when visualization is disabled (default).
 * 
 * @author Hal Hildebrand
 */
public interface Visualizable {
    
    /**
     * Get visualization data for the current state of the algorithm.
     * 
     * By default, this returns empty, meaning no visualization data is available.
     * Implementations may override to provide actual visualization data when
     * visualization is enabled.
     * 
     * @return optional visualization data, empty by default
     */
    default Optional<VisualizationData> getVisualizationData() {
        return Optional.empty(); // JIT will optimize this away when not overridden
    }
    
    /**
     * Check if visualization is currently enabled for this algorithm.
     * 
     * @return false by default (visualization disabled)
     */
    default boolean isVisualizationEnabled() {
        return false; // Default: no visualization overhead
    }
    
    /**
     * Enable or disable visualization data collection.
     * 
     * By default, this is a no-op. Implementations may override to actually
     * enable/disable visualization data collection.
     * 
     * When enabled, algorithms may collect additional data for visualization
     * purposes, which may have a small performance impact. When disabled (default),
     * there should be zero performance overhead.
     * 
     * @param enabled true to enable visualization, false to disable
     */
    default void setVisualizationEnabled(boolean enabled) {
        // No-op by default - JIT will optimize this away
    }
    
    /**
     * Update visualization data after an algorithm step.
     * 
     * This method is called by algorithms after significant steps (e.g., pattern
     * processing, weight updates) to update visualization data. By default, this
     * is a no-op with zero overhead.
     * 
     * Implementations may override to collect visualization data when enabled.
     */
    default void updateVisualizationData() {
        // No-op by default - zero overhead
    }
    
    /**
     * Clear any collected visualization data.
     * 
     * Called when the algorithm is reset or cleared. By default, this is a no-op.
     */
    default void clearVisualizationData() {
        // No-op by default
    }
    
    /**
     * Get a human-readable description of the visualization capabilities.
     * 
     * @return description of what can be visualized, empty by default
     */
    default String getVisualizationDescription() {
        return "No visualization available";
    }
}