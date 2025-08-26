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
package com.hellblazer.art.core;

/**
 * Base interface for ARTMAP variants used in hierarchical learning.
 * 
 * This interface defines the common operations that all ARTMAP-style
 * learning systems must support, including training state management,
 * category counting, and basic lifecycle operations.
 * 
 * @author Hal Hildebrand
 */
public interface BaseARTMAP {
    
    /**
     * Check if this ARTMAP has been trained.
     * 
     * @return true if trained, false otherwise
     */
    boolean isTrained();
    
    /**
     * Get the number of categories created during training.
     * 
     * @return the number of categories
     */
    int getCategoryCount();
    
    /**
     * Clear all learned patterns and reset to untrained state.
     */
    void clear();
}