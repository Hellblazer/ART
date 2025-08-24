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
 * Parameters for ART-2 neural network.
 * 
 * ART-2 processes continuous analog inputs using:
 * - vigilance: Controls category granularity (0 < ρ ≤ 1)
 * - learningRate: Controls weight update plasticity (0 < β ≤ 1) 
 * - maxCategories: Maximum number of categories to create
 * 
 * @param vigilance the vigilance parameter ρ (0 < ρ ≤ 1)
 * @param learningRate the learning rate β (0 < β ≤ 1)
 * @param maxCategories maximum number of categories
 * 
 * @author Hal Hildebrand
 */
public record ART2Parameters(
    double vigilance,
    double learningRate,
    int maxCategories
) {
    
    /**
     * Create ART-2 parameters with validation.
     * 
     * @param vigilance the vigilance parameter (0 < ρ ≤ 1)
     * @param learningRate the learning rate (0 < β ≤ 1)  
     * @param maxCategories maximum categories (> 0)
     * @throws IllegalArgumentException if parameters are invalid
     */
    public ART2Parameters {
        if (vigilance <= 0.0 || vigilance > 1.0 || !Double.isFinite(vigilance)) {
            throw new IllegalArgumentException("vigilance must be in (0, 1], got: " + vigilance);
        }
        if (learningRate <= 0.0 || learningRate > 1.0 || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learning rate must be in (0, 1], got: " + learningRate);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("maxCategories must be positive, got: " + maxCategories);
        }
    }
}