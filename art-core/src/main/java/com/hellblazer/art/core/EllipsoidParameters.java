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
 * Parameters for EllipsoidART - ellipsoidal category regions with Mahalanobis distance.
 * 
 * @param vigilance Vigilance parameter (0 < rho <= 1)
 * @param learningRate Learning rate for weight updates (0 < beta <= 1)
 * @param dimensions Number of input dimensions
 * @param minVariance Minimum variance for regularization (prevents numerical issues)
 * @param maxVariance Maximum variance to prevent ellipsoid explosion
 * @param shapeAdaptationRate Rate of ellipsoid shape adaptation (> 0)
 * @param maxCategories Maximum number of categories allowed
 * 
 * @author Hal Hildebrand
 */
public record EllipsoidParameters(
    double vigilance,
    double learningRate, 
    int dimensions,
    double minVariance,
    double maxVariance,
    double shapeAdaptationRate,
    int maxCategories
) {
    
    public EllipsoidParameters {
        // Validate vigilance parameter
        if (vigilance <= 0.0 || vigilance > 1.0 || !Double.isFinite(vigilance)) {
            throw new IllegalArgumentException("vigilance must be in (0, 1], got: " + vigilance);
        }
        
        // Validate learning rate
        if (learningRate <= 0.0 || learningRate > 1.0 || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learning rate must be in (0, 1], got: " + learningRate);
        }
        
        // Validate dimensions
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions must be positive, got: " + dimensions);
        }
        
        // Validate variance bounds
        if (minVariance >= maxVariance) {
            throw new IllegalArgumentException("minVariance must be less than maxVariance, got: " + 
                                             minVariance + " >= " + maxVariance);
        }
        
        if (minVariance <= 0.0 || !Double.isFinite(minVariance)) {
            throw new IllegalArgumentException("minVariance must be positive and finite, got: " + minVariance);
        }
        
        if (maxVariance <= 0.0 || !Double.isFinite(maxVariance)) {
            throw new IllegalArgumentException("maxVariance must be positive and finite, got: " + maxVariance);
        }
        
        // Validate shape adaptation rate
        if (shapeAdaptationRate <= 0.0 || !Double.isFinite(shapeAdaptationRate)) {
            throw new IllegalArgumentException("shape adaptation rate must be positive and finite, got: " + shapeAdaptationRate);
        }
        
        // Validate max categories
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("maxCategories must be positive, got: " + maxCategories);
        }
    }
}