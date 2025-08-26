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
package com.hellblazer.art.core.results;

import java.util.Map;

/**
 * Activation result for EllipsoidART with ellipsoid-specific information.
 * 
 * @param categoryIndex Index of the activated category
 * @param activationValue Activation value (typically based on Mahalanobis distance)
 * @param mahalanobisDistance Mahalanobis distance to ellipsoid center
 * @param ellipsoidVolume Volume of the ellipsoid
 * @param confidenceEllipse Parameters for confidence ellipse visualization
 * 
 * @author Hal Hildebrand
 */
public record EllipsoidActivationResult(
    int categoryIndex,
    double activationValue,
    double mahalanobisDistance,
    double ellipsoidVolume,
    Map<String, Object> confidenceEllipse
) implements ActivationResult {
    
    public EllipsoidActivationResult {
        if (categoryIndex < -1) {
            throw new IllegalArgumentException("categoryIndex must be >= -1, got: " + categoryIndex);
        }
        if (!Double.isFinite(activationValue) || activationValue < 0) {
            throw new IllegalArgumentException("activationValue must be finite and non-negative, got: " + activationValue);
        }
        if (!Double.isFinite(mahalanobisDistance) || mahalanobisDistance < 0) {
            throw new IllegalArgumentException("mahalanobisDistance must be finite and non-negative, got: " + mahalanobisDistance);
        }
        if (!Double.isFinite(ellipsoidVolume) || ellipsoidVolume <= 0) {
            throw new IllegalArgumentException("ellipsoidVolume must be finite and positive, got: " + ellipsoidVolume);
        }
        if (confidenceEllipse == null) {
            throw new IllegalArgumentException("confidenceEllipse cannot be null");
        }
    }
    
    /**
     * Get visualization data for the ellipsoid.
     * 
     * @return Map containing visualization parameters
     */
    public Map<String, Object> getVisualizationData() {
        return Map.of(
            "mahalanobisDistance", mahalanobisDistance,
            "ellipsoidVolume", ellipsoidVolume,
            "confidenceEllipse", confidenceEllipse,
            "categoryIndex", categoryIndex,
            "activationValue", activationValue
        );
    }
}