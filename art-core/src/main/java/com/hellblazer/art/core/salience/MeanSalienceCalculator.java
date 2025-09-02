/*
 * Copyright (c) 2024 Hal Hildebrand. All rights reserved.
 * 
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core.salience;

import java.util.Arrays;

/**
 * Mean-based salience calculator with variance weighting
 */
public class MeanSalienceCalculator implements SalienceCalculator {
    
    @Override
    public double[] calculate(ClusterStatistics stats, SparseVector input) {
        int dimension = stats.getDimension();
        double[] salience = new double[dimension];
        
        if (stats.getSampleCount() < 2) {
            // Return uniform salience for insufficient statistics
            Arrays.fill(salience, 1.0 / dimension);
            return salience;
        }
        
        for (int i = 0; i < dimension; i++) {
            // Distance from mean indicates salience
            double distance = Math.abs(input.get(i) - stats.getFeatureMean(i));
            double normalizedDistance = distance / (1.0 + distance);
            
            // Inverse distance: closer to mean = higher salience for stable features
            salience[i] = 1.0 - normalizedDistance;
            
            // Weight by feature variance (low variance = more reliable)
            double variance = stats.getFeatureVariance(i);
            salience[i] *= Math.exp(-variance);
        }
        
        // Normalize to sum to 1.0
        double sum = Arrays.stream(salience).sum();
        if (sum > 0) {
            for (int i = 0; i < salience.length; i++) {
                salience[i] /= sum;
            }
        } else {
            // Fallback to uniform if all zeros
            Arrays.fill(salience, 1.0 / dimension);
        }
        
        return salience;
    }
}