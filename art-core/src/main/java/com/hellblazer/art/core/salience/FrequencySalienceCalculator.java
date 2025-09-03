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
 * Frequency-based salience calculator
 */
public class FrequencySalienceCalculator implements SalienceCalculator {
    
    @Override
    public double[] calculate(ClusterStatistics stats, SparseVector input) {
        int dimension = stats.getDimension();
        double[] salience = new double[dimension];
        
        if (stats.getSampleCount() == 0) {
            // Return uniform salience for empty statistics
            Arrays.fill(salience, 1.0 / dimension);
            return salience;
        }
        
        // Find maximum frequency for normalization
        double maxFreq = 0.0;
        for (int i = 0; i < dimension; i++) {
            maxFreq = Math.max(maxFreq, stats.getFeatureFrequency(i));
        }
        
        if (maxFreq == 0.0) {
            maxFreq = 1.0; // Avoid division by zero
        }
        
        for (int i = 0; i < dimension; i++) {
            // Higher frequency = higher salience
            salience[i] = stats.getFeatureFrequency(i) / maxFreq;
            
            // Apply smoothing to avoid zero salience
            // Using 0.9 weight for frequency, 0.1 minimum
            salience[i] = 0.1 + 0.9 * salience[i];
        }
        
        return salience;
    }
}