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

/**
 * Combined statistical salience calculator
 */
public class StatisticalSalienceCalculator implements SalienceCalculator {
    
    @Override
    public double[] calculate(ClusterStatistics stats, SparseVector input) {
        int dimension = stats.getDimension();
        double[] salience = new double[dimension];
        
        if (stats.getSampleCount() < 2) {
            // Return uniform salience for insufficient statistics
            for (int i = 0; i < dimension; i++) {
                salience[i] = 1.0 / dimension;
            }
            return salience;
        }
        
        for (int i = 0; i < dimension; i++) {
            // Combine multiple statistical measures
            double frequency = stats.getFrequencyRatio(i);
            double mean = stats.getFeatureMean(i);
            double variance = stats.getFeatureVariance(i);
            
            // Information content: low variance, high frequency = high salience
            double informationContent = frequency / (1.0 + variance);
            
            // Signal-to-noise ratio
            double snr = Math.abs(mean) / (Math.sqrt(variance) + 1e-10);
            
            // Combine measures with weights
            salience[i] = 0.4 * frequency +              // 40% frequency
                         0.3 * informationContent +      // 30% information content
                         0.3 * Math.tanh(snr);           // 30% SNR (bounded by tanh)
        }
        
        return salience;
    }
}