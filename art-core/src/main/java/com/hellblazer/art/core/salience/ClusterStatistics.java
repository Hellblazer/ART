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
 * Maintains cluster-wise statistics using Welford's online algorithm
 */
public class ClusterStatistics {
    private final double[] featureMean;
    private final double[] featureFrequency;
    private final double[] featureM2; // For variance calculation (Welford's algorithm)
    private final int dimension;
    private int sampleCount;
    private static final double EPSILON = 1e-10;
    
    public ClusterStatistics(int dimension) {
        this.dimension = dimension;
        this.featureMean = new double[dimension];
        this.featureFrequency = new double[dimension];
        this.featureM2 = new double[dimension];
        this.sampleCount = 0;
    }
    
    public synchronized void updateStatistics(double[] input) {
        if (input.length != dimension) {
            throw new IllegalArgumentException("Input dimension mismatch: expected " + 
                                             dimension + ", got " + input.length);
        }
        
        sampleCount++;
        
        for (int i = 0; i < dimension; i++) {
            // Welford's online algorithm for mean and variance
            double delta = input[i] - featureMean[i];
            featureMean[i] += delta / sampleCount;
            double delta2 = input[i] - featureMean[i];
            featureM2[i] += delta * delta2;
            
            // Frequency update (count non-zero entries)
            if (Math.abs(input[i]) > EPSILON) {
                featureFrequency[i]++;
            }
        }
    }
    
    public void updateBatch(double[][] batch) {
        for (double[] sample : batch) {
            updateStatistics(sample);
        }
    }
    
    public int getDimension() {
        return dimension;
    }
    
    public int getSampleCount() {
        return sampleCount;
    }
    
    public double getFeatureMean(int index) {
        if (index < 0 || index >= dimension) {
            throw new IndexOutOfBoundsException("Index: " + index);
        }
        return featureMean[index];
    }
    
    public double getFeatureFrequency(int index) {
        if (index < 0 || index >= dimension) {
            throw new IndexOutOfBoundsException("Index: " + index);
        }
        return featureFrequency[index];
    }
    
    public double getFeatureVariance(int index) {
        if (index < 0 || index >= dimension) {
            throw new IndexOutOfBoundsException("Index: " + index);
        }
        if (sampleCount < 2) {
            return 0.0;
        }
        return featureM2[index] / sampleCount;
    }
    
    public double getFeatureStandardDeviation(int index) {
        return Math.sqrt(getFeatureVariance(index));
    }
    
    public double getCoefficientOfVariation(int index) {
        double mean = getFeatureMean(index);
        if (Math.abs(mean) < EPSILON) {
            return 0.0;
        }
        return getFeatureStandardDeviation(index) / Math.abs(mean);
    }
    
    public double getFrequencyRatio(int index) {
        if (sampleCount == 0) {
            return 0.0;
        }
        return featureFrequency[index] / sampleCount;
    }
    
    public double getInformationContent(int index) {
        if (sampleCount < 2) {
            return 0.0;
        }
        double frequency = getFrequencyRatio(index);
        double variance = getFeatureVariance(index);
        
        // Information content: high frequency, low variance = high information
        return frequency / (1.0 + variance);
    }
    
    public synchronized void reset() {
        Arrays.fill(featureMean, 0.0);
        Arrays.fill(featureFrequency, 0.0);
        Arrays.fill(featureM2, 0.0);
        sampleCount = 0;
    }
    
    public synchronized ClusterStatistics copy() {
        var copy = new ClusterStatistics(dimension);
        System.arraycopy(this.featureMean, 0, copy.featureMean, 0, dimension);
        System.arraycopy(this.featureFrequency, 0, copy.featureFrequency, 0, dimension);
        System.arraycopy(this.featureM2, 0, copy.featureM2, 0, dimension);
        copy.sampleCount = this.sampleCount;
        return copy;
    }
    
    public static ClusterStatistics merge(ClusterStatistics stats1, ClusterStatistics stats2) {
        if (stats1.dimension != stats2.dimension) {
            throw new IllegalArgumentException("Cannot merge statistics with different dimensions");
        }
        
        var merged = new ClusterStatistics(stats1.dimension);
        int n1 = stats1.sampleCount;
        int n2 = stats2.sampleCount;
        int n = n1 + n2;
        
        if (n == 0) {
            return merged;
        }
        
        merged.sampleCount = n;
        
        for (int i = 0; i < stats1.dimension; i++) {
            // Merge means
            merged.featureMean[i] = (n1 * stats1.featureMean[i] + n2 * stats2.featureMean[i]) / n;
            
            // Merge frequencies
            merged.featureFrequency[i] = stats1.featureFrequency[i] + stats2.featureFrequency[i];
            
            // Merge variances (parallel algorithm)
            double delta = stats2.featureMean[i] - stats1.featureMean[i];
            merged.featureM2[i] = stats1.featureM2[i] + stats2.featureM2[i] + 
                                 delta * delta * n1 * n2 / n;
        }
        
        return merged;
    }
    
    public String serialize() {
        // Simple serialization for testing
        StringBuilder sb = new StringBuilder();
        sb.append(dimension).append(";");
        sb.append(sampleCount).append(";");
        sb.append(Arrays.toString(featureMean)).append(";");
        sb.append(Arrays.toString(featureFrequency)).append(";");
        sb.append(Arrays.toString(featureM2));
        return sb.toString();
    }
    
    public static ClusterStatistics deserialize(String data) {
        String[] parts = data.split(";");
        int dim = Integer.parseInt(parts[0]);
        var stats = new ClusterStatistics(dim);
        stats.sampleCount = Integer.parseInt(parts[1]);
        
        // Parse arrays
        String meanStr = parts[2].replace("[", "").replace("]", "");
        String freqStr = parts[3].replace("[", "").replace("]", "");
        String m2Str = parts[4].replace("[", "").replace("]", "");
        
        String[] meanVals = meanStr.split(", ");
        String[] freqVals = freqStr.split(", ");
        String[] m2Vals = m2Str.split(", ");
        
        for (int i = 0; i < dim; i++) {
            stats.featureMean[i] = Double.parseDouble(meanVals[i]);
            stats.featureFrequency[i] = Double.parseDouble(freqVals[i]);
            stats.featureM2[i] = Double.parseDouble(m2Vals[i]);
        }
        
        return stats;
    }
    
    @Override
    public String toString() {
        return String.format("ClusterStatistics{dimension=%d, sampleCount=%d, meanRange=[%.3f, %.3f]}",
                           dimension, sampleCount,
                           Arrays.stream(featureMean).min().orElse(0),
                           Arrays.stream(featureMean).max().orElse(0));
    }
}