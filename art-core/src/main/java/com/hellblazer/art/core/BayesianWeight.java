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

import java.util.Objects;

/**
 * BayesianWeight represents a category in Bayesian ART networks.
 * Stores multivariate Gaussian parameters including mean, covariance, and sample statistics.
 * 
 * @author Hal Hildebrand
 */
public record BayesianWeight(
    DenseVector mean,
    Matrix covariance,
    long sampleCount,
    double precision
) implements WeightVector {
    
    public BayesianWeight {
        Objects.requireNonNull(mean, "Mean cannot be null");
        Objects.requireNonNull(covariance, "Covariance cannot be null");
        
        if (sampleCount < 0) {
            throw new IllegalArgumentException("Sample count cannot be negative: " + sampleCount);
        }
        if (precision <= 0.0) {
            throw new IllegalArgumentException("Precision must be positive: " + precision);
        }
    }
    
    @Override
    public double get(int index) {
        return mean.get(index);
    }
    
    @Override
    public int dimension() {
        return mean.dimension();
    }
    
    @Override
    public double l1Norm() {
        return mean.l1Norm();
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        if (input == null) {
            throw new IllegalArgumentException("Input pattern cannot be null");
        }
        if (!(parameters instanceof BayesianParameters bayesianParams)) {
            throw new IllegalArgumentException("Parameters must be BayesianParameters");
        }
        
        // Bayesian update using conjugate prior (normal-inverse-gamma)
        var inputData = switch (input) {
            case DenseVector dv -> dv.data();
        };
        var learningRate = bayesianParams.learningRate();
        
        // Update mean using weighted average
        var newMeanData = new double[mean.dimension()];
        for (int i = 0; i < mean.dimension() && i < inputData.length; i++) {
            newMeanData[i] = mean.get(i) + learningRate * (inputData[i] - mean.get(i));
        }
        // Fill remaining dimensions if input is smaller
        for (int i = inputData.length; i < mean.dimension(); i++) {
            newMeanData[i] = mean.get(i);
        }
        
        // Update covariance (simplified online update)
        var newCovData = new double[covariance.getRowCount()][covariance.getColumnCount()];
        
        // Copy existing covariance
        for (int i = 0; i < covariance.getRowCount(); i++) {
            for (int j = 0; j < covariance.getColumnCount(); j++) {
                newCovData[i][j] = covariance.get(i, j);
            }
        }
        
        // Update diagonal elements (variances) based on input
        for (int i = 0; i < Math.min(inputData.length, covariance.getRowCount()); i++) {
            var error = inputData[i] - mean.get(i);
            var varianceUpdate = learningRate * error * error;
            // Exponential moving average for variance
            newCovData[i][i] = (1.0 - learningRate) * covariance.get(i, i) + varianceUpdate;
            
            // Ensure minimum variance
            if (newCovData[i][i] < bayesianParams.noiseVariance()) {
                newCovData[i][i] = bayesianParams.noiseVariance();
            }
        }
        
        return new BayesianWeight(
            new DenseVector(newMeanData),
            new Matrix(newCovData),
            sampleCount + 1,
            precision * (1.0 + learningRate) // Increase precision with more samples
        );
    }
}