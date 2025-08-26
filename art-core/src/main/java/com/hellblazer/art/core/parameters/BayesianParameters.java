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
package com.hellblazer.art.core.parameters;

import com.hellblazer.art.core.utils.Matrix;
import java.util.Arrays;
import java.util.Objects;

/**
 * BayesianParameters encapsulates configuration for Bayesian ART networks.
 * Contains vigilance, prior distributions, noise parameters, and capacity limits.
 * 
 * @author Hal Hildebrand
 */
public record BayesianParameters(
    double vigilance,
    double[] priorMean,
    Matrix priorCovariance,
    double noiseVariance,
    double priorPrecision,
    int maxCategories
) {
    
    public BayesianParameters {
        if (vigilance < 0.0 || vigilance > 1.0 || Double.isNaN(vigilance) || Double.isInfinite(vigilance)) {
            throw new IllegalArgumentException("Invalid vigilance parameter: " + vigilance);
        }
        
        Objects.requireNonNull(priorMean, "Prior mean cannot be null");
        Objects.requireNonNull(priorCovariance, "Prior covariance cannot be null");
        
        if (priorMean.length != priorCovariance.getRowCount()) {
            throw new IllegalArgumentException("Prior mean and covariance dimensions must match");
        }
        
        if (noiseVariance <= 0.0 || Double.isNaN(noiseVariance) || Double.isInfinite(noiseVariance)) {
            throw new IllegalArgumentException("Invalid noise variance: " + noiseVariance);
        }
        
        if (priorPrecision <= 0.0 || Double.isNaN(priorPrecision) || Double.isInfinite(priorPrecision)) {
            throw new IllegalArgumentException("Invalid prior precision: " + priorPrecision);
        }
        
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Invalid max categories: " + maxCategories);
        }
        
        // Copy array to ensure immutability
        priorMean = Arrays.copyOf(priorMean, priorMean.length);
    }
    
    // Convenience methods for BayesianART compatibility
    public double learningRate() {
        return 0.1; // Default learning rate - could be made a parameter
    }
    
    public int dimensions() {
        return priorMean.length;
    }
    
    // Additional methods for hierarchical inference
    public double priorAlpha() {
        return 1.0; // Default Dirichlet concentration parameter
    }
    
    public double priorBeta() {
        return 1.0; // Default beta parameter for hierarchical priors
    }
}