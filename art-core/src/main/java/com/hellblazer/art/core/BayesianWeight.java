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
 * BayesianWeight for BayesianART - MINIMAL STUB FOR TEST COMPILATION
 * This is a minimal implementation to allow tests to compile.
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
        throw new UnsupportedOperationException("Not implemented yet");
    }
}