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
package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.WeightVector;
/**
 * Weight vector for ART-2 neural network categories.
 * 
 * ART-2 uses normalized weight vectors representing category prototypes.
 * Each weight is a unit vector in the input space.
 * 
 * @param vector the normalized weight vector
 * 
 * @author Hal Hildebrand
 */
public record ART2Weight(DenseVector vector) implements WeightVector {
    
    /**
     * Create ART-2 weight with validation.
     * 
     * @param vector the weight vector (should be normalized)
     * @throws IllegalArgumentException if vector is null or invalid
     */
    public ART2Weight {
        if (vector == null) {
            throw new IllegalArgumentException("Weight vector cannot be null");
        }
        if (vector.getDimension() == 0) {
            throw new IllegalArgumentException("Weight vector cannot be empty");
        }
        
        // Validate that all components are finite
        var values = vector.values();
        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException("Weight vector contains non-finite value at index " + i);
            }
        }
    }
    
    /**
     * Create a normalized ART-2 weight from an input vector.
     * 
     * @param input the input vector to normalize
     * @return normalized ART-2 weight
     * @throws IllegalArgumentException if input is null or zero vector
     */
    public static ART2Weight fromInput(DenseVector input) {
        if (input == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        
        var values = input.values();
        var norm = 0.0;
        for (var value : values) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        
        if (norm == 0.0) {
            throw new IllegalArgumentException("Cannot create weight from zero vector");
        }
        
        var normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = values[i] / norm;
        }
        
        return new ART2Weight(new DenseVector(normalized));
    }
    
    @Override
    public int dimension() {
        return vector.getDimension();
    }
    
    @Override
    public double get(int index) {
        return vector.get(index);
    }
    
    @Override
    public double l1Norm() {
        return vector.l1Norm();
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        // ART2Weight updates are handled by ART2.updateWeights() method
        // Individual weights cannot update themselves without the full ART2 context
        // This method returns this weight unchanged as per WeightVector interface contract
        return this;
    }
    
    public int getDimension() {
        return vector.getDimension();
    }
    
    public double[] values() {
        return vector.values();
    }
}