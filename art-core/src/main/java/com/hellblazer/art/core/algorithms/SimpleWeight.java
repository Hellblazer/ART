package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;

/**
 * Simple weight vector implementation for FusionART channels.
 * Stores a fixed array of weight values.
 */
public class SimpleWeight implements WeightVector {
    private final double[] values;
    
    public SimpleWeight(double[] values) {
        this.values = values.clone();
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= values.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + values.length);
        }
        return values[index];
    }
    
    @Override
    public int dimension() {
        return values.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : values) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        // Simple learning rule
        double alpha = 0.01; // Default learning rate
        if (parameters instanceof FusionParameters fusionParams) {
            alpha = fusionParams.getLearningRate();
        }
        
        var updatedValues = new double[values.length];
        int inputDim = Math.min(input.dimension(), values.length);
        
        for (int i = 0; i < inputDim; i++) {
            updatedValues[i] = alpha * Math.min(input.get(i), values[i]) + (1.0 - alpha) * values[i];
        }
        
        // Copy remaining values if weight dimension is larger
        for (int i = inputDim; i < values.length; i++) {
            updatedValues[i] = values[i];
        }
        
        return new SimpleWeight(updatedValues);
    }
}