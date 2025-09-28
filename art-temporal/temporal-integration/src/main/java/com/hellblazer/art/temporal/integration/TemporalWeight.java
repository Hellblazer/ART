package com.hellblazer.art.temporal.integration;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;

/**
 * Temporal weight vector implementation for TemporalART.
 * Provides both WeightVector interface and getData() method for compatibility.
 */
public class TemporalWeight implements WeightVector {
    private final double[] data;

    public TemporalWeight(double[] data) {
        this.data = data.clone();
    }

    /**
     * Get the underlying data array.
     * @return clone of the data array
     */
    public double[] getData() {
        return data.clone();
    }

    @Override
    public double get(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + data.length);
        }
        return data[index];
    }

    @Override
    public int dimension() {
        return data.length;
    }

    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : data) {
            sum += Math.abs(value);
        }
        return sum;
    }

    @Override
    public WeightVector update(Pattern input, Object parameters) {
        double learningRate = 0.1;
        if (parameters instanceof TemporalARTParameters params) {
            learningRate = params.getLearningRate();
        }

        var updatedData = new double[data.length];
        int inputDim = Math.min(input.dimension(), data.length);

        for (int i = 0; i < inputDim; i++) {
            updatedData[i] = (1 - learningRate) * data[i] + learningRate * input.get(i);
        }

        // Copy remaining values if weight dimension is larger
        for (int i = inputDim; i < data.length; i++) {
            updatedData[i] = data[i];
        }

        return new TemporalWeight(updatedData);
    }
}