package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import java.util.Arrays;
import java.util.Objects;

/**
 * FuzzyWeight represents a weight vector for FuzzyART using complement coding.
 * The vector stores both the original input and its complement, doubling the dimension.
 * Updates use fuzzy min operations with beta learning rate.
 */
public record FuzzyWeight(double[] data, int originalDimension) implements WeightVector {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public FuzzyWeight {
        Objects.requireNonNull(data, "FuzzyWeight data cannot be null");
        if (data.length == 0) {
            throw new IllegalArgumentException("FuzzyWeight data cannot be empty");
        }
        if (data.length % 2 != 0) {
            throw new IllegalArgumentException("FuzzyWeight data length must be even for complement coding");
        }
        if (originalDimension <= 0) {
            throw new IllegalArgumentException("Original dimension must be positive, got: " + originalDimension);
        }
        if (data.length != originalDimension * 2) {
            throw new IllegalArgumentException("Data length must be 2 * originalDimension, got: " + 
                data.length + " for originalDimension: " + originalDimension);
        }
        
        // Copy array to ensure immutability
        data = Arrays.copyOf(data, data.length);
    }
    
    /**
     * Create a FuzzyWeight from complement-coded data.
     * @param data the complement-coded data (length must be even)
     * @param originalDimension the original dimension before complement coding
     * @return new FuzzyWeight instance
     */
    public static FuzzyWeight of(double[] data, int originalDimension) {
        return new FuzzyWeight(data, originalDimension);
    }
    
    /**
     * Create a FuzzyWeight from an input vector, automatically applying complement coding.
     * The resulting weight vector will have dimension 2 * input.dimension().
     * Complement coding: [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]
     * 
     * @param input the input vector to complement code
     * @return new FuzzyWeight with complement coding applied
     */
    public static FuzzyWeight fromInput(Pattern input) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        var original = input.dimension();
        var complementCoded = new double[original * 2];
        
        // Copy original values
        for (int i = 0; i < original; i++) {
            complementCoded[i] = input.get(i);
        }
        
        // Add complement values
        for (int i = 0; i < original; i++) {
            complementCoded[original + i] = 1.0 - input.get(i);
        }
        
        return new FuzzyWeight(complementCoded, original);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for vector of size " + data.length);
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
    
    /**
     * Update this FuzzyWeight using the fuzzy ART learning rule.
     * New weight = β * min(input, weight) + (1-β) * weight
     * 
     * @param input the input vector (must be complement-coded with same dimension)
     * @param parameters FuzzyParameters containing the beta learning rate
     * @return new updated FuzzyWeight
     */
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Support both FuzzyParameters and MutableFuzzyParameters
        double beta;
        if (parameters instanceof FuzzyParameters fuzzyParams) {
            beta = fuzzyParams.beta();
        } else if (parameters instanceof com.hellblazer.art.core.parameters.MutableFuzzyParameters mutableParams) {
            beta = mutableParams.beta();
        } else {
            throw new IllegalArgumentException("Parameters must be FuzzyParameters or MutableFuzzyParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != data.length) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match weight dimension " + data.length);
        }
        var newData = new double[data.length];
        
        // Apply fuzzy learning rule: β * min(input, weight) + (1-β) * weight
        for (int i = 0; i < data.length; i++) {
            var minValue = Math.min(input.get(i), data[i]);
            newData[i] = beta * minValue + (1.0 - beta) * data[i];
        }
        
        return new FuzzyWeight(newData, originalDimension);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof FuzzyWeight other)) return false;
        return originalDimension == other.originalDimension && Arrays.equals(data, other.data);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(originalDimension, Arrays.hashCode(data));
    }
    
    @Override
    public String toString() {
        return "FuzzyWeight{originalDim=" + originalDimension + 
               ", data=" + Arrays.toString(data) + "}";
    }
}