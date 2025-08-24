package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable data bounds for vector normalization.
 * Stores minimum and maximum values for each dimension to enable
 * proper data normalization for ART algorithms.
 */
public record DataBounds(double[] min, double[] max) {
    
    /**
     * Constructor that validates and copies input arrays.
     * @param min minimum values for each dimension (will be copied)
     * @param max maximum values for each dimension (will be copied)
     * @throws NullPointerException if min or max is null
     * @throws IllegalArgumentException if arrays are empty, different lengths, or min > max
     */
    public DataBounds {
        Objects.requireNonNull(min, "Min bounds cannot be null");
        Objects.requireNonNull(max, "Max bounds cannot be null");
        
        if (min.length == 0) {
            throw new IllegalArgumentException("Bounds cannot be empty");
        }
        
        if (min.length != max.length) {
            throw new IllegalArgumentException("Min and max bounds must have same dimension: " +
                min.length + " vs " + max.length);
        }
        
        // Validate that min <= max for all dimensions
        for (int i = 0; i < min.length; i++) {
            if (min[i] > max[i]) {
                throw new IllegalArgumentException("Min value " + min[i] + 
                    " must not exceed max value " + max[i] + " at dimension " + i);
            }
        }
        
        // Copy arrays to ensure immutability
        min = Arrays.copyOf(min, min.length);
        max = Arrays.copyOf(max, max.length);
    }
    
    /**
     * Get the dimensionality of these bounds.
     * @return the number of dimensions
     */
    public int dimension() {
        return min.length;
    }
    
    /**
     * Get the minimum value for the specified dimension.
     * @param index the dimension index (0-based)
     * @return the minimum value
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public double min(int index) {
        if (index < 0 || index >= min.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for bounds of size " + min.length);
        }
        return min[index];
    }
    
    /**
     * Get the maximum value for the specified dimension.
     * @param index the dimension index (0-based)
     * @return the maximum value
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public double max(int index) {
        if (index < 0 || index >= max.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for bounds of size " + max.length);
        }
        return max[index];
    }
    
    /**
     * Get the range (max - min) for the specified dimension.
     * @param index the dimension index (0-based)
     * @return the range for the dimension
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public double range(int index) {
        if (index < 0 || index >= min.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for bounds of size " + min.length);
        }
        return max[index] - min[index];
    }
    
    /**
     * Check if a vector is contained within these bounds.
     * A vector is contained if all its components are within [min, max] for each dimension.
     * 
     * @param vector the vector to check
     * @return true if the vector is contained within bounds
     * @throws NullPointerException if vector is null
     * @throws IllegalArgumentException if vector dimension doesn't match
     */
    public boolean contains(Pattern vector) {
        Objects.requireNonNull(vector, "Vector cannot be null");
        if (vector.dimension() != min.length) {
            throw new IllegalArgumentException("Vector dimension " + vector.dimension() + 
                " does not match bounds dimension " + min.length);
        }
        
        for (int i = 0; i < min.length; i++) {
            double value = vector.get(i);
            if (value < min[i] || value > max[i]) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Expand these bounds to include the specified vector.
     * If the vector is already contained, returns the same bounds.
     * Otherwise, returns new bounds that include both the original bounds and the vector.
     * 
     * @param vector the vector to include
     * @return new bounds that include the vector, or this if already contained
     * @throws NullPointerException if vector is null
     * @throws IllegalArgumentException if vector dimension doesn't match
     */
    public DataBounds expand(Pattern vector) {
        Objects.requireNonNull(vector, "Vector cannot be null");
        if (vector.dimension() != min.length) {
            throw new IllegalArgumentException("Vector dimension " + vector.dimension() + 
                " does not match bounds dimension " + min.length);
        }
        
        var newMin = Arrays.copyOf(min, min.length);
        var newMax = Arrays.copyOf(max, max.length);
        boolean changed = false;
        
        for (int i = 0; i < min.length; i++) {
            double value = vector.get(i);
            if (value < newMin[i]) {
                newMin[i] = value;
                changed = true;
            }
            if (value > newMax[i]) {
                newMax[i] = value;
                changed = true;
            }
        }
        
        return changed ? new DataBounds(newMin, newMax) : this;
    }
    
    /**
     * Create DataBounds from minimum and maximum arrays.
     * @param min minimum values for each dimension
     * @param max maximum values for each dimension
     * @return new DataBounds instance
     * @throws NullPointerException if min or max is null
     * @throws IllegalArgumentException if validation fails
     */
    public static DataBounds of(double[] min, double[] max) {
        return new DataBounds(min, max);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DataBounds other)) return false;
        return Arrays.equals(min, other.min) && Arrays.equals(max, other.max);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(min) * 31 + Arrays.hashCode(max);
    }
    
    @Override
    public String toString() {
        return "DataBounds{min=" + Arrays.toString(min) + ", max=" + Arrays.toString(max) + "}";
    }
}