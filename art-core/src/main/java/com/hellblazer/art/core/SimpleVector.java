package com.hellblazer.art.core;

import java.util.Arrays;

/**
 * Simple test class to verify compilation setup.
 */
public class SimpleVector {
    private final double[] data;
    
    public SimpleVector(double... data) {
        this.data = Arrays.copyOf(data, data.length);
    }
    
    public double get(int index) {
        return data[index];
    }
    
    public int dimension() {
        return data.length;
    }
}