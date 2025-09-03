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

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Sparse vector implementation for efficient large-scale data handling
 */
public class SparseVector {
    private final Map<Integer, Double> nonZeroElements;
    private final int dimension;
    private static final double EPSILON = 1e-10;
    
    public SparseVector(int dimension) {
        this.dimension = dimension;
        this.nonZeroElements = new ConcurrentHashMap<>();
    }
    
    public SparseVector(double[] denseArray, double sparsityThreshold) {
        this.dimension = denseArray.length;
        this.nonZeroElements = new ConcurrentHashMap<>();
        
        for (int i = 0; i < denseArray.length; i++) {
            if (Math.abs(denseArray[i]) > sparsityThreshold) {
                nonZeroElements.put(i, denseArray[i]);
            }
        }
    }
    
    public double get(int index) {
        if (index < 0 || index >= dimension) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Dimension: " + dimension);
        }
        return nonZeroElements.getOrDefault(index, 0.0);
    }
    
    public void set(int index, double value) {
        if (index < 0 || index >= dimension) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Dimension: " + dimension);
        }
        if (Math.abs(value) > EPSILON) {
            nonZeroElements.put(index, value);
        } else {
            nonZeroElements.remove(index);
        }
    }
    
    public int getDimension() {
        return dimension;
    }
    
    public int getNonZeroCount() {
        return nonZeroElements.size();
    }
    
    public Set<Integer> getNonZeroIndices() {
        return new HashSet<>(nonZeroElements.keySet());
    }
    
    public SparseVector complement() {
        var result = new SparseVector(dimension * 2);
        
        // Original values
        for (var entry : nonZeroElements.entrySet()) {
            result.set(entry.getKey(), entry.getValue());
        }
        
        // Complement values (1 - original)
        for (int i = 0; i < dimension; i++) {
            double complementValue = 1.0 - get(i);
            if (Math.abs(complementValue) > EPSILON) {
                result.set(dimension + i, complementValue);
            }
        }
        
        return result;
    }
    
    public SparseVector fuzzyAnd(SparseVector other) {
        if (other.dimension != this.dimension) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        var result = new SparseVector(dimension);
        
        // Only iterate over potentially non-zero elements
        Set<Integer> indices = new HashSet<>(nonZeroElements.keySet());
        indices.addAll(other.nonZeroElements.keySet());
        
        for (Integer idx : indices) {
            double minValue = Math.min(get(idx), other.get(idx));
            if (Math.abs(minValue) > EPSILON) {
                result.set(idx, minValue);
            }
        }
        
        return result;
    }
    
    public double normL1() {
        return nonZeroElements.values().stream()
                             .mapToDouble(Math::abs)
                             .sum();
    }
    
    public double normL2() {
        return Math.sqrt(nonZeroElements.values().stream()
                                       .mapToDouble(v -> v * v)
                                       .sum());
    }
    
    public SparseVector normalizeL1() {
        double norm = normL1();
        if (norm < EPSILON) {
            return new SparseVector(dimension);
        }
        
        var result = new SparseVector(dimension);
        for (var entry : nonZeroElements.entrySet()) {
            result.set(entry.getKey(), entry.getValue() / norm);
        }
        return result;
    }
    
    public SparseVector normalizeL2() {
        double norm = normL2();
        if (norm < EPSILON) {
            return new SparseVector(dimension);
        }
        
        var result = new SparseVector(dimension);
        for (var entry : nonZeroElements.entrySet()) {
            result.set(entry.getKey(), entry.getValue() / norm);
        }
        return result;
    }
    
    public double dot(SparseVector other) {
        if (other.dimension != this.dimension) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        double sum = 0.0;
        // Only iterate over elements that are non-zero in both vectors
        for (var entry : nonZeroElements.entrySet()) {
            int idx = entry.getKey();
            if (other.nonZeroElements.containsKey(idx)) {
                sum += entry.getValue() * other.get(idx);
            }
        }
        return sum;
    }
    
    public SparseVector add(SparseVector other) {
        if (other.dimension != this.dimension) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        var result = new SparseVector(dimension);
        
        // Add all elements from this vector
        for (var entry : nonZeroElements.entrySet()) {
            result.set(entry.getKey(), entry.getValue());
        }
        
        // Add elements from other vector
        for (var entry : other.nonZeroElements.entrySet()) {
            int idx = entry.getKey();
            result.set(idx, result.get(idx) + entry.getValue());
        }
        
        return result;
    }
    
    public SparseVector multiply(double scalar) {
        var result = new SparseVector(dimension);
        for (var entry : nonZeroElements.entrySet()) {
            result.set(entry.getKey(), entry.getValue() * scalar);
        }
        return result;
    }
    
    public double mean() {
        return nonZeroElements.values().stream()
                             .mapToDouble(Double::doubleValue)
                             .sum() / dimension;
    }
    
    public double variance() {
        double m = mean();
        double sumSq = 0.0;
        
        // Variance including zeros
        for (int i = 0; i < dimension; i++) {
            double diff = get(i) - m;
            sumSq += diff * diff;
        }
        
        return sumSq / dimension;
    }
    
    public SparseVector applySalience(double[] salience) {
        if (salience.length != dimension) {
            throw new IllegalArgumentException("Salience dimension mismatch");
        }
        
        var result = new SparseVector(dimension);
        for (var entry : nonZeroElements.entrySet()) {
            int idx = entry.getKey();
            result.set(idx, entry.getValue() * salience[idx]);
        }
        return result;
    }
    
    public double getSparsityRatio() {
        return (double) getNonZeroCount() / dimension;
    }
    
    public long getMemoryUsage() {
        // Approximate memory usage in bytes
        // Map overhead + entries (key + value)
        return 48 + nonZeroElements.size() * 32L;
    }
    
    public double[] toDenseArray() {
        double[] dense = new double[dimension];
        for (var entry : nonZeroElements.entrySet()) {
            dense[entry.getKey()] = entry.getValue();
        }
        return dense;
    }
    
    public Pattern asPattern() {
        return new DenseVector(toDenseArray());
    }
}