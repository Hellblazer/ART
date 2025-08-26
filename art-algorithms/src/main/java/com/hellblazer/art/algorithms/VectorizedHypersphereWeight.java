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
package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import java.util.Arrays;
import java.util.Objects;

/**
 * Vectorized weight representation for VectorizedHypersphereART.
 * 
 * This class extends the standard HypersphereWeight with additional
 * performance tracking and optimization features for SIMD operations.
 * 
 * Key Features:
 * - Immutable hypersphere representation (center + radius)
 * - Creation and update time tracking for performance analysis
 * - Update counter for learning statistics
 * - Optimized for SIMD distance calculations
 * - Full compatibility with HypersphereWeight semantics
 * 
 * The hypersphere learning rule:
 * - If input point is within radius: no change
 * - If input point is outside radius: expand radius to include point
 * - Center remains fixed (preserves original category prototype)
 */
public record VectorizedHypersphereWeight(
    double[] center,
    double radius,
    long creationTime,
    int updateCount
) implements WeightVector {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public VectorizedHypersphereWeight {
        Objects.requireNonNull(center, "Center cannot be null");
        if (center.length == 0) {
            throw new IllegalArgumentException("Center cannot be empty");
        }
        if (radius < 0.0) {
            throw new IllegalArgumentException("Radius must be non-negative, got: " + radius);
        }
        if (updateCount < 0) {
            throw new IllegalArgumentException("Update count must be non-negative, got: " + updateCount);
        }
        
        // Defensive copy to ensure immutability
        center = Arrays.copyOf(center, center.length);
    }
    
    /**
     * Create a VectorizedHypersphereWeight with specified center and radius.
     */
    public static VectorizedHypersphereWeight of(double[] center, double radius) {
        return new VectorizedHypersphereWeight(center, radius, System.currentTimeMillis(), 0);
    }
    
    /**
     * Create a VectorizedHypersphereWeight centered at the given point with zero radius.
     */
    public static VectorizedHypersphereWeight atPoint(double[] center) {
        return new VectorizedHypersphereWeight(center, 0.0, System.currentTimeMillis(), 0);
    }
    
    /**
     * Create from standard HypersphereWeight.
     */
    public static VectorizedHypersphereWeight from(HypersphereWeight weight) {
        return new VectorizedHypersphereWeight(weight.center(), weight.radius(), 
                                               System.currentTimeMillis(), 0);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= center.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for vector of size " + center.length);
        }
        return center[index];
    }
    
    @Override
    public int dimension() {
        return center.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : center) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    /**
     * Expand this hypersphere to include the given point.
     * Returns a new weight with updated radius if expansion is needed.
     */
    public VectorizedHypersphereWeight expandToInclude(Pattern point) {
        Objects.requireNonNull(point, "Point cannot be null");
        if (point.dimension() != center.length) {
            throw new IllegalArgumentException("Point dimension " + point.dimension() + 
                " must match center dimension " + center.length);
        }
        
        var distance = calculateEuclideanDistance(point);
        if (distance <= radius) {
            // Point is already within hypersphere
            return this;
        }
        
        // Expand radius to include the point
        return new VectorizedHypersphereWeight(center, distance, creationTime, updateCount + 1);
    }
    
    /**
     * Calculate Euclidean distance from center to given point.
     * d(x, c) = √(∑(x_i - c_i)²)
     */
    public double calculateEuclideanDistance(Pattern point) {
        if (point.dimension() != center.length) {
            throw new IllegalArgumentException("Point dimension " + point.dimension() + 
                " must match center dimension " + center.length);
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < center.length; i++) {
            var diff = point.get(i) - center[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Test if a point is within this hypersphere.
     */
    public boolean contains(Pattern point) {
        return calculateEuclideanDistance(point) <= radius;
    }
    
    /**
     * Calculate the volume of this hypersphere (for debugging/analysis).
     * V_n = π^(n/2) * r^n / Γ(n/2 + 1)
     * Simplified approximation for common dimensions.
     */
    public double approximateVolume() {
        int n = center.length;
        if (radius == 0.0) return 0.0;
        if (n == 1) return 2 * radius;
        if (n == 2) return Math.PI * radius * radius;
        if (n == 3) return (4.0/3.0) * Math.PI * Math.pow(radius, 3);
        
        // General approximation for higher dimensions
        return Math.pow(Math.PI, n/2.0) * Math.pow(radius, n) / Math.exp(n/2.0);
    }
    
    /**
     * Calculate expansion factor since creation.
     */
    public double getExpansionFactor(double initialRadius) {
        if (initialRadius == 0.0) return radius == 0.0 ? 1.0 : Double.POSITIVE_INFINITY;
        return radius / initialRadius;
    }
    
    /**
     * Get age of this weight in milliseconds.
     */
    public long getAge() {
        return System.currentTimeMillis() - creationTime;
    }
    
    /**
     * Check if this hypersphere has been updated (radius expanded).
     */
    public boolean hasBeenUpdated() {
        return updateCount > 0;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedHypersphereParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedHypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != center.length) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match center dimension " + center.length);
        }
        
        // Use the hypersphere expansion rule
        return expandToInclude(input);
    }
    
    /**
     * Convert to standard HypersphereWeight for compatibility.
     */
    public HypersphereWeight toHypersphereWeight() {
        return HypersphereWeight.of(Arrays.copyOf(center, center.length), radius);
    }
    
    /**
     * Create a copy with updated radius.
     */
    public VectorizedHypersphereWeight withRadius(double newRadius) {
        if (newRadius < 0.0) {
            throw new IllegalArgumentException("Radius must be non-negative");
        }
        return new VectorizedHypersphereWeight(center, newRadius, creationTime, updateCount + 1);
    }
    
    /**
     * Create a copy with reset update statistics.
     */
    public VectorizedHypersphereWeight resetStats() {
        return new VectorizedHypersphereWeight(center, radius, System.currentTimeMillis(), 0);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedHypersphereWeight other)) return false;
        return Double.compare(radius, other.radius) == 0 && 
               Arrays.equals(center, other.center) &&
               creationTime == other.creationTime &&
               updateCount == other.updateCount;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(center), radius, creationTime, updateCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedHypersphereWeight{center=%s, radius=%.3f, age=%dms, updates=%d}", 
                           Arrays.toString(center), radius, getAge(), updateCount);
    }
    
    /**
     * Create a detailed string representation for debugging.
     */
    public String toDetailedString() {
        return String.format(
            "VectorizedHypersphereWeight{\n" +
            "  center=%s,\n" +
            "  radius=%.6f,\n" +
            "  dimension=%d,\n" +
            "  volume≈%.6f,\n" +
            "  creationTime=%d,\n" +
            "  age=%dms,\n" +
            "  updateCount=%d,\n" +
            "  l1Norm=%.6f\n" +
            "}",
            Arrays.toString(center), radius, dimension(), approximateVolume(),
            creationTime, getAge(), updateCount, l1Norm()
        );
    }
}