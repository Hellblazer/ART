package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * HypersphereWeight represents a weight vector for HypersphereART using geometric representation.
 * Maintains a center point and radius that defines a hypersphere in the input space.
 * Updates expand the radius to include new points when necessary.
 */
public record HypersphereWeight(double[] center, double radius) implements WeightVector {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public HypersphereWeight {
        Objects.requireNonNull(center, "Center cannot be null");
        if (center.length == 0) {
            throw new IllegalArgumentException("Center cannot be empty");
        }
        if (radius < 0.0) {
            throw new IllegalArgumentException("Radius must be non-negative, got: " + radius);
        }
        
        // Copy array to ensure immutability
        center = Arrays.copyOf(center, center.length);
    }
    
    /**
     * Create a HypersphereWeight with specified center and radius.
     * @param center the center point of the hypersphere
     * @param radius the radius of the hypersphere (must be non-negative)
     * @return new HypersphereWeight instance
     */
    public static HypersphereWeight of(double[] center, double radius) {
        return new HypersphereWeight(center, radius);
    }
    
    /**
     * Create a HypersphereWeight centered at the given point with zero radius.
     * @param center the center point
     * @return new HypersphereWeight with zero radius
     */
    public static HypersphereWeight atPoint(double[] center) {
        return new HypersphereWeight(center, 0.0);
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
     * If the point is already within the hypersphere, returns this weight unchanged.
     * Otherwise, returns a new weight with radius expanded to include the point.
     * 
     * @param point the point to include
     * @return new HypersphereWeight that includes the point
     */
    public HypersphereWeight expandToInclude(Vector point) {
        Objects.requireNonNull(point, "Point cannot be null");
        if (point.dimension() != center.length) {
            throw new IllegalArgumentException("Point dimension " + point.dimension() + 
                " must match center dimension " + center.length);
        }
        
        var distance = euclideanDistance(point);
        if (distance <= radius) {
            // Point is already inside or on the hypersphere boundary
            return this;
        }
        
        // Expand radius to include the point
        return new HypersphereWeight(center, distance);
    }
    
    /**
     * Calculate the Euclidean distance from the center to the given point.
     * @param point the point to calculate distance to
     * @return the Euclidean distance
     */
    private double euclideanDistance(Vector point) {
        double sumSquares = 0.0;
        for (int i = 0; i < center.length; i++) {
            var diff = point.get(i) - center[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Update this HypersphereWeight using the hypersphere ART learning rule.
     * Currently, this simply expands the radius to include the new input point.
     * More sophisticated updates could adjust the center position.
     * 
     * @param input the input vector
     * @param parameters HypersphereParameters (not used in current implementation)
     * @return new updated HypersphereWeight
     */
    @Override
    public WeightVector update(Vector input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof HypersphereParameters)) {
            throw new IllegalArgumentException("Parameters must be HypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != center.length) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match center dimension " + center.length);
        }
        
        // Expand hypersphere to include the new point
        return expandToInclude(input);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof HypersphereWeight other)) return false;
        return Double.compare(radius, other.radius) == 0 && Arrays.equals(center, other.center);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(center), radius);
    }
    
    @Override
    public String toString() {
        return "HypersphereWeight{center=" + Arrays.toString(center) + 
               ", radius=" + radius + "}";
    }
}