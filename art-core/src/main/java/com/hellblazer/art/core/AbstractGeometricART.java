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

import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.utils.MathOperations;
import java.util.List;
import java.util.Objects;

/**
 * Abstract base class for geometric ART algorithms that use geometric shapes for clustering.
 * 
 * This class provides common functionality for algorithms that model categories
 * as geometric regions (hyperspheres, ellipsoids, hyperrectangles) in the input space
 * rather than using fuzzy operations or statistical distributions.
 * 
 * Key features provided:
 * - Geometric distance computation utilities
 * - Shape-based activation functions
 * - Distance-based vigilance testing
 * - Geometric parameter updates
 * - Performance tracking for geometric operations
 * - Common geometric transformations
 * 
 * Algorithms extending this class: HypersphereART, EllipsoidART, and similar
 * geometric clustering approaches.
 * 
 * @param <P> the parameter type for the specific geometric algorithm
 */
public abstract class AbstractGeometricART<P> extends BaseART {
    
    // Performance tracking
    private long totalGeometricOperations = 0L;
    private long shapeUpdates = 0L;
    private long distanceComputations = 0L;
    private double avgGeometricTime = 0.0;
    private long geometricComputeTime = 0L;
    
    /**
     * Create a new AbstractGeometricART with empty categories.
     */
    protected AbstractGeometricART() {
        super();
    }
    
    /**
     * Create a new AbstractGeometricART with initial categories.
     * 
     * @param initialCategories initial weight vectors (will be copied)
     */
    protected AbstractGeometricART(List<? extends WeightVector> initialCategories) {
        super(Objects.requireNonNull(initialCategories, "Initial categories cannot be null"));
    }
    
    // === BaseART Template Method Implementation ===
    
    @Override
    protected final double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");

        if (!getParameterClass().isInstance(parameters)) {
            throw new IllegalArgumentException("Invalid parameter type: expected " +
                getParameterClass().getSimpleName() + " but got " + parameters.getClass().getSimpleName());
        }

        @SuppressWarnings("unchecked")
        P typedParams = (P) parameters;
        
        long startTime = System.nanoTime();
        try {
            trackGeometricOperation();
            trackDistanceComputation();
            return computeGeometricActivation(input, weight, typedParams);
        } finally {
            updateGeometricTime(startTime);
        }
    }
    
    @Override
    protected final MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");

        if (!getParameterClass().isInstance(parameters)) {
            throw new IllegalArgumentException("Invalid parameter type: expected " +
                getParameterClass().getSimpleName() + " but got " + parameters.getClass().getSimpleName());
        }

        @SuppressWarnings("unchecked")
        P typedParams = (P) parameters;
        
        long startTime = System.nanoTime();
        try {
            trackGeometricOperation();
            trackDistanceComputation();
            return computeGeometricVigilance(input, weight, typedParams);
        } finally {
            updateGeometricTime(startTime);
        }
    }
    
    @Override
    protected final WeightVector updateWeights(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");

        if (!getParameterClass().isInstance(parameters)) {
            throw new IllegalArgumentException("Invalid parameter type: expected " +
                getParameterClass().getSimpleName() + " but got " + parameters.getClass().getSimpleName());
        }

        @SuppressWarnings("unchecked")
        P typedParams = (P) parameters;
        
        long startTime = System.nanoTime();
        try {
            trackShapeUpdate();
            return computeGeometricWeightUpdate(input, weight, typedParams);
        } finally {
            updateGeometricTime(startTime);
        }
    }
    
    @Override
    protected final WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");

        if (!getParameterClass().isInstance(parameters)) {
            throw new IllegalArgumentException("Invalid parameter type: expected " +
                getParameterClass().getSimpleName() + " but got " + parameters.getClass().getSimpleName());
        }

        @SuppressWarnings("unchecked")
        P typedParams = (P) parameters;
        
        trackShapeUpdate();
        return createGeometricWeightVector(input, typedParams);
    }
    
    // === Abstract Methods for Geometric Algorithms ===

    /**
     * Get the expected parameter class for this algorithm.
     *
     * @return the Class object for the parameter type
     */
    protected abstract Class<P> getParameterClass();

    /**
     * Compute activation based on geometric distance/similarity measures.
     *
     * @param input the input pattern
     * @param weight the category weight (geometric parameters)
     * @param parameters algorithm-specific parameters
     * @return activation value (typically inverse of distance)
     */
    protected abstract double computeGeometricActivation(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Compute vigilance test using geometric distance criteria.
     * 
     * @param input the input pattern
     * @param weight the category weight (geometric parameters)
     * @param parameters algorithm-specific parameters
     * @return vigilance test result
     */
    protected abstract MatchResult computeGeometricVigilance(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Update geometric shape parameters based on new data point.
     * 
     * @param input the input pattern
     * @param weight current geometric parameters
     * @param parameters algorithm-specific parameters
     * @return updated geometric parameters
     */
    protected abstract WeightVector computeGeometricWeightUpdate(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Create initial geometric weight vector from first input pattern.
     * 
     * @param input the input pattern
     * @param parameters algorithm-specific parameters
     * @return initial geometric weight vector
     */
    protected abstract WeightVector createGeometricWeightVector(Pattern input, P parameters);
    
    // === Common Geometric Utility Methods ===
    
    /**
     * Calculate Euclidean distance between two patterns.
     * 
     * @param pattern1 first pattern
     * @param pattern2 second pattern
     * @return Euclidean distance
     */
    protected final double calculateEuclideanDistance(Pattern pattern1, Pattern pattern2) {
        if (pattern1.dimension() != pattern2.dimension()) {
            throw new IllegalArgumentException("Pattern dimensions must match");
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < pattern1.dimension(); i++) {
            double diff = pattern1.get(i) - pattern2.get(i);
            sumSquares += diff * diff;
        }
        
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Calculate squared Euclidean distance (faster when square root not needed).
     * 
     * @param pattern1 first pattern
     * @param pattern2 second pattern
     * @return squared Euclidean distance
     */
    protected final double calculateSquaredEuclideanDistance(Pattern pattern1, Pattern pattern2) {
        if (pattern1.dimension() != pattern2.dimension()) {
            throw new IllegalArgumentException("Pattern dimensions must match");
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < pattern1.dimension(); i++) {
            double diff = pattern1.get(i) - pattern2.get(i);
            sumSquares += diff * diff;
        }
        
        return sumSquares;
    }
    
    /**
     * Calculate Manhattan (L1) distance between two patterns.
     * 
     * @param pattern1 first pattern
     * @param pattern2 second pattern
     * @return Manhattan distance
     */
    protected final double calculateManhattanDistance(Pattern pattern1, Pattern pattern2) {
        if (pattern1.dimension() != pattern2.dimension()) {
            throw new IllegalArgumentException("Pattern dimensions must match");
        }
        
        double sum = 0.0;
        for (int i = 0; i < pattern1.dimension(); i++) {
            sum += Math.abs(pattern1.get(i) - pattern2.get(i));
        }
        
        return sum;
    }
    
    /**
     * Calculate Minkowski distance with specified p-norm.
     * 
     * @param pattern1 first pattern
     * @param pattern2 second pattern
     * @param p the p-norm parameter (p=1: Manhattan, p=2: Euclidean)
     * @return Minkowski distance
     */
    protected final double calculateMinkowskiDistance(Pattern pattern1, Pattern pattern2, double p) {
        if (pattern1.dimension() != pattern2.dimension()) {
            throw new IllegalArgumentException("Pattern dimensions must match");
        }
        if (p <= 0) {
            throw new IllegalArgumentException("p must be positive");
        }
        
        double sum = 0.0;
        for (int i = 0; i < pattern1.dimension(); i++) {
            double diff = Math.abs(pattern1.get(i) - pattern2.get(i));
            sum += Math.pow(diff, p);
        }
        
        return Math.pow(sum, 1.0 / p);
    }
    
    /**
     * Calculate weighted Euclidean distance using diagonal weights.
     * 
     * @param pattern1 first pattern
     * @param pattern2 second pattern (or mean)
     * @param weights diagonal weight values
     * @return weighted Euclidean distance
     */
    protected final double calculateWeightedEuclideanDistance(Pattern pattern1, Pattern pattern2, double[] weights) {
        if (pattern1.dimension() != pattern2.dimension()) {
            throw new IllegalArgumentException("Pattern dimensions must match");
        }
        if (pattern1.dimension() != weights.length) {
            throw new IllegalArgumentException("Weights dimension must match pattern dimension");
        }
        
        double sumSquares = 0.0;
        for (int i = 0; i < pattern1.dimension(); i++) {
            double diff = pattern1.get(i) - pattern2.get(i);
            sumSquares += weights[i] * diff * diff;
        }
        
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Check if a point is inside a hypersphere.
     * 
     * @param point the point to test
     * @param center hypersphere center
     * @param radius hypersphere radius
     * @return true if point is inside the hypersphere
     */
    protected final boolean isInsideHypersphere(Pattern point, Pattern center, double radius) {
        double distance = calculateEuclideanDistance(point, center);
        return distance <= radius;
    }
    
    /**
     * Check if a point is inside a hyperrectangle (axis-aligned box).
     * 
     * @param point the point to test
     * @param minBounds minimum bounds for each dimension
     * @param maxBounds maximum bounds for each dimension
     * @return true if point is inside the hyperrectangle
     */
    protected final boolean isInsideHyperrectangle(Pattern point, double[] minBounds, double[] maxBounds) {
        if (point.dimension() != minBounds.length || point.dimension() != maxBounds.length) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        for (int i = 0; i < point.dimension(); i++) {
            double value = point.get(i);
            if (value < minBounds[i] || value > maxBounds[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Compute centroid (mean) of a set of patterns.
     * 
     * @param patterns list of patterns
     * @return centroid pattern
     */
    protected final Pattern computeCentroid(List<Pattern> patterns) {
        if (patterns.isEmpty()) {
            throw new IllegalArgumentException("Pattern list cannot be empty");
        }
        
        var dimension = patterns.get(0).dimension();
        var centroid = new double[dimension];
        
        // Sum all patterns
        for (var pattern : patterns) {
            if (pattern.dimension() != dimension) {
                throw new IllegalArgumentException("All patterns must have same dimension");
            }
            for (int i = 0; i < dimension; i++) {
                centroid[i] += pattern.get(i);
            }
        }
        
        // Average
        for (int i = 0; i < dimension; i++) {
            centroid[i] /= patterns.size();
        }
        
        return new com.hellblazer.art.core.DenseVector(centroid);
    }
    
    /**
     * Update geometric bounds to include a new point.
     * 
     * @param currentMinBounds current minimum bounds
     * @param currentMaxBounds current maximum bounds
     * @param newPoint new point to include
     * @return updated bounds [minBounds, maxBounds]
     */
    protected final double[][] updateBounds(double[] currentMinBounds, double[] currentMaxBounds, Pattern newPoint) {
        var dimension = newPoint.dimension();
        var newMinBounds = new double[dimension];
        var newMaxBounds = new double[dimension];
        
        for (int i = 0; i < dimension; i++) {
            double value = newPoint.get(i);
            newMinBounds[i] = Math.min(currentMinBounds[i], value);
            newMaxBounds[i] = Math.max(currentMaxBounds[i], value);
        }
        
        return new double[][]{newMinBounds, newMaxBounds};
    }
    
    // === Performance Tracking ===
    
    /**
     * Track a geometric operation for performance monitoring.
     */
    protected final void trackGeometricOperation() {
        totalGeometricOperations++;
    }
    
    /**
     * Track a distance computation.
     */
    protected final void trackDistanceComputation() {
        distanceComputations++;
    }
    
    /**
     * Track a shape parameter update.
     */
    protected final void trackShapeUpdate() {
        shapeUpdates++;
    }
    
    /**
     * Update geometric computation time tracking.
     */
    private void updateGeometricTime(long startTime) {
        var elapsed = System.nanoTime() - startTime;
        var elapsedMs = elapsed / 1_000_000.0;
        
        // Simple moving average
        if (totalGeometricOperations > 0) {
            avgGeometricTime = ((avgGeometricTime * (totalGeometricOperations - 1)) + elapsedMs) / totalGeometricOperations;
        }
        
        geometricComputeTime += elapsed;
    }
    
    /**
     * Get performance statistics for geometric operations.
     * 
     * @return map of performance metrics
     */
    public final java.util.Map<String, Object> getGeometricPerformanceStats() {
        return java.util.Map.of(
            "totalGeometricOperations", totalGeometricOperations,
            "shapeUpdates", shapeUpdates,
            "distanceComputations", distanceComputations,
            "avgGeometricTimeMs", avgGeometricTime,
            "totalGeometricTimeMs", geometricComputeTime / 1_000_000.0,
            "categoryCount", getCategoryCount()
        );
    }
    
    /**
     * Reset geometric performance tracking counters.
     */
    public final void resetGeometricPerformanceTracking() {
        totalGeometricOperations = 0L;
        shapeUpdates = 0L;
        distanceComputations = 0L;
        avgGeometricTime = 0.0;
        geometricComputeTime = 0L;
    }
}