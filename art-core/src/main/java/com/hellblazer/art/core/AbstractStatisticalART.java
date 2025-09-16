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
 * Abstract base class for statistical ART algorithms that use probability distributions.
 * 
 * This class provides common functionality for algorithms that model categories
 * using statistical distributions (Gaussian, Bayesian, etc.) rather than simple
 * geometric shapes or fuzzy operations.
 * 
 * Key features provided:
 * - Statistical weight vector management
 * - Probability-based activation computation
 * - Likelihood-based vigilance testing
 * - Distribution parameter updates
 * - Performance tracking for statistical operations
 * 
 * Algorithms extending this class: GaussianART, BayesianART, and similar
 * probabilistic clustering approaches.
 * 
 * @param <P> the parameter type for the specific statistical algorithm
 */
public abstract class AbstractStatisticalART<P> extends BaseART {
    
    // Performance tracking
    private long totalStatisticalOperations = 0L;
    private long distributionUpdates = 0L;
    private double avgStatisticalTime = 0.0;
    private long statisticalComputeTime = 0L;
    
    /**
     * Create a new AbstractStatisticalART with empty categories.
     */
    protected AbstractStatisticalART() {
        super();
    }
    
    /**
     * Create a new AbstractStatisticalART with initial categories.
     * 
     * @param initialCategories initial weight vectors (will be copied)
     */
    protected AbstractStatisticalART(List<? extends WeightVector> initialCategories) {
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
            trackStatisticalOperation();
            return computeStatisticalActivation(input, weight, typedParams);
        } finally {
            updateStatisticalTime(startTime);
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
            trackStatisticalOperation();
            return computeStatisticalVigilance(input, weight, typedParams);
        } finally {
            updateStatisticalTime(startTime);
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
            trackDistributionUpdate();
            return computeStatisticalWeightUpdate(input, weight, typedParams);
        } finally {
            updateStatisticalTime(startTime);
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
        
        trackDistributionUpdate();
        return createStatisticalWeightVector(input, typedParams);
    }
    
    // === Abstract Methods for Statistical Algorithms ===

    /**
     * Get the expected parameter class for this algorithm.
     *
     * @return the Class object for the parameter type
     */
    protected abstract Class<P> getParameterClass();

    /**
     * Compute activation using statistical/probabilistic measures.
     *
     * @param input the input pattern
     * @param weight the category weight (statistical parameters)
     * @param parameters algorithm-specific parameters
     * @return activation value (typically probability or likelihood)
     */
    protected abstract double computeStatisticalActivation(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Compute vigilance test using statistical criteria.
     * 
     * @param input the input pattern
     * @param weight the category weight (statistical parameters)
     * @param parameters algorithm-specific parameters
     * @return vigilance test result
     */
    protected abstract MatchResult computeStatisticalVigilance(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Update statistical distribution parameters based on new data.
     * 
     * @param input the input pattern
     * @param weight current statistical parameters
     * @param parameters algorithm-specific parameters
     * @return updated statistical parameters
     */
    protected abstract WeightVector computeStatisticalWeightUpdate(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Create initial statistical weight vector from first input pattern.
     * 
     * @param input the input pattern
     * @param parameters algorithm-specific parameters
     * @return initial statistical weight vector
     */
    protected abstract WeightVector createStatisticalWeightVector(Pattern input, P parameters);
    
    // === Common Statistical Utility Methods ===
    
    /**
     * Calculate multivariate Gaussian probability density (simplified version).
     * 
     * @param input input pattern
     * @param mean distribution mean
     * @param variance diagonal variance values
     * @return probability density value
     */
    protected final double calculateGaussianProbability(Pattern input, double[] mean, double[] variance) {
        var dimension = input.dimension();
        
        // Calculate squared Mahalanobis distance for diagonal covariance
        double mahalanobis = 0.0;
        double logDet = 0.0;
        
        for (int i = 0; i < dimension; i++) {
            double diff = input.get(i) - mean[i];
            double var = Math.max(variance[i], 1e-8); // Prevent division by zero
            mahalanobis += (diff * diff) / var;
            logDet += Math.log(var);
        }
        
        // Calculate log probability to avoid numerical issues
        double logProb = -0.5 * (dimension * Math.log(2 * Math.PI) + logDet + mahalanobis);
        return Math.exp(logProb);
    }
    
    /**
     * Calculate Euclidean distance between input and mean.
     * 
     * @param input input pattern
     * @param mean distribution mean
     * @return Euclidean distance
     */
    protected final double calculateEuclideanDistance(Pattern input, double[] mean) {
        var dimension = input.dimension();
        
        double sumSquares = 0.0;
        for (int i = 0; i < dimension; i++) {
            double diff = input.get(i) - mean[i];
            sumSquares += diff * diff;
        }
        
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Update running statistics (mean, variance) using online algorithms.
     * 
     * @param currentMean current mean estimate
     * @param currentVariance current variance estimate
     * @param newValue new data point
     * @param sampleCount total sample count including new value
     * @return updated statistics [mean, variance]
     */
    protected final double[] updateOnlineStatistics(double currentMean, double currentVariance, 
                                                   double newValue, long sampleCount) {
        if (sampleCount == 1) {
            return new double[]{newValue, 0.0};
        }
        
        // Welford's online algorithm
        var delta = newValue - currentMean;
        var newMean = currentMean + delta / sampleCount;
        var delta2 = newValue - newMean;
        var newVariance = currentVariance + (delta * delta2 - currentVariance) / sampleCount;
        
        return new double[]{newMean, newVariance};
    }
    
    // === Performance Tracking ===
    
    /**
     * Track a statistical operation for performance monitoring.
     */
    protected final void trackStatisticalOperation() {
        totalStatisticalOperations++;
    }
    
    /**
     * Track a distribution parameter update.
     */
    protected final void trackDistributionUpdate() {
        distributionUpdates++;
    }
    
    /**
     * Update statistical computation time tracking.
     */
    private void updateStatisticalTime(long startTime) {
        var elapsed = System.nanoTime() - startTime;
        var elapsedMs = elapsed / 1_000_000.0;
        
        // Simple moving average
        if (totalStatisticalOperations > 0) {
            avgStatisticalTime = ((avgStatisticalTime * (totalStatisticalOperations - 1)) + elapsedMs) / totalStatisticalOperations;
        }
        
        statisticalComputeTime += elapsed;
    }
    
    /**
     * Get performance statistics for statistical operations.
     * 
     * @return map of performance metrics
     */
    public final java.util.Map<String, Object> getStatisticalPerformanceStats() {
        return java.util.Map.of(
            "totalStatisticalOperations", totalStatisticalOperations,
            "distributionUpdates", distributionUpdates,
            "avgStatisticalTimeMs", avgStatisticalTime,
            "totalStatisticalTimeMs", statisticalComputeTime / 1_000_000.0,
            "categoryCount", getCategoryCount()
        );
    }
    
    /**
     * Reset statistical performance tracking counters.
     */
    public final void resetStatisticalPerformanceTracking() {
        totalStatisticalOperations = 0L;
        distributionUpdates = 0L;
        avgStatisticalTime = 0.0;
        statisticalComputeTime = 0L;
    }
}