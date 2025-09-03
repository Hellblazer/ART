package com.hellblazer.art.performance;

import com.hellblazer.art.core.Pattern;

/**
 * Common interface for all vectorized ARTMAP algorithms in the performance module.
 * Provides supervised learning capabilities with performance tracking.
 * 
 * @param <S> Type of performance statistics
 * @param <P> Type of algorithm parameters
 * @param <R> Type of result returned by learn and predict operations
 */
public interface VectorizedARTMAPAlgorithm<S, P, R> extends AutoCloseable {
    
    /**
     * Learn a supervised pattern pair (input -> output).
     * 
     * @param input the input pattern
     * @param output the output/target pattern
     * @param parameters algorithm-specific parameters
     * @return result of the learning operation
     */
    R learn(Pattern input, Pattern output, P parameters);
    
    /**
     * Predict the output for a given input pattern.
     * 
     * @param input the input pattern
     * @param parameters algorithm-specific parameters
     * @return result of the prediction
     */
    R predict(Pattern input, P parameters);
    
    /**
     * Get the current number of categories/clusters.
     * 
     * @return number of categories
     */
    int getCategoryCount();
    
    /**
     * Get current performance statistics.
     * 
     * @return performance statistics
     */
    S getPerformanceStats();
    
    /**
     * Reset performance tracking counters.
     */
    void resetPerformanceTracking();
    
    /**
     * Get the algorithm parameters.
     * 
     * @return algorithm parameters
     */
    P getParameters();
    
    /**
     * Close and clean up resources.
     */
    @Override
    void close();
}