package com.hellblazer.art.core;

import java.util.Optional;

/**
 * Functional interface for match reset functions in ART algorithms.
 * 
 * Match reset functions provide additional criteria for determining whether
 * a category should be considered for matching beyond the standard vigilance test.
 * This is used in advanced ART variants and specialized applications.
 */
@FunctionalInterface
public interface MatchResetFunction {
    
    /**
     * Determine whether a category should be considered for matching.
     * 
     * @param input the input pattern
     * @param weight the category weight vector
     * @param categoryIndex the index of the category being tested
     * @param parameters the algorithm parameters
     * @param cache optional cache from previous computations
     * @return true if the category should be considered, false to skip it
     */
    boolean shouldConsiderCategory(
        Pattern input, 
        WeightVector weight, 
        int categoryIndex, 
        Object parameters, 
        Optional<Object> cache
    );
}