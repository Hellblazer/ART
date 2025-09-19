package com.hellblazer.art.core.parameters;

/**
 * Interface for providing FuzzyART parameters.
 * Implemented by both FuzzyParameters (immutable) and MutableFuzzyParameters.
 */
public interface FuzzyParameterProvider {
    double vigilance();
    double alpha();
    double beta();
}