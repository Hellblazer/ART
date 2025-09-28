package com.hellblazer.art.temporal.core;

import java.util.Map;
import java.util.Optional;

/**
 * Base interface for system parameters with validation and bounds checking.
 */
public interface Parameters {
    /**
     * Validate that all parameters are within acceptable bounds.
     * @throws IllegalArgumentException if any parameter is invalid
     */
    void validate();

    /**
     * Get parameter value by name.
     */
    Optional<Double> getParameter(String name);

    /**
     * Get all parameters as a map.
     */
    Map<String, Double> getAllParameters();

    /**
     * Create a copy with modified parameter.
     */
    Parameters withParameter(String name, double value);

    /**
     * Get parameter with default value if not present.
     */
    default double getParameterOrDefault(String name, double defaultValue) {
        return getParameter(name).orElse(defaultValue);
    }
}