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
package com.hellblazer.art.core.artmap;

/**
 * Parameters for DeepARTMAP hierarchical learning.
 * 
 * This record encapsulates the configuration parameters specific to DeepARTMAP,
 * including hierarchical learning controls, layer management settings, and
 * performance optimization flags.
 * 
 * @param vigilance             the base vigilance parameter for hierarchical layers (default: 0.75)
 * @param learningRate         the learning rate for weight updates (default: 0.1)  
 * @param maxCategories        the maximum number of categories per layer (default: 1000)
 * @param enableDeepMapping    whether to enable deep label mapping operations (default: true)
 * 
 * @author Hal Hildebrand
 */
public record DeepARTMAPParameters(
    double vigilance,
    double learningRate,
    int maxCategories,
    boolean enableDeepMapping
) {
    
    /**
     * Default DeepARTMAP parameters with standard settings.
     */
    public static final DeepARTMAPParameters DEFAULT = new DeepARTMAPParameters(0.75, 0.1, 1000, true);
    
    /**
     * Create DeepARTMAP parameters with default values.
     */
    public DeepARTMAPParameters() {
        this(DEFAULT.vigilance, DEFAULT.learningRate, DEFAULT.maxCategories, DEFAULT.enableDeepMapping);
    }
    
    /**
     * Compact constructor with parameter validation.
     */
    public DeepARTMAPParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("vigilance must be between 0.0 and 1.0, got: " + vigilance);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("learningRate must be between 0.0 and 1.0, got: " + learningRate);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("maxCategories must be positive, got: " + maxCategories);
        }
    }
    
    /**
     * Create parameters with specified vigilance, using defaults for other values.
     * 
     * @param vigilance the vigilance parameter
     * @return new parameters instance
     */
    public static DeepARTMAPParameters withVigilance(double vigilance) {
        return new DeepARTMAPParameters(vigilance, DEFAULT.learningRate, DEFAULT.maxCategories, DEFAULT.enableDeepMapping);
    }
    
    /**
     * Create parameters with specified learning rate, using defaults for other values.
     * 
     * @param learningRate the learning rate
     * @return new parameters instance
     */
    public static DeepARTMAPParameters withLearningRate(double learningRate) {
        return new DeepARTMAPParameters(DEFAULT.vigilance, learningRate, DEFAULT.maxCategories, DEFAULT.enableDeepMapping);
    }
    
    /**
     * Create parameters with specified max categories, using defaults for other values.
     * 
     * @param maxCategories the maximum categories per layer
     * @return new parameters instance
     */
    public static DeepARTMAPParameters withMaxCategories(int maxCategories) {
        return new DeepARTMAPParameters(DEFAULT.vigilance, DEFAULT.learningRate, maxCategories, DEFAULT.enableDeepMapping);
    }
    
    /**
     * Create a copy of these parameters with a different vigilance value.
     * 
     * @param newVigilance the new vigilance value
     * @return new parameters instance
     */
    public DeepARTMAPParameters copyWithVigilance(double newVigilance) {
        return new DeepARTMAPParameters(newVigilance, learningRate, maxCategories, enableDeepMapping);
    }
    
    /**
     * Create a copy of these parameters with a different learning rate.
     * 
     * @param newLearningRate the new learning rate
     * @return new parameters instance
     */
    public DeepARTMAPParameters copyWithLearningRate(double newLearningRate) {
        return new DeepARTMAPParameters(vigilance, newLearningRate, maxCategories, enableDeepMapping);
    }
    
    /**
     * Create a copy of these parameters with a different max categories.
     * 
     * @param newMaxCategories the new max categories
     * @return new parameters instance
     */
    public DeepARTMAPParameters copyWithMaxCategories(int newMaxCategories) {
        return new DeepARTMAPParameters(vigilance, learningRate, newMaxCategories, enableDeepMapping);
    }
    
    /**
     * Create a copy of these parameters with deep mapping enabled/disabled.
     * 
     * @param enabled whether to enable deep mapping
     * @return new parameters instance
     */
    public DeepARTMAPParameters withDeepMapping(boolean enabled) {
        return new DeepARTMAPParameters(vigilance, learningRate, maxCategories, enabled);
    }
}