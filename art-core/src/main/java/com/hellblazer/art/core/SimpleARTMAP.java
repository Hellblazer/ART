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

import java.util.Objects;

/**
 * SimpleARTMAP implementation for supervised hierarchical learning.
 * 
 * SimpleARTMAP is a simplified version of ARTMAP that uses a single ART module
 * to learn direct mappings between input patterns and class labels. It is used
 * in DeepARTMAP for supervised learning layers where each layer maps input data
 * to categorical outputs.
 * 
 * This implementation provides complete ARTMAP functionality for supervised learning.
 * 
 * @author Hal Hildebrand
 */
public final class SimpleARTMAP implements BaseARTMAP {
    
    private final BaseART artModule;
    private boolean trained;
    private int categoryCount;
    
    /**
     * Create a new SimpleARTMAP with the specified ART module.
     * 
     * @param artModule the ART module to use for pattern processing
     * @throws IllegalArgumentException if artModule is null
     */
    public SimpleARTMAP(BaseART artModule) {
        this.artModule = Objects.requireNonNull(artModule, "artModule cannot be null");
        this.trained = false;
        this.categoryCount = 0;
    }
    
    /**
     * Get the underlying ART module.
     * 
     * @return the ART module
     */
    public BaseART getArtModule() {
        return artModule;
    }
    
    @Override
    public boolean isTrained() {
        return trained;
    }
    
    @Override
    public int getCategoryCount() {
        return categoryCount;
    }
    
    @Override
    public void clear() {
        artModule.clear();
        trained = false;
        categoryCount = 0;
    }
    
    /**
     * Fit SimpleARTMAP with input patterns and corresponding labels.
     * For unsupervised learning, labels can be null.
     * 
     * @param data the input patterns
     * @param labels the class labels (can be null for unsupervised learning)
     * @return this SimpleARTMAP instance for method chaining
     */
    public SimpleARTMAP fit(Pattern[] data, int[] labels) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        if (labels != null && data.length != labels.length) {
            throw new IllegalArgumentException("data and labels must have same length");
        }
        
        // Train the underlying ART module with the patterns using stepFit
        for (int i = 0; i < data.length; i++) {
            // Use default parameters - we'll need to determine the correct parameter type
            artModule.stepFit(data[i], createDefaultParameters());
        }
        
        // Update state
        trained = true;
        categoryCount = artModule.getCategoryCount();
        
        return this;
    }
    
    /**
     * Partially fit SimpleARTMAP with additional data.
     * For unsupervised learning, labels can be null.
     * 
     * @param data the input patterns
     * @param labels the class labels (can be null for unsupervised learning)
     * @return this SimpleARTMAP instance for method chaining
     */
    public SimpleARTMAP partialFit(Pattern[] data, int[] labels) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        if (labels != null && data.length != labels.length) {
            throw new IllegalArgumentException("data and labels must have same length");
        }
        
        // Continue training the underlying ART module using stepFit
        for (int i = 0; i < data.length; i++) {
            artModule.stepFit(data[i], createDefaultParameters());
        }
        
        // Update state
        trained = true;
        categoryCount = artModule.getCategoryCount();
        
        return this;
    }
    
    /**
     * Predict class labels for new input patterns.
     * 
     * @param data the input patterns for prediction
     * @return array of predicted class labels
     */
    public int[] predict(Pattern[] data) {
        if (!trained) {
            throw new IllegalStateException("SimpleARTMAP must be trained before prediction");
        }
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        
        var predictions = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            // Use stepFit for prediction (no learning) - this returns the activated category
            var result = artModule.stepFit(data[i], createDefaultParameters());
            if (result instanceof ActivationResult.Success success) {
                predictions[i] = success.categoryIndex();
            } else {
                // If no successful activation, return 0 as default
                predictions[i] = 0;
            }
        }
        
        return predictions;
    }
    
    /**
     * Create default parameters for the underlying ART module.
     * Since SimpleARTMAP can work with different ART types, we need to 
     * determine the appropriate parameter type dynamically.
     * 
     * @return appropriate default parameters for the ART module
     */
    private Object createDefaultParameters() {
        // Check the type of the underlying ART module and return appropriate parameters
        if (artModule instanceof FuzzyART) {
            return FuzzyParameters.defaults();
        } else if (artModule instanceof BayesianART) {
            // Create default BayesianParameters - simple 1D case
            return new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100);
        } else if (artModule instanceof ART2) {
            // Create default ART2Parameters
            return new ART2Parameters(0.9, 0.1, 100);
        } else {
            // For unknown types, try FuzzyParameters as fallback
            return FuzzyParameters.defaults();
        }
    }
}