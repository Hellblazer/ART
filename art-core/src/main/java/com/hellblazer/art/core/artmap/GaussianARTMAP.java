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

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.GaussianART;
import com.hellblazer.art.core.parameters.GaussianParameters;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;

import java.util.Arrays;
import java.util.Objects;

/**
 * GaussianARTMAP for probabilistic supervised learning.
 * 
 * This class implements GaussianARTMAP, which is a specialized version of SimpleARTMAP
 * that uses GaussianART as its underlying clustering module. It is designed for
 * continuous data and provides probabilistic classification based on Gaussian distributions.
 * 
 * GaussianARTMAP combines:
 * - GaussianART for probabilistic clustering with Gaussian distributions
 * - SimpleARTMAP's map field for cluster-to-label associations
 * - Match tracking for handling label conflicts
 * 
 * Key features:
 * - Probabilistic classification based on likelihood
 * - Adaptive variance learning for each cluster
 * - Handles continuous data naturally
 * - Incremental learning support
 * 
 * @author Hal Hildebrand
 */
public class GaussianARTMAP {
    
    private SimpleARTMAP artmap;
    private final double rho;
    private final double[] sigmaInit;
    private final double alpha;
    
    /**
     * Result of prediction including likelihood information.
     */
    public record PredictionResult(int clusterIndex, int classLabel, double likelihood) {}
    
    /**
     * Create a new GaussianARTMAP with specified parameters.
     * 
     * @param rho vigilance parameter (0.0 to 1.0)
     * @param sigmaInit initial standard deviations for each dimension
     * @param alpha small constant to avoid division by zero (typically 1e-10)
     */
    public GaussianARTMAP(double rho, double[] sigmaInit, double alpha) {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException("Vigilance (rho) must be between 0.0 and 1.0");
        }
        
        Objects.requireNonNull(sigmaInit, "sigmaInit cannot be null");
        if (sigmaInit.length == 0) {
            throw new IllegalArgumentException("sigmaInit must have at least one element");
        }
        
        // Validate sigma values
        for (int i = 0; i < sigmaInit.length; i++) {
            if (sigmaInit[i] <= 0.0) {
                throw new IllegalArgumentException(
                    "All sigma values must be positive. Element " + i + " is " + sigmaInit[i]
                );
            }
        }
        
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Alpha must be positive");
        }
        
        this.rho = rho;
        this.sigmaInit = Arrays.copyOf(sigmaInit, sigmaInit.length);
        this.alpha = alpha;
        
        // Will be initialized when we know the input dimension
        this.artmap = null;
    }
    
    /**
     * Initialize the internal ARTMAP with the correct parameters.
     */
    private void initializeARTMAP(int inputDimension) {
        if (this.artmap != null) {
            return;  // Already initialized
        }
        
        // Validate that sigmaInit matches input dimension
        if (sigmaInit.length != inputDimension) {
            throw new IllegalArgumentException(
                "sigmaInit dimension (" + sigmaInit.length + 
                ") must match input dimension (" + inputDimension + ")"
            );
        }
        
        // Create GaussianART module
        var gaussianART = new GaussianART();
        
        // Create SimpleARTMAP parameters
        var mapParams = new SimpleARTMAPParameters(0.95, 1e-10);
        
        this.artmap = new SimpleARTMAP(gaussianART, mapParams);
    }
    
    /**
     * Fit the model to the training data.
     * 
     * @param X training data matrix where each row is a pattern
     * @param y class labels for each pattern
     * @return this instance for method chaining
     */
    public GaussianARTMAP fit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP with input dimension
        initializeARTMAP(X[0].length);
        
        // Convert to Pattern objects and train
        for (int i = 0; i < X.length; i++) {
            var pattern = Pattern.of(X[i]);
            var artParams = new GaussianParameters(rho, sigmaInit);
            
            artmap.train(pattern, y[i], artParams);
        }
        
        return this;
    }
    
    /**
     * Incrementally fit the model with new data.
     * 
     * @param X new training data matrix
     * @param y new class labels
     * @return this instance for method chaining
     */
    public GaussianARTMAP partialFit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP if not already initialized
        initializeARTMAP(X[0].length);
        
        // Train incrementally
        for (int i = 0; i < X.length; i++) {
            var pattern = Pattern.of(X[i]);
            var artParams = new GaussianParameters(rho, sigmaInit);
            
            artmap.train(pattern, y[i], artParams);
        }
        
        return this;
    }
    
    /**
     * Predict the class label for a single pattern.
     * 
     * @param x input pattern
     * @return predicted class label
     */
    public int predict(double[] x) {
        Objects.requireNonNull(x, "Input pattern cannot be null");
        
        if (artmap == null) {
            throw new IllegalStateException("Model has not been trained yet. Call fit() first.");
        }
        
        var pattern = Pattern.of(x);
        var artParams = new GaussianParameters(rho, sigmaInit);
        
        return artmap.predict(pattern, artParams);
    }
    
    /**
     * Predict class label with likelihood information.
     * 
     * @param x input pattern
     * @return prediction result with cluster index, class label, and likelihood
     */
    public PredictionResult predictWithLikelihood(double[] x) {
        Objects.requireNonNull(x, "Input pattern cannot be null");
        
        if (artmap == null) {
            throw new IllegalStateException("Model has not been trained yet. Call fit() first.");
        }
        
        var pattern = Pattern.of(x);
        var artParams = new GaussianParameters(rho, sigmaInit);
        
        // Use the internal ART module to get cluster and activation
        var moduleA = artmap.getModuleA();
        var result = moduleA.stepPredict(pattern, artParams);
        
        if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
            int clusterIndex = success.categoryIndex();
            double likelihood = success.activationValue();  // Gaussian likelihood
            int classLabel = artmap.predict(pattern, artParams);
            return new PredictionResult(clusterIndex, classLabel, likelihood);
        } else {
            // No match found
            return new PredictionResult(-1, -1, 0.0);
        }
    }
    
    /**
     * Get the number of categories (clusters) formed.
     * 
     * @return number of categories
     */
    public int getCategoryCount() {
        return artmap == null ? 0 : artmap.getCategoryCount();
    }
    
    /**
     * Clear all learned patterns and reset the model.
     */
    public void clear() {
        if (artmap != null) {
            artmap.clear();
        }
    }
    
    /**
     * Validate input data and labels.
     * 
     * @param X input data matrix
     * @param y labels array
     */
    private void validateData(double[][] X, int[] y) {
        Objects.requireNonNull(X, "Input data cannot be null");
        Objects.requireNonNull(y, "Labels cannot be null");
        
        if (X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be empty");
        }
        
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "Number of samples (" + X.length + ") must match number of labels (" + y.length + ")"
            );
        }
        
        // Check for consistent dimensions
        int dim = X[0].length;
        for (int i = 1; i < X.length; i++) {
            if (X[i].length != dim) {
                throw new IllegalArgumentException(
                    "All patterns must have the same dimension. Pattern 0 has " + dim + 
                    " dimensions but pattern " + i + " has " + X[i].length
                );
            }
        }
    }
}