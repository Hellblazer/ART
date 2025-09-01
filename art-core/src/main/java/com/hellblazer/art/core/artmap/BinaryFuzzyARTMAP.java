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
import com.hellblazer.art.core.algorithms.BinaryFuzzyART;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;
import com.hellblazer.art.core.MatchTrackingMethod;

import java.util.Arrays;
import java.util.Objects;

/**
 * BinaryFuzzyARTMAP for binary pattern classification.
 * 
 * This class implements BinaryFuzzyARTMAP, which is a specialized version of SimpleARTMAP
 * that uses BinaryFuzzyART as its underlying clustering module. It is optimized for
 * binary input patterns and provides efficient supervised learning capabilities.
 * 
 * BinaryFuzzyARTMAP combines:
 * - BinaryFuzzyART for binary pattern clustering with complement coding
 * - SimpleARTMAP's map field for cluster-to-label associations
 * - Match tracking for handling label conflicts
 * 
 * Key features:
 * - Optimized for binary (0/1) input patterns
 * - Automatic complement coding for improved stability
 * - Fast learning with adjustable vigilance
 * - Incremental learning support
 * 
 * @author Hal Hildebrand
 */
public class BinaryFuzzyARTMAP {
    
    private SimpleARTMAP artmap;
    private final double rho;
    private final double alpha;
    
    /**
     * Result of prediction including both cluster index and class label.
     */
    public record PredictionResult(int clusterIndex, int classLabel) {}
    
    /**
     * Create a new BinaryFuzzyARTMAP with specified parameters.
     * 
     * @param rho vigilance parameter (0.0 to 1.0)
     * @param alpha choice parameter (> 0.0, typically small like 0.01)
     */
    public BinaryFuzzyARTMAP(double rho, double alpha) {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException("Vigilance (rho) must be between 0.0 and 1.0");
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Choice parameter (alpha) must be positive");
        }
        
        this.rho = rho;
        this.alpha = alpha;
        
        // We'll initialize the ART module with actual data dimension later
        this.artmap = null;  // Will be created in fit()
    }
    
    /**
     * Initialize the internal ARTMAP with the correct input size.
     */
    private void initializeARTMAP(int inputDimension) {
        if (this.artmap != null) {
            return;  // Already initialized
        }
        
        // Create BinaryFuzzyART module with parameters
        // Note: BinaryFuzzyART uses complement coding, so actual dimension is 2 * inputDimension
        var artParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(rho)
            .alpha(alpha)
            .beta(1.0)  // Fast learning
            .build();
        
        var binaryFuzzyART = new BinaryFuzzyART(inputDimension, artParams);
        
        // Create SimpleARTMAP parameters using factory method
        var mapParams = new SimpleARTMAPParameters(0.95, 1e-10);
        
        this.artmap = new SimpleARTMAP(binaryFuzzyART, mapParams);
    }
    
    /**
     * Fit the model to the training data.
     * 
     * @param X training data matrix where each row is a binary pattern
     * @param y class labels for each pattern
     * @return this instance for method chaining
     */
    public BinaryFuzzyARTMAP fit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP with input dimension
        initializeARTMAP(X[0].length);
        
        // Convert to Pattern objects and train
        for (int i = 0; i < X.length; i++) {
            var pattern = createBinaryPattern(X[i]);
            var artParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(rho)
                .alpha(alpha)
                .beta(1.0)
                .build();
            
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
    public BinaryFuzzyARTMAP partialFit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP if not already initialized
        initializeARTMAP(X[0].length);
        
        // Train incrementally
        for (int i = 0; i < X.length; i++) {
            var pattern = createBinaryPattern(X[i]);
            var artParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(rho)
                .alpha(alpha)
                .beta(1.0)
                .build();
            
            artmap.train(pattern, y[i], artParams);
        }
        
        return this;
    }
    
    /**
     * Predict the class label for a single pattern.
     * 
     * @param x input binary pattern
     * @return predicted class label
     */
    public int predict(double[] x) {
        Objects.requireNonNull(x, "Input pattern cannot be null");
        
        if (artmap == null) {
            throw new IllegalStateException("Model has not been trained yet. Call fit() first.");
        }
        
        var pattern = createBinaryPattern(x);
        var artParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(rho)
            .alpha(alpha)
            .beta(1.0)
            .build();
        
        return artmap.predict(pattern, artParams);
    }
    
    /**
     * Predict both cluster index and class label for a pattern.
     * 
     * @param x input binary pattern
     * @return prediction result with cluster index and class label
     */
    public PredictionResult predictAB(double[] x) {
        Objects.requireNonNull(x, "Input pattern cannot be null");
        
        if (artmap == null) {
            throw new IllegalStateException("Model has not been trained yet. Call fit() first.");
        }
        
        var pattern = createBinaryPattern(x);
        var artParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(rho)
            .alpha(alpha)
            .beta(1.0)
            .build();
        
        // Use the internal ART module to get cluster and then map to label
        var moduleA = artmap.getModuleA();
        var result = moduleA.stepPredict(pattern, artParams);
        
        if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
            int clusterIndex = success.categoryIndex();
            int classLabel = artmap.predict(pattern, artParams);
            return new PredictionResult(clusterIndex, classLabel);
        } else {
            // No match found
            return new PredictionResult(-1, -1);
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
     * Create a binary pattern with complement coding.
     * 
     * @param data input array
     * @return Pattern object with complement coding
     */
    private Pattern createBinaryPattern(double[] data) {
        // Apply complement coding for BinaryFuzzyART
        int originalDim = data.length;
        double[] complementCoded = new double[originalDim * 2];
        
        // First half: original data (ensure binary)
        for (int i = 0; i < originalDim; i++) {
            complementCoded[i] = (data[i] > 0.5) ? 1.0 : 0.0;
        }
        
        // Second half: complement
        for (int i = 0; i < originalDim; i++) {
            complementCoded[originalDim + i] = 1.0 - complementCoded[i];
        }
        
        return Pattern.of(complementCoded);
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