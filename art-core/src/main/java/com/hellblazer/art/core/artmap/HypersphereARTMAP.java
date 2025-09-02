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
import com.hellblazer.art.core.algorithms.HypersphereART;
import com.hellblazer.art.core.parameters.HypersphereParameters;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;

import java.util.Objects;

/**
 * HypersphereARTMAP for spherical cluster-based supervised learning.
 * 
 * This class implements HypersphereARTMAP, which is a specialized version of SimpleARTMAP
 * that uses HypersphereART as its underlying clustering module. It is designed for
 * supervised learning with spherical clusters bounded by a maximum radius.
 * 
 * HypersphereARTMAP combines:
 * - HypersphereART for spherical clustering with radius bounds
 * - SimpleARTMAP's map field for cluster-to-label associations
 * - Match tracking for handling label conflicts
 * 
 * Key features:
 * - Spherical clusters with bounded radius (r_hat parameter)
 * - Efficient distance-based classification
 * - Incremental learning support
 * - Choice parameter (alpha) for tie-breaking
 * - Learning rate control (beta parameter)
 * 
 * Reference: Anagnostopoulos & Georgiopoulos (2000) "Hypersphere ART and ARTMAP for 
 * unsupervised and supervised incremental learning"
 * 
 * @author Hal Hildebrand
 */
public class HypersphereARTMAP {
    
    private SimpleARTMAP artmap;
    private final double rho;      // Vigilance parameter
    private final double alpha;    // Choice parameter
    private final double beta;     // Learning rate
    private final double rHat;     // Maximum radius bound
    
    /**
     * Create a new HypersphereARTMAP with specified parameters.
     * 
     * @param rho vigilance parameter (0.0 to 1.0)
     * @param alpha choice parameter for tie-breaking (typically 1e-10)
     * @param beta learning rate parameter (0.0 to 1.0)
     * @param rHat global upper bound on cluster radius (must be > 0)
     */
    public HypersphereARTMAP(double rho, double alpha, double beta, double rHat) {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException("Vigilance (rho) must be between 0.0 and 1.0");
        }
        
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be between 0.0 and 1.0");
        }
        
        if (rHat <= 0.0) {
            throw new IllegalArgumentException("r_hat must be positive");
        }
        
        this.rho = rho;
        this.alpha = alpha;
        this.beta = beta;
        this.rHat = rHat;
        
        // Will be initialized when we know the input dimension
        this.artmap = null;
    }
    
    /**
     * Initialize the internal ARTMAP with the correct parameters.
     */
    private void initializeARTMAP() {
        if (this.artmap != null) {
            return;  // Already initialized
        }
        
        // Create HypersphereART module with adaptive radius
        var hypersphereART = new HypersphereART();
        
        // Create SimpleARTMAP parameters
        var mapParams = new SimpleARTMAPParameters(0.95, 1e-10);
        
        this.artmap = new SimpleARTMAP(hypersphereART, mapParams);
    }
    
    /**
     * Fit the model to the training data.
     * 
     * @param X training data matrix where each row is a pattern
     * @param y class labels for each pattern
     * @return this instance for method chaining
     */
    public HypersphereARTMAP fit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP
        initializeARTMAP();
        
        // Convert to Pattern objects and train
        for (int i = 0; i < X.length; i++) {
            var pattern = Pattern.of(X[i]);
            var artParams = new HypersphereParameters(rho, rHat, true);  // adaptive=true
            
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
    public HypersphereARTMAP partialFit(double[][] X, int[] y) {
        validateData(X, y);
        
        // Initialize ARTMAP if not already initialized
        initializeARTMAP();
        
        // Train incrementally
        for (int i = 0; i < X.length; i++) {
            var pattern = Pattern.of(X[i]);
            var artParams = new HypersphereParameters(rho, rHat, true);
            
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
        var artParams = new HypersphereParameters(rho, rHat, true);
        
        return artmap.predict(pattern, artParams);
    }
    
    /**
     * Predict class labels for multiple patterns.
     * 
     * @param X input data matrix
     * @return array of predicted class labels
     */
    public int[] predict(double[][] X) {
        Objects.requireNonNull(X, "Input data cannot be null");
        
        var predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        
        return predictions;
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