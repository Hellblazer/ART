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

import java.util.Map;

/**
 * ScikitClusterer interface for scikit-learn compatibility in ART networks.
 * Provides standard ML interface for clustering operations including fit, predict, and evaluation.
 * 
 * @author Hal Hildebrand
 */
public interface ScikitClusterer<X> extends AutoCloseable {
    
    /**
     * Fit the clusterer to the training data.
     */
    ScikitClusterer<X> fit(X[] X_data);
    
    /**
     * Fit the clusterer to 2D double array data.
     */
    ScikitClusterer<X> fit(double[][] X_data);
    
    /**
     * Predict cluster labels for the input data.
     */
    Integer[] predict(X[] X_data);
    
    /**
     * Predict cluster labels for 2D double array data.
     */
    Integer[] predict(double[][] X_data);
    
    /**
     * Predict cluster probabilities for the input data.
     */
    double[][] predict_proba(X[] X_data);
    
    /**
     * Predict cluster probabilities for 2D double array data.
     */
    double[][] predict_proba(double[][] X_data);
    
    /**
     * Get cluster centers.
     */
    X[] cluster_centers();
    
    /**
     * Calculate clustering metrics.
     */
    Map<String, Double> clustering_metrics(X[] X_data, Integer[] labels);
    
    /**
     * Calculate clustering metrics for 2D double array data.
     */
    Map<String, Double> clustering_metrics(double[][] X_data, Integer[] labels);
    
    /**
     * Get model parameters.
     */
    Map<String, Object> get_params();
    
    /**
     * Set model parameters.
     */
    ScikitClusterer<X> set_params(Map<String, Object> params);
    
    /**
     * Check if the model is fitted.
     */
    boolean is_fitted();
    
    @Override
    default void close() throws Exception {
        // Default empty implementation
    }
}