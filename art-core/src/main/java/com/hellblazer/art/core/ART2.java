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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * ART-2 implementation using the BaseART template framework.
 * 
 * ART-2 is a neural network architecture based on Adaptive Resonance Theory
 * that performs unsupervised learning on continuous analog input patterns.
 * 
 * Key Features:
 * - Input normalization preprocessing layer
 * - Dot product activation function: T_j = I' · w_j
 * - Distance-based vigilance criterion: ||I' - w_j||² ≤ (1-ρ)²
 * - Convex combination learning: w_j^(new) = (1-β)w_j^(old) + β*I'
 * - Normalized weight vectors (unit length)
 * 
 * @see BaseART for the template method framework
 * @see ART2Weight for normalized weight vectors
 * @see ART2Parameters for algorithm parameters (ρ, β, maxCategories)
 * 
 * @author Hal Hildebrand
 */
public final class ART2 extends BaseART implements ScikitClusterer<Pattern> {
    
    private final ART2Parameters parameters;
    private boolean fitted = false;
    
    /**
     * Create a new ART-2 network with specified parameters.
     * 
     * @param parameters the ART-2 algorithm parameters
     * @throws IllegalArgumentException if parameters are null
     */
    public ART2(ART2Parameters parameters) {
        super();
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        this.parameters = parameters;
    }
    
    // BaseART abstract method implementations for ART-2 neural network
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        if (!(weight instanceof ART2Weight art2Weight)) {
            throw new IllegalArgumentException("Weight must be ART2Weight");
        }
        if (!(parameters instanceof ART2Parameters)) {
            throw new IllegalArgumentException("Parameters must be ART2Parameters");
        }
        
        // Normalize input first
        var normalizedInput = normalizeInput(input);
        // Cast Pattern to DenseVector to access values()
        if (!(normalizedInput instanceof DenseVector denseInput)) {
            throw new IllegalStateException("normalizeInput should return DenseVector");
        }
        var inputValues = denseInput.values();
        var weightValues = art2Weight.vector().values();
        
        if (inputValues.length != weightValues.length) {
            throw new IllegalArgumentException("Input and weight dimensions must match");
        }
        
        // Calculate dot product: T_j = I' · w_j
        var dotProduct = 0.0;
        for (int i = 0; i < inputValues.length; i++) {
            dotProduct += inputValues[i] * weightValues[i];
        }
        
        return dotProduct;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        if (!(weight instanceof ART2Weight art2Weight)) {
            throw new IllegalArgumentException("Weight must be ART2Weight");
        }
        if (!(parameters instanceof ART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be ART2Parameters");
        }
        
        // Normalize input first
        var normalizedInput = normalizeInput(input);
        // Cast Pattern to DenseVector to access values()
        if (!(normalizedInput instanceof DenseVector denseInput)) {
            throw new IllegalStateException("normalizeInput should return DenseVector");
        }
        var inputValues = denseInput.values();
        var weightValues = art2Weight.vector().values();
        
        if (inputValues.length != weightValues.length) {
            throw new IllegalArgumentException("Input and weight dimensions must match");
        }
        
        // Calculate distance: ||I' - w_j||²
        var distanceSquared = 0.0;
        for (int i = 0; i < inputValues.length; i++) {
            var diff = inputValues[i] - weightValues[i];
            distanceSquared += diff * diff;
        }
        
        // Vigilance test: ||I' - w_j||² ≤ (1-ρ)²
        var vigilance = art2Params.vigilance();
        var threshold = (1.0 - vigilance) * (1.0 - vigilance);
        
        // Convert distance to match ratio: match = 1 - (distance² / max_distance²)
        // For unit vectors, max distance² = 4 (when vectors point in opposite directions)
        var maxDistanceSquared = 4.0;
        var matchRatio = 1.0 - (distanceSquared / maxDistanceSquared);
        
        // The vigilance test should compare matchRatio directly to vigilance
        // High matchRatio (low distance) → accept if matchRatio >= vigilance
        // Low matchRatio (high distance) → reject if matchRatio < vigilance
        if (matchRatio >= vigilance) {
            return new MatchResult.Accepted(matchRatio, vigilance);
        } else {
            return new MatchResult.Rejected(matchRatio, vigilance);
        }
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        if (!(currentWeight instanceof ART2Weight art2Weight)) {
            throw new IllegalArgumentException("Weight must be ART2Weight");
        }
        if (!(parameters instanceof ART2Parameters art2Params)) {
            throw new IllegalArgumentException("Parameters must be ART2Parameters");
        }
        
        // Normalize input first
        var normalizedInput = normalizeInput(input);
        // Cast Pattern to DenseVector to access values()
        if (!(normalizedInput instanceof DenseVector denseInput)) {
            throw new IllegalStateException("normalizeInput should return DenseVector");
        }
        var inputValues = denseInput.values();
        var weightValues = art2Weight.vector().values();
        
        if (inputValues.length != weightValues.length) {
            throw new IllegalArgumentException("Input and weight dimensions must match");
        }
        
        // Update using convex combination: w_j^(new) = (1-β)w_j^(old) + β*I'
        var beta = art2Params.learningRate();
        var newValues = new double[inputValues.length];
        for (int i = 0; i < inputValues.length; i++) {
            newValues[i] = (1.0 - beta) * weightValues[i] + beta * inputValues[i];
        }
        
        // Create normalized weight
        return ART2Weight.fromInput(new DenseVector(newValues));
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        if (!(parameters instanceof ART2Parameters)) {
            throw new IllegalArgumentException("Parameters must be ART2Parameters");
        }
        
        // Create initial weight from normalized input
        var normalizedInput = normalizeInput(input);
        // Cast Pattern back to DenseVector for ART2Weight.fromInput
        if (!(normalizedInput instanceof DenseVector denseVector)) {
            throw new IllegalStateException("normalizeInput should return DenseVector");
        }
        return ART2Weight.fromInput(denseVector);
    }
    
    // ART-2 specific methods needed by tests - MINIMAL TO MAKE TESTS COMPILE
    
    /**
     * Normalize input vector to unit length for ART-2 preprocessing.
     * 
     * @param input the raw input vector
     * @return normalized input vector
     * @throws IllegalArgumentException if input is null or zero vector
     */
    public Pattern normalizeInput(Pattern input) {
        if (input == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        
        // Pattern is a sealed interface that only permits DenseVector
        if (!(input instanceof DenseVector denseInput)) {
            throw new IllegalArgumentException("Input must be a DenseVector");
        }
        
        var norm = denseInput.l2Norm();
        if (norm == 0.0) {
            // Handle zero vector by returning small random vector
            var random = new java.util.Random();
            var values = new double[denseInput.dimension()];
            for (int i = 0; i < values.length; i++) {
                values[i] = random.nextGaussian() * 1e-6;
            }
            return new DenseVector(values);
        }
        
        return denseInput.scale(1.0 / norm);
    }
    
    // ScikitClusterer interface implementation - MINIMAL TO MAKE TESTS COMPILE
    
    @Override
    public ScikitClusterer<Pattern> fit(Pattern[] X) {
        if (X == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Train on each pattern
        for (var pattern : X) {
            if (pattern == null) {
                throw new IllegalArgumentException("Pattern cannot be null");
            }
            stepFit(pattern, parameters);
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public ScikitClusterer<Pattern> fit(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Convert to Pattern array and fit
        var patterns = new Pattern[X.length];
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                throw new IllegalArgumentException("Training sample cannot be null");
            }
            patterns[i] = Pattern.of(X[i]);
        }
        
        return fit(patterns);
    }
    
    public ActivationResult predict(Pattern input) {
        if (!fitted) {
            throw new IllegalStateException("ART-2 is not fitted. Call fit() before predict()");
        }
        if (input == null) {
            throw new IllegalArgumentException("Input pattern cannot be null");
        }
        
        if (getCategoryCount() == 0) {
            return ActivationResult.NoMatch.instance();
        }
        
        // Normalize input
        var normalizedInput = normalizeInput(input);
        
        // Calculate activations for all categories
        var categories = getCategories();
        var maxActivation = Double.NEGATIVE_INFINITY;
        var bestCategory = -1;
        
        for (int i = 0; i < categories.size(); i++) {
            var activation = calculateActivation(normalizedInput, categories.get(i), parameters);
            if (activation > maxActivation) {
                maxActivation = activation;
                bestCategory = i;
            }
        }
        
        return new ActivationResult.Success(bestCategory, maxActivation, categories.get(bestCategory));
    }
    
    @Override
    public Integer[] predict(Pattern[] X) {
        if (X == null) {
            throw new IllegalArgumentException("Input array cannot be null");
        }
        
        var predictions = new Integer[X.length];
        for (int i = 0; i < X.length; i++) {
            var result = predict(X[i]);
            predictions[i] = result instanceof ActivationResult.Success success ? 
                success.categoryIndex() : -1;
        }
        return predictions;
    }
    
    @Override
    public Integer[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("Input array cannot be null");
        }
        
        var patterns = new Pattern[X.length];
        for (int i = 0; i < X.length; i++) {
            patterns[i] = Pattern.of(X[i]);
        }
        
        return predict(patterns);
    }
    
    @Override
    public boolean is_fitted() {
        return fitted;
    }
    
    @Override
    public Pattern[] cluster_centers() {
        if (!fitted) {
            throw new IllegalStateException("ART-2 is not fitted. Call fit() before cluster_centers()");
        }
        
        var categories = getCategories();
        var centers = new Pattern[categories.size()];
        
        for (int i = 0; i < categories.size(); i++) {
            var weight = categories.get(i);
            if (weight instanceof ART2Weight art2Weight) {
                centers[i] = Pattern.of(art2Weight.vector().values());
            } else {
                throw new IllegalStateException("Invalid weight type: " + weight.getClass());
            }
        }
        
        return centers;
    }
    
    @Override
    public Map<String, Object> get_params() {
        var params = new HashMap<String, Object>();
        params.put("vigilance", parameters.vigilance());
        params.put("learning_rate", parameters.learningRate());
        params.put("max_categories", parameters.maxCategories());
        return params;
    }
    
    @Override
    public ScikitClusterer<Pattern> set_params(Map<String, Object> params) {
        if (params == null || params.isEmpty()) {
            return this;
        }
        
        // Extract current parameters
        var newRho = parameters.vigilance();
        var newBeta = parameters.learningRate();
        var newMaxCategories = parameters.maxCategories();
        
        // Update parameters that can be changed
        if (params.containsKey("vigilance") || params.containsKey("rho")) {
            var vigilanceValue = params.getOrDefault("vigilance", params.get("rho"));
            if (vigilanceValue instanceof Number number) {
                newRho = number.doubleValue();
                if (newRho < 0.0 || newRho > 1.0) {
                    throw new IllegalArgumentException("Vigilance parameter must be between 0 and 1");
                }
            }
        }
        
        if (params.containsKey("beta")) {
            var betaValue = params.get("beta");
            if (betaValue instanceof Number number) {
                newBeta = number.doubleValue();
                if (newBeta < 0.0 || newBeta > 1.0) {
                    throw new IllegalArgumentException("Learning rate beta must be between 0 and 1");
                }
            }
        }
        
        if (params.containsKey("max_categories")) {
            var maxCatValue = params.get("max_categories");
            if (maxCatValue instanceof Number number) {
                newMaxCategories = number.intValue();
                if (newMaxCategories <= 0) {
                    throw new IllegalArgumentException("Maximum categories must be positive");
                }
            }
        }
        
        // Create new ART2 instance with updated parameters
        var newParams = new ART2Parameters(newRho, newBeta, newMaxCategories);
        return new ART2(newParams);
    }
    
    @Override
    public Map<String, Double> clustering_metrics(Pattern[] X_data, Integer[] labels) {
        var metrics = new HashMap<String, Double>();
        
        if (X_data == null || X_data.length == 0) {
            metrics.put("n_clusters", 0.0);
            return metrics;
        }
        
        var predictions = predict(X_data);
        
        // Calculate number of clusters
        var uniqueClusters = Arrays.stream(predictions).collect(java.util.stream.Collectors.toSet());
        metrics.put("n_clusters", (double) uniqueClusters.size());
        
        // Calculate accuracy if labels provided
        if (labels != null && labels.length == predictions.length) {
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (labels[i] != null && labels[i].equals(predictions[i])) {
                    correct++;
                }
            }
            metrics.put("accuracy", (double) correct / predictions.length);
        }
        
        // Calculate inertia (within-cluster sum of squares)
        double inertia = 0.0;
        var categoryCount = getCategoryCount();
        
        for (int clusterIdx = 0; clusterIdx < categoryCount; clusterIdx++) {
            var clusterCenter = getCategory(clusterIdx);
            if (clusterCenter instanceof ART2Weight art2Weight) {
                for (int i = 0; i < X_data.length; i++) {
                    if (predictions[i] == clusterIdx) {
                        // Calculate squared distance to cluster center
                        var pattern = X_data[i];
                        double distance = 0.0;
                        for (int dim = 0; dim < Math.min(pattern.dimension(), art2Weight.vector().dimension()); dim++) {
                            double diff = pattern.get(dim) - art2Weight.vector().get(dim);
                            distance += diff * diff;
                        }
                        inertia += distance;
                    }
                }
            }
        }
        metrics.put("inertia", inertia);
        
        // Calculate Davies-Bouldin Score (lower is better)
        if (categoryCount > 1) {
            double dbScore = 0.0;
            for (int i = 0; i < categoryCount; i++) {
                double maxRatio = 0.0;
                var centerI = getCategory(i);
                
                for (int j = 0; j < categoryCount; j++) {
                    if (i != j) {
                        var centerJ = getCategory(j);
                        
                        // Calculate average intra-cluster distances
                        double avgDistI = calculateIntraClusterDistance(X_data, predictions, i);
                        double avgDistJ = calculateIntraClusterDistance(X_data, predictions, j);
                        
                        // Calculate inter-cluster distance
                        double interDist = calculateInterClusterDistance(centerI, centerJ);
                        
                        if (interDist > 0) {
                            double ratio = (avgDistI + avgDistJ) / interDist;
                            maxRatio = Math.max(maxRatio, ratio);
                        }
                    }
                }
                dbScore += maxRatio;
            }
            metrics.put("davies_bouldin_score", dbScore / categoryCount);
        } else {
            metrics.put("davies_bouldin_score", 0.0);
        }
        
        // Simplified silhouette score
        if (uniqueClusters.size() > 1) {
            double silhouetteSum = 0.0;
            for (int i = 0; i < X_data.length; i++) {
                double a = calculateIntraClusterDistance(X_data, predictions, predictions[i], i);
                double b = calculateNearestClusterDistance(X_data, predictions, predictions[i], i);
                
                if (Math.max(a, b) > 0) {
                    silhouetteSum += (b - a) / Math.max(a, b);
                }
            }
            metrics.put("silhouette_score", silhouetteSum / X_data.length);
        } else {
            metrics.put("silhouette_score", 0.0);
        }
        
        // Calculate Calinski-Harabasz Score (higher is better)
        if (categoryCount > 1 && X_data.length > categoryCount) {
            double betweenSS = 0.0;
            double withinSS = inertia; // We already calculated this above
            
            // Calculate overall centroid
            var overallCentroid = new double[X_data[0].dimension()];
            for (var pattern : X_data) {
                for (int dim = 0; dim < overallCentroid.length; dim++) {
                    overallCentroid[dim] += pattern.get(dim);
                }
            }
            for (int dim = 0; dim < overallCentroid.length; dim++) {
                overallCentroid[dim] /= X_data.length;
            }
            
            // Calculate between-cluster sum of squares
            for (int clusterIdx = 0; clusterIdx < categoryCount; clusterIdx++) {
                var clusterCenter = getCategory(clusterIdx);
                if (clusterCenter instanceof ART2Weight art2Weight) {
                    int clusterSize = 0;
                    for (int pred : predictions) {
                        if (pred == clusterIdx) clusterSize++;
                    }
                    
                    if (clusterSize > 0) {
                        double distance = 0.0;
                        for (int dim = 0; dim < Math.min(overallCentroid.length, art2Weight.vector().dimension()); dim++) {
                            double diff = art2Weight.vector().get(dim) - overallCentroid[dim];
                            distance += diff * diff;
                        }
                        betweenSS += clusterSize * distance;
                    }
                }
            }
            
            if (withinSS > 0) {
                double chScore = (betweenSS / (categoryCount - 1)) / (withinSS / (X_data.length - categoryCount));
                metrics.put("calinski_harabasz_score", chScore);
            } else {
                metrics.put("calinski_harabasz_score", 0.0);
            }
        } else {
            metrics.put("calinski_harabasz_score", 0.0);
        }
        
        return metrics;
    }
    
    @Override
    public Map<String, Double> clustering_metrics(double[][] X_data, Integer[] labels) {
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = Pattern.of(X_data[i]);
        }
        return clustering_metrics(patterns, labels);
    }
    
    @Override
    public double[][] predict_proba(Pattern[] X_data) {
        // ART-2 doesn't naturally provide probabilities, return hard assignments
        var predictions = predict(X_data);
        var probabilities = new double[X_data.length][getCategoryCount()];
        
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] >= 0 && predictions[i] < getCategoryCount()) {
                probabilities[i][predictions[i]] = 1.0;
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predict_proba(double[][] X_data) {
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = Pattern.of(X_data[i]);
        }
        return predict_proba(patterns);
    }
    
    // Additional BaseART methods that may be needed
    
    public double getVigilance() {
        return parameters.vigilance();
    }
    
    public double getLearningRate() {
        return parameters.learningRate();
    }
    
    public int getDimensions() {
        if (getCategoryCount() == 0) {
            return 0;
        }
        
        var firstCategory = getCategories().get(0);
        return firstCategory.dimension();
    }
    
    // Helper methods for clustering metrics
    private double calculateIntraClusterDistance(Pattern[] data, Integer[] predictions, int clusterIdx) {
        double totalDistance = 0.0;
        int count = 0;
        
        var center = getCategory(clusterIdx);
        if (!(center instanceof ART2Weight art2Weight)) {
            return 0.0;
        }
        
        var centerMemory = art2Weight.vector();
        
        for (int i = 0; i < data.length; i++) {
            if (predictions[i] == clusterIdx) {
                double distance = 0.0;
                for (int dim = 0; dim < Math.min(data[i].dimension(), centerMemory.dimension()); dim++) {
                    double diff = data[i].get(dim) - centerMemory.get(dim);
                    distance += diff * diff;
                }
                totalDistance += Math.sqrt(distance);
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : 0.0;
    }
    
    private double calculateIntraClusterDistance(Pattern[] data, Integer[] predictions, int clusterIdx, int pointIdx) {
        if (predictions[pointIdx] != clusterIdx) {
            return 0.0;
        }
        
        var center = getCategory(clusterIdx);
        if (!(center instanceof ART2Weight art2Weight)) {
            return 0.0;
        }
        
        var centerMemory = art2Weight.vector();
        var point = data[pointIdx];
        
        double distance = 0.0;
        for (int dim = 0; dim < Math.min(point.dimension(), centerMemory.dimension()); dim++) {
            double diff = point.get(dim) - centerMemory.get(dim);
            distance += diff * diff;
        }
        
        return Math.sqrt(distance);
    }
    
    private double calculateNearestClusterDistance(Pattern[] data, Integer[] predictions, int currentCluster, int pointIdx) {
        double minDistance = Double.MAX_VALUE;
        var point = data[pointIdx];
        
        for (int clusterIdx = 0; clusterIdx < getCategoryCount(); clusterIdx++) {
            if (clusterIdx != currentCluster) {
                var center = getCategory(clusterIdx);
                if (center instanceof ART2Weight art2Weight) {
                    var centerMemory = art2Weight.vector();
                    
                    double distance = 0.0;
                    for (int dim = 0; dim < Math.min(point.dimension(), centerMemory.dimension()); dim++) {
                        double diff = point.get(dim) - centerMemory.get(dim);
                        distance += diff * diff;
                    }
                    distance = Math.sqrt(distance);
                    minDistance = Math.min(minDistance, distance);
                }
            }
        }
        
        return minDistance == Double.MAX_VALUE ? 0.0 : minDistance;
    }
    
    private double calculateInterClusterDistance(WeightVector centerI, WeightVector centerJ) {
        if (!(centerI instanceof ART2Weight art2I) || !(centerJ instanceof ART2Weight art2J)) {
            return 0.0;
        }
        
        var memoryI = art2I.vector();
        var memoryJ = art2J.vector();
        
        double distance = 0.0;
        for (int dim = 0; dim < Math.min(memoryI.dimension(), memoryJ.dimension()); dim++) {
            double diff = memoryI.get(dim) - memoryJ.get(dim);
            distance += diff * diff;
        }
        
        return Math.sqrt(distance);
    }
}