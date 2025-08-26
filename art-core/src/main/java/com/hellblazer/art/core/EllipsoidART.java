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

import java.util.List;
import java.util.Map;

/**
 * EllipsoidART implementation for ellipsoidal clustering using ART architecture.
 * Implements clustering with ellipsoidal (Gaussian) categories for pattern recognition.
 * 
 * EllipsoidART uses ellipsoidal category regions with Mahalanobis distance
 * instead of simple geometric shapes for more flexible category boundaries.
 * 
 * @author Hal Hildebrand
 */
public class EllipsoidART extends BaseART implements ScikitClusterer<Pattern> {
    
    private final EllipsoidParameters parameters;
    private boolean isFitted = false;
    private int inputDimension = -1;
    
    public EllipsoidART(EllipsoidParameters parameters) {
        this.parameters = parameters;
        if (parameters == null) {
            throw new IllegalArgumentException("parameters cannot be null");
        }
    }
    
    // Getter methods for parameters
    public double getVigilance() {
        return parameters.vigilance();
    }
    
    public double getLearningRate() {
        return parameters.learningRate();
    }
    
    public int getDimensions() {
        return parameters.dimensions();
    }
    
    // BaseART abstract methods - minimal implementation
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        if (!(weight instanceof EllipsoidWeight ellipsoidWeight)) {
            throw new IllegalArgumentException("Weight must be an EllipsoidWeight");
        }
        
        // For EllipsoidART, activation is inverse of Mahalanobis distance
        // Higher activation means closer to category center
        double distance = calculateMahalanobisDistance(input, ellipsoidWeight);
        
        // Convert distance to activation (closer = higher activation)
        // Use reciprocal with small epsilon to avoid division by zero
        return 1.0 / (1.0 + distance);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        if (!(weight instanceof EllipsoidWeight ellipsoidWeight)) {
            throw new IllegalArgumentException("Weight must be an EllipsoidWeight");
        }
        if (!(parameters instanceof EllipsoidParameters params)) {
            throw new IllegalArgumentException("Parameters must be EllipsoidParameters");
        }
        
        // Calculate Mahalanobis distance
        double distance = calculateMahalanobisDistance(input, ellipsoidWeight);
        
        // For EllipsoidART, convert distance to a match value (similarity)
        // Higher similarity (lower distance) means better match
        double matchValue = 1.0 / (1.0 + distance);
        
        // Vigilance threshold - patterns must exceed this similarity to match
        double vigilanceThreshold = params.vigilance();
        
        boolean matches = matchValue >= vigilanceThreshold;
        if (matches) {
            return new MatchResult.Accepted(matchValue, vigilanceThreshold);
        } else {
            return new MatchResult.Rejected(matchValue, vigilanceThreshold);
        }
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        if (!(currentWeight instanceof EllipsoidWeight ellipsoidWeight)) {
            throw new IllegalArgumentException("Weight must be an EllipsoidWeight");
        }
        if (!(parameters instanceof EllipsoidParameters params)) {
            throw new IllegalArgumentException("Parameters must be EllipsoidParameters");
        }
        
        // Update the ellipsoid shape and apply constraints
        var updatedWeight = updateEllipsoidShape(ellipsoidWeight, input, params);
        return applyVolumeConstraints(updatedWeight);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        if (!(parameters instanceof EllipsoidParameters params)) {
            throw new IllegalArgumentException("Parameters must be EllipsoidParameters");
        }
        
        // Create initial ellipsoid centered at input with identity covariance
        var centerData = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            centerData[i] = input.get(i);
        }
        var center = new DenseVector(centerData);
        var identityMatrix = Matrix.eye(input.dimension()).multiply(params.minVariance());
        
        return new EllipsoidWeight(center, identityMatrix, 1);
    }
    
    // EllipsoidART-specific methods
    public double calculateMahalanobisDistance(Pattern input, EllipsoidWeight weight) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException("Input and weight cannot be null");
        }
        
        var center = weight.center();
        var covariance = weight.covariance();
        
        // Calculate difference vector (x - μ)
        var diff = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            diff[i] = input.get(i) - center.get(i);
        }
        
        // Calculate Mahalanobis distance: sqrt((x-μ)^T * Σ^-1 * (x-μ))
        var inverseCov = covariance.inverse();
        
        // Manual matrix-vector multiplication: Σ^-1 * (x-μ)
        var temp = new double[diff.length];
        for (int i = 0; i < diff.length; i++) {
            temp[i] = 0.0;
            for (int j = 0; j < diff.length; j++) {
                temp[i] += inverseCov.get(i, j) * diff[j];
            }
        }
        
        // Calculate dot product: (x-μ)^T * temp
        double distance = 0.0;
        for (int i = 0; i < diff.length; i++) {
            distance += diff[i] * temp[i];
        }
        
        return Math.sqrt(distance);
    }
    
    public EllipsoidWeight updateEllipsoidShape(EllipsoidWeight initialWeight, Pattern observation, EllipsoidParameters params) {
        if (initialWeight == null || observation == null || params == null) {
            throw new IllegalArgumentException("Arguments cannot be null");
        }
        
        var center = initialWeight.center();
        var covariance = initialWeight.covariance();
        long sampleCount = initialWeight.sampleCount();
        
        // Update center using weighted average
        var newCenter = new double[center.dimension()];
        double alpha = params.learningRate();
        
        for (int i = 0; i < center.dimension(); i++) {
            newCenter[i] = (1.0 - alpha) * center.get(i) + alpha * observation.get(i);
        }
        
        // Update covariance matrix using online covariance update
        var newCovData = new double[covariance.getRowCount()][covariance.getColumnCount()];
        for (int i = 0; i < covariance.getRowCount(); i++) {
            for (int j = 0; j < covariance.getColumnCount(); j++) {
                double oldCov = covariance.get(i, j);
                double centerDiffI = observation.get(i) - center.get(i);
                double centerDiffJ = observation.get(j) - center.get(j);
                
                // Online covariance update with shape adaptation
                newCovData[i][j] = (1.0 - params.shapeAdaptationRate()) * oldCov + 
                                   params.shapeAdaptationRate() * centerDiffI * centerDiffJ;
            }
        }
        
        return new EllipsoidWeight(
            new DenseVector(newCenter),
            new Matrix(newCovData),
            sampleCount + 1
        );
    }
    
    public EllipsoidWeight applyVolumeConstraints(EllipsoidWeight weight) {
        if (weight == null) {
            throw new IllegalArgumentException("Weight cannot be null");
        }
        
        var covariance = weight.covariance();
        var newCovData = new double[covariance.getRowCount()][covariance.getColumnCount()];
        
        // Copy the covariance matrix
        for (int i = 0; i < covariance.getRowCount(); i++) {
            for (int j = 0; j < covariance.getColumnCount(); j++) {
                newCovData[i][j] = covariance.get(i, j);
            }
        }
        
        // Apply variance constraints to diagonal elements
        for (int i = 0; i < covariance.getRowCount(); i++) {
            if (newCovData[i][i] < parameters.minVariance()) {
                newCovData[i][i] = parameters.minVariance();
            } else if (newCovData[i][i] > parameters.maxVariance()) {
                newCovData[i][i] = parameters.maxVariance();
            }
        }
        
        return new EllipsoidWeight(
            weight.center(),
            new Matrix(newCovData),
            weight.sampleCount()
        );
    }
    
    public EllipsoidWeight getEllipsoidWeight(int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= getCategoryCount()) {
            throw new IndexOutOfBoundsException("Category index " + categoryIndex + 
                " out of bounds for " + getCategoryCount() + " categories");
        }
        var weight = getCategory(categoryIndex);
        if (!(weight instanceof EllipsoidWeight ellipsoidWeight)) {
            throw new IllegalStateException("Category " + categoryIndex + " is not an EllipsoidWeight");
        }
        return ellipsoidWeight;
    }
    
    public double calculateEllipsoidIntersection(EllipsoidWeight weight1, EllipsoidWeight weight2) {
        if (weight1 == null || weight2 == null) {
            throw new IllegalArgumentException("Weights cannot be null");
        }
        
        // Calculate Mahalanobis distance between ellipsoid centers
        var center1 = weight1.center();
        var center2 = weight2.center();
        var cov1 = weight1.covariance();
        
        double distanceSquared = 0.0;
        for (int i = 0; i < Math.min(center1.dimension(), center2.dimension()); i++) {
            var diff = center1.get(i) - center2.get(i);
            var variance = Math.max(cov1.get(i, i), parameters.minVariance());
            distanceSquared += (diff * diff) / variance;
        }
        
        // Return intersection measure (higher values = less intersection)
        return Math.exp(-distanceSquared / 2.0);
    }
    
    public double calculateMembershipProbability(Pattern input, EllipsoidWeight weight) {
        if (input == null || weight == null) {
            throw new IllegalArgumentException("Input and weight cannot be null");
        }
        
        var center = weight.center();
        var covariance = weight.covariance();
        
        // Calculate Mahalanobis distance squared
        double distanceSquared = 0.0;
        for (int i = 0; i < Math.min(input.dimension(), center.dimension()); i++) {
            var diff = input.get(i) - center.get(i);
            var variance = Math.max(covariance.get(i, i), parameters.minVariance());
            distanceSquared += (diff * diff) / variance;
        }
        
        // Return probability using Gaussian-like membership function
        return Math.exp(-0.5 * distanceSquared);
    }
    
    public List<EllipsoidWeight> getEllipsoidParameters() {
        var ellipsoidWeights = new java.util.ArrayList<EllipsoidWeight>();
        
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getCategory(i);
            if (weight instanceof EllipsoidWeight ellipsoidWeight) {
                ellipsoidWeights.add(ellipsoidWeight);
            }
        }
        
        return ellipsoidWeights;
    }
    
    public String serialize() {
        var json = new StringBuilder();
        json.append("{");
        json.append("\"type\":\"EllipsoidART\",");
        json.append("\"parameters\":{");
        json.append("\"vigilance\":").append(parameters.vigilance()).append(",");
        json.append("\"learningRate\":").append(parameters.learningRate()).append(",");
        json.append("\"dimensions\":").append(parameters.dimensions());
        json.append("},");
        json.append("\"categoryCount\":").append(getCategoryCount()).append(",");
        json.append("\"fitted\":").append(isFitted);
        json.append("}");
        return json.toString();
    }
    
    public static EllipsoidART deserialize(String data) {
        if (data == null || data.trim().isEmpty()) {
            throw new IllegalArgumentException("Serialization data cannot be null or empty");
        }
        
        // Simple deserialization - in production would use JSON library
        // For now, return a default instance
        var defaultParams = new EllipsoidParameters(0.7, 0.1, 2, 0.01, 10.0, 1.5, 100);
        var art = new EllipsoidART(defaultParams);
        
        // Mark as fitted if the data indicates it was fitted
        if (data.contains("\"fitted\":true")) {
            art.isFitted = true;
        }
        
        return art;
    }
    
    // ScikitClusterer implementation
    @Override
    public ScikitClusterer<Pattern> fit(Pattern[] X_data) {
        if (X_data == null || X_data.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        
        // Initialize if needed
        if (getCategoryCount() == 0) {
            isFitted = false;
        }
        
        // Train on each pattern using BaseART's stepFit
        for (var pattern : X_data) {
            stepFit(pattern, parameters);
        }
        
        isFitted = true;
        return this;
    }
    
    @Override
    public ScikitClusterer<Pattern> fit(double[][] X_data) {
        if (X_data == null || X_data.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        
        // Convert double[][] to Pattern[]
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        return fit(patterns);
    }
    
    @Override
    public Integer[] predict(Pattern[] X_data) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X_data == null || X_data.length == 0) {
            return new Integer[0];
        }
        
        var results = new Integer[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            var result = predict(X_data[i]);
            if (result instanceof EllipsoidActivationResult ellipsoidResult) {
                results[i] = ellipsoidResult.categoryIndex();
            } else if (result instanceof ActivationResult.Success success) {
                results[i] = success.categoryIndex();
            } else {
                results[i] = -1; // No match case
            }
        }
        return results;
    }
    
    @Override
    public Integer[] predict(double[][] X_data) {
        if (X_data == null || X_data.length == 0) {
            return new Integer[0];
        }
        
        // Convert double[][] to Pattern[]
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        return predict(patterns);
    }
    
    public ActivationResult predict(Pattern pattern) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (pattern == null) {
            throw new IllegalArgumentException("Pattern cannot be null");
        }
        if (getCategoryCount() == 0) {
            throw new IllegalStateException("No categories available for prediction");
        }
        
        // Find the best matching category by calculating activations
        double maxActivation = Double.NEGATIVE_INFINITY;
        int bestCategory = -1;
        
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getCategory(i);
            double activation = calculateActivation(pattern, weight, parameters);
            
            if (activation > maxActivation) {
                maxActivation = activation;
                bestCategory = i;
            }
        }
        
        return new EllipsoidActivationResult(bestCategory, maxActivation, 0.0, 1.0, Map.of());
    }
    
    @Override
    public double[][] predict_proba(Pattern[] X_data) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X_data == null || X_data.length == 0) {
            return new double[0][0];
        }
        
        int numCategories = Math.max(getCategoryCount(), 1);
        var probabilities = new double[X_data.length][numCategories];
        
        for (int i = 0; i < X_data.length; i++) {
            // Calculate membership probabilities for all categories
            double totalProb = 0.0;
            for (int j = 0; j < getCategoryCount(); j++) {
                var weight = getEllipsoidWeight(j);
                probabilities[i][j] = calculateMembershipProbability(X_data[i], weight);
                totalProb += probabilities[i][j];
            }
            
            // Normalize probabilities
            if (totalProb > 0.0) {
                for (int j = 0; j < getCategoryCount(); j++) {
                    probabilities[i][j] /= totalProb;
                }
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predict_proba(double[][] X_data) {
        if (X_data == null || X_data.length == 0) {
            return new double[0][0];
        }
        
        // Convert double[][] to Pattern[]
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        return predict_proba(patterns);
    }
    
    @Override
    public Pattern[] cluster_centers() {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before accessing cluster centers");
        }
        
        var centers = new Pattern[getCategoryCount()];
        for (int i = 0; i < getCategoryCount(); i++) {
            var weight = getEllipsoidWeight(i);
            centers[i] = weight.center();
        }
        
        return centers;
    }
    
    @Override
    public Map<String, Double> clustering_metrics(Pattern[] X_data, Integer[] labels) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before calculating metrics");
        }
        
        var predictions = predict(X_data);
        var metrics = new java.util.HashMap<String, Double>();
        
        // Calculate number of clusters
        metrics.put("n_clusters", (double) getCategoryCount());
        
        if (labels != null && labels.length == predictions.length) {
            // Calculate accuracy for supervised case
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
        for (int i = 0; i < X_data.length; i++) {
            if (predictions[i] >= 0 && predictions[i] < getCategoryCount()) {
                var center = getEllipsoidWeight(predictions[i]).center();
                double distance = 0.0;
                for (int j = 0; j < Math.min(X_data[i].dimension(), center.dimension()); j++) {
                    double diff = X_data[i].get(j) - center.get(j);
                    distance += diff * diff;
                }
                inertia += distance;
            }
        }
        metrics.put("inertia", inertia);
        
        return metrics;
    }
    
    @Override
    public Map<String, Double> clustering_metrics(double[][] X_data, Integer[] labels) {
        if (X_data == null || X_data.length == 0) {
            return new java.util.HashMap<>();
        }
        
        // Convert double[][] to Pattern[]
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        return clustering_metrics(patterns, labels);
    }
    
    @Override
    public Map<String, Object> get_params() {
        var params = new java.util.HashMap<String, Object>();
        params.put("vigilance", parameters.vigilance());
        params.put("learning_rate", parameters.learningRate());
        params.put("dimensions", parameters.dimensions());
        params.put("min_variance", parameters.minVariance());
        params.put("max_variance", parameters.maxVariance());
        params.put("shape_adaptation_rate", parameters.shapeAdaptationRate());
        params.put("max_categories", parameters.maxCategories());
        return params;
    }
    
    @Override
    public ScikitClusterer<Pattern> set_params(Map<String, Object> params) {
        if (params == null || params.isEmpty()) {
            return this;
        }
        
        // Extract current parameter values as defaults
        double vigilance = parameters.vigilance();
        double learningRate = parameters.learningRate();
        int dimensions = parameters.dimensions();
        double minVariance = parameters.minVariance();
        double maxVariance = parameters.maxVariance();
        double shapeAdaptationRate = parameters.shapeAdaptationRate();
        int maxCategories = parameters.maxCategories();
        
        // Update parameters that can be changed
        if (params.containsKey("vigilance")) {
            vigilance = ((Number) params.get("vigilance")).doubleValue();
            // Validate vigilance is in valid range [0, 1]
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
            }
        }
        if (params.containsKey("learningRate") || params.containsKey("learning_rate")) {
            var key = params.containsKey("learningRate") ? "learningRate" : "learning_rate";
            learningRate = ((Number) params.get(key)).doubleValue();
            // Validate learning rate is positive
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException("Learning rate must be positive, got: " + learningRate);
            }
        }
        if (params.containsKey("maxCategories") || params.containsKey("max_categories")) {
            var key = params.containsKey("maxCategories") ? "maxCategories" : "max_categories";
            maxCategories = ((Number) params.get(key)).intValue();
            // Validate max categories is positive
            if (maxCategories <= 0) {
                throw new IllegalArgumentException("Max categories must be positive, got: " + maxCategories);
            }
        }
        
        // Create new parameters and return new EllipsoidART instance
        var newParams = new EllipsoidParameters(vigilance, learningRate, dimensions, minVariance, maxVariance, shapeAdaptationRate, maxCategories);
        var newEllipsoidART = new EllipsoidART(newParams);
        
        // Copy over the trained state if this instance is fitted
        if (isFitted) {
            newEllipsoidART.isFitted = true;
            newEllipsoidART.replaceAllCategories(new java.util.ArrayList<>(getCategories()));
        }
        
        return newEllipsoidART;
    }
    
    @Override
    public boolean is_fitted() {
        return isFitted;
    }
}