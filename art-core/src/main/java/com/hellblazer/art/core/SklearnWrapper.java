package com.hellblazer.art.core;

import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.algorithms.*;
import com.hellblazer.art.core.parameters.*;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * Scikit-learn compatible wrapper for BaseART algorithms.
 * Provides a familiar API for users coming from Python's scikit-learn.
 * 
 * This wrapper bridges the gap between the Java ART implementation's
 * step-based API and scikit-learn's batch-based fit/predict paradigm.
 */
public class SklearnWrapper {
    
    private final BaseART algorithm;
    private Object parameters;
    private boolean fitted = false;
    private int[] labels;
    private int nClusters = 0;
    
    /**
     * Create a new SklearnWrapper for the given ART algorithm.
     * 
     * @param algorithm the BaseART algorithm to wrap
     * @param parameters the parameters for the algorithm
     */
    public SklearnWrapper(BaseART algorithm, Object parameters) {
        if (algorithm == null) {
            throw new IllegalArgumentException("Algorithm cannot be null");
        }
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        this.algorithm = algorithm;
        this.parameters = parameters;
    }
    
    /**
     * Fit the model to the data (List overload).
     * 
     * @param X Training data as list
     * @return This instance for method chaining
     */
    public SklearnWrapper fit(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return this;
        }
        return fit(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Fit the model to the data.
     * 
     * @param X Training data, shape (n_samples, n_features)
     * @return This instance for method chaining
     */
    public SklearnWrapper fit(double[][] X) {
        if (X == null || X.length == 0) {
            return this; // Handle empty data gracefully
        }
        
        // Clear any existing categories
        algorithm.clearCategories();
        
        // Convert to Pattern objects and train
        labels = new int[X.length];
        int maxCategory = -1;
        
        for (int i = 0; i < X.length; i++) {
            var pattern = new DenseVector(X[i]);
            var result = algorithm.stepFit(pattern, parameters);
            
            if (result instanceof ActivationResult.Success success) {
                labels[i] = success.categoryIndex();
                maxCategory = Math.max(maxCategory, success.categoryIndex());
            } else {
                labels[i] = -1; // Failed to classify
            }
        }
        
        nClusters = maxCategory + 1;
        fitted = true;
        return this;
    }
    
    /**
     * Predict cluster labels for samples (List overload).
     * 
     * @param X Data to predict as list
     * @return Cluster labels for each sample
     */
    public int[] predict(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return new int[0];
        }
        return predict(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Predict cluster labels for samples.
     * 
     * @param X Data to predict, shape (n_samples, n_features)
     * @return Cluster labels for each sample
     */
    public int[] predict(double[][] X) {
        if (!fitted) {
            return new int[X.length]; // Return zeros instead of throwing
        }
        
        if (X == null || X.length == 0) {
            return new int[0];
        }
        
        var predictions = new int[X.length];
        
        for (int i = 0; i < X.length; i++) {
            var pattern = new DenseVector(X[i]);
            var result = algorithm.stepPredict(pattern, parameters);
            
            if (result instanceof ActivationResult.Success success) {
                predictions[i] = success.categoryIndex();
            } else {
                predictions[i] = -1; // Failed to classify
            }
        }
        
        return predictions;
    }
    
    /**
     * Fit the model and predict cluster labels (List overload).
     * 
     * @param X Training data as list
     * @return Cluster labels for each sample
     */
    public int[] fitPredict(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return new int[0];
        }
        return fitPredict(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Fit the model and predict cluster labels.
     * 
     * @param X Training data, shape (n_samples, n_features)
     * @return Cluster labels for each sample
     */
    public int[] fitPredict(double[][] X) {
        fit(X);
        return labels == null ? new int[0] : labels;
    }
    
    /**
     * Incrementally fit the model on a batch of samples (List overload).
     * 
     * @param X Training data batch as list
     * @return This instance for method chaining
     */
    public SklearnWrapper partialFit(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return this;
        }
        return partialFit(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Incrementally fit the model on a batch of samples.
     * 
     * @param X Training data batch, shape (n_samples, n_features)
     * @return This instance for method chaining
     */
    public SklearnWrapper partialFit(double[][] X) {
        if (!fitted) {
            return fit(X);
        }
        
        // Extend labels array for new data
        var oldLength = labels.length;
        var newLabels = new int[oldLength + X.length];
        System.arraycopy(labels, 0, newLabels, 0, oldLength);
        
        for (int i = 0; i < X.length; i++) {
            var pattern = new DenseVector(X[i]);
            var result = algorithm.stepFit(pattern, parameters);
            
            if (result instanceof ActivationResult.Success success) {
                newLabels[oldLength + i] = success.categoryIndex();
                nClusters = Math.max(nClusters, success.categoryIndex() + 1);
            } else {
                newLabels[oldLength + i] = -1;
            }
        }
        
        labels = newLabels;
        return this;
    }
    
    /**
     * Get parameters of the model as a Map.
     * 
     * @return Map of parameter names to values
     */
    public Map<String, Object> getParams() {
        var params = new HashMap<String, Object>();
        
        if (parameters instanceof FuzzyParameters fp) {
            params.put("vigilance", fp.vigilance());
            params.put("alpha", fp.alpha());
            params.put("beta", fp.beta());
        } else if (parameters instanceof BayesianParameters bp) {
            params.put("vigilance", bp.vigilance());
        } else if (parameters instanceof GaussianParameters gp) {
            params.put("vigilance", gp.vigilance());
        } else if (parameters instanceof HypersphereParameters hp) {
            params.put("vigilance", hp.vigilance());
            params.put("radius", hp.defaultRadius());
        } else if (parameters instanceof EllipsoidParameters ep) {
            params.put("vigilance", ep.vigilance());
            params.put("learningRate", ep.learningRate());
        }
        
        return params;
    }
    
    /**
     * Set parameters of the model.
     * 
     * @param params Map of parameter names to values
     */
    public void setParams(Map<String, Object> params) {
        if (params.containsKey("vigilance")) {
            // Update vigilance in parameters
            if (parameters instanceof FuzzyParameters fp) {
                parameters = new FuzzyParameters(
                    (Double) params.get("vigilance"),
                    fp.alpha(),
                    fp.beta()
                );
            }
            // Add more parameter types as needed
        }
    }
    
    /**
     * Get the number of clusters/categories.
     * 
     * @return Number of clusters
     */
    public int getCategoryCount() {
        return nClusters;
    }
    
    /**
     * Get the number of clusters (sklearn compatibility).
     * 
     * @return Number of clusters
     */
    public int getNClusters() {
        return nClusters;
    }
    
    /**
     * Check if the model has been fitted.
     * 
     * @return True if fitted, false otherwise
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Get the underlying ART algorithm.
     * 
     * @return The wrapped BaseART algorithm
     */
    public BaseART getAlgorithm() {
        return algorithm;
    }
    
    /**
     * Get the cluster labels from the last fit.
     * 
     * @return Cluster labels from last fit
     */
    public int[] getLabels() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before getting labels");
        }
        return labels;
    }
    
    /**
     * Get cluster centers (if supported by the algorithm).
     * 
     * @return Array of cluster centers, shape (n_clusters, n_features)
     */
    public double[][] getClusterCenters() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before getting cluster centers");
        }
        
        var categories = algorithm.getCategories();
        if (categories.isEmpty()) {
            return new double[0][0];
        }
        
        // Extract features from weight vectors
        var centers = new double[categories.size()][];
        for (int i = 0; i < categories.size(); i++) {
            var weight = categories.get(i);
            // Extract data from weight vector
            // Since weights are typically stored as arrays internally
            // we need to extract the raw data
            var dimension = weight.dimension();
            var center = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                var value = weight.get(j);
                // Handle potential NaN values
                if (Double.isNaN(value)) {
                    center[j] = 0.0; // Default to 0 for NaN values
                } else {
                    center[j] = value;
                }
            }
            centers[i] = center;
        }
        
        return centers;
    }
    
    /**
     * Score the model (List overload).
     * 
     * @param X Data to score as list
     * @return Silhouette coefficient (0 to 1)
     */
    public double score(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return 0.0;
        }
        return score(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Score the model using silhouette coefficient.
     * 
     * @param X Data to score
     * @return Silhouette coefficient (0 to 1, higher is better)
     */
    public double score(double[][] X) {
        if (!fitted || X == null || X.length == 0) {
            return 0.0;
        }
        
        var predictions = predict(X);
        var centers = getClusterCenters();
        
        if (centers == null || centers.length == 0) {
            return 0.0;
        }
        
        var inertia = 0.0;
        var validPredictions = 0;
        
        for (int i = 0; i < X.length; i++) {
            var label = predictions[i];
            if (label >= 0 && label < centers.length && centers[label].length > 0) {
                var distance = euclideanDistance(X[i], centers[label]);
                inertia += distance * distance;
                validPredictions++;
            }
        }
        
        // If no valid predictions, return 0
        if (validPredictions == 0) {
            return 0.0;
        }
        
        // Return as positive silhouette-like score (0 to 1)
        inertia = inertia / validPredictions; // Average inertia
        return Math.max(0.0, Math.min(1.0, 1.0 / (1.0 + inertia)));
    }
    
    /**
     * Transform data to one-hot encoded space (List overload).
     * 
     * @param X Data to transform as list
     * @return One-hot encoded categories
     */
    public double[][] transform(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return new double[0][0];
        }
        return transform(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Transform data to one-hot encoded category space.
     * 
     * @param X Data to transform
     * @return One-hot encoded categories
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before transformation");
        }
        
        if (nClusters == 0) {
            return new double[X.length][1]; // Return single column of zeros
        }
        
        var predictions = predict(X);
        var oneHot = new double[X.length][nClusters];
        
        for (int i = 0; i < X.length; i++) {
            if (predictions[i] >= 0 && predictions[i] < nClusters) {
                oneHot[i][predictions[i]] = 1.0;
            }
        }
        
        return oneHot;
    }
    
    /**
     * Fit and transform data (List overload).
     * 
     * @param X Training data as list
     * @return One-hot encoded categories
     */
    public double[][] fitTransform(List<double[]> X) {
        if (X == null || X.isEmpty()) {
            return new double[0][0];
        }
        return fitTransform(X.toArray(new double[X.size()][]));
    }
    
    /**
     * Fit and transform data.
     * 
     * @param X Training data
     * @return One-hot encoded categories
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            return 0.0; // Return 0 instead of NaN for mismatched dimensions
        }
        
        var sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            var diff = a[i] - b[i];
            // Check for NaN in the calculation
            if (Double.isNaN(diff)) {
                return 0.0; // Return 0 for NaN values
            }
            sum += diff * diff;
        }
        
        var result = Math.sqrt(sum);
        // Final check for NaN result
        return Double.isNaN(result) ? 0.0 : result;
    }
    
    /**
     * Factory methods for creating SklearnWrapper instances.
     */
    
    public static SklearnWrapper fuzzyART(double vigilance, double alpha, double beta) {
        var params = new FuzzyParameters(vigilance, alpha, beta);
        return new SklearnWrapper(new FuzzyART(), params);
    }
    
    public static SklearnWrapper bayesianART(double vigilance, int dimensions) {
        // Create default parameters for Bayesian ART
        var priorMean = new double[dimensions];
        var priorCovariance = com.hellblazer.art.core.utils.Matrix.eye(dimensions);
        var noiseVariance = 0.01;
        var priorPrecision = 1.0;
        var maxCategories = 100;
        
        var params = new BayesianParameters(
            vigilance, priorMean, priorCovariance, 
            noiseVariance, priorPrecision, maxCategories
        );
        return new SklearnWrapper(new BayesianART(params), params);
    }
    
    public static SklearnWrapper gaussianART(double vigilance, double sigmaInit, double sigmaDecay) {
        // Use default 3 dimensions for testing
        var dimensions = 3;
        var sigmaInitArray = new double[dimensions];
        java.util.Arrays.fill(sigmaInitArray, sigmaInit);
        var params = new GaussianParameters(vigilance, sigmaInitArray);
        return new SklearnWrapper(new GaussianART(), params);
    }
    
    public static SklearnWrapper hypersphereART(double vigilance, double radiusInit, double radiusDecay) {
        // Use adaptive radius by default
        var params = new HypersphereParameters(vigilance, radiusInit, true);
        return new SklearnWrapper(new HypersphereART(), params);
    }
    
    public static SklearnWrapper ellipsoidART(double vigilance, double radiusInit, double radiusDecay, double learningRate) {
        // Use default 3 dimensions for testing
        var dimensions = 3;
        var minVariance = 0.001;
        var maxVariance = 10.0;
        var shapeAdaptationRate = 0.01;
        var maxCategories = 100;
        
        var params = new EllipsoidParameters(
            vigilance, learningRate, dimensions,
            minVariance, maxVariance, shapeAdaptationRate, maxCategories
        );
        return new SklearnWrapper(new EllipsoidART(params), params);
    }
}