package com.hellblazer.art.nlp.util;

import com.hellblazer.art.core.DenseVector;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

/**
 * Utility methods for DenseVector operations in NLP context.
 * Provides common vector transformations and aggregations.
 */
public final class VectorUtils {
    
    private VectorUtils() {
        // Utility class - no instantiation
    }
    
    /**
     * Create a DenseVector from a double array.
     * 
     * @param values The array values
     * @return New DenseVector instance
     */
    public static DenseVector fromArray(double[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Values array cannot be null or empty");
        }
        return new DenseVector(Arrays.copyOf(values, values.length));
    }
    
    /**
     * Create a zero vector of specified dimension.
     * 
     * @param dimension The vector dimension
     * @return Zero vector
     */
    public static DenseVector zeros(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + dimension);
        }
        return new DenseVector(new double[dimension]);
    }
    
    /**
     * Create a ones vector of specified dimension.
     * 
     * @param dimension The vector dimension
     * @return Ones vector
     */
    public static DenseVector ones(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + dimension);
        }
        var values = new double[dimension];
        Arrays.fill(values, 1.0);
        return new DenseVector(values);
    }
    
    /**
     * Compute element-wise average of multiple vectors.
     * All vectors must have the same dimension.
     * 
     * @param vectors List of vectors to average
     * @return Average vector
     */
    public static DenseVector average(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors list cannot be null or empty");
        }
        
        var first = vectors.get(0);
        var dimension = first.dimension();
        var result = new double[dimension];
        
        for (var vector : vectors) {
            if (vector.dimension() != dimension) {
                throw new IllegalArgumentException("All vectors must have same dimension");
            }
            for (int i = 0; i < dimension; i++) {
                result[i] += vector.get(i);
            }
        }
        
        // Divide by count to get average
        var count = vectors.size();
        for (int i = 0; i < dimension; i++) {
            result[i] /= count;
        }
        
        return new DenseVector(result);
    }
    
    /**
     * Compute weighted average of vectors.
     * 
     * @param vectors List of vectors
     * @param weights Corresponding weights (must sum to 1.0)
     * @return Weighted average vector
     */
    public static DenseVector weightedAverage(List<DenseVector> vectors, List<Double> weights) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors list cannot be null or empty");
        }
        if (weights == null || weights.size() != vectors.size()) {
            throw new IllegalArgumentException("Weights must match vectors count");
        }
        
        // Verify weights sum to approximately 1.0
        var weightSum = weights.stream().mapToDouble(Double::doubleValue).sum();
        if (Math.abs(weightSum - 1.0) > 1e-6) {
            throw new IllegalArgumentException("Weights must sum to 1.0, got: " + weightSum);
        }
        
        var first = vectors.get(0);
        var dimension = first.dimension();
        var result = new double[dimension];
        
        for (int j = 0; j < vectors.size(); j++) {
            var vector = vectors.get(j);
            var weight = weights.get(j);
            
            if (vector.dimension() != dimension) {
                throw new IllegalArgumentException("All vectors must have same dimension");
            }
            
            for (int i = 0; i < dimension; i++) {
                result[i] += vector.get(i) * weight;
            }
        }
        
        return new DenseVector(result);
    }
    
    /**
     * Concatenate multiple vectors into a single vector.
     * 
     * @param vectors Vectors to concatenate
     * @return Concatenated vector
     */
    public static DenseVector concatenate(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Vectors list cannot be null or empty");
        }
        
        var totalSize = vectors.stream().mapToInt(DenseVector::dimension).sum();
        var result = new double[totalSize];
        
        int offset = 0;
        for (var vector : vectors) {
            var size = vector.dimension();
            for (int i = 0; i < size; i++) {
                result[offset + i] = vector.get(i);
            }
            offset += size;
        }
        
        return new DenseVector(result);
    }
    
    /**
     * Normalize vector to unit length (L2 normalization).
     * 
     * @param vector Vector to normalize
     * @return Normalized vector
     */
    public static DenseVector normalize(DenseVector vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        
        var magnitude = vector.l2Norm();
        if (magnitude == 0.0) {
            return vector; // Zero vector stays zero
        }
        
        var dimension = vector.dimension();
        var result = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            result[i] = vector.get(i) / magnitude;
        }
        
        return new DenseVector(result);
    }
    
    /**
     * Apply complement coding to vector [x, 1-x].
     * Input vector values should be in [0, 1].
     * 
     * @param vector Input vector with values in [0, 1]
     * @return Complement coded vector with twice the dimension
     */
    public static DenseVector complementCode(DenseVector vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        
        var dimension = vector.dimension();
        var result = new double[dimension * 2];
        
        // First half: original values
        for (int i = 0; i < dimension; i++) {
            var value = vector.get(i);
            // Clamp to [0, 1] for safety
            result[i] = Math.max(0.0, Math.min(1.0, value));
        }
        
        // Second half: complement values
        for (int i = 0; i < dimension; i++) {
            result[i + dimension] = 1.0 - result[i];
        }
        
        return new DenseVector(result);
    }
    
    /**
     * Compute dot product of two vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return Dot product
     */
    public static double dotProduct(DenseVector a, DenseVector b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors cannot be null");
        }
        if (a.dimension() != b.dimension()) {
            throw new IllegalArgumentException("Vectors must have same dimension");
        }
        
        double result = 0.0;
        var dimension = a.dimension();
        for (int i = 0; i < dimension; i++) {
            result += a.get(i) * b.get(i);
        }
        return result;
    }
    
    /**
     * Compute cosine similarity between two vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity [-1, 1]
     */
    public static double cosineSimilarity(DenseVector a, DenseVector b) {
        var dot = dotProduct(a, b);
        var magnitudeA = a.l2Norm();
        var magnitudeB = b.l2Norm();
        
        if (magnitudeA == 0.0 || magnitudeB == 0.0) {
            return 0.0; // Undefined, but return 0 for practical purposes
        }
        
        return dot / (magnitudeA * magnitudeB);
    }
    
    /**
     * Check if vector contains only finite values (no NaN or infinity).
     * 
     * @param vector Vector to check
     * @return true if all values are finite
     */
    public static boolean isFinite(DenseVector vector) {
        if (vector == null) {
            return false;
        }
        
        var dimension = vector.dimension();
        for (int i = 0; i < dimension; i++) {
            if (!Double.isFinite(vector.get(i))) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Create vector with random values in [0, 1].
     * 
     * @param dimension Vector dimension
     * @return Random vector
     */
    public static DenseVector random(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + dimension);
        }
        
        var values = new double[dimension];
        var random = new java.util.Random();
        for (int i = 0; i < dimension; i++) {
            values[i] = random.nextDouble();
        }
        
        return new DenseVector(values);
    }
}