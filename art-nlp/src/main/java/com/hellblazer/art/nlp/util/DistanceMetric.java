package com.hellblazer.art.nlp.util;

import com.hellblazer.art.core.DenseVector;

/**
 * Interface for distance metrics used in ART-NLP algorithms.
 * Provides different distance/similarity measures for vector comparisons.
 */
public interface DistanceMetric {
    
    /**
     * Calculate distance between two vectors.
     * Lower values indicate more similar vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return Distance value (typically >= 0)
     */
    double distance(DenseVector a, DenseVector b);
    
    /**
     * Calculate similarity between two vectors.
     * Higher values indicate more similar vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return Similarity value
     */
    default double similarity(DenseVector a, DenseVector b) {
        return 1.0 / (1.0 + distance(a, b));
    }
    
    /**
     * Euclidean distance metric implementation.
     */
    DistanceMetric EUCLIDEAN = new DistanceMetric() {
        @Override
        public double distance(DenseVector a, DenseVector b) {
            if (a.dimension() != b.dimension()) {
                throw new IllegalArgumentException("Vectors must have same dimension");
            }
            
            double sum = 0.0;
            for (int i = 0; i < a.dimension(); i++) {
                var diff = a.get(i) - b.get(i);
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
        
        @Override
        public String toString() {
            return "EuclideanDistance";
        }
    };
    
    /**
     * Manhattan (L1) distance metric implementation.
     */
    DistanceMetric MANHATTAN = new DistanceMetric() {
        @Override
        public double distance(DenseVector a, DenseVector b) {
            if (a.dimension() != b.dimension()) {
                throw new IllegalArgumentException("Vectors must have same dimension");
            }
            
            double sum = 0.0;
            for (int i = 0; i < a.dimension(); i++) {
                sum += Math.abs(a.get(i) - b.get(i));
            }
            return sum;
        }
        
        @Override
        public String toString() {
            return "ManhattanDistance";
        }
    };
    
    /**
     * Cosine distance metric implementation (1 - cosine similarity).
     */
    DistanceMetric COSINE = new DistanceMetric() {
        @Override
        public double distance(DenseVector a, DenseVector b) {
            if (a.dimension() != b.dimension()) {
                throw new IllegalArgumentException("Vectors must have same dimension");
            }
            
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            
            for (int i = 0; i < a.dimension(); i++) {
                var aVal = a.get(i);
                var bVal = b.get(i);
                dotProduct += aVal * bVal;
                normA += aVal * aVal;
                normB += bVal * bVal;
            }
            
            if (normA == 0.0 || normB == 0.0) {
                return 1.0; // Maximum distance for zero vectors
            }
            
            var cosineSimilarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
            return 1.0 - cosineSimilarity;
        }
        
        @Override
        public double similarity(DenseVector a, DenseVector b) {
            return 1.0 - distance(a, b); // Cosine similarity directly
        }
        
        @Override
        public String toString() {
            return "CosineDistance";
        }
    };
    
    /**
     * Chebyshev (maximum) distance metric implementation.
     */
    DistanceMetric CHEBYSHEV = new DistanceMetric() {
        @Override
        public double distance(DenseVector a, DenseVector b) {
            if (a.dimension() != b.dimension()) {
                throw new IllegalArgumentException("Vectors must have same dimension");
            }
            
            double maxDiff = 0.0;
            for (int i = 0; i < a.dimension(); i++) {
                var diff = Math.abs(a.get(i) - b.get(i));
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
            return maxDiff;
        }
        
        @Override
        public String toString() {
            return "ChebyshevDistance";
        }
    };
    
    /**
     * Get distance metric by name.
     * 
     * @param name Metric name (case-insensitive)
     * @return Distance metric instance
     */
    static DistanceMetric fromString(String name) {
        if (name == null) {
            return EUCLIDEAN;
        }
        
        return switch (name.toLowerCase().trim()) {
            case "euclidean", "l2" -> EUCLIDEAN;
            case "manhattan", "l1" -> MANHATTAN;
            case "cosine" -> COSINE;
            case "chebyshev", "maximum", "linf" -> CHEBYSHEV;
            default -> EUCLIDEAN;
        };
    }
}