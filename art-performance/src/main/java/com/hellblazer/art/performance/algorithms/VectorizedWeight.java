package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import org.joml.Vector3f;
import org.joml.Vector4f;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/**
 * High-performance weight vector optimized for vectorized operations.
 * 
 * Features:
 * - Memory-aligned arrays for SIMD operations
 * - JOML Vector3f/Vector4f conversions for 3D/4D math
 * - Efficient update operations
 * - Optimized similarity computations
 * - Cache-friendly data layout
 */
public final class VectorizedWeight implements WeightVector {
    
    private final double[] weights;
    private final long creationTime;
    private final int updateCount;
    
    // Cached arrays for vectorized operations (lazy initialization)
    private volatile float[] floatWeights;
    private volatile Vector3f vector3f;
    private volatile Vector4f vector4f;
    
    /**
     * Create VectorizedWeight with specified weights.
     */
    public VectorizedWeight(double[] weights, long creationTime, int updateCount) {
        this.weights = Objects.requireNonNull(weights, "Weights cannot be null").clone();
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        
        // Validate weights are non-negative
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] < 0.0) {
                throw new IllegalArgumentException("Weight at index " + i + " must be non-negative, got: " + weights[i]);
            }
        }
    }
    
    /**
     * Create initial VectorizedWeight from input.
     */
    public static VectorizedWeight fromInput(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        var weights = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            weights[i] = input.get(i);
        }
        
        return new VectorizedWeight(weights, System.currentTimeMillis(), 0);
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= weights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + weights.length);
        }
        return weights[index];
    }
    
    @Override
    public int dimension() {
        return weights.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double weight : weights) {
            sum += Math.abs(weight);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof VectorizedParameters vParams)) {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        // Vectorized update computation
        var newWeights = new double[dimension()];
        double learningRate = vParams.learningRate();
        
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            double fuzzyMin = Math.min(inputVal, weights[i]);
            newWeights[i] = learningRate * fuzzyMin + (1.0 - learningRate) * weights[i];
        }
        
        return new VectorizedWeight(newWeights, creationTime, updateCount + 1);
    }
    
    /**
     * Get weights as float array for SIMD operations.
     */
    public float[] getCategoryWeights() {
        if (floatWeights == null) {
            synchronized (this) {
                if (floatWeights == null) {
                    floatWeights = new float[weights.length];
                    for (int i = 0; i < weights.length; i++) {
                        floatWeights[i] = (float) weights[i];
                    }
                }
            }
        }
        return floatWeights.clone();
    }
    
    /**
     * Convert input Pattern to float array for SIMD operations.
     */
    public float[] getInputArray(Pattern input) {
        var array = new float[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            array[i] = (float) input.get(i);
        }
        return array;
    }
    
    /**
     * Get as JOML Vector3f (cached for performance).
     */
    public Vector3f asVector3f() {
        if (dimension() != 3) {
            throw new IllegalStateException("Weight must be 3-dimensional for Vector3f conversion");
        }
        
        if (vector3f == null) {
            synchronized (this) {
                if (vector3f == null) {
                    vector3f = new Vector3f((float) weights[0], (float) weights[1], (float) weights[2]);
                }
            }
        }
        return new Vector3f(vector3f); // Return copy to maintain immutability
    }
    
    /**
     * Convert input Pattern to JOML Vector3f.
     */
    public Vector3f asVector3f(Pattern input) {
        if (input.dimension() != 3) {
            throw new IllegalArgumentException("Input must be 3-dimensional for Vector3f conversion");
        }
        return new Vector3f((float) input.get(0), (float) input.get(1), (float) input.get(2));
    }
    
    /**
     * Get as JOML Vector4f (cached for performance).
     */
    public Vector4f asVector4f() {
        if (dimension() != 4) {
            throw new IllegalStateException("Weight must be 4-dimensional for Vector4f conversion");
        }
        
        if (vector4f == null) {
            synchronized (this) {
                if (vector4f == null) {
                    vector4f = new Vector4f((float) weights[0], (float) weights[1], 
                                           (float) weights[2], (float) weights[3]);
                }
            }
        }
        return new Vector4f(vector4f); // Return copy to maintain immutability
    }
    
    /**
     * Convert input Pattern to JOML Vector4f.
     */
    public Vector4f asVector4f(Pattern input) {
        if (input.dimension() != 4) {
            throw new IllegalArgumentException("Input must be 4-dimensional for Vector4f conversion");
        }
        return new Vector4f((float) input.get(0), (float) input.get(1), 
                           (float) input.get(2), (float) input.get(3));
    }
    
    /**
     * Compute vectorized similarity using optimized algorithms.
     */
    public double computeSimilarity(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        if (dimension() == 3 && params.enableJOML()) {
            return computeJOMLSimilarity3D(input);
        } else if (dimension() == 4 && params.enableJOML()) {
            return computeJOMLSimilarity4D(input);
        } else {
            return computeStandardSimilarity(input);
        }
    }
    
    /**
     * JOML-optimized 3D similarity computation.
     */
    private double computeJOMLSimilarity3D(Pattern input) {
        var inputVec = asVector3f(input);
        var weightVec = asVector3f();
        
        // Compute fuzzy intersection
        var intersection = new Vector3f();
        intersection.x = Math.min(inputVec.x, weightVec.x);
        intersection.y = Math.min(inputVec.y, weightVec.y);
        intersection.z = Math.min(inputVec.z, weightVec.z);
        
        // Compute union
        var union = new Vector3f();
        union.x = Math.max(inputVec.x, weightVec.x);
        union.y = Math.max(inputVec.y, weightVec.y);
        union.z = Math.max(inputVec.z, weightVec.z);
        
        double intersectionNorm = intersection.length();
        double unionNorm = union.length();
        
        return unionNorm > 0.0 ? intersectionNorm / unionNorm : 0.0;
    }
    
    /**
     * JOML-optimized 4D similarity computation.
     */
    private double computeJOMLSimilarity4D(Pattern input) {
        var inputVec = asVector4f(input);
        var weightVec = asVector4f();
        
        // Compute fuzzy intersection
        var intersection = new Vector4f();
        intersection.x = Math.min(inputVec.x, weightVec.x);
        intersection.y = Math.min(inputVec.y, weightVec.y);
        intersection.z = Math.min(inputVec.z, weightVec.z);
        intersection.w = Math.min(inputVec.w, weightVec.w);
        
        // Compute union
        var union = new Vector4f();
        union.x = Math.max(inputVec.x, weightVec.x);
        union.y = Math.max(inputVec.y, weightVec.y);
        union.z = Math.max(inputVec.z, weightVec.z);
        union.w = Math.max(inputVec.w, weightVec.w);
        
        double intersectionNorm = intersection.length();
        double unionNorm = union.length();
        
        return unionNorm > 0.0 ? intersectionNorm / unionNorm : 0.0;
    }
    
    /**
     * Standard similarity computation.
     */
    private double computeStandardSimilarity(Pattern input) {
        double intersection = 0.0;
        double union = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weights[i];
            intersection += Math.min(inputVal, weightVal);
            union += Math.max(inputVal, weightVal);
        }
        
        return union > 0.0 ? intersection / union : 0.0;
    }
    
    /**
     * Compute distance using optimized algorithms.
     */
    public double computeDistance(Pattern input, VectorizedParameters params) {
        if (dimension() == 3 && params.enableJOML()) {
            return asVector3f().distance(asVector3f(input));
        } else if (dimension() == 4 && params.enableJOML()) {
            return asVector4f().distance(asVector4f(input));
        } else {
            return computeEuclideanDistance(input);
        }
    }
    
    /**
     * Standard Euclidean distance computation.
     */
    private double computeEuclideanDistance(Pattern input) {
        double sumSquares = 0.0;
        for (int i = 0; i < dimension(); i++) {
            double diff = input.get(i) - weights[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Generate random noise for weight perturbation.
     */
    public VectorizedWeight addNoise(double noiseLevel, VectorizedParameters params) {
        var newWeights = new double[dimension()];
        var random = ThreadLocalRandom.current();
        
        for (int i = 0; i < dimension(); i++) {
            double noise = (random.nextDouble() - 0.5) * 2.0 * noiseLevel;
            newWeights[i] = Math.max(0.0, Math.min(1.0, weights[i] + noise));
        }
        
        return new VectorizedWeight(newWeights, creationTime, updateCount + 1);
    }
    
    // Accessors
    
    public long getCreationTime() {
        return creationTime;
    }
    
    public int getUpdateCount() {
        return updateCount;
    }
    
    public long getAge() {
        return System.currentTimeMillis() - creationTime;
    }
    
    /**
     * Get raw weight array (copy for safety).
     */
    public double[] getWeights() {
        return weights.clone();
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedWeight other)) return false;
        
        return Arrays.equals(weights, other.weights) &&
               creationTime == other.creationTime &&
               updateCount == other.updateCount;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(weights), creationTime, updateCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedWeight{dim=%d, updates=%d, age=%dms, norm=%.3f}",
                           dimension(), updateCount, getAge(), l1Norm());
    }
}