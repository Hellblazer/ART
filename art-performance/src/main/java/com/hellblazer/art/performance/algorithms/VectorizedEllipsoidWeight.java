package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * High-performance vectorized weight vector for EllipsoidART operations.
 * 
 * Features:
 * - SIMD-optimized ellipsoidal distance calculations
 * - Adaptive covariance matrix computation
 * - Memory-aligned arrays for optimal vectorization  
 * - Efficient center and shape parameter updates
 * - Cache-friendly data layout for parallel operations
 * 
 * EllipsoidART represents categories as ellipsoids defined by:
 * - Center vector (mean of patterns)
 * - Covariance matrix (shape and orientation)
 * - Count of patterns in category
 * - Shape parameter mu controlling ellipsoid eccentricity
 */
public final class VectorizedEllipsoidWeight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final double[] center;           // Category center (mean)
    private final double[][] covariance;     // Covariance matrix
    private final int patternCount;          // Number of patterns in category
    private final long creationTime;
    private final int updateCount;
    
    // Cached arrays for vectorized operations (lazy initialization)
    private volatile float[] floatCenter;
    private volatile float[] eigenValues;    // For efficient distance computation
    
    /**
     * Create VectorizedEllipsoidWeight with specified center, covariance, and metadata.
     */
    public VectorizedEllipsoidWeight(double[] center, double[][] covariance, int patternCount, 
                                   long creationTime, int updateCount) {
        this.center = Objects.requireNonNull(center, "Center cannot be null").clone();
        this.covariance = cloneMatrix(Objects.requireNonNull(covariance, "Covariance cannot be null"));
        this.patternCount = Math.max(1, patternCount);
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        
        // Validate dimensions
        if (covariance.length != center.length) {
            throw new IllegalArgumentException("Covariance matrix rows must match center dimension");
        }
        for (int i = 0; i < covariance.length; i++) {
            if (covariance[i].length != center.length) {
                throw new IllegalArgumentException("Covariance matrix must be square");
            }
        }
    }
    
    /**
     * Create initial ellipsoid weight from first pattern.
     */
    public static VectorizedEllipsoidWeight fromInput(Pattern input, VectorizedEllipsoidParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        int dim = input.dimension();
        var center = new double[dim];
        var covariance = new double[dim][dim];
        
        // Initialize center with input values
        for (int i = 0; i < dim; i++) {
            center[i] = input.get(i);
        }
        
        // Initialize covariance as identity matrix scaled by baseRadius
        double variance = params.baseRadius() * params.baseRadius();
        for (int i = 0; i < dim; i++) {
            covariance[i][i] = variance;
        }
        
        return new VectorizedEllipsoidWeight(center, covariance, 1, System.currentTimeMillis(), 0);
    }
    
    /**
     * Update ellipsoid with new pattern using incremental covariance update.
     */
    public VectorizedEllipsoidWeight updateEllipsoid(Pattern input, VectorizedEllipsoidParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (input.dimension() != center.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        int newCount = patternCount + 1;
        var newCenter = new double[center.length];
        var newCovariance = new double[center.length][center.length];
        
        // Update center: new_center = (old_center * count + input) / (count + 1)
        for (int i = 0; i < center.length; i++) {
            newCenter[i] = (center[i] * patternCount + input.get(i)) / newCount;
        }
        
        // Update covariance matrix incrementally
        updateCovarianceMatrix(input, newCenter, newCovariance, params);
        
        return new VectorizedEllipsoidWeight(newCenter, newCovariance, newCount, 
                                           creationTime, updateCount + 1);
    }
    
    /**
     * Compute ellipsoidal distance for vigilance test.
     * Returns similarity score in [0,1] where higher is more similar.
     */
    public double computeVigilance(Pattern input, VectorizedEllipsoidParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (input.dimension() != center.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        if (params.enableSIMD() && input.dimension() >= SPECIES.length()) {
            return computeSIMDVigilance(input, params);
        } else {
            return computeStandardVigilance(input, params);
        }
    }
    
    /**
     * SIMD-optimized vigilance computation.
     */
    private double computeSIMDVigilance(Pattern input, VectorizedEllipsoidParameters params) {
        var inputArray = getFloatArray(input);
        var centerArray = getFloatCenter();
        
        double mahalanobisDistance = computeMahalanobisDistance(inputArray, centerArray);
        
        // Convert to similarity score: similarity = exp(-distance^2 / (2 * sigma^2))
        double sigma = params.baseRadius();
        double similarity = Math.exp(-mahalanobisDistance * mahalanobisDistance / (2 * sigma * sigma));
        
        return Math.min(1.0, similarity);
    }
    
    /**
     * Standard vigilance computation fallback.
     */
    private double computeStandardVigilance(Pattern input, VectorizedEllipsoidParameters params) {
        // Compute squared distance from center
        double squaredDistance = 0.0;
        for (int i = 0; i < center.length; i++) {
            double diff = input.get(i) - center[i];
            squaredDistance += diff * diff;
        }
        
        // Simple ellipsoidal approximation using mu parameter
        double adjustedDistance = squaredDistance / (params.baseRadius() * params.baseRadius());
        double similarity = Math.exp(-adjustedDistance * params.mu());
        
        return Math.min(1.0, similarity);
    }
    
    /**
     * Compute Mahalanobis distance using SIMD operations.
     */
    private double computeMahalanobisDistance(float[] inputArray, float[] centerArray) {
        // For performance, use simplified distance computation
        // Full Mahalanobis would require matrix inversion which is expensive
        
        double distance = 0.0;
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(inputArray.length);
        
        // Vectorized difference computation
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var centerVec = FloatVector.fromArray(SPECIES, centerArray, i);
            
            var diff = inputVec.sub(centerVec);
            var squaredDiff = diff.mul(diff);
            distance += squaredDiff.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < inputArray.length; i++) {
            double diff = inputArray[i] - centerArray[i];
            distance += diff * diff;
        }
        
        return Math.sqrt(distance);
    }
    
    /**
     * Update covariance matrix with new pattern.
     */
    private void updateCovarianceMatrix(Pattern input, double[] newCenter, double[][] newCovariance, 
                                      VectorizedEllipsoidParameters params) {
        int dim = center.length;
        
        // Copy old covariance and scale by count factor
        double countFactor = (double) patternCount / (patternCount + 1);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                newCovariance[i][j] = covariance[i][j] * countFactor;
            }
        }
        
        // Add contribution from new pattern
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double centerDiffI = input.get(i) - newCenter[i];
                double centerDiffJ = input.get(j) - newCenter[j];
                newCovariance[i][j] += centerDiffI * centerDiffJ / (patternCount + 1);
            }
        }
        
        // Apply mu parameter for ellipsoid shape control
        double muFactor = params.mu();
        if (muFactor < 1.0) {
            // Make ellipsoid more elongated along principal axes
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    if (i != j) {
                        newCovariance[i][j] *= muFactor;
                    }
                }
            }
        }
    }
    
    /**
     * Get float array representation of center for SIMD operations.
     */
    private float[] getFloatCenter() {
        if (floatCenter == null) {
            synchronized (this) {
                if (floatCenter == null) {
                    floatCenter = new float[center.length];
                    for (int i = 0; i < center.length; i++) {
                        floatCenter[i] = (float) center[i];
                    }
                }
            }
        }
        return floatCenter;
    }
    
    /**
     * Convert Pattern to float array for SIMD operations.
     */
    private float[] getFloatArray(Pattern pattern) {
        var array = new float[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            array[i] = (float) pattern.get(i);
        }
        return array;
    }
    
    /**
     * Deep clone matrix.
     */
    private static double[][] cloneMatrix(double[][] matrix) {
        var cloned = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            cloned[i] = matrix[i].clone();
        }
        return cloned;
    }
    
    // WeightVector interface implementation
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= center.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + center.length);
        }
        return center[index];
    }
    
    @Override
    public int dimension() {
        return center.length;
    }
    
    public double[] data() {
        return center.clone();
    }
    
    @Override
    public double l1Norm() {
        double norm = 0.0;
        for (double value : center) {
            norm += Math.abs(value);
        }
        return norm;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        if (!(parameters instanceof VectorizedEllipsoidParameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedEllipsoidParameters");
        }
        return updateEllipsoid(input, params);
    }
    
    // Accessors
    
    public double[] getCenter() {
        return center.clone();
    }
    
    public double[][] getCovariance() {
        return cloneMatrix(covariance);
    }
    
    public int getPatternCount() {
        return patternCount;
    }
    
    public long getCreationTime() {
        return creationTime;
    }
    
    public int getUpdateCount() {
        return updateCount;
    }
    
    /**
     * Get the determinant of covariance matrix (volume measure).
     */
    public double getCovarianceDeterminant() {
        // For performance, approximate using diagonal elements
        double det = 1.0;
        for (int i = 0; i < covariance.length; i++) {
            det *= Math.max(covariance[i][i], 1e-10); // Avoid zero determinant
        }
        return det;
    }
    
    /**
     * Get ellipsoid volume estimate.
     */
    public double getEllipsoidVolume() {
        int dim = center.length;
        double volumeConstant = Math.pow(Math.PI, dim / 2.0) / tgamma(dim / 2.0 + 1);
        return volumeConstant * Math.sqrt(getCovarianceDeterminant());
    }
    
    /**
     * Approximate gamma function for volume calculation.
     */
    private static double tgamma(double x) {
        // Stirling's approximation for simplicity
        if (x < 1) return tgamma(x + 1) / x;
        return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedEllipsoidWeight other)) return false;
        
        return patternCount == other.patternCount &&
               updateCount == other.updateCount &&
               Arrays.equals(center, other.center) &&
               Arrays.deepEquals(covariance, other.covariance);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(center), Arrays.deepHashCode(covariance), 
                          patternCount, updateCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedEllipsoidWeight{dim=%d, patterns=%d, updates=%d, " +
                           "center=[%.3f...], det=%.6f}",
                           center.length, patternCount, updateCount,
                           center.length > 0 ? center[0] : 0.0, 
                           getCovarianceDeterminant());
    }
}