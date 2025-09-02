package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * High-performance vectorized weight vector for GaussianART operations.
 * 
 * Features:
 * - SIMD-optimized Gaussian probability calculations using Java Vector API
 * - Incremental mean and covariance updates with numerical stability
 * - Memory-aligned arrays for optimal vectorization
 * - Efficient multivariate Gaussian PDF computation
 * - Cache-friendly data layout and operations
 * 
 * This class represents a Gaussian cluster with mean vector, covariance matrix,
 * and sample count for incremental learning statistics.
 */
public final class VectorizedGaussianWeight implements WeightVector {
    
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double EPSILON = 1e-10; // Numerical stability threshold
    
    private final double[] mean;
    private final double[][] covariance;
    private final long sampleCount;
    private final double determinant;
    private final double logDeterminant;
    private final long creationTime;
    private final int updateCount;
    
    // Cached inverse covariance matrix for SIMD operations
    private volatile double[][] invCovariance;
    
    /**
     * Create VectorizedGaussianWeight with specified parameters.
     */
    public VectorizedGaussianWeight(double[] mean, double[][] covariance, long sampleCount, 
                                   double determinant, double logDeterminant, 
                                   long creationTime, int updateCount) {
        this.mean = Objects.requireNonNull(mean, "Mean cannot be null").clone();
        this.covariance = deepCopyMatrix(Objects.requireNonNull(covariance, "Covariance cannot be null"));
        this.sampleCount = sampleCount;
        this.determinant = determinant;
        this.logDeterminant = logDeterminant;
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        
        // Validate dimensions
        if (mean.length == 0) {
            throw new IllegalArgumentException("Mean vector cannot be empty");
        }
        if (covariance.length != mean.length) {
            throw new IllegalArgumentException("Covariance matrix size must match mean vector dimension");
        }
        for (int i = 0; i < covariance.length; i++) {
            if (covariance[i].length != mean.length) {
                throw new IllegalArgumentException("Covariance matrix must be square");
            }
        }
        if (sampleCount <= 0) {
            throw new IllegalArgumentException("Sample count must be positive, got: " + sampleCount);
        }
        if (determinant <= 0.0) {
            throw new IllegalArgumentException("Determinant must be positive, got: " + determinant);
        }
    }
    
    /**
     * Create initial VectorizedGaussianWeight from input pattern.
     */
    public static VectorizedGaussianWeight fromInput(Pattern input, VectorizedGaussianParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        int dimension = input.dimension();
        var mean = new double[dimension];
        var covariance = new double[dimension][dimension];
        
        // Initialize mean from input
        for (int i = 0; i < dimension; i++) {
            mean[i] = input.get(i);
        }
        
        // Initialize covariance as diagonal matrix with minimum variance
        double initialVariance = Math.max(params.rho_b(), 0.1); // Ensure reasonable initial variance
        double determinant = 1.0;
        
        for (int i = 0; i < dimension; i++) {
            Arrays.fill(covariance[i], 0.0);
            covariance[i][i] = initialVariance;
            determinant *= initialVariance;
        }
        
        double logDeterminant = Math.log(determinant);
        
        return new VectorizedGaussianWeight(mean, covariance, 1L, determinant, logDeterminant, 
                                          System.currentTimeMillis(), 0);
    }
    
    /**
     * Deep copy a matrix for immutability.
     */
    private static double[][] deepCopyMatrix(double[][] matrix) {
        var copy = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            copy[i] = matrix[i].clone();
        }
        return copy;
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= mean.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + mean.length);
        }
        return mean[index];
    }
    
    @Override
    public int dimension() {
        return mean.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : mean) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (parameters instanceof VectorizedGaussianParameters vParams) {
            return updateGaussian(input, vParams);
        } else {
            throw new IllegalArgumentException("Parameters must be VectorizedGaussianParameters");
        }
    }
    
    /**
     * Incremental Gaussian learning update with numerical stability.
     * Updates mean and covariance using online algorithms.
     */
    public VectorizedGaussianWeight updateGaussian(Pattern input, VectorizedGaussianParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return updateSIMD(input, params);
        } else {
            return updateStandard(input, params);
        }
    }
    
    /**
     * SIMD-optimized Gaussian update.
     */
    private VectorizedGaussianWeight updateSIMD(Pattern input, VectorizedGaussianParameters params) {
        // For now, delegate to standard update as SIMD matrix operations are complex
        // Future optimization: implement SIMD-optimized incremental covariance updates
        return updateStandard(input, params);
    }
    
    /**
     * Standard incremental Gaussian learning update.
     * Uses Welford's online algorithm for numerical stability.
     */
    private VectorizedGaussianWeight updateStandard(Pattern input, VectorizedGaussianParameters params) {
        int n = dimension();
        long newSampleCount = sampleCount + 1;
        var newMean = new double[n];
        var newCovariance = new double[n][n];
        
        // Incremental mean update: μ_new = μ_old + (x - μ_old) / n_new
        var delta = new double[n];
        for (int i = 0; i < n; i++) {
            delta[i] = input.get(i) - mean[i];
            newMean[i] = mean[i] + delta[i] / newSampleCount;
        }
        
        // Incremental covariance update using Welford's online algorithm
        var delta2 = new double[n];
        for (int i = 0; i < n; i++) {
            delta2[i] = input.get(i) - newMean[i];
        }
        
        // Update covariance: C_new = C_old + (delta * delta2^T - C_old) * gamma
        double gamma = params.gamma();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double outerProduct = delta[i] * delta2[j];
                newCovariance[i][j] = covariance[i][j] + gamma * (outerProduct - covariance[i][j]);
                
                // Apply variance constraints
                if (i == j) {
                    // Apply minimum variance constraint only
                    newCovariance[i][j] = Math.max(newCovariance[i][j], params.rho_b());
                }
            }
        }
        
        // Recompute determinant for the new covariance matrix
        double newDeterminant = computeDeterminant(newCovariance);
        double newLogDeterminant = Math.log(Math.max(newDeterminant, EPSILON));
        
        return new VectorizedGaussianWeight(newMean, newCovariance, newSampleCount, 
                                          newDeterminant, newLogDeterminant, 
                                          creationTime, updateCount + 1);
    }
    
    /**
     * Compute Gaussian probability density function.
     * p(x | μ, Σ) = (2π)^(-k/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^(-1) (x-μ))
     */
    public double computeProbabilityDensity(Pattern input, VectorizedGaussianParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return computeProbabilityDensitySIMD(input);
        } else {
            return computeProbabilityDensityStandard(input);
        }
    }
    
    /**
     * SIMD-optimized probability density computation.
     */
    private double computeProbabilityDensitySIMD(Pattern input) {
        int n = dimension();
        var diff = new double[n];
        
        // Compute (x - μ) with SIMD
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(n);
        
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = DoubleVector.fromArray(SPECIES, getInputArray(input), i);
            var meanVec = DoubleVector.fromArray(SPECIES, mean, i);
            var diffVec = inputVec.sub(meanVec);
            diffVec.intoArray(diff, i);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < n; i++) {
            diff[i] = input.get(i) - mean[i];
        }
        
        // Compute quadratic form (x-μ)^T Σ^(-1) (x-μ)
        double quadraticForm = computeQuadraticForm(diff);
        
        // Compute probability density
        double normalization = Math.pow(TWO_PI, -n / 2.0) / Math.sqrt(determinant);
        return normalization * Math.exp(-0.5 * quadraticForm);
    }
    
    /**
     * Standard probability density computation.
     */
    private double computeProbabilityDensityStandard(Pattern input) {
        int n = dimension();
        var diff = new double[n];
        
        // Compute (x - μ)
        for (int i = 0; i < n; i++) {
            diff[i] = input.get(i) - mean[i];
        }
        
        // Compute quadratic form (x-μ)^T Σ^(-1) (x-μ)
        double quadraticForm = computeQuadraticForm(diff);
        
        // Compute probability density
        double normalization = Math.pow(TWO_PI, -n / 2.0) / Math.sqrt(determinant);
        return normalization * Math.exp(-0.5 * quadraticForm);
    }
    
    /**
     * Compute quadratic form (x-μ)^T Σ^(-1) (x-μ).
     */
    private double computeQuadraticForm(double[] diff) {
        var invCov = getInverseCovariance();
        double result = 0.0;
        
        for (int i = 0; i < diff.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < diff.length; j++) {
                sum += invCov[i][j] * diff[j];
            }
            result += diff[i] * sum;
        }
        
        return result;
    }
    
    /**
     * Get or compute inverse covariance matrix (cached).
     */
    private double[][] getInverseCovariance() {
        if (invCovariance == null) {
            synchronized (this) {
                if (invCovariance == null) {
                    invCovariance = computeMatrixInverse(covariance);
                }
            }
        }
        return invCovariance;
    }
    
    /**
     * Compute vigilance test for GaussianART.
     * Returns probability density value for comparison with vigilance threshold.
     */
    public double computeVigilance(Pattern input, VectorizedGaussianParameters params) {
        return computeProbabilityDensity(input, params);
    }
    
    /**
     * Compute determinant of a matrix using LU decomposition.
     */
    private static double computeDeterminant(double[][] matrix) {
        int n = matrix.length;
        var lu = deepCopyMatrix(matrix);
        
        double det = 1.0;
        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(lu[j][i]) > Math.abs(lu[pivot][i])) {
                    pivot = j;
                }
            }
            
            // Swap rows if needed
            if (pivot != i) {
                var temp = lu[i];
                lu[i] = lu[pivot];
                lu[pivot] = temp;
                det = -det;
            }
            
            // Check for singular matrix
            if (Math.abs(lu[i][i]) < EPSILON) {
                return EPSILON; // Return small positive value instead of zero
            }
            
            det *= lu[i][i];
            
            // Eliminate column
            for (int j = i + 1; j < n; j++) {
                double factor = lu[j][i] / lu[i][i];
                for (int k = i + 1; k < n; k++) {
                    lu[j][k] -= factor * lu[i][k];
                }
            }
        }
        
        return Math.abs(det);
    }
    
    /**
     * Compute matrix inverse using Gauss-Jordan elimination.
     */
    private static double[][] computeMatrixInverse(double[][] matrix) {
        int n = matrix.length;
        var augmented = new double[n][2 * n];
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, n);
            augmented[i][n + i] = 1.0;
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(augmented[j][i]) > Math.abs(augmented[pivot][i])) {
                    pivot = j;
                }
            }
            
            // Swap rows
            if (pivot != i) {
                var temp = augmented[i];
                augmented[i] = augmented[pivot];
                augmented[pivot] = temp;
            }
            
            // Scale pivot row
            double pivotValue = augmented[i][i];
            if (Math.abs(pivotValue) < EPSILON) {
                // Singular matrix - add small diagonal term for numerical stability
                augmented[i][i] = EPSILON;
                pivotValue = EPSILON;
            }
            
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivotValue;
            }
            
            // Eliminate column
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double factor = augmented[j][i];
                    for (int k = 0; k < 2 * n; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        var inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(augmented[i], n, inverse[i], 0, n);
        }
        
        return inverse;
    }
    
    /**
     * Convert Pattern to double array for SIMD operations.
     */
    private double[] getInputArray(Pattern input) {
        var array = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            array[i] = input.get(i);
        }
        return array;
    }
    
    // Accessors
    
    public double[] getMean() {
        return mean.clone();
    }
    
    public double[][] getCovariance() {
        return deepCopyMatrix(covariance);
    }
    
    public long getSampleCount() {
        return sampleCount;
    }
    
    public double getDeterminant() {
        return determinant;
    }
    
    public double getLogDeterminant() {
        return logDeterminant;
    }
    
    public long getCreationTime() {
        return creationTime;
    }
    
    public int getUpdateCount() {
        return updateCount;
    }
    
    public long getAge() {
        return System.currentTimeMillis() - creationTime;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedGaussianWeight other)) return false;
        
        return Arrays.equals(mean, other.mean) &&
               Arrays.deepEquals(covariance, other.covariance) &&
               sampleCount == other.sampleCount &&
               Double.compare(determinant, other.determinant) == 0 &&
               creationTime == other.creationTime &&
               updateCount == other.updateCount;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(mean), Arrays.deepHashCode(covariance), 
                          sampleCount, determinant, creationTime, updateCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedGaussianWeight{dim=%d, samples=%d, det=%.6f, updates=%d, age=%dms}",
                           dimension(), sampleCount, determinant, updateCount, getAge());
    }
}