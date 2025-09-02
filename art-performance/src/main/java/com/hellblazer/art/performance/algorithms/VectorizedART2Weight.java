package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * High-performance vectorized weight vector for ART2 operations.
 * 
 * ART2 uses normalized weight vectors (unit length) and includes
 * preprocessing with contrast enhancement (theta) and noise suppression (epsilon).
 * 
 * Features:
 * - SIMD-optimized vector operations using Java Vector API
 * - Unit vector normalization for ART2 semantics
 * - Efficient dot product and distance calculations
 * - Memory-aligned arrays for optimal vectorization
 * - Cache-friendly data layout and operations
 */
public final class VectorizedART2Weight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final double[] weights;
    private final long creationTime;
    private final int updateCount;
    
    // Cached arrays for vectorized operations (lazy initialization)
    private volatile float[] floatWeights;
    private volatile double cachedL2Norm;
    private volatile boolean l2NormCached = false;
    
    /**
     * Create VectorizedART2Weight with specified normalized weights and metadata.
     */
    public VectorizedART2Weight(double[] weights, long creationTime, int updateCount) {
        this.weights = Objects.requireNonNull(weights, "Weights cannot be null").clone();
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        
        // Validate weights are valid numbers
        for (int i = 0; i < weights.length; i++) {
            if (!Double.isFinite(weights[i])) {
                throw new IllegalArgumentException("Weight at index " + i + " must be finite, got: " + weights[i]);
            }
        }
        
        // Ensure the weight vector is normalized (unit length)
        normalizeWeights();
    }
    
    /**
     * Create initial VectorizedART2Weight from preprocessed input.
     */
    public static VectorizedART2Weight fromInput(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        // Apply ART2 preprocessing: contrast enhancement and noise suppression
        var preprocessed = preprocessART2Input(input, params);
        var weights = new double[preprocessed.dimension()];
        
        for (int i = 0; i < preprocessed.dimension(); i++) {
            weights[i] = preprocessed.get(i);
        }
        
        return new VectorizedART2Weight(weights, System.currentTimeMillis(), 0);
    }
    
    /**
     * Apply ART2 preprocessing to input pattern.
     * Includes contrast enhancement (theta) and noise suppression (epsilon).
     */
    public static Pattern preprocessART2Input(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        var processed = new double[input.dimension()];
        
        // Apply contrast enhancement and noise suppression
        for (int i = 0; i < input.dimension(); i++) {
            double value = input.get(i);
            
            // Contrast enhancement using theta
            // Enhanced value = value + theta * (value - mean)
            // Simplified: enhanced = value * (1 + theta) for positive contrast
            double enhanced = value * (1.0 + params.theta());
            
            // Noise suppression using epsilon threshold
            // Suppress small values below epsilon
            double suppressed = Math.abs(enhanced) < params.epsilon() ? 0.0 : enhanced;
            
            processed[i] = suppressed;
        }
        
        // Normalize to unit length (required for ART2)
        return normalizeToUnitLength(Pattern.of(processed));
    }
    
    /**
     * Normalize pattern to unit length (L2 norm = 1).
     */
    public static Pattern normalizeToUnitLength(Pattern pattern) {
        if (pattern instanceof DenseVector denseVector) {
            return denseVector.scale(1.0 / Math.max(denseVector.l2Norm(), 1e-10));
        }
        
        // Fallback for other Pattern types
        double norm = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double val = pattern.get(i);
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        
        if (norm < 1e-10) {
            // Handle zero vector by returning uniform small values
            var values = new double[pattern.dimension()];
            double uniformValue = 1.0 / Math.sqrt(pattern.dimension());
            Arrays.fill(values, uniformValue);
            return Pattern.of(values);
        }
        
        var normalized = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            normalized[i] = pattern.get(i) / norm;
        }
        
        return Pattern.of(normalized);
    }
    
    /**
     * Ensure weight vector is normalized (unit length).
     */
    private void normalizeWeights() {
        double norm = l2Norm();
        
        if (norm < 1e-10) {
            // Handle zero vector case
            double uniformValue = 1.0 / Math.sqrt(weights.length);
            Arrays.fill(weights, uniformValue);
        } else if (Math.abs(norm - 1.0) > 1e-6) {
            // Normalize only if not already unit length
            for (int i = 0; i < weights.length; i++) {
                weights[i] /= norm;
            }
            // Reset cached values
            l2NormCached = false;
            floatWeights = null;
        }
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
    
    public double l2Norm() {
        if (!l2NormCached) {
            double sum = 0.0;
            for (double weight : weights) {
                sum += weight * weight;
            }
            cachedL2Norm = Math.sqrt(sum);
            l2NormCached = true;
        }
        return cachedL2Norm;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (parameters instanceof VectorizedART2Parameters art2Params) {
            return updateART2(input, art2Params);
        } else {
            throw new IllegalArgumentException("Parameters must be VectorizedART2Parameters");
        }
    }
    
    /**
     * ART2 learning update with convex combination.
     * w_new = (1-β)*w_old + β*I' (where I' is preprocessed input)
     */
    public VectorizedART2Weight updateART2(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        // Preprocess input first
        var preprocessedInput = preprocessART2Input(input, params);
        
        if (preprocessedInput.dimension() != dimension()) {
            throw new IllegalArgumentException("Preprocessed input dimension must match weight dimension");
        }
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return updateSIMD(preprocessedInput, params);
        } else {
            return updateStandard(preprocessedInput, params);
        }
    }
    
    /**
     * SIMD-optimized ART2 learning update.
     */
    private VectorizedART2Weight updateSIMD(Pattern input, VectorizedART2Parameters params) {
        var inputArray = convertToFloatArray(input);
        var weightArray = getFloatWeights();
        var newWeights = new double[dimension()];
        
        // Learning rate for ART2 (typically small, around 0.1)
        float beta = 0.1f; // Fixed learning rate for ART2
        float oneMinusBeta = 1.0f - beta;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension());
        
        // Vectorized convex combination: w_new = (1-β)*w_old + β*I'
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Apply convex combination
            var betaInput = inputVec.mul(beta);
            var oneMinusBetaWeight = weightVec.mul(oneMinusBeta);
            var newWeightVec = betaInput.add(oneMinusBetaWeight);
            
            // Store results
            newWeightVec.intoArray(weightArray, i);
        }
        
        // Convert back to double and handle remaining elements
        for (int i = 0; i < upperBound; i++) {
            newWeights[i] = weightArray[i];
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weights[i];
            newWeights[i] = (1.0 - beta) * weightVal + beta * inputVal;
        }
        
        return new VectorizedART2Weight(newWeights, creationTime, updateCount + 1);
    }
    
    /**
     * Standard ART2 learning update fallback.
     */
    private VectorizedART2Weight updateStandard(Pattern input, VectorizedART2Parameters params) {
        var newWeights = new double[dimension()];
        double beta = 0.1; // Fixed learning rate for ART2
        
        // Apply convex combination: w_new = (1-β)*w_old + β*I'
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weights[i];
            newWeights[i] = (1.0 - beta) * weightVal + beta * inputVal;
        }
        
        return new VectorizedART2Weight(newWeights, creationTime, updateCount + 1);
    }
    
    /**
     * Compute dot product activation: T_j = I' · w_j
     */
    public double computeActivation(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        // Preprocess input first
        var preprocessedInput = preprocessART2Input(input, params);
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return computeActivationSIMD(preprocessedInput);
        } else {
            return computeActivationStandard(preprocessedInput);
        }
    }
    
    /**
     * SIMD-optimized dot product computation.
     */
    private double computeActivationSIMD(Pattern input) {
        var inputArray = convertToFloatArray(input);
        var weightArray = getFloatWeights();
        
        double dotProduct = 0.0;
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension());
        
        // Vectorized dot product
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute element-wise product and sum
            var product = inputVec.mul(weightVec);
            dotProduct += product.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension(); i++) {
            dotProduct += input.get(i) * weights[i];
        }
        
        return dotProduct;
    }
    
    /**
     * Standard dot product computation fallback.
     */
    private double computeActivationStandard(Pattern input) {
        double dotProduct = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            dotProduct += input.get(i) * weights[i];
        }
        
        return dotProduct;
    }
    
    /**
     * Compute vigilance test: ||I' - w_j||² ≤ (1-ρ)²
     */
    public double computeVigilance(Pattern input, VectorizedART2Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        // Preprocess input first
        var preprocessedInput = preprocessART2Input(input, params);
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return computeVigilanceSIMD(preprocessedInput);
        } else {
            return computeVigilanceStandard(preprocessedInput);
        }
    }
    
    /**
     * SIMD-optimized vigilance computation.
     */
    private double computeVigilanceSIMD(Pattern input) {
        var inputArray = convertToFloatArray(input);
        var weightArray = getFloatWeights();
        
        double distanceSquared = 0.0;
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension());
        
        // Vectorized distance computation
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute squared difference
            var diff = inputVec.sub(weightVec);
            var squaredDiff = diff.mul(diff);
            distanceSquared += squaredDiff.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension(); i++) {
            double diff = input.get(i) - weights[i];
            distanceSquared += diff * diff;
        }
        
        // Convert distance to match ratio for consistency with reference
        // High similarity (low distance) → high match ratio
        // For unit vectors, max distance² = 4 (vectors pointing in opposite directions)
        double maxDistanceSquared = 4.0;
        return 1.0 - (distanceSquared / maxDistanceSquared);
    }
    
    /**
     * Standard vigilance computation fallback.
     */
    private double computeVigilanceStandard(Pattern input) {
        double distanceSquared = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double diff = input.get(i) - weights[i];
            distanceSquared += diff * diff;
        }
        
        // Convert distance to match ratio
        double maxDistanceSquared = 4.0;
        return 1.0 - (distanceSquared / maxDistanceSquared);
    }
    
    /**
     * Get weights as float array for SIMD operations.
     */
    public float[] getFloatWeights() {
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
     * Convert Pattern to float array for SIMD operations.
     */
    private float[] convertToFloatArray(Pattern pattern) {
        var array = new float[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            array[i] = (float) pattern.get(i);
        }
        return array;
    }
    
    /**
     * Get raw weight array (copy for safety).
     */
    public double[] getWeights() {
        return weights.clone();
    }
    
    /**
     * Get the underlying DenseVector for compatibility.
     */
    public DenseVector getDenseVector() {
        return new DenseVector(weights.clone());
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
     * Check if this weight vector is properly normalized.
     */
    public boolean isNormalized() {
        return Math.abs(l2Norm() - 1.0) < 1e-6;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedART2Weight other)) return false;
        
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
        return String.format("VectorizedART2Weight{dim=%d, updates=%d, age=%dms, l2Norm=%.3f, normalized=%s}",
                           dimension(), updateCount, getAge(), l2Norm(), isNormalized());
    }
}