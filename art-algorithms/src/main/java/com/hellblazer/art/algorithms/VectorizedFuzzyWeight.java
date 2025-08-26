package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/**
 * High-performance vectorized weight vector for FuzzyART operations.
 * 
 * Features:
 * - SIMD-optimized fuzzy operations using Java Vector API
 * - Complement coding support for FuzzyART semantics
 * - Memory-aligned arrays for optimal vectorization
 * - Efficient update operations with vectorized fuzzy min
 * - Cache-friendly data layout and operations
 */
public final class VectorizedFuzzyWeight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final double[] weights;
    private final int originalDimension;
    private final long creationTime;
    private final int updateCount;
    
    // Cached arrays for vectorized operations (lazy initialization)
    private volatile float[] floatWeights;
    
    /**
     * Create VectorizedFuzzyWeight with specified weights and metadata.
     */
    public VectorizedFuzzyWeight(double[] weights, int originalDimension, long creationTime, int updateCount) {
        this.weights = Objects.requireNonNull(weights, "Weights cannot be null").clone();
        this.originalDimension = originalDimension;
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        
        // Validate complement coding structure
        if (weights.length % 2 != 0) {
            throw new IllegalArgumentException("FuzzyWeight must have even dimension for complement coding");
        }
        if (originalDimension * 2 != weights.length) {
            throw new IllegalArgumentException("Weight dimension must be 2 * originalDimension");
        }
        
        // Validate weights are non-negative
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] < 0.0) {
                throw new IllegalArgumentException("Weight at index " + i + " must be non-negative, got: " + weights[i]);
            }
        }
    }
    
    /**
     * Create initial VectorizedFuzzyWeight from input with complement coding.
     */
    public static VectorizedFuzzyWeight fromInput(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        var complementCoded = getComplementCoded(input);
        var weights = new double[complementCoded.dimension()];
        
        for (int i = 0; i < complementCoded.dimension(); i++) {
            weights[i] = complementCoded.get(i);
        }
        
        return new VectorizedFuzzyWeight(weights, input.dimension(), System.currentTimeMillis(), 0);
    }
    
    /**
     * Apply complement coding to input pattern.
     * [x1, x2, ..., xn] -> [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]
     */
    public static Pattern getComplementCoded(Pattern input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        var complementCoded = new double[input.dimension() * 2];
        
        // Copy original values
        for (int i = 0; i < input.dimension(); i++) {
            complementCoded[i] = input.get(i);
        }
        
        // Add complement values
        for (int i = 0; i < input.dimension(); i++) {
            complementCoded[input.dimension() + i] = 1.0 - input.get(i);
        }
        
        return Pattern.of(complementCoded);
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
        
        if (parameters instanceof VectorizedParameters vParams) {
            return updateFuzzy(input, vParams);
        } else {
            throw new IllegalArgumentException("Parameters must be VectorizedParameters");
        }
    }
    
    /**
     * Vectorized fuzzy learning update: w_new = β * min(input, w_old) + (1-β) * w_old
     */
    public VectorizedFuzzyWeight updateFuzzy(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        // Get complement-coded input to match weight dimensions
        var complementCoded = getComplementCoded(input);
        
        if (complementCoded.dimension() != dimension()) {
            throw new IllegalArgumentException("Complement-coded input dimension must match weight dimension");
        }
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return updateSIMD(complementCoded, params);
        } else {
            return updateStandard(complementCoded, params);
        }
    }
    
    /**
     * SIMD-optimized fuzzy learning update.
     */
    private VectorizedFuzzyWeight updateSIMD(Pattern input, VectorizedParameters params) {
        var inputArray = convertToFloatArray(input);
        var weightArray = getCategoryWeights();
        var newWeights = new double[dimension()];
        
        float beta = (float) params.learningRate();
        float oneMinusBeta = 1.0f - beta;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension());
        
        // Vectorized fuzzy min learning rule
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute fuzzy minimum: min(input, weight)
            var fuzzyMin = inputVec.min(weightVec);
            
            // Apply learning rule: β * min(input, weight) + (1-β) * weight
            var betaMin = fuzzyMin.mul(beta);
            var oneMinusBetaWeight = weightVec.mul(oneMinusBeta);
            var newWeightVec = betaMin.add(oneMinusBetaWeight);
            
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
            double fuzzyMin = Math.min(inputVal, weightVal);
            newWeights[i] = beta * fuzzyMin + (1.0 - beta) * weightVal;
        }
        
        return new VectorizedFuzzyWeight(newWeights, originalDimension, creationTime, updateCount + 1);
    }
    
    /**
     * Standard fuzzy learning update fallback.
     */
    private VectorizedFuzzyWeight updateStandard(Pattern input, VectorizedParameters params) {
        var newWeights = new double[dimension()];
        double beta = params.learningRate();
        
        // Apply fuzzy learning rule: β * min(input, weight) + (1-β) * weight
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weights[i];
            double fuzzyMin = Math.min(inputVal, weightVal);
            newWeights[i] = beta * fuzzyMin + (1.0 - beta) * weightVal;
        }
        
        return new VectorizedFuzzyWeight(newWeights, originalDimension, creationTime, updateCount + 1);
    }
    
    /**
     * Compute vigilance test: |I ∧ w| / |I| >= ρ
     */
    public double computeVigilance(Pattern input, VectorizedParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        // Get complement-coded input to match weight dimensions
        var complementCoded = getComplementCoded(input);
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return computeVigilanceSIMD(complementCoded);
        } else {
            return computeVigilanceStandard(complementCoded);
        }
    }
    
    /**
     * SIMD-optimized vigilance computation.
     */
    private double computeVigilanceSIMD(Pattern input) {
        var inputArray = convertToFloatArray(input);
        var weightArray = getCategoryWeights();
        
        double intersectionSum = 0.0;
        double inputSum = 0.0;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension());
        
        // Vectorized intersection and input norm computation
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, weightArray, i);
            
            // Compute fuzzy intersection: min(input, weight)
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(VectorOperators.ADD);
            
            // Sum input values
            inputSum += inputVec.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension(); i++) {
            double inputVal = inputArray[i];
            double weightVal = weightArray[i];
            intersectionSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }
        
        // Vigilance test: |I ∧ w| / |I|
        return inputSum > 0.0 ? intersectionSum / inputSum : 0.0;
    }
    
    /**
     * Standard vigilance computation fallback.
     */
    private double computeVigilanceStandard(Pattern input) {
        double intersection = 0.0;
        double inputNorm = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weights[i];
            intersection += Math.min(inputVal, weightVal);
            inputNorm += inputVal;
        }
        
        return inputNorm > 0.0 ? intersection / inputNorm : 0.0;
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
     * Generate random noise for weight perturbation.
     */
    public VectorizedFuzzyWeight addNoise(double noiseLevel, VectorizedParameters params) {
        var newWeights = new double[dimension()];
        var random = ThreadLocalRandom.current();
        
        for (int i = 0; i < dimension(); i++) {
            double noise = (random.nextDouble() - 0.5) * 2.0 * noiseLevel;
            newWeights[i] = Math.max(0.0, Math.min(1.0, weights[i] + noise));
        }
        
        return new VectorizedFuzzyWeight(newWeights, originalDimension, creationTime, updateCount + 1);
    }
    
    // Accessors
    
    public int getOriginalDimension() {
        return originalDimension;
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
        if (!(obj instanceof VectorizedFuzzyWeight other)) return false;
        
        return Arrays.equals(weights, other.weights) &&
               originalDimension == other.originalDimension &&
               creationTime == other.creationTime &&
               updateCount == other.updateCount;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(weights), originalDimension, creationTime, updateCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedFuzzyWeight{dim=%d, originalDim=%d, updates=%d, age=%dms, norm=%.3f}",
                           dimension(), originalDimension, updateCount, getAge(), l1Norm());
    }
}