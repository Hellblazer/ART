package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.MatchResult;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Objects;

/**
 * Vectorized weight structure for ART1 binary pattern recognition algorithm.
 * 
 * ART1 uses dual weight vectors for binary pattern processing:
 * - Bottom-up weights: Used for category choice/activation calculation (T_j = |I ∧ w_j| / (L + |w_j|))
 * - Top-down weights: Used for match criterion and learning (binary values only)
 * 
 * This implementation provides SIMD-optimized operations for high-performance
 * binary pattern processing while maintaining full compatibility with ART1 semantics.
 * 
 * Features:
 * - SIMD-accelerated dot products and logical operations
 * - Binary constraint enforcement for top-down weights
 * - Optimized memory layout for cache efficiency
 * - Comprehensive validation for algorithm correctness
 */
public class VectorizedART1Weight implements WeightVector {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final double[] bottomUpWeights;    // Choice function weights (continuous values)
    private final int[] topDownWeights;        // Match criterion weights (binary: 0 or 1)
    private final long creationTimestamp;      // For performance tracking
    private final int usageCount;              // For cache management
    
    /**
     * Create a VectorizedART1Weight with specified bottom-up and top-down weights.
     * 
     * @param bottomUpWeights Bottom-up weights for activation calculation
     * @param topDownWeights Top-down binary weights for matching
     * @param creationTimestamp Timestamp for performance tracking
     * @param usageCount Usage counter for cache management
     * @throws IllegalArgumentException if dimensions don't match or weights are invalid
     * @throws NullPointerException if any weight array is null
     */
    public VectorizedART1Weight(double[] bottomUpWeights, int[] topDownWeights, 
                               long creationTimestamp, int usageCount) {
        Objects.requireNonNull(bottomUpWeights, "Bottom-up weights cannot be null");
        Objects.requireNonNull(topDownWeights, "Top-down weights cannot be null");
        
        if (bottomUpWeights.length != topDownWeights.length) {
            throw new IllegalArgumentException(
                String.format("Bottom-up and top-down weights must have same dimension: %d vs %d", 
                             bottomUpWeights.length, topDownWeights.length));
        }
        
        if (bottomUpWeights.length == 0) {
            throw new IllegalArgumentException("Weight vectors cannot be empty");
        }
        
        // Validate binary constraint for top-down weights
        for (int i = 0; i < topDownWeights.length; i++) {
            if (topDownWeights[i] != 0 && topDownWeights[i] != 1) {
                throw new IllegalArgumentException(
                    String.format("Top-down weights must be binary (0 or 1), found %d at index %d", 
                                 topDownWeights[i], i));
            }
        }
        
        // Validate bottom-up weights are non-negative (typical for ART1)
        for (int i = 0; i < bottomUpWeights.length; i++) {
            if (bottomUpWeights[i] < 0.0) {
                throw new IllegalArgumentException(
                    String.format("Bottom-up weights should be non-negative, found %.6f at index %d", 
                                 bottomUpWeights[i], i));
            }
        }
        
        this.bottomUpWeights = Arrays.copyOf(bottomUpWeights, bottomUpWeights.length);
        this.topDownWeights = Arrays.copyOf(topDownWeights, topDownWeights.length);
        this.creationTimestamp = creationTimestamp;
        this.usageCount = usageCount;
    }
    
    /**
     * Create a VectorizedART1Weight from an input pattern for new category initialization.
     * Following ART1 algorithm, initial weights are set directly from the binary input.
     * 
     * @param input Binary input pattern (must contain only 0.0 or 1.0 values)
     * @param params VectorizedART1Parameters for configuration
     * @return VectorizedART1Weight initialized for the input pattern
     * @throws IllegalArgumentException if input is not binary
     * @throws NullPointerException if input or params is null
     */
    public static VectorizedART1Weight fromInput(Pattern input, VectorizedART1Parameters params) {
        Objects.requireNonNull(input, "Input pattern cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        validateBinaryPattern(input);
        
        int dimension = input.dimension();
        var bottomUp = new double[dimension];
        var topDown = new int[dimension];
        
        // Initialize weights from binary input pattern
        // Initial bottom-up weights equal input values (will be normalized during activation)
        for (int i = 0; i < dimension; i++) {
            double value = input.get(i);
            topDown[i] = value != 0.0 ? 1 : 0;
            bottomUp[i] = value; // Direct copy of input, formula applied during updates
        }
        
        return new VectorizedART1Weight(bottomUp, topDown, System.currentTimeMillis(), 0);
    }
    
    /**
     * Validate that a pattern contains only binary values (0.0 or 1.0).
     * 
     * @param pattern Pattern to validate
     * @throws IllegalArgumentException if pattern contains non-binary values
     */
    private static void validateBinaryPattern(Pattern pattern) {
        for (int i = 0; i < pattern.dimension(); i++) {
            double value = pattern.get(i);
            if (value != 0.0 && value != 1.0) {
                throw new IllegalArgumentException(
                    String.format("ART1 requires binary input patterns, found %.6f at index %d", value, i));
            }
        }
    }
    
    /**
     * Compute vectorized activation using ART1 choice function: T_j = |I ∧ w_j| / (L + |w_j|).
     * Uses SIMD operations for high performance on large patterns.
     * 
     * @param input Binary input pattern
     * @param params VectorizedART1Parameters containing L parameter
     * @return Activation value for this category
     */
    public double computeActivation(Pattern input, VectorizedART1Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        validateBinaryPattern(input);
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException(
                String.format("Input dimension %d does not match weight dimension %d", 
                             input.dimension(), dimension()));
        }
        
        if (params.enableSIMD() && dimension() >= SPECIES.length()) {
            return computeSIMDActivation(input, params);
        } else {
            return computeStandardActivation(input, params);
        }
    }
    
    /**
     * SIMD-optimized activation computation for large patterns.
     */
    private double computeSIMDActivation(Pattern input, VectorizedART1Parameters params) {
        int dimension = dimension();
        
        // Convert input to float array for SIMD processing
        var inputArray = new float[dimension];
        var topDownFloatArray = new float[dimension];
        
        for (int i = 0; i < dimension; i++) {
            inputArray[i] = (float) input.get(i);
            topDownFloatArray[i] = (float) topDownWeights[i];
        }
        
        double intersectionSum = 0.0;
        double weightSum = 0.0;
        
        int vectorLength = SPECIES.length();
        int upperBound = SPECIES.loopBound(dimension);
        
        // Vectorized loop for binary AND and weight sum
        for (int i = 0; i < upperBound; i += vectorLength) {
            var inputVec = FloatVector.fromArray(SPECIES, inputArray, i);
            var weightVec = FloatVector.fromArray(SPECIES, topDownFloatArray, i);
            
            // Binary AND: min(input, weight) for binary values
            var intersection = inputVec.min(weightVec);
            intersectionSum += intersection.reduceLanes(VectorOperators.ADD);
            
            // Sum top-down weights for normalization
            weightSum += weightVec.reduceLanes(VectorOperators.ADD);
        }
        
        // Handle remaining elements
        for (int i = upperBound; i < dimension; i++) {
            double inputVal = input.get(i);
            int weightVal = topDownWeights[i];
            intersectionSum += Math.min(inputVal, weightVal);
            weightSum += weightVal;
        }
        
        // ART1 choice function: T_j = |I ∧ w_j| / (L + |w_j|)
        return intersectionSum / (params.L() + weightSum);
    }
    
    /**
     * Standard activation computation fallback for small patterns or when SIMD is disabled.
     */
    private double computeStandardActivation(Pattern input, VectorizedART1Parameters params) {
        double intersectionSum = 0.0;
        double weightSum = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            int weightVal = topDownWeights[i];
            
            // Binary AND operation: min for binary values
            intersectionSum += Math.min(inputVal, weightVal);
            weightSum += weightVal;
        }
        
        // ART1 choice function: T_j = |I ∧ w_j| / (L + |w_j|)
        return intersectionSum / (params.L() + weightSum);
    }
    
    /**
     * Compute vigilance match criterion: |I ∧ w_j| / |I| >= ρ.
     * 
     * @param input Binary input pattern
     * @param params VectorizedART1Parameters containing vigilance parameter
     * @return Vigilance ratio [0, 1]
     */
    public double computeVigilance(Pattern input, VectorizedART1Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        validateBinaryPattern(input);
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException(
                String.format("Input dimension %d does not match weight dimension %d", 
                             input.dimension(), dimension()));
        }
        
        double intersectionSum = 0.0;
        double inputSum = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double inputVal = input.get(i);
            int weightVal = topDownWeights[i];
            
            // Binary AND for intersection
            intersectionSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }
        
        // Avoid division by zero for all-zero inputs
        if (inputSum == 0.0) {
            return 0.0;
        }
        
        // Vigilance criterion: |I ∧ w_j| / |I|
        return intersectionSum / inputSum;
    }
    
    /**
     * Check vigilance criterion and return MatchResult.
     * 
     * @param input Binary input pattern
     * @param params VectorizedART1Parameters containing vigilance threshold
     * @return MatchResult.Accepted if vigilance test passes, MatchResult.Rejected otherwise
     */
    public MatchResult checkVigilance(Pattern input, VectorizedART1Parameters params) {
        double vigilanceRatio = computeVigilance(input, params);
        
        if (vigilanceRatio >= params.vigilance()) {
            return new MatchResult.Accepted(vigilanceRatio, params.vigilance());
        } else {
            return new MatchResult.Rejected(vigilanceRatio, params.vigilance());
        }
    }
    
    /**
     * Update weights using ART1 learning rule: new_weight = input AND old_weight.
     * This implements the conservative learning rule where weights can only decrease.
     * 
     * @param input Binary input pattern that matched this category
     * @param params VectorizedART1Parameters for configuration
     * @return Updated VectorizedART1Weight with learned pattern
     */
    public VectorizedART1Weight updateWithLearning(Pattern input, VectorizedART1Parameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        validateBinaryPattern(input);
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException(
                String.format("Input dimension %d does not match weight dimension %d", 
                             input.dimension(), dimension()));
        }
        
        int dimension = dimension();
        var newTopDown = new int[dimension];
        var newBottomUp = new double[dimension];
        
        int activeCount = 0;
        
        // ART1 learning rule: new_weight = input AND old_weight
        for (int i = 0; i < dimension; i++) {
            int inputBit = input.get(i) != 0.0 ? 1 : 0;
            int oldWeight = topDownWeights[i];
            
            // Binary AND operation
            newTopDown[i] = inputBit & oldWeight;
            if (newTopDown[i] == 1) {
                activeCount++;
            }
        }
        
        // Update bottom-up weights based on new top-down pattern using correct ART1 formula
        for (int i = 0; i < dimension; i++) {
            if (activeCount > 0) {
                newBottomUp[i] = newTopDown[i] / (params.L() - 1.0 + activeCount);
            } else {
                newBottomUp[i] = 0.0;
            }
        }
        
        return new VectorizedART1Weight(newBottomUp, newTopDown, 
                                       creationTimestamp, usageCount + 1);
    }
    
    /**
     * Get the bottom-up weights used for activation computation.
     * 
     * @return Copy of bottom-up weight array
     */
    public double[] getBottomUpWeights() {
        return Arrays.copyOf(bottomUpWeights, bottomUpWeights.length);
    }
    
    /**
     * Get the top-down binary weights used for matching.
     * 
     * @return Copy of top-down weight array
     */
    public int[] getTopDownWeights() {
        return Arrays.copyOf(topDownWeights, topDownWeights.length);
    }
    
    /**
     * Get creation timestamp for performance tracking.
     * 
     * @return Creation timestamp in milliseconds
     */
    public long getCreationTimestamp() {
        return creationTimestamp;
    }
    
    /**
     * Get usage count for cache management.
     * 
     * @return Number of times this weight has been used
     */
    public int getUsageCount() {
        return usageCount;
    }
    
    // WeightVector interface implementation
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= bottomUpWeights.length) {
            throw new IndexOutOfBoundsException(
                String.format("Index %d out of bounds for dimension %d", index, bottomUpWeights.length));
        }
        return bottomUpWeights[index];
    }
    
    @Override
    public int dimension() {
        return bottomUpWeights.length;
    }
    
    @Override
    public double l1Norm() {
        double norm = 0.0;
        for (double weight : bottomUpWeights) {
            norm += Math.abs(weight);
        }
        return norm;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        if (!(parameters instanceof VectorizedART1Parameters params)) {
            throw new IllegalArgumentException("Parameters must be VectorizedART1Parameters");
        }
        return updateWithLearning(input, params);
    }
    
    public double[] data() {
        return getBottomUpWeights();
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof VectorizedART1Weight other)) return false;
        
        return Arrays.equals(bottomUpWeights, other.bottomUpWeights) &&
               Arrays.equals(topDownWeights, other.topDownWeights);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(bottomUpWeights) * 31 + Arrays.hashCode(topDownWeights);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedART1Weight{dim=%d, usage=%d, timestamp=%d, topDown=%s}", 
                           dimension(), usageCount, creationTimestamp, 
                           Arrays.toString(Arrays.copyOf(topDownWeights, Math.min(8, topDownWeights.length))));
    }
}