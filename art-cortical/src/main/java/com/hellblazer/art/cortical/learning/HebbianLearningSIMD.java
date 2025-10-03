package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized Hebbian learning using Java Vector API.
 *
 * <h2>Vectorization Strategy</h2>
 *
 * <p>Standard Hebbian (scalar):
 * <pre>
 * for (j : postSize) {
 *     for (i : preSize) {
 *         Δw[j][i] = α × x_i × y_j
 *         w[j][i] += Δw[j][i] - β × w[j][i]
 *     }
 * }
 * </pre>
 *
 * <p>SIMD Hebbian (vectorized):
 * <pre>
 * for (j : postSize) {
 *     y_j = postActivation[j]
 *     for (i : 0 to preSize step VLENGTH) {
 *         // Load vector of pre-activations
 *         x_vec = DoubleVector.fromArray(preActivation, i)
 *
 *         // Vectorized outer product: Δw = α × x_vec × y_j
 *         delta_vec = x_vec.mul(α × y_j)
 *
 *         // Load current weights
 *         w_vec = DoubleVector.fromArray(weights[j], i)
 *
 *         // Vectorized decay: w_vec - β × w_vec
 *         decayed_vec = w_vec.sub(w_vec.mul(β))
 *
 *         // Apply update and clip
 *         new_w_vec = decayed_vec.add(delta_vec).max(min).min(max)
 *
 *         // Store result
 *         new_w_vec.intoArray(newWeights[j], i)
 *     }
 * }
 * </pre>
 *
 * <h2>Expected Performance</h2>
 * <ul>
 *   <li><b>Sequential</b>: 1.0x baseline</li>
 *   <li><b>SIMD</b>: 3-8x speedup (depends on vector length)</li>
 *   <li><b>Vector lengths</b>:
 *     <ul>
 *       <li>AVX-512: 8 doubles (8x theoretical)</li>
 *       <li>AVX2: 4 doubles (4x theoretical)</li>
 *       <li>SSE: 2 doubles (2x theoretical)</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Drop-in replacement for HebbianLearning
 * var learning = new HebbianLearningSIMD(0.0001, 0.0, 1.0);
 *
 * var newWeights = learning.update(
 *     preActivation,
 *     postActivation,
 *     currentWeights,
 *     learningRate
 * );
 * }</pre>
 *
 * <h2>Fallback Behavior</h2>
 * <p>If SIMD is not available or preSize is very small, automatically falls back
 * to scalar implementation for safety and correctness.
 *
 * @author Phase 4D: Learning Vectorization
 * @see HebbianLearning
 */
public class HebbianLearningSIMD implements LearningRule {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLENGTH = SPECIES.length();
    private static final int MIN_SIZE_FOR_SIMD = 8;  // Minimum preSize to use SIMD

    private final double decayRate;
    private final double minWeight;
    private final double maxWeight;
    private final HebbianLearning scalarFallback;

    /**
     * Create SIMD Hebbian learning rule.
     *
     * @param decayRate Weight decay rate [0, 1]
     * @param minWeight Minimum weight bound
     * @param maxWeight Maximum weight bound
     */
    public HebbianLearningSIMD(double decayRate, double minWeight, double maxWeight) {
        if (decayRate < 0.0 || decayRate > 1.0) {
            throw new IllegalArgumentException("decayRate must be in [0, 1]: " + decayRate);
        }
        if (minWeight >= maxWeight) {
            throw new IllegalArgumentException(
                "minWeight must be < maxWeight: " + minWeight + " >= " + maxWeight);
        }

        this.decayRate = decayRate;
        this.minWeight = minWeight;
        this.maxWeight = maxWeight;
        this.scalarFallback = new HebbianLearning(decayRate, minWeight, maxWeight);
    }

    /**
     * Create with default parameters.
     */
    public HebbianLearningSIMD() {
        this(0.0001, 0.0, 1.0);
    }

    @Override
    public WeightMatrix update(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            double learningRate) {

        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("learningRate must be in [0, 1]: " + learningRate);
        }

        // Fast path: no learning
        if (learningRate == 0.0) {
            return currentWeights;
        }

        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        // Validate dimensions
        if (currentWeights.getCols() != preSize || currentWeights.getRows() != postSize) {
            throw new IllegalArgumentException(
                "Weight matrix dimensions don't match: " +
                "weights(" + currentWeights.getRows() + "x" + currentWeights.getCols() + ") " +
                "vs pre(" + preSize + ") × post(" + postSize + ")");
        }

        // Fallback to scalar for small matrices
        if (preSize < MIN_SIZE_FOR_SIMD) {
            return scalarFallback.update(preActivation, postActivation, currentWeights, learningRate);
        }

        // SIMD vectorized update
        return updateSIMD(preActivation, postActivation, currentWeights, learningRate, preSize, postSize);
    }

    /**
     * SIMD vectorized weight update.
     */
    private WeightMatrix updateSIMD(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            double learningRate,
            int preSize,
            int postSize) {

        var newWeights = new WeightMatrix(postSize, preSize);

        // Pre-compute constants
        var effectiveDecay = decayRate * learningRate;
        var oneMinusDecay = 1.0 - effectiveDecay;

        // Broadcast constants as vectors
        var minVec = DoubleVector.broadcast(SPECIES, minWeight);
        var maxVec = DoubleVector.broadcast(SPECIES, maxWeight);
        var decayVec = DoubleVector.broadcast(SPECIES, oneMinusDecay);

        // Convert patterns to arrays for efficient access
        var preArray = preActivation.toArray();
        var postArray = postActivation.toArray();

        // For each post-synaptic neuron
        for (int j = 0; j < postSize; j++) {
            var postAct = postArray[j];
            var hebbianScale = learningRate * postAct;

            // Broadcast Hebbian scale for this neuron
            var hebbianVec = DoubleVector.broadcast(SPECIES, hebbianScale);

            // Vectorized loop over pre-synaptic neurons
            int i = 0;
            for (; i < SPECIES.loopBound(preSize); i += VLENGTH) {
                // Load pre-activations (x_i)
                var preVec = DoubleVector.fromArray(SPECIES, preArray, i);

                // Load current weights
                var weightArray = getCurrentWeightRow(currentWeights, j, preSize);
                var weightVec = DoubleVector.fromArray(SPECIES, weightArray, i);

                // Hebbian update: Δw = α × x_i × y_j
                var hebbianDelta = preVec.mul(hebbianVec);

                // Apply decay and Hebbian: w_new = w * (1-decay) + Δw
                var updated = weightVec.mul(decayVec).add(hebbianDelta);

                // Clip to bounds: max(min, min(max, w))
                updated = updated.max(minVec).min(maxVec);

                // Store result
                storeWeightRow(newWeights, j, i, updated);
            }

            // Handle remainder (tail) with scalar code
            for (; i < preSize; i++) {
                var preAct = preArray[i];
                var currentWeight = currentWeights.get(j, i);

                // Hebbian delta
                var hebbianDelta = hebbianScale * preAct;

                // Apply decay and Hebbian
                var updated = currentWeight * oneMinusDecay + hebbianDelta;

                // Clip to bounds
                updated = Math.max(minWeight, Math.min(maxWeight, updated));

                newWeights.set(j, i, updated);
            }
        }

        return newWeights;
    }

    /**
     * Get a row of weights as an array for SIMD access.
     */
    private double[] getCurrentWeightRow(WeightMatrix weights, int row, int cols) {
        var array = new double[cols];
        for (int i = 0; i < cols; i++) {
            array[i] = weights.get(row, i);
        }
        return array;
    }

    /**
     * Store a vector into a weight matrix row.
     */
    private void storeWeightRow(WeightMatrix weights, int row, int startCol, DoubleVector vec) {
        var temp = new double[VLENGTH];
        vec.intoArray(temp, 0);
        for (int i = 0; i < VLENGTH && (startCol + i) < weights.getCols(); i++) {
            weights.set(row, startCol + i, temp[i]);
        }
    }

    @Override
    public String getName() {
        return "Hebbian-SIMD";
    }

    @Override
    public boolean requiresNormalization() {
        return false;
    }

    @Override
    public double[] getRecommendedLearningRateRange() {
        return new double[]{0.001, 0.1};
    }

    public double getDecayRate() {
        return decayRate;
    }

    public double getMinWeight() {
        return minWeight;
    }

    public double getMaxWeight() {
        return maxWeight;
    }

    /**
     * Get the vector species being used for SIMD operations.
     */
    public VectorSpecies<Double> getVectorSpecies() {
        return SPECIES;
    }

    /**
     * Get the vector length (number of doubles per vector).
     */
    public int getVectorLength() {
        return VLENGTH;
    }

    @Override
    public String toString() {
        return "HebbianLearningSIMD[decay=" + decayRate +
               ", bounds=[" + minWeight + ", " + maxWeight + "]" +
               ", vectorLength=" + VLENGTH + "]";
    }
}
