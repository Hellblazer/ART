package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;

/**
 * Hebbian learning rule with weight decay and bounds.
 *
 * <p>Implements the classic Hebbian plasticity principle:
 * "Cells that fire together, wire together" (Hebb, 1949).
 *
 * <h2>Learning Dynamics</h2>
 * <p>The weight update follows:
 * <pre>
 * Δw_ij = α × x_i × y_j - β × w_ij
 * w_ij := clip(w_ij + Δw_ij, w_min, w_max)
 * </pre>
 * where:
 * <ul>
 *   <li>α: learning rate (typically 0.001-0.1)</li>
 *   <li>x_i: pre-synaptic activation (input neuron i)</li>
 *   <li>y_j: post-synaptic activation (output neuron j)</li>
 *   <li>β: weight decay rate (typically 0.0001)</li>
 *   <li>w_ij: synaptic weight from neuron i to j</li>
 *   <li>w_min, w_max: weight bounds (typically 0.0, 1.0)</li>
 * </ul>
 *
 * <h2>Weight Decay</h2>
 * <p>Weight decay is critical for stability:
 * <ul>
 *   <li>Prevents weight explosion (runaway growth)</li>
 *   <li>Implements synaptic scaling (homeostatic plasticity)</li>
 *   <li>Allows old memories to fade (catastrophic forgetting mitigation)</li>
 * </ul>
 *
 * <h2>Biological Plausibility</h2>
 * <p>This rule captures several biological phenomena:
 * <ul>
 *   <li><b>LTP</b>: Hebbian term strengthens correlated synapses</li>
 *   <li><b>Synaptic Scaling</b>: Decay term implements homeostasis</li>
 *   <li><b>Weight Bounds</b>: Physical constraints on synaptic strength</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create Hebbian learning with moderate decay
 * var learning = new HebbianLearning(
 *     0.0001,  // decay rate
 *     0.0,     // min weight
 *     1.0      // max weight
 * );
 *
 * // Apply learning during training
 * for (var pattern : trainingSet) {
 *     var output = layer.process(pattern);
 *     var newWeights = learning.update(pattern, output, layer.getWeights(), 0.01);
 *     layer.setWeights(newWeights);
 * }
 * }</pre>
 *
 * @author Phase 3A: Core Learning Infrastructure
 * @see LearningRule
 */
public class HebbianLearning implements LearningRule {

    private final double decayRate;
    private final double minWeight;
    private final double maxWeight;

    /**
     * Create Hebbian learning rule with specified parameters.
     *
     * @param decayRate Weight decay rate [0, 1], typically 0.0001
     * @param minWeight Minimum weight bound, typically 0.0
     * @param maxWeight Maximum weight bound, typically 1.0
     * @throws IllegalArgumentException if parameters are invalid
     */
    public HebbianLearning(double decayRate, double minWeight, double maxWeight) {
        if (decayRate < 0.0 || decayRate > 1.0) {
            throw new IllegalArgumentException(
                "decayRate must be in [0, 1]: " + decayRate
            );
        }
        if (minWeight >= maxWeight) {
            throw new IllegalArgumentException(
                "minWeight must be < maxWeight: " + minWeight + " >= " + maxWeight
            );
        }

        this.decayRate = decayRate;
        this.minWeight = minWeight;
        this.maxWeight = maxWeight;
    }

    /**
     * Create Hebbian learning with default parameters.
     *
     * <p>Defaults:
     * <ul>
     *   <li>Decay rate: 0.0001 (gentle decay)</li>
     *   <li>Weight bounds: [0.0, 1.0] (standard normalization)</li>
     * </ul>
     */
    public HebbianLearning() {
        this(0.0001, 0.0, 1.0);
    }

    @Override
    public WeightMatrix update(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            double learningRate) {

        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException(
                "learningRate must be in [0, 1]: " + learningRate
            );
        }

        // Fast path: no learning if rate is zero
        if (learningRate == 0.0) {
            return currentWeights;
        }

        // Get dimensions
        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        // Validate dimensions
        if (currentWeights.getCols() != preSize) {
            throw new IllegalArgumentException(
                "Weight cols (" + currentWeights.getCols() +
                ") must match pre-activation dimension (" + preSize + ")"
            );
        }
        if (currentWeights.getRows() != postSize) {
            throw new IllegalArgumentException(
                "Weight rows (" + currentWeights.getRows() +
                ") must match post-activation dimension (" + postSize + ")"
            );
        }

        // Compute Hebbian update: Δw_ij = α × x_i × y_j
        var hebbianDelta = computeHebbianDelta(preActivation, postActivation, learningRate);

        // Compute weight decay: -β × α × w_ij
        // (decay is also scaled by learning rate for consistency)
        var decayDelta = computeDecayDelta(currentWeights, learningRate);

        // Apply updates with clipping: w_ij := clip(w_ij + Δw_hebbian - Δw_decay, min, max)
        return applyUpdates(currentWeights, hebbianDelta, decayDelta);
    }

    /**
     * Compute Hebbian strengthening: Δw_ij = α × x_i × y_j
     */
    private WeightMatrix computeHebbianDelta(
            Pattern preActivation,
            Pattern postActivation,
            double learningRate) {

        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        // Create delta matrix
        var delta = new WeightMatrix(postSize, preSize);

        // Outer product: Δw[j][i] = α × x_i × y_j
        for (int j = 0; j < postSize; j++) {
            double postAct = postActivation.get(j);
            for (int i = 0; i < preSize; i++) {
                double preAct = preActivation.get(i);
                delta.set(j, i, learningRate * preAct * postAct);
            }
        }

        return delta;
    }

    /**
     * Compute weight decay: -β × α × w_ij
     */
    private WeightMatrix computeDecayDelta(WeightMatrix weights, double learningRate) {
        // Decay is also scaled by learning rate
        double effectiveDecay = decayRate * learningRate;

        var postSize = weights.getRows();
        var preSize = weights.getCols();
        var delta = new WeightMatrix(postSize, preSize);

        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                delta.set(j, i, -effectiveDecay * weights.get(j, i));
            }
        }

        return delta;
    }

    /**
     * Apply updates with clipping to weight bounds.
     */
    private WeightMatrix applyUpdates(
            WeightMatrix currentWeights,
            WeightMatrix hebbianDelta,
            WeightMatrix decayDelta) {

        var postSize = currentWeights.getRows();
        var preSize = currentWeights.getCols();
        var newWeights = new WeightMatrix(postSize, preSize);

        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                double current = currentWeights.get(j, i);
                double hebbian = hebbianDelta.get(j, i);
                double decay = decayDelta.get(j, i);

                // Apply updates
                double updated = current + hebbian + decay;

                // Clip to bounds
                newWeights.set(j, i, Math.max(minWeight, Math.min(maxWeight, updated)));
            }
        }

        return newWeights;
    }

    @Override
    public String getName() {
        return "Hebbian";
    }

    @Override
    public boolean requiresNormalization() {
        // With decay, explicit normalization is not required
        return false;
    }

    @Override
    public double[] getRecommendedLearningRateRange() {
        // Hebbian learning is stable for wide range of learning rates
        return new double[]{0.001, 0.1};
    }

    /**
     * Get weight decay rate.
     */
    public double getDecayRate() {
        return decayRate;
    }

    /**
     * Get minimum weight bound.
     */
    public double getMinWeight() {
        return minWeight;
    }

    /**
     * Get maximum weight bound.
     */
    public double getMaxWeight() {
        return maxWeight;
    }

    @Override
    public String toString() {
        return "HebbianLearning[decay=" + decayRate +
               ", bounds=[" + minWeight + ", " + maxWeight + "]]";
    }
}
