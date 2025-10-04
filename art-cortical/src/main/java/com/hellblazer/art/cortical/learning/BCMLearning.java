package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;

/**
 * BCM (Bienenstock-Cooper-Munro) learning rule with sliding threshold.
 *
 * <p>Implements the BCM theory of synaptic plasticity (Bienenstock, Cooper & Munro, 1982),
 * which explains how neurons can develop selectivity through experience-dependent learning.
 *
 * <h2>Mathematical Formulation</h2>
 *
 * <pre>
 * Δw_ij = α × φ(y_j, θ_j) × x_i
 *
 * where:
 * - φ(y, θ) = y × (y - θ)  [BCM plasticity function]
 * - y_j: post-synaptic activation
 * - θ_j: modification threshold (sliding)
 * - x_i: pre-synaptic activation
 * - α: learning rate
 *
 * Threshold update:
 * θ_j = E[y_j²]  (running average of squared activity)
 * </pre>
 *
 * <h2>Key Properties</h2>
 *
 * <h3>Sliding Threshold</h3>
 * <p>The modification threshold θ adapts based on average activity:
 * <ul>
 *   <li>If y &gt; θ: LTP (Long-Term Potentiation) - weights increase</li>
 *   <li>If y &lt; θ: LTD (Long-Term Depression) - weights decrease</li>
 *   <li>If y = θ: No change (equilibrium)</li>
 * </ul>
 *
 * <h3>Selectivity Development</h3>
 * <p>BCM naturally develops:
 * <ul>
 *   <li><b>Feature Selectivity</b>: Neurons become selective to specific patterns</li>
 *   <li><b>Homeostasis</b>: Activity stabilizes around the threshold</li>
 *   <li><b>Competition</b>: Winner-take-all dynamics emerge</li>
 *   <li><b>Decorrelation</b>: Different neurons learn different features</li>
 * </ul>
 *
 * <h2>Biological Motivation</h2>
 *
 * <p>BCM learning matches experimental observations:
 * <ul>
 *   <li><b>Visual Cortex Development</b>: Orientation selectivity in V1</li>
 *   <li><b>Ocular Dominance</b>: Competition between eyes during development</li>
 *   <li><b>Critical Period</b>: Experience-dependent plasticity windows</li>
 *   <li><b>Homeostatic Plasticity</b>: Activity-dependent threshold regulation</li>
 * </ul>
 *
 * <h2>Comparison with Hebbian Learning</h2>
 *
 * <table border="1">
 *   <tr>
 *     <th>Property</th>
 *     <th>Hebbian</th>
 *     <th>BCM</th>
 *   </tr>
 *   <tr>
 *     <td>Update rule</td>
 *     <td>Δw = α × x × y</td>
 *     <td>Δw = α × φ(y, θ) × x</td>
 *   </tr>
 *   <tr>
 *     <td>LTD (depression)</td>
 *     <td>Requires separate mechanism</td>
 *     <td>Built-in (when y &lt; θ)</td>
 *   </tr>
 *   <tr>
 *     <td>Stability</td>
 *     <td>Requires weight bounds/decay</td>
 *     <td>Self-stabilizing via threshold</td>
 *   </tr>
 *   <tr>
 *     <td>Selectivity</td>
 *     <td>Develops slowly</td>
 *     <td>Develops rapidly</td>
 *   </tr>
 * </table>
 *
 * <h2>Usage Examples</h2>
 *
 * <h3>Visual Cortex Development</h3>
 * <pre>{@code
 * // Layer 4: Develop orientation selectivity
 * var bcm = new BCMLearning(
 *     0.5,      // threshold decay rate (slow adaptation)
 *     0.0001,   // weight decay
 *     0.0, 1.0  // weight bounds
 * );
 * layer4.enableLearning(bcm);
 *
 * // Train on oriented edges
 * for (var edge : orientedEdges) {
 *     layer4.processAndLearn(edge);
 * }
 * }</pre>
 *
 * <h3>Competitive Learning</h3>
 * <pre>{@code
 * // Layer 2/3: Winner-take-all category formation
 * var bcm = BCMLearning.createCompetitive();  // Fast threshold
 * layer23.enableLearning(bcm);
 * }</pre>
 *
 * <h3>Homeostatic Plasticity</h3>
 * <pre>{@code
 * // Layer 6: Stable feedback learning
 * var bcm = BCMLearning.createHomeostatic();  // Very slow threshold
 * layer6.enableLearning(bcm);
 * }</pre>
 *
 * <h2>Integration with Resonance</h2>
 * <p>BCM combines naturally with resonance-gated learning:
 * <pre>{@code
 * var bcm = new BCMLearning(0.5, 0.0001, 0.0, 1.0);
 * var gated = new ResonanceGatedLearning(bcm, 0.7);
 * circuit.enableLearning(gated);
 * }</pre>
 *
 * <h2>References</h2>
 * <ul>
 *   <li>Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
 *       Theory for the development of neuron selectivity: orientation specificity
 *       and binocular interaction in visual cortex.
 *       Journal of Neuroscience, 2(1), 32-48.</li>
 *   <li>Cooper, L. N., & Bear, M. F. (2012). The BCM theory of synapse modification
 *       at 30: interaction of theory with experiment.
 *       Nature Reviews Neuroscience, 13(11), 798-810.</li>
 *   <li>Yeung, L. C., Shouval, H. Z., Blais, B. S., & Cooper, L. N. (2004).
 *       Synaptic homeostasis and input selectivity follow from a calcium-dependent
 *       plasticity model. PNAS, 101(41), 14943-14948.</li>
 * </ul>
 *
 * @author Phase 3D: Advanced Learning Rules
 * @see HebbianLearning
 * @see InstarOutstarLearning
 */
public class BCMLearning implements LearningRule {

    private final double thresholdDecayRate;
    private final double weightDecayRate;
    private final double minWeight;
    private final double maxWeight;

    // Per-neuron sliding thresholds (indexed by post-synaptic neuron)
    private double[] modificationThresholds;

    /**
     * Create BCM learning rule.
     *
     * @param thresholdDecayRate Threshold adaptation rate [0, 1] (typically 0.1-0.9)
     * @param weightDecayRate Weight decay rate [0, 1] (typically 0.0001-0.001)
     * @param minWeight Minimum weight bound (typically 0.0)
     * @param maxWeight Maximum weight bound (typically 1.0)
     * @throws IllegalArgumentException if parameters invalid
     */
    public BCMLearning(
            double thresholdDecayRate,
            double weightDecayRate,
            double minWeight,
            double maxWeight) {

        if (thresholdDecayRate < 0.0 || thresholdDecayRate > 1.0) {
            throw new IllegalArgumentException(
                "thresholdDecayRate must be in [0, 1]: " + thresholdDecayRate);
        }
        if (weightDecayRate < 0.0 || weightDecayRate > 1.0) {
            throw new IllegalArgumentException(
                "weightDecayRate must be in [0, 1]: " + weightDecayRate);
        }
        if (minWeight < 0.0 || maxWeight > 1.0 || minWeight >= maxWeight) {
            throw new IllegalArgumentException(
                "Invalid weight bounds: min=" + minWeight + ", max=" + maxWeight);
        }

        this.thresholdDecayRate = thresholdDecayRate;
        this.weightDecayRate = weightDecayRate;
        this.minWeight = minWeight;
        this.maxWeight = maxWeight;
        this.modificationThresholds = null;  // Initialized on first use
    }

    /**
     * Create BCM with fast threshold adaptation (competitive learning).
     *
     * @return BCM with fast threshold (0.8), small weight decay
     */
    public static BCMLearning createCompetitive() {
        return new BCMLearning(0.8, 0.0001, 0.0, 1.0);
    }

    /**
     * Create BCM with medium threshold adaptation (balanced).
     *
     * @return BCM with medium threshold (0.5), medium weight decay
     */
    public static BCMLearning createBalanced() {
        return new BCMLearning(0.5, 0.0005, 0.0, 1.0);
    }

    /**
     * Create BCM with slow threshold adaptation (homeostatic).
     *
     * @return BCM with slow threshold (0.1), small weight decay
     */
    public static BCMLearning createHomeostatic() {
        return new BCMLearning(0.1, 0.0001, 0.0, 1.0);
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

        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        if (currentWeights.getCols() != preSize || currentWeights.getRows() != postSize) {
            throw new IllegalArgumentException(
                "Weight matrix dimensions don't match: " +
                "weights(" + currentWeights.getRows() + "x" + currentWeights.getCols() + ") " +
                "vs pre(" + preSize + ") × post(" + postSize + ")"
            );
        }

        // Initialize thresholds on first use
        if (modificationThresholds == null || modificationThresholds.length != postSize) {
            modificationThresholds = new double[postSize];
            // Initialize to small positive values
            for (int j = 0; j < postSize; j++) {
                modificationThresholds[j] = 0.1;
            }
        }

        // Update thresholds: θ_j = (1-τ) × θ_j + τ × y_j²
        for (int j = 0; j < postSize; j++) {
            double ySquared = postActivation.get(j) * postActivation.get(j);
            modificationThresholds[j] = (1.0 - thresholdDecayRate) * modificationThresholds[j]
                                      + thresholdDecayRate * ySquared;
        }

        // Create new weight matrix
        var newWeights = new WeightMatrix(postSize, preSize);

        // Apply BCM plasticity: Δw_ij = α × φ(y_j, θ_j) × x_i
        for (int j = 0; j < postSize; j++) {
            double y = postActivation.get(j);
            double theta = modificationThresholds[j];

            // BCM plasticity function: φ(y, θ) = y × (y - θ)
            double phi = y * (y - theta);

            for (int i = 0; i < preSize; i++) {
                double x = preActivation.get(i);
                double oldWeight = currentWeights.get(j, i);

                // BCM update
                double delta = learningRate * phi * x;

                // Weight decay
                double decayed = oldWeight * (1.0 - weightDecayRate * learningRate);

                // Apply update
                double updated = decayed + delta;

                // Clip to bounds
                updated = Math.max(minWeight, Math.min(maxWeight, updated));

                newWeights.set(j, i, updated);
            }
        }

        return newWeights;
    }

    /**
     * Get current modification thresholds for each post-synaptic neuron.
     *
     * @return array of thresholds, or null if not yet initialized
     */
    public double[] getModificationThresholds() {
        return modificationThresholds != null
            ? modificationThresholds.clone()
            : null;
    }

    /**
     * Reset modification thresholds to initial values.
     */
    public void resetThresholds() {
        if (modificationThresholds != null) {
            for (int j = 0; j < modificationThresholds.length; j++) {
                modificationThresholds[j] = 0.1;
            }
        }
    }

    /**
     * Get threshold decay rate.
     *
     * @return threshold adaptation rate [0, 1]
     */
    public double getThresholdDecayRate() {
        return thresholdDecayRate;
    }

    /**
     * Get weight decay rate.
     *
     * @return weight decay rate [0, 1]
     */
    public double getWeightDecayRate() {
        return weightDecayRate;
    }

    @Override
    public String getName() {
        return "BCM";
    }

    @Override
    public boolean requiresNormalization() {
        // BCM is self-stabilizing via sliding threshold
        return false;
    }

    @Override
    public double[] getRecommendedLearningRateRange() {
        // BCM works well with moderate to high learning rates
        return new double[]{0.01, 0.5};
    }

    @Override
    public String toString() {
        return "BCMLearning[" +
               "thresholdDecay=" + thresholdDecayRate +
               ", weightDecay=" + weightDecayRate +
               ", bounds=[" + minWeight + ", " + maxWeight + "]" +
               "]";
    }
}
