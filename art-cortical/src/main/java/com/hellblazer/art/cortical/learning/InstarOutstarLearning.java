package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;

/**
 * Instar/Outstar learning rule from Adaptive Resonance Theory.
 *
 * <p>Implements Grossberg's (1976) instar and outstar learning laws:
 * <ul>
 *   <li><b>Instar (Bottom-Up)</b>: Learn to recognize input patterns</li>
 *   <li><b>Outstar (Top-Down)</b>: Learn to predict/generate output patterns</li>
 * </ul>
 *
 * <h2>Mathematical Formulation</h2>
 *
 * <h3>Instar Learning (Bottom-Up Recognition)</h3>
 * <p>Learns to match input patterns:
 * <pre>
 * Δw_ji = α × y_j × (x_i - w_ji)
 *
 * where:
 * - w_ji: weight from input i to category j
 * - x_i: input activation
 * - y_j: category activation (post-synaptic)
 * - α: learning rate
 * </pre>
 *
 * <p>This moves weights toward the input pattern, implementing template matching.
 *
 * <h3>Outstar Learning (Top-Down Prediction)</h3>
 * <p>Learns to generate/predict patterns:
 * <pre>
 * Δw_ij = α × y_j × (x_i - w_ij)
 *
 * where:
 * - w_ij: weight from category j to output i
 * - x_i: target output pattern
 * - y_j: category activation
 * - α: learning rate
 * </pre>
 *
 * <p>This moves weights toward the output pattern, implementing pattern completion.
 *
 * <h2>Biological Motivation</h2>
 * <p>Neuroscience evidence for instar/outstar dynamics:
 * <ul>
 *   <li><b>Instar</b>: Bottom-up feature detection (LGN → V1)</li>
 *   <li><b>Outstar</b>: Top-down predictions (V1 → LGN)</li>
 *   <li><b>Both</b>: Bidirectional cortical connections</li>
 * </ul>
 *
 * <h2>Usage Examples</h2>
 *
 * <h3>Bottom-Up Recognition (Instar Only)</h3>
 * <pre>{@code
 * // Layer 4: Learn to recognize thalamic input patterns
 * var instarLearning = new InstarOutstarLearning(
 *     LearningMode.INSTAR,
 *     0.0,   // no weight decay for pure instar
 *     0.0,   // min weight
 *     1.0    // max weight
 * );
 * layer4.enableLearning(instarLearning);
 * }</pre>
 *
 * <h3>Top-Down Prediction (Outstar Only)</h3>
 * <pre>{@code
 * // Layer 6: Learn to predict/modulate lower layer patterns
 * var outstarLearning = new InstarOutstarLearning(
 *     LearningMode.OUTSTAR,
 *     0.0001,  // small decay for stability
 *     0.0,
 *     1.0
 * );
 * layer6.enableLearning(outstarLearning);
 * }</pre>
 *
 * <h3>Bidirectional Learning (Both)</h3>
 * <pre>{@code
 * // Layer 2/3: Learn both recognition and prediction
 * var bidir = new InstarOutstarLearning(
 *     LearningMode.BOTH,
 *     0.0001,
 *     0.0,
 *     1.0
 * );
 * layer23.enableLearning(bidir);
 * }</pre>
 *
 * <h2>Relation to ART Algorithms</h2>
 * <p>This is the fundamental learning rule underlying:
 * <ul>
 *   <li>ART1, ART2, ART2A (binary/continuous patterns)</li>
 *   <li>FuzzyART (fuzzy pattern matching)</li>
 *   <li>ARTMAP (supervised category learning)</li>
 * </ul>
 *
 * <h2>References</h2>
 * <ul>
 *   <li>Grossberg, S. (1976). Adaptive pattern classification and universal recoding:
 *       I. Parallel development and coding of neural feature detectors.
 *       Biological Cybernetics, 23(3), 121-134.</li>
 *   <li>Grossberg, S. (2013). Adaptive Resonance Theory: How a brain learns to
 *       consciously attend, learn, and recognize a changing world.
 *       Neural Networks, 37, 1-47.</li>
 *   <li>Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture
 *       for a self-organizing neural pattern recognition machine.
 *       Computer Vision, Graphics, and Image Processing, 37(1), 54-115.</li>
 * </ul>
 *
 * @author Phase 3D: Advanced Learning Rules
 * @see HebbianLearning
 * @see ResonanceGatedLearning
 */
public class InstarOutstarLearning implements LearningRule {

    /**
     * Learning mode: INSTAR (bottom-up), OUTSTAR (top-down), or BOTH.
     */
    public enum LearningMode {
        /** Instar learning only: learn to recognize input patterns (bottom-up) */
        INSTAR,

        /** Outstar learning only: learn to predict/generate patterns (top-down) */
        OUTSTAR,

        /** Both instar and outstar: bidirectional learning */
        BOTH
    }

    private final LearningMode mode;
    private final double decayRate;
    private final double minWeight;
    private final double maxWeight;

    /**
     * Create instar/outstar learning rule.
     *
     * @param mode Learning mode (INSTAR, OUTSTAR, or BOTH)
     * @param decayRate Weight decay rate [0, 1] (typically 0.0-0.001)
     * @param minWeight Minimum weight bound (typically 0.0)
     * @param maxWeight Maximum weight bound (typically 1.0)
     * @throws IllegalArgumentException if parameters invalid
     */
    public InstarOutstarLearning(
            LearningMode mode,
            double decayRate,
            double minWeight,
            double maxWeight) {

        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        if (decayRate < 0.0 || decayRate > 1.0) {
            throw new IllegalArgumentException("decayRate must be in [0, 1]: " + decayRate);
        }
        if (minWeight < 0.0 || maxWeight > 1.0 || minWeight >= maxWeight) {
            throw new IllegalArgumentException(
                "Invalid weight bounds: min=" + minWeight + ", max=" + maxWeight);
        }

        this.mode = mode;
        this.decayRate = decayRate;
        this.minWeight = minWeight;
        this.maxWeight = maxWeight;
    }

    /**
     * Create instar-only learning (default for Layer 4).
     *
     * @return Instar learning with no decay, weights in [0, 1]
     */
    public static InstarOutstarLearning createInstar() {
        return new InstarOutstarLearning(LearningMode.INSTAR, 0.0, 0.0, 1.0);
    }

    /**
     * Create outstar-only learning (default for Layer 6).
     *
     * @return Outstar learning with small decay, weights in [0, 1]
     */
    public static InstarOutstarLearning createOutstar() {
        return new InstarOutstarLearning(LearningMode.OUTSTAR, 0.0001, 0.0, 1.0);
    }

    /**
     * Create bidirectional learning (default for Layer 2/3).
     *
     * @return Both instar and outstar, with small decay
     */
    public static InstarOutstarLearning createBidirectional() {
        return new InstarOutstarLearning(LearningMode.BOTH, 0.0001, 0.0, 1.0);
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

        // Create new weight matrix for update
        var newWeights = new WeightMatrix(postSize, preSize);

        // Apply learning based on mode
        if (mode == LearningMode.INSTAR || mode == LearningMode.BOTH) {
            applyInstarLearning(preActivation, postActivation, currentWeights, newWeights, learningRate);
        }

        if (mode == LearningMode.OUTSTAR || mode == LearningMode.BOTH) {
            applyOutstarLearning(preActivation, postActivation, currentWeights, newWeights, learningRate);
        }

        // If mode is BOTH, average the two updates
        if (mode == LearningMode.BOTH) {
            // Already combined in the apply methods with 0.5 weighting
        }

        return newWeights;
    }

    /**
     * Apply instar learning: Δw_ji = α × y_j × (x_i - w_ji)
     *
     * <p>Moves weights toward input pattern (template matching).
     */
    private void applyInstarLearning(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            WeightMatrix newWeights,
            double learningRate) {

        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        // Weight for BOTH mode (0.5 each), 1.0 for INSTAR only
        double modeWeight = (mode == LearningMode.BOTH) ? 0.5 : 1.0;

        for (int j = 0; j < postSize; j++) {
            double postAct = postActivation.get(j);

            for (int i = 0; i < preSize; i++) {
                double preAct = preActivation.get(i);
                double oldWeight = currentWeights.get(j, i);

                // Instar update: w += α × y × (x - w)
                double delta = learningRate * postAct * (preAct - oldWeight);

                // Apply decay: w *= (1 - β × α)
                double decayed = oldWeight * (1.0 - decayRate * learningRate);

                // Combine: new = old + delta, with decay
                double updated = decayed + delta * modeWeight;

                // Clip to bounds
                updated = Math.max(minWeight, Math.min(maxWeight, updated));

                newWeights.set(j, i, updated);
            }
        }
    }

    /**
     * Apply outstar learning: Δw_ij = α × y_j × (x_i - w_ij)
     *
     * <p>Moves weights toward output pattern (pattern completion).
     */
    private void applyOutstarLearning(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            WeightMatrix newWeights,
            double learningRate) {

        var preSize = preActivation.dimension();
        var postSize = postActivation.dimension();

        // Weight for BOTH mode (0.5 each), 1.0 for OUTSTAR only
        double modeWeight = (mode == LearningMode.BOTH) ? 0.5 : 1.0;

        for (int j = 0; j < postSize; j++) {
            double postAct = postActivation.get(j);

            for (int i = 0; i < preSize; i++) {
                double preAct = preActivation.get(i);

                // For outstar, we want to learn the input pattern at this category
                // This is identical to instar mathematically, but conceptually different
                double oldWeight = currentWeights.get(j, i);

                // Outstar update: same form as instar
                double delta = learningRate * postAct * (preAct - oldWeight);

                double decayed = oldWeight * (1.0 - decayRate * learningRate);
                double updated = decayed + delta * modeWeight;

                updated = Math.max(minWeight, Math.min(maxWeight, updated));

                // If OUTSTAR only, set directly; if BOTH, add to instar contribution
                if (mode == LearningMode.OUTSTAR) {
                    newWeights.set(j, i, updated);
                } else {
                    // BOTH mode: add outstar contribution (delta only) to existing instar
                    double currentValue = newWeights.get(j, i);
                    newWeights.set(j, i, Math.max(minWeight, Math.min(maxWeight, currentValue + delta * modeWeight)));
                }
            }
        }
    }

    /**
     * Get the learning mode.
     *
     * @return learning mode (INSTAR, OUTSTAR, or BOTH)
     */
    public LearningMode getMode() {
        return mode;
    }

    /**
     * Get decay rate.
     *
     * @return weight decay rate [0, 1]
     */
    public double getDecayRate() {
        return decayRate;
    }

    @Override
    public String getName() {
        return "InstarOutstar[" + mode + "]";
    }

    @Override
    public boolean requiresNormalization() {
        // Instar/outstar naturally keeps weights bounded, no normalization needed
        return false;
    }

    @Override
    public double[] getRecommendedLearningRateRange() {
        // ART algorithms typically use higher learning rates than Hebbian
        return switch (mode) {
            case INSTAR -> new double[]{0.1, 0.9};     // Fast bottom-up learning
            case OUTSTAR -> new double[]{0.05, 0.5};   // Medium top-down learning
            case BOTH -> new double[]{0.1, 0.7};       // Balanced bidirectional
        };
    }

    @Override
    public String toString() {
        return "InstarOutstarLearning[" +
               "mode=" + mode +
               ", decay=" + decayRate +
               ", bounds=[" + minWeight + ", " + maxWeight + "]" +
               "]";
    }
}
