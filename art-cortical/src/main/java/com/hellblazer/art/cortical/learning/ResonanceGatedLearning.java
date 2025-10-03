package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;

/**
 * Learning rule wrapper that gates learning based on conscious resonance.
 *
 * <p>Implements the principle that learning should only occur during
 * conscious perception, as indicated by high consciousness likelihood
 * from the enhanced resonance detector (Phase 2D).
 *
 * <h2>Biological Motivation</h2>
 * <p>Neuroscience evidence suggests that:
 * <ul>
 *   <li>Conscious perception is required for long-term memory formation</li>
 *   <li>Subliminal stimuli (below consciousness threshold) show minimal learning</li>
 *   <li>Attention and consciousness modulate synaptic plasticity</li>
 *   <li>Resonance between bottom-up and top-down signals enables learning</li>
 * </ul>
 *
 * <h2>Gating Mechanism</h2>
 * <p>Learning is gated by consciousness likelihood:
 * <pre>
 * if (consciousness >= threshold):
 *     α_eff = α_base × consciousness
 *     apply_learning(α_eff)
 * else:
 *     no_learning()
 * </pre>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create base Hebbian learning
 * var hebbianLearning = new HebbianLearning(0.0001, 0.0, 1.0);
 *
 * // Wrap with resonance gating
 * var gatedLearning = new ResonanceGatedLearning(
 *     hebbianLearning,
 *     0.7  // Consciousness threshold
 * );
 *
 * // Learning only occurs during conscious resonance
 * var context = new LearningContext(...);
 * if (context.hasResonance() && context.isLikelyConscious(0.7)) {
 *     // This will apply learning scaled by consciousness
 *     var newWeights = gatedLearning.updateWithContext(
 *         context, currentWeights, baseLearningRate
 *     );
 * }
 * }</pre>
 *
 * <h2>Integration with Phase 2D</h2>
 * <p>This wrapper integrates with the enhanced resonance detection
 * from Phase 2D:
 * <ul>
 *   <li>Uses {@link com.hellblazer.art.cortical.resonance.ResonanceState}</li>
 *   <li>Gates learning based on consciousness likelihood</li>
 *   <li>Modulates learning rate by consciousness level</li>
 *   <li>Prevents learning from noise and weak patterns</li>
 * </ul>
 *
 * @author Phase 3B: Resonance-Gated Learning
 * @see LearningContext
 * @see com.hellblazer.art.cortical.resonance.ResonanceState
 */
public class ResonanceGatedLearning implements LearningRule {

    private final LearningRule baseLearning;
    private final double resonanceThreshold;

    /**
     * Create resonance-gated learning wrapper.
     *
     * @param baseLearning Base learning rule to wrap
     * @param resonanceThreshold Minimum consciousness likelihood for learning [0, 1]
     * @throws IllegalArgumentException if parameters invalid
     */
    public ResonanceGatedLearning(LearningRule baseLearning, double resonanceThreshold) {
        if (baseLearning == null) {
            throw new IllegalArgumentException("baseLearning cannot be null");
        }
        if (resonanceThreshold < 0.0 || resonanceThreshold > 1.0) {
            throw new IllegalArgumentException(
                "resonanceThreshold must be in [0, 1]: " + resonanceThreshold
            );
        }

        this.baseLearning = baseLearning;
        this.resonanceThreshold = resonanceThreshold;
    }

    /**
     * Update weights with resonance gating via LearningContext.
     *
     * <p>This is the preferred method when using with cortical circuits,
     * as it has access to resonance state from the context.
     *
     * @param context Learning context with resonance state
     * @param currentWeights Current weight matrix
     * @param baseLearningRate Base learning rate (before modulation)
     * @return Updated weights, or unchanged if gating prevents learning
     */
    public WeightMatrix updateWithContext(
            LearningContext context,
            WeightMatrix currentWeights,
            double baseLearningRate) {

        // Check if resonance detection is enabled
        if (!context.hasResonanceDetection()) {
            // No gating - apply base learning directly
            return baseLearning.update(
                context.preActivation(),
                context.postActivation(),
                currentWeights,
                baseLearningRate
            );
        }

        // Get consciousness likelihood
        double consciousness = context.getConsciousnessLikelihood();

        // Gate learning by consciousness threshold
        if (consciousness < resonanceThreshold) {
            // Below threshold - no learning
            return currentWeights;
        }

        // Above threshold - apply learning scaled by consciousness
        double modulatedRate = baseLearningRate * consciousness;

        return baseLearning.update(
            context.preActivation(),
            context.postActivation(),
            currentWeights,
            modulatedRate
        );
    }

    @Override
    public WeightMatrix update(
            Pattern preActivation,
            Pattern postActivation,
            WeightMatrix currentWeights,
            double learningRate) {

        // When called without context, we can't gate by resonance
        // This falls back to base learning (used for testing/simple scenarios)
        return baseLearning.update(
            preActivation,
            postActivation,
            currentWeights,
            learningRate
        );
    }

    /**
     * Get the base (unwrapped) learning rule.
     *
     * @return base learning rule
     */
    public LearningRule getBaseLearning() {
        return baseLearning;
    }

    /**
     * Get resonance threshold for learning.
     *
     * @return consciousness likelihood threshold [0, 1]
     */
    public double getResonanceThreshold() {
        return resonanceThreshold;
    }

    @Override
    public String getName() {
        return "ResonanceGated[" + baseLearning.getName() + "]";
    }

    @Override
    public boolean requiresNormalization() {
        return baseLearning.requiresNormalization();
    }

    @Override
    public double[] getRecommendedLearningRateRange() {
        // Since we modulate by consciousness, recommend slightly higher base rates
        var baseRange = baseLearning.getRecommendedLearningRateRange();
        return new double[]{
            baseRange[0] * 1.5,  // Increase min by 50%
            baseRange[1] * 1.5   // Increase max by 50%
        };
    }

    @Override
    public String toString() {
        return "ResonanceGatedLearning[" +
               "base=" + baseLearning.getName() +
               ", threshold=" + resonanceThreshold +
               "]";
    }
}
