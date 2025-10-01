package com.hellblazer.art.laminar.canonical;

/**
 * Parameters controlling prediction generation.
 *
 * Based on ART theory and canonical laminar circuit dynamics.
 * Controls how top-down expectations are generated from category activations.
 *
 * @param topDownGain strength of top-down modulation [0,1]
 * @param expectationThreshold minimum category activation to contribute [0,1]
 * @param maxTemplateUpdates learning limit per category
 * @param templateLearningRate how fast templates adapt [0,1]
 *
 * @author Hal Hildebrand
 */
public record PredictionParameters(
    double topDownGain,
    double expectationThreshold,
    int maxTemplateUpdates,
    double templateLearningRate
) {
    /**
     * Default parameters based on ART theory.
     *
     * - topDownGain: 0.5 (top-down ~50% of bottom-up strength)
     * - expectationThreshold: 0.1 (ignore weak categories)
     * - maxTemplateUpdates: 100 (sufficient for convergence)
     * - templateLearningRate: 0.1 (gradual adaptation)
     */
    public PredictionParameters() {
        this(0.5, 0.1, 100, 0.1);
    }

    /**
     * Validate parameter ranges.
     */
    public PredictionParameters {
        if (topDownGain < 0.0 || topDownGain > 1.0) {
            throw new IllegalArgumentException("topDownGain must be in [0,1]");
        }
        if (expectationThreshold < 0.0 || expectationThreshold > 1.0) {
            throw new IllegalArgumentException("expectationThreshold must be in [0,1]");
        }
        if (maxTemplateUpdates < 1) {
            throw new IllegalArgumentException("maxTemplateUpdates must be positive");
        }
        if (templateLearningRate < 0.0 || templateLearningRate > 1.0) {
            throw new IllegalArgumentException("templateLearningRate must be in [0,1]");
        }
    }
}
