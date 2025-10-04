package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;

/**
 * Complete state representation for a laminar layer including temporal context.
 *
 * Integrates current activation with temporal context from chunked history,
 * enabling layers to maintain both instantaneous and historical information.
 *
 * @author Hal Hildebrand
 */
public record LayerState(
    Pattern currentActivation,
    Pattern temporalContext,
    double timestamp
) {

    /**
     * Create state with only current activation (no temporal context).
     */
    public static LayerState fromActivation(Pattern activation, double timestamp) {
        return new LayerState(activation, null, timestamp);
    }

    /**
     * Create state with both activation and temporal context.
     */
    public static LayerState withContext(Pattern activation, Pattern context, double timestamp) {
        return new LayerState(activation, context, timestamp);
    }

    /**
     * Check if this state includes temporal context.
     */
    public boolean hasTemporalContext() {
        return temporalContext != null;
    }

    /**
     * Combine current activation with temporal context using weighted sum.
     *
     * @param contextWeight Weight for temporal context (0.0 = only activation, 1.0 = only context)
     * @return Combined pattern
     */
    public Pattern combine(double contextWeight) {
        if (!hasTemporalContext()) {
            return currentActivation;
        }

        if (contextWeight <= 0.0) {
            return currentActivation;
        }

        if (contextWeight >= 1.0) {
            return temporalContext;
        }

        // Weighted combination
        double activationWeight = 1.0 - contextWeight;
        int dimension = currentActivation.dimension();
        double[] combined = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            combined[i] = activationWeight * currentActivation.get(i) +
                         contextWeight * temporalContext.get(i);
        }

        return new com.hellblazer.art.core.DenseVector(combined);
    }

    /**
     * Get the effective pattern (with context if available, otherwise activation).
     * Uses default context weight of 0.3 (30% temporal context).
     */
    public Pattern getEffectivePattern() {
        return combine(0.3);
    }

    /**
     * Update activation while keeping temporal context.
     */
    public LayerState withActivation(Pattern newActivation) {
        return new LayerState(newActivation, temporalContext, timestamp);
    }

    /**
     * Update temporal context while keeping activation.
     */
    public LayerState withTemporalContext(Pattern newContext) {
        return new LayerState(currentActivation, newContext, timestamp);
    }

    /**
     * Update timestamp while keeping patterns.
     */
    public LayerState withTimestamp(double newTimestamp) {
        return new LayerState(currentActivation, temporalContext, newTimestamp);
    }
}