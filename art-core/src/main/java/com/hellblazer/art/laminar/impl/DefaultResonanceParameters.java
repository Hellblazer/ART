package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.IResonanceParameters;

/**
 * Default resonance parameters.
 *
 * @author Hal Hildebrand
 */
public record DefaultResonanceParameters(
        double vigilance,
        double matchThreshold,
        int maxSearchCycles,
        boolean useFastLearning
) implements IResonanceParameters {

    public static final DefaultResonanceParameters DEFAULT = new DefaultResonanceParameters(
            0.8, 0.7, 100, true
    );

    @Override
    public double getVigilance() {
        return vigilance;
    }

    @Override
    public double getMatchThreshold() {
        return matchThreshold;
    }

    @Override
    public int getMaxSearchCycles() {
        return maxSearchCycles;
    }

    @Override
    public boolean useFastLearning() {
        return useFastLearning;
    }
}