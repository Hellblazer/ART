package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Default pathway parameters.
 *
 * @author Hal Hildebrand
 */
public record DefaultPathwayParameters(
        double connectionStrength,
        double initialGain,
        int propagationDelay,
        double signalAttenuation,
        double learningRate,
        boolean hebbian,
        boolean competitive,
        double maxGain,
        double minGain,
        double gainDecay
) implements IPathwayParameters {

    public static final DefaultPathwayParameters DEFAULT = new DefaultPathwayParameters(
            1.0, 1.0, 0, 0.0, 0.01, true, false, 2.0, 0.1, 0.01
    );

    @Override
    public double getConnectionStrength() {
        return connectionStrength;
    }

    @Override
    public double getInitialGain() {
        return initialGain;
    }

    @Override
    public int getPropagationDelay() {
        return propagationDelay;
    }

    @Override
    public double getSignalAttenuation() {
        return signalAttenuation;
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public boolean isHebbian() {
        return hebbian;
    }

    @Override
    public boolean isCompetitive() {
        return competitive;
    }

    @Override
    public double getMaxGain() {
        return maxGain;
    }

    @Override
    public double getMinGain() {
        return minGain;
    }

    @Override
    public double getGainDecay() {
        return gainDecay;
    }
}