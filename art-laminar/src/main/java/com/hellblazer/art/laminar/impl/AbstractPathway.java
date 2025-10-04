package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.Pathway;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.core.WeightMatrix;
import com.hellblazer.art.laminar.parameters.PathwayParameters;

/**
 * Abstract pathway implementation.
 *
 * @author Hal Hildebrand
 */
public abstract class AbstractPathway implements Pathway {
    protected final String id;
    protected final String sourceLayerId;
    protected final String targetLayerId;
    protected final PathwayType type;
    protected double gain;
    protected boolean adaptive;
    protected WeightMatrix weights;

    public AbstractPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
        this.id = id;
        this.sourceLayerId = sourceLayerId;
        this.targetLayerId = targetLayerId;
        this.type = type;
        this.gain = 1.0;
        this.adaptive = true;
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public String getSourceLayerId() {
        return sourceLayerId;
    }

    @Override
    public String getTargetLayerId() {
        return targetLayerId;
    }

    @Override
    public PathwayType getType() {
        return type;
    }

    @Override
    public Pattern propagate(Pattern signal, PathwayParameters parameters) {
        // Apply gain to signal
        var result = new double[signal.dimension()];
        var effectiveGain = parameters != null ? parameters.getGain() : gain;

        for (int i = 0; i < signal.dimension(); i++) {
            result[i] = signal.get(i) * effectiveGain;
        }

        return new DenseVector(result);
    }

    @Override
    public double getGain() {
        return gain;
    }

    @Override
    public void setGain(double gain) {
        this.gain = gain;
    }

    @Override
    public boolean isAdaptive() {
        return adaptive;
    }

    @Override
    public void updateWeights(Pattern sourceActivation, Pattern targetActivation, double learningRate) {
        if (!adaptive || weights == null) {
            return;
        }

        // Simple Hebbian learning
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                if (i < targetActivation.dimension() && j < sourceActivation.dimension()) {
                    var current = weights.get(i, j);
                    var delta = learningRate * sourceActivation.get(j) * targetActivation.get(i);
                    weights.set(i, j, current + delta);
                }
            }
        }
    }

    @Override
    public void reset() {
        gain = 1.0;
        if (weights != null) {
            // Reset weights to small random values
            for (int i = 0; i < weights.getRows(); i++) {
                for (int j = 0; j < weights.getCols(); j++) {
                    weights.set(i, j, Math.random() * 0.1);
                }
            }
        }
    }
}