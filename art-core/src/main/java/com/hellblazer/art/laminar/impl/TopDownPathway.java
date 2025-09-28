package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Top-down pathway implementation for feedback signal propagation.
 * Carries expectations, predictions, and attentional signals from higher to lower layers.
 *
 * @author Hal Hildebrand
 */
public class TopDownPathway extends AbstractPathway {

    private final boolean useGating;
    private Pattern gatingSignal;

    public TopDownPathway(String sourceLayerId, String targetLayerId,
                         int sourceSize, int targetSize) {
        this(sourceLayerId, targetLayerId, sourceSize, targetSize, true);
    }

    public TopDownPathway(String sourceLayerId, String targetLayerId,
                         int sourceSize, int targetSize, boolean useGating) {
        super(PathwayType.TOP_DOWN, sourceLayerId, targetLayerId, sourceSize, targetSize);
        this.useGating = useGating;
        this.gatingSignal = null;
    }

    @Override
    protected Pattern processSignal(Pattern signal, IPathwayParameters parameters) {
        // Apply expectation gating if enabled
        if (useGating && gatingSignal != null) {
            signal = applyGating(signal, gatingSignal);
        }

        // Top-down signals often act as modulatory rather than driving
        signal = applyModulation(signal, parameters);

        return signal;
    }

    /**
     * Apply gating based on match between expectation and input.
     */
    private Pattern applyGating(Pattern expectation, Pattern gate) {
        var result = new double[expectation.dimension()];

        for (int i = 0; i < expectation.dimension(); i++) {
            var expValue = expectation.get(i);
            var gateValue = i < gate.dimension() ? gate.get(i) : 1.0;

            // Gate controls how much expectation passes through
            result[i] = expValue * gateValue;
        }

        return new DenseVector(result);
    }

    /**
     * Apply modulatory transformation to top-down signal.
     */
    private Pattern applyModulation(Pattern signal, IPathwayParameters parameters) {
        var modulationStrength = parameters.getConnectionStrength();
        var result = new double[signal.dimension()];

        for (int i = 0; i < signal.dimension(); i++) {
            var value = signal.get(i);

            // Sigmoid-like modulation to keep signals bounded
            result[i] = modulationStrength * value / (1.0 + Math.abs(value));
        }

        return new DenseVector(result);
    }

    /**
     * Set the gating signal for expectation control.
     */
    public void setGatingSignal(Pattern gatingSignal) {
        this.gatingSignal = gatingSignal;
    }

    /**
     * Clear the gating signal.
     */
    public void clearGatingSignal() {
        this.gatingSignal = null;
    }

    @Override
    public void updateWeights(Pattern source, Pattern target, double learningRate) {
        if (!isAdaptive()) return;

        // Top-down weights learn the prototype pattern
        for (int i = 0; i < weights.getRows(); i++) {
            if (i < target.dimension() && target.get(i) > 0) {
                for (int j = 0; j < weights.getCols(); j++) {
                    var sourceValue = j < source.dimension() ? source.get(j) : 0.0;
                    var currentWeight = weights.get(i, j);

                    // Learn expectation pattern
                    var newWeight = currentWeight + learningRate * sourceValue * (1.0 - currentWeight);

                    weights.set(i, j, Math.max(0.0, Math.min(1.0, newWeight)));
                }
            }
        }
    }

    public boolean isUsingGating() {
        return useGating;
    }
}