package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;
import java.util.UUID;

/**
 * Abstract base class for pathway implementations with common functionality.
 * Handles signal propagation between layers in laminar circuits.
 *
 * @author Hal Hildebrand
 */
public abstract class AbstractPathway implements IPathway {

    protected final String id;
    protected final PathwayType type;
    protected final String sourceLayerId;
    protected final String targetLayerId;
    protected WeightMatrix weights;
    protected double gain;
    protected int delay;
    protected boolean adaptive;
    protected boolean enabled;

    protected AbstractPathway(PathwayType type, String sourceLayerId, String targetLayerId,
                            int sourceSize, int targetSize) {
        this(UUID.randomUUID().toString(), type, sourceLayerId, targetLayerId, sourceSize, targetSize);
    }

    protected AbstractPathway(String id, PathwayType type, String sourceLayerId,
                            String targetLayerId, int sourceSize, int targetSize) {
        this.id = id;
        this.type = type;
        this.sourceLayerId = sourceLayerId;
        this.targetLayerId = targetLayerId;
        this.weights = new WeightMatrix(targetSize, sourceSize);
        this.gain = 1.0;
        this.delay = 0;
        this.adaptive = true;
        this.enabled = true;
        initializeWeights();
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public PathwayType getType() {
        return type;
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
    public Pattern propagate(Pattern input, IPathwayParameters parameters) {
        if (!enabled) {
            return new DenseVector(new double[weights.getRows()]);
        }

        // Apply signal attenuation
        var attenuatedInput = input.scale(1.0 - parameters.getSignalAttenuation());

        // Apply weight transformation
        var transformed = applyWeights(attenuatedInput);

        // Apply gain modulation
        var modulated = transformed.scale(gain * parameters.getConnectionStrength());

        // Apply any pathway-specific processing
        return processSignal(modulated, parameters);
    }

    @Override
    public int getDelay() {
        return delay;
    }

    @Override
    public void setDelay(int delay) {
        this.delay = Math.max(0, delay);
    }

    @Override
    public WeightMatrix getWeights() {
        return weights.copy();
    }

    @Override
    public void updateWeights(Pattern source, Pattern target, double learningRate) {
        if (!adaptive) return;

        // Default Hebbian learning with bounds
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                var srcValue = j < source.dimension() ? source.get(j) : 0.0;
                var tgtValue = i < target.dimension() ? target.get(i) : 0.0;

                var deltaW = learningRate * tgtValue * srcValue;
                var newWeight = weights.get(i, j) + deltaW;

                // Bound weights to [0, 1]
                weights.set(i, j, Math.max(0.0, Math.min(1.0, newWeight)));
            }
        }
    }

    @Override
    public boolean isAdaptive() {
        return adaptive;
    }

    @Override
    public void setAdaptive(boolean adaptive) {
        this.adaptive = adaptive;
    }

    @Override
    public void applyGain(double gain) {
        this.gain = Math.max(0.0, gain);
    }

    @Override
    public double getGain() {
        return gain;
    }

    @Override
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    @Override
    public boolean isEnabled() {
        return enabled;
    }

    // Protected methods for subclasses

    /**
     * Initialize weights with pathway-specific pattern.
     */
    protected void initializeWeights() {
        switch (type) {
            case BOTTOM_UP -> initializeBottomUpWeights();
            case TOP_DOWN -> initializeTopDownWeights();
            case HORIZONTAL -> initializeHorizontalWeights();
            case DIAGONAL -> initializeDiagonalWeights();
            case MODULATORY -> initializeModulatoryWeights();
        }
    }

    /**
     * Apply weight matrix to input pattern.
     */
    protected Pattern applyWeights(Pattern input) {
        var output = new double[weights.getRows()];

        for (int i = 0; i < weights.getRows(); i++) {
            var sum = 0.0;
            for (int j = 0; j < Math.min(weights.getCols(), input.dimension()); j++) {
                sum += weights.get(i, j) * input.get(j);
            }
            output[i] = sum;
        }

        return new DenseVector(output);
    }

    /**
     * Process signal with pathway-specific transformations.
     * Subclasses should override this for custom processing.
     */
    protected abstract Pattern processSignal(Pattern signal, IPathwayParameters parameters);

    // Weight initialization methods

    protected void initializeBottomUpWeights() {
        // Convergent weights - many-to-one mappings
        weights.randomize(0.0, 0.1);
        weights.normalize();
    }

    protected void initializeTopDownWeights() {
        // Divergent weights - one-to-many mappings
        weights.randomize(0.0, 0.2);
    }

    protected void initializeHorizontalWeights() {
        // Lateral connections - typically inhibitory
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                if (i == j) {
                    weights.set(i, j, 0.0);  // No self-connections
                } else {
                    weights.set(i, j, -0.1 * Math.random());  // Inhibitory
                }
            }
        }
    }

    protected void initializeDiagonalWeights() {
        // Skip connections - direct pathways
        weights.randomize(0.0, 0.05);
    }

    protected void initializeModulatoryWeights() {
        // Gain control - uniform weak connections
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                weights.set(i, j, 0.1);
            }
        }
    }

    /**
     * Update gain with decay towards baseline.
     */
    public void updateGain(IPathwayParameters parameters) {
        var targetGain = parameters.getInitialGain();
        var decay = parameters.getGainDecay();

        // Decay towards baseline
        gain = gain + decay * (targetGain - gain);

        // Clamp to valid range
        gain = Math.max(parameters.getMinGain(), Math.min(parameters.getMaxGain(), gain));
    }
}