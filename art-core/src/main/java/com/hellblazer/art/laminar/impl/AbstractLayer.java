package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.ILayerParameters;
import com.hellblazer.art.laminar.events.ILayerActivationListener;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Abstract base class for layer implementations with common functionality.
 *
 * @author Hal Hildebrand
 */
public abstract class AbstractLayer implements ILayer {

    protected final String id;
    protected final LayerType type;
    protected final int size;
    protected Pattern activation;
    protected WeightMatrix weights;
    protected boolean plastic = true;
    protected final List<ILayerActivationListener> listeners = new CopyOnWriteArrayList<>();
    protected long updateCount = 0;

    protected AbstractLayer(String id, LayerType type, int size) {
        this.id = id != null ? id : UUID.randomUUID().toString();
        this.type = type;
        this.size = size;
        this.activation = new DenseVector(new double[size]);
        this.weights = new WeightMatrix(size, size);
        initializeWeights();
    }

    protected AbstractLayer(LayerType type, int size) {
        this(null, type, size);
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public LayerType getType() {
        return type;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Pattern getActivation() {
        // Pattern is immutable, so we can return directly
        return activation;
    }

    @Override
    public void setActivation(Pattern activation) {
        if (activation.dimension() != size) {
            throw new IllegalArgumentException("Activation size must match layer size");
        }
        var oldActivation = this.activation;
        this.activation = activation;
        notifyActivationChange(oldActivation, this.activation);
    }

    @Override
    public void reset() {
        var oldActivation = this.activation;
        this.activation = new DenseVector(new double[size]);
        notifyActivationChange(oldActivation, this.activation);
    }

    @Override
    public void updateActivation(Pattern excitation, Pattern inhibition,
                                ILayerParameters parameters, double dt) {
        var oldActivation = this.activation;

        // Grossberg shunting equation: dx/dt = -Ax + (B-x)E - (x+C)I
        var newActivation = new double[size];

        for (int i = 0; i < size; i++) {
            var x = activation.get(i);
            var E = excitation.get(i);
            var I = inhibition.get(i);

            var A = parameters.getDecayRate();
            var B = parameters.getUpperBound();
            var C = parameters.getLowerBound();

            // Shunting dynamics
            var dxdt = -A * x + (B - x) * E - (x + C) * I;

            // Apply noise if specified
            if (parameters.getNoiseLevel() > 0) {
                dxdt += applyNoise(parameters);
            }

            // Euler integration
            newActivation[i] = Math.max(0, Math.min(B, x + dt * dxdt));
        }

        this.activation = new DenseVector(newActivation);

        // Apply normalization if requested
        if (parameters.useNormalization()) {
            this.activation = normalizeActivation(this.activation, parameters.getNormalizationType());
        }

        updateCount++;
        notifyActivationChange(oldActivation, this.activation);

        // Check for threshold and saturation
        checkThresholdAndSaturation(parameters);
    }

    @Override
    public WeightMatrix getWeights() {
        return weights.copy();
    }

    @Override
    public void updateWeights(Pattern learningSignal, double learningRate) {
        if (!plastic) return;

        // Default Hebbian learning rule
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                var deltaW = learningRate * activation.get(i) * learningSignal.get(j);
                var newWeight = weights.get(i, j) + deltaW;
                weights.set(i, j, Math.max(0, Math.min(1, newWeight)));
            }
        }
    }

    @Override
    public boolean isPlastic() {
        return plastic;
    }

    @Override
    public void setPlastic(boolean plastic) {
        this.plastic = plastic;
    }

    @Override
    public LayerStatistics getStatistics() {
        var activationArray = new double[size];
        for (int i = 0; i < size; i++) {
            activationArray[i] = activation.get(i);
        }

        var mean = calculateMean(activationArray);
        var variance = calculateVariance(activationArray, mean);
        var max = calculateMax(activationArray);
        var min = calculateMin(activationArray);
        var activeNeurons = countActiveNeurons(activationArray, 0.01);
        var sparsity = 1.0 - ((double) activeNeurons / size);

        return LayerStatistics.builder()
                .withLayerId(id)
                .withMeanActivation(mean)
                .withMaxActivation(max)
                .withMinActivation(min)
                .withActivationVariance(variance)
                .withActiveNeurons(activeNeurons)
                .withTotalNeurons(size)
                .withSparsity(sparsity)
                .withUpdateCount(updateCount)
                .build();
    }

    @Override
    public void addActivationListener(ILayerActivationListener listener) {
        listeners.add(listener);
    }

    // Abstract methods for subclasses to implement
    @Override
    public abstract Pattern processBottomUp(Pattern input, ILayerParameters parameters);

    @Override
    public abstract Pattern processTopDown(Pattern feedback, ILayerParameters parameters);

    @Override
    public abstract Pattern processLateral(Pattern lateral, ILayerParameters parameters);

    // Protected helper methods
    protected void initializeWeights() {
        weights.randomize(0.0, 0.1);
    }

    protected void notifyActivationChange(Pattern oldActivation, Pattern newActivation) {
        if (!listeners.isEmpty()) {
            var timestamp = System.currentTimeMillis();
            for (var listener : listeners) {
                try {
                    listener.onActivationChange(id, oldActivation, newActivation, timestamp);
                } catch (Exception e) {
                    // Log and continue - don't let listener errors break processing
                    System.err.println("Error in activation listener: " + e.getMessage());
                }
            }
        }
    }

    protected void checkThresholdAndSaturation(ILayerParameters parameters) {
        var maxActivation = calculateMax(activation);

        if (maxActivation >= parameters.getActivationThreshold()) {
            for (var listener : listeners) {
                try {
                    listener.onThresholdReached(id, activation);
                } catch (Exception e) {
                    System.err.println("Error in threshold listener: " + e.getMessage());
                }
            }
        }

        if (maxActivation >= parameters.getSaturationLevel()) {
            for (var listener : listeners) {
                try {
                    listener.onSaturation(id, maxActivation);
                } catch (Exception e) {
                    System.err.println("Error in saturation listener: " + e.getMessage());
                }
            }
        }
    }

    protected double applyNoise(ILayerParameters parameters) {
        return switch (parameters.getNoiseType()) {
            case GAUSSIAN -> parameters.getNoiseLevel() *
                            (Math.random() * 2.0 - 1.0) * Math.sqrt(-2 * Math.log(Math.random()));
            case UNIFORM -> parameters.getNoiseLevel() * (Math.random() * 2.0 - 1.0);
            case NONE -> 0.0;
        };
    }

    protected Pattern normalizeActivation(Pattern activation, ILayerParameters.NormalizationType type) {
        return switch (type) {
            case L1 -> normalizeL1(activation);
            case L2 -> normalizeL2(activation);
            case MAX -> normalizeMax(activation);
            case NONE -> activation;
        };
    }

    private Pattern normalizeL1(Pattern pattern) {
        var norm = pattern.l1Norm();
        return norm > 0 ? pattern.scale(1.0 / norm) : pattern;
    }

    private Pattern normalizeL2(Pattern pattern) {
        var norm = pattern.l2Norm();
        return norm > 0 ? pattern.scale(1.0 / norm) : pattern;
    }

    private Pattern normalizeMax(Pattern pattern) {
        var max = calculateMax(pattern);
        return max > 0 ? pattern.scale(1.0 / max) : pattern;
    }

    // Statistics helper methods
    protected double calculateMean(double[] values) {
        var sum = 0.0;
        for (var value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    protected double calculateVariance(double[] values, double mean) {
        var sum = 0.0;
        for (var value : values) {
            var diff = value - mean;
            sum += diff * diff;
        }
        return sum / values.length;
    }

    protected double calculateMax(double[] values) {
        var max = Double.NEGATIVE_INFINITY;
        for (var value : values) {
            max = Math.max(max, value);
        }
        return max;
    }

    protected double calculateMax(Pattern pattern) {
        var max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < pattern.dimension(); i++) {
            max = Math.max(max, pattern.get(i));
        }
        return max;
    }

    protected double calculateMin(double[] values) {
        var min = Double.POSITIVE_INFINITY;
        for (var value : values) {
            min = Math.min(min, value);
        }
        return min;
    }

    protected int countActiveNeurons(double[] values, double threshold) {
        var count = 0;
        for (var value : values) {
            if (value > threshold) {
                count++;
            }
        }
        return count;
    }
}