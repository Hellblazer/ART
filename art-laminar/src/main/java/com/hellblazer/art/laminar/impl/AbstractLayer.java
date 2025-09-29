package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.Layer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.core.WeightMatrix;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.temporal.dynamics.ShuntingDynamicsImpl;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;
import com.hellblazer.art.temporal.core.ActivationState;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Abstract layer implementation using ShuntingDynamicsImpl.
 * Reduces code from 200+ lines to ~130 lines through delegation.
 *
 * @author Hal Hildebrand
 */
public abstract class AbstractLayer implements Layer {

    protected final String id;
    protected final int size;
    protected final LayerType type;
    protected WeightMatrix weights;
    protected Pattern activation;
    protected final List<LayerActivationListener> listeners;

    // Delegate to ShuntingDynamicsImpl instead of manual implementation
    private final ShuntingDynamicsImpl shuntingDynamics;

    public AbstractLayer(String id, int size, LayerType type) {
        this.id = id;
        this.size = size;
        this.type = type;
        this.weights = new WeightMatrix(size, size);
        this.activation = new DenseVector(new double[size]);
        this.listeners = new CopyOnWriteArrayList<>();

        // Initialize shunting dynamics with default parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.5)
            .timeStep(0.01)
            .build();
        this.shuntingDynamics = new ShuntingDynamicsImpl(shuntingParams, size);
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public LayerType getType() {
        return type;
    }

    @Override
    public Pattern getActivation() {
        return activation;
    }

    @Override
    public void setActivation(Pattern activation) {
        var oldActivation = this.activation;
        this.activation = activation;

        // Update shunting dynamics state
        var activationArray = new double[size];
        for (int i = 0; i < Math.min(activation.dimension(), size); i++) {
            activationArray[i] = activation.get(i);
        }
        shuntingDynamics.setState(new ActivationState(activationArray));

        // Notify listeners
        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }
    }

    @Override
    public WeightMatrix getWeights() {
        return weights;
    }

    @Override
    public void setWeights(WeightMatrix weights) {
        this.weights = weights;
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        // Convert input to activation state
        var inputArray = new double[size];
        for (int i = 0; i < Math.min(input.dimension(), size); i++) {
            inputArray[i] = input.get(i);
        }

        // Set external input for shunting dynamics
        shuntingDynamics.setExcitatoryInput(inputArray);

        // Evolve dynamics using ShuntingDynamicsImpl
        var currentState = new ActivationState(inputArray);
        var deltaT = 0.01; // Default time step
        var evolvedState = shuntingDynamics.evolve(currentState, deltaT);

        // Convert back to Pattern
        var result = evolvedState.getActivations();
        activation = new DenseVector(result);
        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Top-down processing applies modulation
        var result = new double[size];
        var modulationStrength = 0.5; // Default modulation

        for (int i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = i < activation.dimension() ? activation.get(i) : 0.0;

            // Modulate current activation by expectation
            result[i] = current * (1.0 + modulationStrength * expect);
        }

        return new DenseVector(result);
    }

    @Override
    public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
        // Lateral processing already handled by shunting dynamics
        // This just applies additional lateral modulation if needed
        return lateral;
    }

    @Override
    public void updateWeights(Pattern input, double learningRate) {
        // Simple Hebbian learning for weights
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                if (j < input.dimension()) {
                    var currentWeight = weights.get(i, j);
                    var inputValue = input.get(j);
                    var activationValue = activation.get(i);

                    // Hebbian update: Δw = η * x * y
                    var deltaW = learningRate * inputValue * activationValue;
                    weights.set(i, j, currentWeight + deltaW);
                }
            }
        }
    }

    @Override
    public void reset() {
        activation = new DenseVector(new double[size]);
        shuntingDynamics.reset();
    }

    @Override
    public void addActivationListener(LayerActivationListener listener) {
        listeners.add(listener);
    }
}