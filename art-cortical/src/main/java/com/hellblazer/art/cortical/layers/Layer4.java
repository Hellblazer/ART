package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Layer 4 Implementation - Thalamic Driving Input Layer.
 *
 * <p>Layer 4 is the primary recipient of driving input from the thalamus (LGN).
 * It initiates cortical processing with strong, fast dynamics that can directly
 * fire cells without requiring modulatory input.
 *
 * <p>Key characteristics:
 * <ul>
 *   <li>Fast time constants (10-50ms) for rapid response</li>
 *   <li>Strong driving signals that can fire cells independently</li>
 *   <li>Simple feedforward processing</li>
 *   <li>Minimal lateral inhibition in basic circuits</li>
 *   <li>Direct transformation of thalamic input to cortical representation</li>
 * </ul>
 *
 * <p>Biological references:
 * <ul>
 *   <li>Douglas, R. J., & Martin, K. A. (2004). Neuronal circuits of the neocortex.
 *       Annual Review of Neuroscience, 27, 419-451.</li>
 *   <li>Sherman, S. M., & Guillery, R. W. (1998). On the actions that one nerve cell
 *       can have on another: Distinguishing "drivers" from "modulators".
 *       Proceedings of the National Academy of Sciences, 95(12), 7121-7126.</li>
 *   <li>Grossberg, S. (1973). Contour enhancement, short term memory, and
 *       constancies in reverberating neural networks. Studies in Applied Mathematics.</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 3)
 */
public final class Layer4 implements Layer {

    private final String id;
    private final int size;
    private final WeightMatrix weights;
    private final List<LayerActivationListener> listeners;

    private ShuntingDynamics dynamics;
    private Pattern activation;
    private Layer4Parameters currentParameters;

    /**
     * Create Layer 4 with given ID and size.
     *
     * @param id layer identifier (typically "L4")
     * @param size number of units/columns in this layer
     * @throws IllegalArgumentException if size <= 0
     */
    public Layer4(String id, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive: " + size);
        }

        this.id = id;
        this.size = size;
        this.weights = new WeightMatrix(size, size);
        this.listeners = new CopyOnWriteArrayList<>();
        this.activation = new DenseVector(new double[size]);

        // Initialize with default fast dynamics
        initializeFastDynamics();
    }

    /**
     * Initialize shunting dynamics with fast time constants for Layer 4.
     */
    private void initializeFastDynamics() {
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(0.0) // No lateral inhibition initially
            .timeStep(0.001) // 1ms time step for fast dynamics
            .build();
        this.dynamics = new ShuntingDynamics(params);
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
        return LayerType.LAYER_4;
    }

    @Override
    public Pattern getActivation() {
        return activation;
    }

    @Override
    public void setActivation(Pattern activation) {
        if (activation.dimension() != size) {
            throw new IllegalArgumentException(
                "Activation dimension " + activation.dimension() +
                " does not match layer size " + size);
        }

        var oldActivation = this.activation;
        this.activation = activation;

        // Notify listeners
        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        if (!(parameters instanceof Layer4Parameters)) {
            // Use defaults if wrong parameter type
            parameters = Layer4Parameters.builder().build();
        }
        currentParameters = (Layer4Parameters) parameters;

        // Update shunting dynamics parameters based on Layer4Parameters
        updateDynamicsParameters(currentParameters);

        // Convert input to activation array and apply driving strength
        var inputArray = new double[size];
        for (var i = 0; i < Math.min(input.dimension(), size); i++) {
            inputArray[i] = input.get(i);
        }

        // Apply driving strength to input (Layer 4 receives strong thalamic drive)
        // Inline scaleInPlace for performance (Spartan design - no dependency)
        var drivingStrength = currentParameters.drivingStrength();
        for (var i = 0; i < inputArray.length; i++) {
            inputArray[i] *= drivingStrength;
        }

        // For Layer 4, driving input directly activates neurons (Sherman & Guillery 1998)
        // Layer 4 has "simple feedforward processing" and "minimal lateral inhibition"
        // Apply sigmoid saturation to transform unbounded input to bounded activation
        var ceiling = currentParameters.ceiling();
        var floor = currentParameters.floor();

        // Apply soft sigmoid saturation for biological plausibility
        // Maps unbounded scaled input to [0, ceiling] range
        // Based on Grossberg (1973) sigmoid saturation function
        var result = new double[size];
        for (var i = 0; i < inputArray.length; i++) {
            if (inputArray[i] > 0) {
                // Sigmoid: ceiling * x / (1 + x) for x > 0
                result[i] = ceiling * inputArray[i] / (1.0 + inputArray[i]);
            } else {
                result[i] = 0.0;
            }
        }

        // Apply lateral inhibition only if configured (minimal for Layer 4)
        if (currentParameters.lateralInhibition() > 0.01) {
            // Use shunting dynamics for lateral competition
            dynamics.reset();
            dynamics.setExcitatoryInput(result);
            var timeStep = Math.min(currentParameters.timeConstant() / 1000.0, 0.01);
            result = dynamics.update(timeStep);
        }
        // Otherwise: pure feedforward (no lateral dynamics)

        // Inline clampInPlace for performance (Spartan design)
        for (var i = 0; i < result.length; i++) {
            if (result[i] < floor) {
                result[i] = floor;
            } else if (result[i] > ceiling) {
                result[i] = ceiling;
            }
        }

        // Update activation and return
        activation = new DenseVector(result);
        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Layer 4 receives minimal top-down modulation
        // It's primarily driven by bottom-up thalamic input
        // Based on Sherman & Guillery (1998) - driving vs. modulatory distinction

        if (activation == null) {
            return new DenseVector(new double[size]);
        }

        // Apply weak modulation (Layer 4 is less affected by top-down)
        var result = new double[size];
        var modulationStrength = 0.1; // Weak modulation for Layer 4

        for (var i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = activation.get(i);

            // Minimal modulation - Layer 4 maintains driving characteristics
            result[i] = current * (1.0 + modulationStrength * expect);

            // Ensure bounds
            if (currentParameters != null) {
                result[i] = Math.max(currentParameters.floor(),
                            Math.min(currentParameters.ceiling(), result[i]));
            }
        }

        return new DenseVector(result);
    }

    @Override
    public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
        // Lateral processing already handled by shunting dynamics
        // This provides additional lateral modulation if needed
        // For Layer 4, lateral effects are minimal (primarily feedforward)

        // Return current activation (lateral handled by dynamics)
        return activation;
    }

    @Override
    public void updateWeights(Pattern input, double learningRate) {
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate must be in [0, 1]: " + learningRate);
        }

        // Simple Hebbian learning for weights
        // Based on Grossberg (1976) outstar learning rule
        for (var i = 0; i < weights.getRows(); i++) {
            for (var j = 0; j < weights.getCols(); j++) {
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
        dynamics.reset();
        currentParameters = null;
    }

    @Override
    public void close() {
        // Clean up resources if needed
        reset();
    }

    /**
     * Get weight matrix for this layer.
     * Primarily used for learning and testing.
     *
     * @return weight matrix
     */
    public WeightMatrix getWeights() {
        return weights;
    }

    /**
     * Set weight matrix for this layer.
     * Used for initialization or loading trained weights.
     *
     * @param weights new weight matrix
     * @throws IllegalArgumentException if dimensions don't match
     */
    public void setWeights(WeightMatrix weights) {
        if (weights.getRows() != size || weights.getCols() != size) {
            throw new IllegalArgumentException(
                "Weight matrix dimensions (" + weights.getRows() + "x" + weights.getCols() +
                ") don't match layer size " + size);
        }
        // Copy weights to avoid external mutation
        for (var i = 0; i < size; i++) {
            for (var j = 0; j < size; j++) {
                this.weights.set(i, j, weights.get(i, j));
            }
        }
    }

    /**
     * Add activation listener for monitoring layer dynamics.
     *
     * @param listener activation change listener
     */
    public void addActivationListener(LayerActivationListener listener) {
        if (listener != null) {
            listeners.add(listener);
        }
    }

    /**
     * Remove activation listener.
     *
     * @param listener listener to remove
     */
    public void removeActivationListener(LayerActivationListener listener) {
        listeners.remove(listener);
    }

    /**
     * Update shunting dynamics parameters based on Layer4Parameters.
     * Creates new dynamics instance with updated parameters.
     *
     * @param params Layer 4 specific parameters
     */
    private void updateDynamicsParameters(Layer4Parameters params) {
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.ceiling())
            .floor(params.floor())
            .selfExcitation(params.selfExcitation())
            .inhibitoryStrength(params.lateralInhibition())
            .timeStep(params.timeConstant() / 1000.0) // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.dynamics = new ShuntingDynamics(shuntingParams);
    }
}
