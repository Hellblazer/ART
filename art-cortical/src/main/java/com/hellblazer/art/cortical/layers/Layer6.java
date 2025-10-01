package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Layer 6 Implementation - Corticothalamic Feedback & Attentional Modulation.
 *
 * <p>CRITICAL: Implements ART matching rule - modulatory only!
 * Layer 6 provides modulatory feedback to Layer 4 and thalamus.
 * It CANNOT fire cells alone - requires coincidence of bottom-up
 * and top-down signals (ART matching rule).
 *
 * <p>Key characteristics:
 * <ul>
 *   <li>Slow time constants (100-500ms) for sustained modulation</li>
 *   <li>Modulatory only - cannot drive cells without bottom-up input</li>
 *   <li>On-center, off-surround dynamics for selective attention</li>
 *   <li>Top-down expectation generation</li>
 *   <li>Attentional gain control</li>
 *   <li>Lower firing rates than other layers</li>
 * </ul>
 *
 * <p>Biological references:
 * <ul>
 *   <li>Sherman, S. M., & Guillery, R. W. (1998). On the actions that one nerve cell
 *       can have on another: Distinguishing "drivers" from "modulators". PNAS 95(12), 7121-7126.</li>
 *   <li>Grossberg, S. (1980). How does a brain build a cognitive code? Psych Review 87(1): 1-51.</li>
 *   <li>Grossberg, S. (2013). Adaptive Resonance Theory. Scholarpedia 8(1): 1569.</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3)
 */
public final class Layer6 implements Layer {

    private final String id;
    private final int size;
    private final WeightMatrix weights;
    private final List<LayerActivationListener> listeners;

    private Pattern activation;
    private Pattern topDownExpectation;
    private double[] modulationState;  // Persistent modulation state
    private Layer6Parameters currentParameters;

    /**
     * Create Layer 6 with given ID and size.
     *
     * @param id layer identifier (typically "L6")
     * @param size number of units/columns in this layer
     * @throws IllegalArgumentException if size <= 0
     */
    public Layer6(String id, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive: " + size);
        }

        this.id = id;
        this.size = size;
        this.weights = new WeightMatrix(size, size);
        this.listeners = new CopyOnWriteArrayList<>();
        this.activation = new DenseVector(new double[size]);
        this.topDownExpectation = new DenseVector(new double[size]);
        this.modulationState = new double[size];
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
        return LayerType.LAYER_6;
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

    /**
     * Set top-down expectation signal from higher areas.
     *
     * @param expectation top-down expectation pattern
     */
    public void setTopDownExpectation(Pattern expectation) {
        this.topDownExpectation = expectation;
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        if (!(parameters instanceof Layer6Parameters)) {
            // Use defaults if wrong parameter type
            parameters = Layer6Parameters.builder().build();
        }
        currentParameters = (Layer6Parameters) parameters;

        // Convert input to array
        var bottomUpArray = new double[size];
        for (var i = 0; i < Math.min(input.dimension(), size); i++) {
            bottomUpArray[i] = input.get(i);
        }

        // Get top-down expectation
        var topDownArray = new double[size];
        for (var i = 0; i < Math.min(topDownExpectation.dimension(), size); i++) {
            topDownArray[i] = topDownExpectation.get(i);
        }

        // CRITICAL: Implement ART matching rule
        // Layer 6 output = bottom-up * (1 + modulation from top-down)
        // If no bottom-up, output MUST be zero (modulatory only!)
        var result = new double[size];

        for (var i = 0; i < size; i++) {
            if (bottomUpArray[i] <= currentParameters.floor()) {
                // NO BOTTOM-UP = NO OUTPUT (CRITICAL!)
                result[i] = 0.0;
            } else {
                // Calculate on-center, off-surround modulation
                var modulation = calculateModulation(i, topDownArray, currentParameters);

                // Apply modulation ONLY when bottom-up is present
                if (modulation > currentParameters.modulationThreshold()) {
                    // Enhanced activation when both signals present (ART matching)
                    result[i] = bottomUpArray[i] * (1.0 + modulation * currentParameters.attentionalGain());
                } else {
                    // Pass through bottom-up with minimal processing
                    result[i] = bottomUpArray[i];
                }

                // Clamp to ceiling
                result[i] = Math.min(currentParameters.ceiling(), result[i]);
            }
        }

        // Update modulation state with slow dynamics
        updateModulationState(topDownArray, currentParameters);

        // Update activation and notify listeners
        var oldActivation = this.activation;
        activation = new DenseVector(result);

        // Notify listeners
        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }

        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Layer 6 uses top-down to set its expectation state
        setTopDownExpectation(expectation);

        // Return current activation modulated by new expectation
        if (activation == null) {
            return new DenseVector(new double[size]);
        }

        var result = new double[size];
        for (var i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = activation.get(i);

            // Layer 6 integrates top-down into its modulation state
            result[i] = current * (1.0 + 0.2 * expect);  // 20% modulation

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
        // Layer 6 has minimal lateral processing
        // On-center, off-surround handled by modulation calculation
        return activation;
    }

    @Override
    public void updateWeights(Pattern input, double learningRate) {
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate must be in [0, 1]: " + learningRate);
        }

        // Hebbian learning with outstar rule (Grossberg 1976)
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
        topDownExpectation = new DenseVector(new double[size]);
        for (var i = 0; i < modulationState.length; i++) {
            modulationState[i] = 0.0;
        }
        currentParameters = null;
    }

    @Override
    public void close() {
        reset();
    }

    /**
     * Calculate on-center, off-surround modulation for a given position.
     *
     * @param centerIndex center position for modulation
     * @param topDown top-down expectation array
     * @param params Layer 6 parameters
     * @return modulation value (rectified, non-negative)
     */
    private double calculateModulation(int centerIndex, double[] topDown, Layer6Parameters params) {
        var onCenter = topDown[centerIndex] * params.onCenterWeight();

        // Calculate off-surround inhibition
        var offSurround = 0.0;
        var surroundSize = 2;  // Look at 2 neighbors on each side

        for (var offset = 1; offset <= surroundSize; offset++) {
            // Left neighbor
            if (centerIndex - offset >= 0) {
                offSurround += topDown[centerIndex - offset];
            }
            // Right neighbor
            if (centerIndex + offset < size) {
                offSurround += topDown[centerIndex + offset];
            }
        }

        // Apply off-surround inhibition
        var modulation = onCenter - (offSurround * params.offSurroundStrength());

        // Include persistent modulation state
        modulation += modulationState[centerIndex] * 0.5;  // 50% contribution from state

        return Math.max(0.0, modulation);  // Rectify
    }

    /**
     * Update persistent modulation state with slow dynamics.
     *
     * @param topDown top-down expectation array
     * @param params Layer 6 parameters
     */
    private void updateModulationState(double[] topDown, Layer6Parameters params) {
        var decayRate = params.decayRate();
        var decayFactor = 1.0 - decayRate * 0.02;
        var integrationFactor = decayRate * 0.02;

        for (var i = 0; i < size; i++) {
            // Leaky integration: state = state * decay + topDown * integration
            modulationState[i] = modulationState[i] * decayFactor + topDown[i] * integrationFactor;
        }
    }

    /**
     * Generate feedback signal to Layer 4.
     * This is the modulatory feedback that implements attention.
     *
     * @param layer6Output current Layer 6 output
     * @param parameters layer parameters (unused)
     * @return feedback pattern for Layer 4
     */
    public Pattern generateFeedbackToLayer4(Pattern layer6Output, LayerParameters parameters) {
        var feedback = new double[size];

        for (var i = 0; i < Math.min(layer6Output.dimension(), size); i++) {
            // Feedback is a weighted version of Layer 6 output
            feedback[i] = layer6Output.get(i) * 0.5;  // 50% strength feedback
        }

        return new DenseVector(feedback);
    }

    /**
     * Get weight matrix for this layer.
     *
     * @return weight matrix
     */
    public WeightMatrix getWeights() {
        return weights;
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
}
