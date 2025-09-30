package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import com.hellblazer.art.laminar.parameters.Layer6Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.laminar.performance.VectorizedArrayOperations;
import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.dynamics.ShuntingDynamicsImpl;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;

/**
 * Layer 6 Implementation - Feedback Modulation.
 *
 * CRITICAL: Implements ART matching rule - modulatory only!
 * Layer 6 provides modulatory feedback to Layer 4 and thalamus.
 * It CANNOT fire cells alone - requires coincidence of bottom-up
 * and top-down signals (ART matching rule).
 *
 * Key characteristics:
 * - Slow time constants (100-500ms) for sustained modulation
 * - Modulatory only - cannot drive cells without bottom-up input
 * - On-center, off-surround dynamics for selective attention
 * - Top-down expectation generation
 * - Attentional gain control
 * - Lower firing rates than other layers
 *
 * Biological basis:
 * Layer 6 corticothalamic neurons provide feedback to thalamus and Layer 4.
 * They implement the ART matching rule by requiring both bottom-up sensory
 * input and top-down expectations to generate output. This prevents
 * hallucinations and ensures stable learning.
 *
 * @author Hal Hildebrand
 */
public class Layer6Implementation extends AbstractLayer {

    private ShuntingDynamicsImpl slowDynamics;
    private Layer6Parameters currentParameters;
    private Pattern topDownExpectation;  // Top-down expectation signal
    private double[] modulationState;    // Persistent modulation state

    public Layer6Implementation(String id, int size) {
        super(id, size, LayerType.CUSTOM);
        this.topDownExpectation = new DenseVector(new double[size]);
        this.modulationState = new double[size];
        initializeSlowDynamics();
    }

    private void initializeSlowDynamics() {
        // Initialize with slow dynamics suitable for Layer 6
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.1)  // Weak self-excitation
            .inhibitoryStrength(0.3)  // Moderate lateral inhibition
            .timeStep(0.02)  // 20ms time step for slow dynamics
            .build();
        this.slowDynamics = new ShuntingDynamicsImpl(params, size);
    }

    /**
     * Set the top-down expectation signal.
     * This represents predictions from higher areas.
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

        // Update shunting dynamics parameters based on Layer6Parameters
        updateDynamicsParameters(currentParameters);

        // Convert input to activation array
        var bottomUpArray = new double[size];
        for (int i = 0; i < Math.min(input.dimension(), size); i++) {
            bottomUpArray[i] = input.get(i);
        }

        // Get top-down expectation
        var topDownArray = new double[size];
        for (int i = 0; i < Math.min(topDownExpectation.dimension(), size); i++) {
            topDownArray[i] = topDownExpectation.get(i);
        }

        // CRITICAL: Implement ART matching rule
        // Layer 6 output = bottom-up * (1 + modulation from top-down)
        // If no bottom-up, output MUST be zero (modulatory only!)
        var result = new double[size];

        for (int i = 0; i < size; i++) {
            if (bottomUpArray[i] <= currentParameters.getFloor()) {
                // NO BOTTOM-UP = NO OUTPUT (CRITICAL!)
                result[i] = 0.0;
            } else {
                // Calculate on-center, off-surround modulation
                var modulation = calculateModulation(i, topDownArray, currentParameters);

                // Apply modulation ONLY when bottom-up is present
                if (modulation > currentParameters.getModulationThreshold()) {
                    // Enhanced activation when both signals present (ART matching)
                    result[i] = bottomUpArray[i] * (1.0 + modulation * currentParameters.getAttentionalGain());
                } else {
                    // Pass through bottom-up with minimal processing
                    result[i] = bottomUpArray[i];
                }
            }
        }

        // Update modulation state with slow dynamics
        updateModulationState(topDownArray, currentParameters);

        // Set as excitatory input for slow dynamics
        slowDynamics.setExcitatoryInput(result);

        // Evolve with slow time constant
        var currentState = new ActivationState(result);
        var timeStep = currentParameters.getTimeConstant() / 5000.0;  // Convert ms to seconds
        var evolvedState = slowDynamics.evolve(currentState, timeStep);

        // Get evolved result
        result = evolvedState.getActivations();

        // Apply final constraints (vectorized where possible)
        var ceiling = currentParameters.getCeiling();
        var floor = currentParameters.getFloor();

        // First apply general clamp (vectorized)
        VectorizedArrayOperations.clampInPlace(result, floor, ceiling);

        // Then ensure modulatory behavior (critical - must check bottom-up)
        for (int i = 0; i < result.length; i++) {
            if (bottomUpArray[i] <= floor) {
                result[i] = 0.0;  // No bottom-up = no output!
            }
        }

        // Update activation and notify listeners
        activation = new DenseVector(result);

        // Notify listeners about activation change
        var oldActivation = getActivation();
        for (var listener : listeners) {
            listener.onActivationChanged(getId(), oldActivation, activation);
        }

        return activation;
    }

    /**
     * Calculate on-center, off-surround modulation.
     */
    private double calculateModulation(int centerIndex, double[] topDown, Layer6Parameters params) {
        var onCenter = topDown[centerIndex] * params.getOnCenterWeight();

        // Calculate off-surround inhibition
        var offSurround = 0.0;
        var surroundSize = 2;  // Look at 2 neighbors on each side

        for (int offset = 1; offset <= surroundSize; offset++) {
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
        var modulation = onCenter - (offSurround * params.getOffSurroundStrength());

        // Include persistent modulation state
        modulation += modulationState[centerIndex] * 0.5;  // 50% contribution from state

        return Math.max(0.0, modulation);  // Rectify
    }

    /**
     * Update persistent modulation state (vectorized).
     */
    private void updateModulationState(double[] topDown, Layer6Parameters params) {
        var decayRate = params.getDecayRate();
        var decayFactor = 1.0 - decayRate * 0.02;
        var integrationFactor = decayRate * 0.02;

        // Vectorized: modulationState = modulationState * decayFactor + topDown * integrationFactor
        var decayed = VectorizedArrayOperations.scale(modulationState, decayFactor);
        var integrated = VectorizedArrayOperations.scale(topDown, integrationFactor);
        modulationState = VectorizedArrayOperations.add(decayed, integrated);
    }

    /**
     * Generate feedback signal to Layer 4.
     * This is the modulatory feedback that implements attention.
     */
    public Pattern generateFeedbackToLayer4(Pattern layer6Output, LayerParameters parameters) {
        var feedback = new double[size];

        for (int i = 0; i < Math.min(layer6Output.dimension(), size); i++) {
            // Feedback is a weighted version of Layer 6 output
            feedback[i] = layer6Output.get(i) * 0.5;  // 50% strength feedback
        }

        return new DenseVector(feedback);
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
        for (int i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = activation.get(i);

            // Layer 6 integrates top-down into its modulation state
            result[i] = current * (1.0 + 0.2 * expect);  // 20% modulation

            // Ensure bounds
            if (currentParameters != null) {
                result[i] = Math.max(currentParameters.getFloor(),
                            Math.min(currentParameters.getCeiling(), result[i]));
            }
        }

        return new DenseVector(result);
    }

    @Override
    public void reset() {
        super.reset();
        if (slowDynamics != null) {
            slowDynamics.reset();
        }
        if (modulationState != null) {
            for (int i = 0; i < modulationState.length; i++) {
                modulationState[i] = 0.0;
            }
        }
        topDownExpectation = new DenseVector(new double[size]);
        currentParameters = null;
    }

    private void updateDynamicsParameters(Layer6Parameters params) {
        // Update the shunting dynamics with Layer 6 specific parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .timeStep(params.getTimeConstant() / 5000.0)  // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.slowDynamics = new ShuntingDynamicsImpl(shuntingParams, size);
    }
}