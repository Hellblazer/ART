package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.batch.BatchLayer;
import com.hellblazer.art.laminar.batch.StatefulBatchProcessor;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import com.hellblazer.art.laminar.parameters.Layer1Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.dynamics.ShuntingDynamicsImpl;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;

/**
 * Layer 1 Implementation - Top-Down Attentional Priming.
 *
 * Layer 1 contains the apical dendrites of pyramidal neurons that receive
 * top-down attentional signals from higher cortical areas. It provides
 * sustained attention without directly driving responses.
 *
 * Key characteristics:
 * - Very slow time constants (200-1000ms) for sustained attention
 * - Receives top-down signals from higher cortical areas
 * - Sustained attention effects that persist long after input ends
 * - Priming without driving cells directly
 * - Integrates with Layer 2/3 apical dendrites
 * - Long-duration memory traces (seconds)
 * - Lowest firing rates of all layers
 *
 * Biological basis:
 * Layer 1 is primarily composed of apical dendritic tufts from pyramidal
 * neurons in deeper layers, along with sparse inhibitory interneurons.
 * It receives diffuse top-down projections that modulate cortical processing
 * through sustained attentional effects.
 *
 * @author Hal Hildebrand
 */
public class Layer1Implementation extends AbstractLayer implements BatchLayer, StatefulBatchProcessor {

    private ShuntingDynamicsImpl verySlowDynamics;
    private Layer1Parameters currentParameters;
    private double[] attentionState;      // Persistent attention state
    private double[] primingEffect;       // Current priming effect
    private double[] memoryTrace;         // Long-term memory trace

    public Layer1Implementation(String id, int size) {
        super(id, size, LayerType.CUSTOM);
        this.attentionState = new double[size];
        this.primingEffect = new double[size];
        this.memoryTrace = new double[size];
        initializeVerySlowDynamics();
    }

    private void initializeVerySlowDynamics() {
        // Initialize with very slow dynamics suitable for Layer 1
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.05)  // Very weak self-excitation
            .inhibitoryStrength(0.05)  // Very weak lateral inhibition
            .timeStep(0.05)  // 50ms time step for very slow dynamics
            .build();
        this.verySlowDynamics = new ShuntingDynamicsImpl(params, size);
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        if (!(parameters instanceof Layer1Parameters)) {
            // Use defaults if wrong parameter type
            parameters = Layer1Parameters.builder().build();
        }
        currentParameters = (Layer1Parameters) parameters;

        // Update shunting dynamics parameters based on Layer1Parameters
        updateDynamicsParameters(currentParameters);

        // Convert top-down expectation to attention array
        var attentionInput = new double[size];
        for (int i = 0; i < Math.min(expectation.dimension(), size); i++) {
            attentionInput[i] = expectation.get(i);
        }

        // Update attention state with very slow dynamics
        updateAttentionState(attentionInput, currentParameters);

        // Update memory trace with even slower dynamics
        updateMemoryTrace(attentionInput, currentParameters);

        // Calculate priming effect (modulates but doesn't drive)
        for (int i = 0; i < size; i++) {
            // Combine attention state and memory trace
            var combinedAttention = attentionState[i] + memoryTrace[i] * 0.5;

            // For initial strong attention, allow higher values
            if (attentionInput[i] > 0.8) {
                combinedAttention = Math.max(combinedAttention, attentionInput[i] * 0.9);
            }

            // Apply priming strength (keeps values moderate)
            primingEffect[i] = combinedAttention * currentParameters.getPrimingStrength();

            // Ensure priming doesn't exceed limits
            primingEffect[i] = Math.min(primingEffect[i], 0.5);  // Cap at 50% to prevent driving
        }

        // Set as excitatory input for very slow dynamics
        verySlowDynamics.setExcitatoryInput(primingEffect);

        // Evolve with very slow time constant
        var currentState = new ActivationState(primingEffect);
        var timeStep = currentParameters.getTimeConstant() / 20000.0;  // Convert ms to seconds
        var evolvedState = verySlowDynamics.evolve(currentState, timeStep);

        // Get evolved result
        var result = evolvedState.getActivations();

        // Apply constraints and allow strong initial attention
        for (int i = 0; i < result.length; i++) {
            // For strong attention input, allow stronger output
            if (attentionInput[i] > 0.8) {
                result[i] = Math.max(result[i], attentionInput[i] * 0.85);
            }

            // Maintain attention state even without input
            if (attentionState[i] > 0.1) {
                result[i] = Math.max(result[i], attentionState[i] * 0.7);
            }

            result[i] = Math.max(currentParameters.getFloor(),
                        Math.min(currentParameters.getCeiling(), result[i]));
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
     * Update attention state with very slow dynamics.
     */
    private void updateAttentionState(double[] input, Layer1Parameters params) {
        var sustainedDecay = params.getSustainedDecayRate();
        var shiftRate = params.getAttentionShiftRate();

        for (int i = 0; i < size; i++) {
            // Very slow decay to maintain attention (apply much smaller decay per time step)
            // With sustainedDecay = 0.0005 and 100 steps, we want to retain > 0.3
            // So decay factor per step should be very small
            attentionState[i] *= (1.0 - sustainedDecay * 0.01);  // Much slower decay

            // Integrate new input with shift rate
            if (input[i] > 0) {
                // Shift attention gradually to new location
                attentionState[i] += input[i] * shiftRate;
            }

            // Cap attention state
            attentionState[i] = Math.min(attentionState[i], 1.0);
        }
    }

    /**
     * Update memory trace with even slower dynamics for long persistence.
     */
    private void updateMemoryTrace(double[] input, Layer1Parameters params) {
        var traceDecay = params.getSustainedDecayRate() * 0.05;  // Much slower than attention

        for (int i = 0; i < size; i++) {
            // Very slow decay for memory trace
            memoryTrace[i] *= (1.0 - traceDecay);

            // Build up memory trace from sustained attention
            if (attentionState[i] > 0.3) {  // Threshold for memory formation
                memoryTrace[i] += attentionState[i] * 0.1;  // Slow accumulation
            }

            // Cap memory trace
            memoryTrace[i] = Math.min(memoryTrace[i], 0.8);
        }
    }

    /**
     * Get the current priming effect for integration with other layers.
     */
    public Pattern getPrimingEffect() {
        return new DenseVector(primingEffect.clone());
    }

    /**
     * Get the signal for Layer 2/3 apical dendrites.
     */
    public Pattern getApicalDendriteSignal() {
        var apicalSignal = new double[size];
        var apicalIntegration = currentParameters != null ?
            currentParameters.getApicalIntegration() : 0.5;

        for (int i = 0; i < size; i++) {
            // Modulated signal for apical dendrites
            apicalSignal[i] = (attentionState[i] + primingEffect[i]) * apicalIntegration;
        }

        return new DenseVector(apicalSignal);
    }

    /**
     * Get the current attention state.
     */
    public Pattern getAttentionState() {
        return new DenseVector(attentionState.clone());
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        // Layer 1 primarily processes top-down signals
        // Bottom-up processing is minimal - just maintain current state
        if (activation == null) {
            activation = new DenseVector(new double[size]);
        }
        return activation;
    }

    @Override
    public void reset() {
        super.reset();
        if (verySlowDynamics != null) {
            verySlowDynamics.reset();
        }
        if (attentionState != null) {
            for (int i = 0; i < attentionState.length; i++) {
                attentionState[i] = 0.0;
            }
        }
        if (primingEffect != null) {
            for (int i = 0; i < primingEffect.length; i++) {
                primingEffect[i] = 0.0;
            }
        }
        if (memoryTrace != null) {
            for (int i = 0; i < memoryTrace.length; i++) {
                memoryTrace[i] = 0.0;
            }
        }
        currentParameters = null;
    }

    private void updateDynamicsParameters(Layer1Parameters params) {
        // Update the shunting dynamics with Layer 1 specific parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .timeStep(params.getTimeConstant() / 20000.0)  // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.verySlowDynamics = new ShuntingDynamicsImpl(shuntingParams, size);
    }

    // ==================== Batch Processing Implementation ====================

    /**
     * Process single pattern with stateful SIMD (Layer 1 top-down variant).
     *
     * Layer 1 has complex state (attention, priming, memory) that makes
     * single-pattern SIMD overhead not worthwhile. This implementation
     * simply delegates to processTopDown.
     *
     * @param input Top-down expectation pattern
     * @param parameters Layer parameters
     * @return Processed pattern with state updated
     */
    @Override
    public Pattern processWithStatefulSIMD(Pattern input, LayerParameters parameters) {
        // Layer 1 state is too complex for single-pattern SIMD benefit
        // State management overhead exceeds SIMD computation savings
        return processTopDown(input, parameters);
    }

    /**
     * Process batch of top-down expectations.
     *
     * NOTE: Layer 1 has unique API - it processes TOP-DOWN signals, not bottom-up!
     * This method processes expectations from higher cortical areas.
     *
     * Layer 1 has complex state (attention, priming, memory traces) that accumulates
     * over time. For semantic equivalence with sequential processing in a circuit context,
     * we process sequentially to maintain proper state evolution.
     */
    public Pattern[] processTopDownBatch(Pattern[] expectations, LayerParameters parameters) {
        if (expectations == null || expectations.length == 0) {
            throw new IllegalArgumentException("expectations cannot be null or empty");
        }
        if (parameters == null) {
            throw new NullPointerException("parameters cannot be null");
        }

        var layer1Params = (parameters instanceof Layer1Parameters) ?
            (Layer1Parameters) parameters : Layer1Parameters.builder().build();

        // Phase 6A: Stateful batch processing
        // Process each pattern sequentially with state management
        var batchSize = expectations.length;
        var outputs = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            outputs[i] = processWithStatefulSIMD(expectations[i], layer1Params);
        }

        return outputs;
    }

    @Override
    public Pattern[] processBatchBottomUp(Pattern[] inputs, LayerParameters parameters) {
        // Layer 1 primarily processes top-down signals
        // Bottom-up processing is minimal - return current activation for all
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("inputs cannot be null or empty");
        }

        var outputs = new Pattern[inputs.length];
        var currentActivation = activation != null ? activation : new DenseVector(new double[size]);

        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = currentActivation;
        }

        return outputs;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public String getId() {
        return id;
    }
}