package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.batch.BatchLayer;
import com.hellblazer.art.laminar.batch.Layer5SIMDBatch;
import com.hellblazer.art.laminar.batch.StatefulBatchProcessor;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import com.hellblazer.art.laminar.parameters.Layer5Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.performance.VectorizedArrayOperations;
import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.dynamics.ShuntingDynamicsImpl;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;

/**
 * Layer 5 Implementation - Output to Higher Areas.
 *
 * Layer 5 projects processed signals from Layer 2/3 to higher cortical areas.
 * This is the output stage of the cortical column, responsible for sending
 * categorized and amplified signals to other brain regions.
 *
 * Key characteristics:
 * - Medium time constants (50-200ms) for sustained output
 * - Receives input from Layer 2/3 pyramidal cells
 * - Amplification and gating for salient features
 * - Output normalization for stable signaling
 * - Category signal generation for classification
 * - Burst firing capability for important signals
 *
 * Biological basis:
 * Layer 5 contains large pyramidal neurons that project to subcortical
 * structures and other cortical areas. These neurons can generate burst
 * firing patterns for signaling important events.
 *
 * @author Hal Hildebrand
 */
public class Layer5Implementation extends AbstractLayer implements BatchLayer, StatefulBatchProcessor {

    private ShuntingDynamicsImpl mediumDynamics;
    private Layer5Parameters currentParameters;
    private double[] previousActivation;  // For state persistence

    public Layer5Implementation(String id, int size) {
        super(id, size, LayerType.CUSTOM);
        this.previousActivation = new double[size];
        initializeMediumDynamics();
    }

    private void initializeMediumDynamics() {
        // Initialize with medium dynamics suitable for Layer 5
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.2)  // Moderate self-excitation
            .inhibitoryStrength(0.1)  // Weak lateral inhibition
            .timeStep(0.01)  // 10ms time step
            .build();
        this.mediumDynamics = new ShuntingDynamicsImpl(params, size);
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        if (!(parameters instanceof Layer5Parameters)) {
            // Use defaults if wrong parameter type
            parameters = Layer5Parameters.builder().build();
        }
        currentParameters = (Layer5Parameters) parameters;

        // Update shunting dynamics parameters based on Layer5Parameters
        updateDynamicsParameters(currentParameters);

        // Convert input to activation array
        var inputArray = new double[size];
        for (int i = 0; i < Math.min(input.dimension(), size); i++) {
            inputArray[i] = input.get(i);
        }

        // Apply amplification gain to input (vectorized)
        var amplificationGain = currentParameters.getAmplificationGain();
        VectorizedArrayOperations.scaleInPlace(inputArray, amplificationGain);

        // Check for burst firing conditions and apply burst amplification
        var burstThreshold = currentParameters.getBurstThreshold();
        var burstAmplification = currentParameters.getBurstAmplification();

        boolean shouldBurst = false;
        for (int i = 0; i < inputArray.length; i++) {
            if (inputArray[i] > burstThreshold) {
                shouldBurst = true;
                break;
            }
        }

        // Apply burst amplification if needed
        if (shouldBurst) {
            for (int i = 0; i < inputArray.length; i++) {
                if (inputArray[i] > burstThreshold) {
                    inputArray[i] *= burstAmplification;
                }
            }
        }

        // Set as excitatory input for medium dynamics
        mediumDynamics.setExcitatoryInput(inputArray);

        // Use previous activation blended with new input for state persistence (vectorized)
        var persistence = 1.0 - currentParameters.getDecayRate() * 0.01;  // Convert to time step
        var decayedPrev = VectorizedArrayOperations.scale(previousActivation, persistence);
        var blendedState = VectorizedArrayOperations.add(inputArray, decayedPrev);
        var currentState = new ActivationState(blendedState);

        // Evolve with medium time constant
        var timeStep = currentParameters.getTimeConstant() / 10000.0;  // Convert ms to seconds, use smaller step
        var evolvedState = mediumDynamics.evolve(currentState, timeStep);

        // Apply output gain and normalization (vectorized)
        var result = evolvedState.getActivations();
        var outputGain = currentParameters.getOutputGain();
        var normalization = currentParameters.getOutputNormalization();

        // Apply output gain (vectorized)
        VectorizedArrayOperations.scaleInPlace(result, outputGain);

        // Calculate sum for normalization (vectorized)
        var sum = VectorizedArrayOperations.sum(result);

        // Apply normalization if sum is significant
        if (sum > 0.01 && normalization > 0) {
            var normalizer = 1.0 / (1.0 + normalization * sum);
            VectorizedArrayOperations.scaleInPlace(result, normalizer);
        }

        // Apply ceiling and floor constraints (vectorized)
        var ceiling = currentParameters.getCeiling();
        var floor = currentParameters.getFloor();
        VectorizedArrayOperations.clampInPlace(result, floor, ceiling);

        // Store for state persistence
        System.arraycopy(result, 0, previousActivation, 0, result.length);

        // Update activation and notify listeners
        activation = new DenseVector(result);

        // Notify listeners about activation change
        var oldActivation = getActivation();
        for (var listener : listeners) {
            listener.onActivationChanged(getId(), oldActivation, activation);
        }

        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Layer 5 receives minimal top-down modulation as it's an output layer
        if (activation == null) {
            return new DenseVector(new double[size]);
        }

        // Apply very weak modulation (Layer 5 is primarily output-focused)
        var result = new double[size];
        var modulationStrength = 0.05;  // Very weak modulation for Layer 5

        for (int i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = activation.get(i);

            // Minimal modulation - Layer 5 maintains output characteristics
            result[i] = current * (1.0 + modulationStrength * expect);

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
        if (mediumDynamics != null) {
            mediumDynamics.reset();
        }
        if (previousActivation != null) {
            for (int i = 0; i < previousActivation.length; i++) {
                previousActivation[i] = 0.0;
            }
        }
        currentParameters = null;
    }

    private void updateDynamicsParameters(Layer5Parameters params) {
        // Update the shunting dynamics with Layer 5 specific parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .timeStep(params.getTimeConstant() / 10000.0)  // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.mediumDynamics = new ShuntingDynamicsImpl(shuntingParams, size);
    }

    private double[] patternToArray(Pattern pattern) {
        var array = new double[size];
        for (int i = 0; i < Math.min(pattern.dimension(), size); i++) {
            array[i] = pattern.get(i);
        }
        return array;
    }

    // ==================== Batch Processing Implementation ====================

    @Override
    public Pattern processWithStatefulSIMD(Pattern input, LayerParameters parameters) {
        if (!isStatefulSIMDBeneficial(input)) {
            return processBottomUp(input, parameters);
        }

        var layer5Params = (parameters instanceof Layer5Parameters) ?
            (Layer5Parameters) parameters : Layer5Parameters.builder().build();

        // Create single-pattern batch with previous state
        var batch = new Pattern[]{input};
        Pattern[] previousStates = null;
        if (previousActivation != null) {
            previousStates = new Pattern[]{new DenseVector(previousActivation.clone())};
        }

        var simdOutputs = Layer5SIMDBatch.processBatchSIMD(batch, previousStates, layer5Params, size);

        if (simdOutputs != null) {
            // Update state
            activation = simdOutputs[0];
            previousActivation = ((DenseVector) simdOutputs[0]).data().clone();
            return simdOutputs[0];
        }

        // Fall back to scalar
        return processBottomUp(input, parameters);
    }

    @Override
    public Pattern[] processBatchBottomUp(Pattern[] inputs, LayerParameters parameters) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("inputs cannot be null or empty");
        }
        if (parameters == null) {
            throw new NullPointerException("parameters cannot be null");
        }

        var layer5Params = (parameters instanceof Layer5Parameters) ?
            (Layer5Parameters) parameters : Layer5Parameters.builder().build();

        // Phase 6A: Stateful batch processing
        var batchSize = inputs.length;
        var outputs = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            outputs[i] = processWithStatefulSIMD(inputs[i], layer5Params);
        }

        return outputs;
    }

    @Deprecated
    private Pattern[] processBatchBottomUpLegacy(Pattern[] inputs, LayerParameters parameters) {
        var layer5Params = (Layer5Parameters) parameters;

        // Create previous states array from current previousActivation
        Pattern[] previousStates = null;
        if (previousActivation != null) {
            previousStates = new Pattern[inputs.length];
            for (int i = 0; i < inputs.length; i++) {
                previousStates[i] = new DenseVector(previousActivation.clone());
            }
        }

        // Try SIMD batch processing (Phase 3 optimization)
        var simdOutputs = Layer5SIMDBatch.processBatchSIMD(inputs, previousStates, layer5Params, size);

        if (simdOutputs != null) {
            // SIMD path was beneficial - use it
            if (simdOutputs.length > 0) {
                activation = simdOutputs[simdOutputs.length - 1];
                // Update previousActivation from last output
                for (int i = 0; i < size; i++) {
                    previousActivation[i] = activation.get(i);
                }
            }
            return simdOutputs;
        }

        // Fall back to sequential processing (Phase 2)
        updateDynamicsParameters(layer5Params);

        var batchSize = inputs.length;
        var outputs = new Pattern[batchSize];

        // Process each pattern
        for (int i = 0; i < batchSize; i++) {
            outputs[i] = processBottomUp(inputs[i], layer5Params);
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