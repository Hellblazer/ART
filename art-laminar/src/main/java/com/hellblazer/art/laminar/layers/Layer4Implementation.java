package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.batch.BatchLayer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.impl.AbstractLayer;
import com.hellblazer.art.laminar.parameters.Layer4Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.performance.VectorizedArrayOperations;
import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.dynamics.ShuntingDynamicsImpl;
import com.hellblazer.art.temporal.dynamics.ShuntingParameters;

/**
 * Layer 4 Implementation - Thalamic Driving Input Layer.
 *
 * Layer 4 is the primary recipient of driving input from the thalamus (LGN).
 * It initiates cortical processing with strong, fast dynamics that can directly
 * fire cells without requiring modulatory input.
 *
 * Key characteristics:
 * - Fast time constants (10-50ms) for rapid response
 * - Strong driving signals that can fire cells independently
 * - Simple feedforward processing
 * - Minimal lateral inhibition in basic circuits
 * - Direct transformation of thalamic input to cortical representation
 * - Batch processing support for efficient multi-pattern processing
 *
 * @author Hal Hildebrand
 */
public class Layer4Implementation extends AbstractLayer implements BatchLayer {

    private ShuntingDynamicsImpl fastDynamics;
    private Layer4Parameters currentParameters;

    public Layer4Implementation(String id, int size) {
        super(id, size, LayerType.CUSTOM);
        initializeFastDynamics();
    }

    private void initializeFastDynamics() {
        // Initialize with fast dynamics suitable for Layer 4
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(0.0) // No lateral inhibition initially
            .timeStep(0.001) // 1ms time step for fast dynamics
            .build();
        this.fastDynamics = new ShuntingDynamicsImpl(params, size);
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
        for (int i = 0; i < Math.min(input.dimension(), size); i++) {
            inputArray[i] = input.get(i);
        }

        // Apply driving strength to input (Layer 4 receives strong thalamic drive)
        // Use vectorized scaling for performance
        var drivingStrength = currentParameters.getDrivingStrength();
        VectorizedArrayOperations.scaleInPlace(inputArray, drivingStrength);

        // Set as excitatory input for fast dynamics
        fastDynamics.setExcitatoryInput(inputArray);

        // For Layer 4, use the input itself as initial state since it's driving input
        // This reflects the direct thalamic drive characteristic
        var currentState = new ActivationState(inputArray);

        // Evolve with fast time constant - use smaller step for stability
        var timeStep = Math.min(currentParameters.getTimeConstant() / 1000.0, 0.01); // Cap at 10ms
        var evolvedState = fastDynamics.evolve(currentState, timeStep);

        // Apply ceiling and floor constraints with soft sigmoid saturation
        var result = evolvedState.getActivations();
        var ceiling = currentParameters.getCeiling();
        var floor = currentParameters.getFloor();

        // Apply soft sigmoid saturation for biological plausibility
        // Maps unbounded activation to [0, ceiling] range
        for (int i = 0; i < result.length; i++) {
            if (result[i] > 0) {
                // Sigmoid: ceiling * x / (1 + x) for x > 0
                result[i] = ceiling * result[i] / (1.0 + result[i]);
            }
        }

        // Vectorized floor constraint
        VectorizedArrayOperations.clampInPlace(result, floor, ceiling);

        // Update activation and return
        activation = new DenseVector(result);
        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Layer 4 receives minimal top-down modulation
        // It's primarily driven by bottom-up thalamic input
        if (activation == null) {
            return new DenseVector(new double[size]);
        }

        // Apply weak modulation (Layer 4 is less affected by top-down)
        var result = new double[size];
        var modulationStrength = 0.1; // Weak modulation for Layer 4

        for (int i = 0; i < size; i++) {
            var expect = i < expectation.dimension() ? expectation.get(i) : 0.0;
            var current = activation.get(i);

            // Minimal modulation - Layer 4 maintains driving characteristics
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
        if (fastDynamics != null) {
            fastDynamics.reset();
        }
        currentParameters = null;
    }

    private void updateDynamicsParameters(Layer4Parameters params) {
        // Update the shunting dynamics with Layer 4 specific parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .timeStep(params.getTimeConstant() / 1000.0) // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.fastDynamics = new ShuntingDynamicsImpl(shuntingParams, size);
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
    public Pattern[] processBatchBottomUp(Pattern[] inputs, LayerParameters parameters) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("inputs cannot be null or empty");
        }
        if (parameters == null) {
            throw new NullPointerException("parameters cannot be null");
        }

        // Use Layer4Parameters
        var layer4Params = (parameters instanceof Layer4Parameters) ?
            (Layer4Parameters) parameters : Layer4Parameters.builder().build();

        // Update dynamics parameters once for entire batch
        updateDynamicsParameters(layer4Params);

        var batchSize = inputs.length;
        var outputs = new Pattern[batchSize];
        var drivingStrength = layer4Params.getDrivingStrength();
        var ceiling = layer4Params.getCeiling();
        var floor = layer4Params.getFloor();
        var timeStep = Math.min(layer4Params.getTimeConstant() / 1000.0, 0.01);

        // Process each pattern with shared dynamics configuration
        for (int i = 0; i < batchSize; i++) {
            var input = inputs[i];
            if (input == null) {
                throw new IllegalArgumentException("inputs[" + i + "] is null");
            }
            if (input.dimension() != size && input.dimension() < size) {
                // Allow smaller inputs (zero-pad), but not larger
                var padded = new double[size];
                for (int j = 0; j < input.dimension(); j++) {
                    padded[j] = input.get(j);
                }
                input = new DenseVector(padded);
            }

            // Convert and scale input (vectorized)
            var inputArray = patternToArray(input);
            VectorizedArrayOperations.scaleInPlace(inputArray, drivingStrength);

            // Set excitatory input
            fastDynamics.setExcitatoryInput(inputArray);

            // Evolve dynamics
            var currentState = new ActivationState(inputArray);
            var evolvedState = fastDynamics.evolve(currentState, timeStep);

            // Apply sigmoid saturation
            var result = evolvedState.getActivations();
            for (int j = 0; j < result.length; j++) {
                if (result[j] > 0) {
                    result[j] = ceiling * result[j] / (1.0 + result[j]);
                }
            }

            // Apply constraints (vectorized)
            VectorizedArrayOperations.clampInPlace(result, floor, ceiling);

            outputs[i] = new DenseVector(result);
        }

        // Update activation to last pattern (maintains single-pattern semantics)
        if (batchSize > 0) {
            activation = outputs[batchSize - 1];
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