package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.temporal.core.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test to understand shunting dynamics behavior.
 */
class ShuntingDebugTest {

    @Test
    void testBasicShuntingStep() {
        var dynamics = new ShuntingDynamics();

        // Create parameters
        var params = ShuntingParameters.builder()
            .decayRate(0.1)
            .upperBound(1.0)
            .lowerBound(0.0)
            .selfExcitation(0.2)
            .lateralInhibition(0.5)
            .build();

        // Create initial state with zero activations and strong excitatory input
        var initialActivations = new double[]{0.0, 0.0, 0.0, 0.0, 0.0};
        var excitatoryInputs = new double[]{1.0, 0.8, 0.6, 0.4, 0.2};  // Strong inputs
        var state = new ShuntingState(initialActivations, excitatoryInputs);

        System.out.println("Initial state:");
        System.out.println("  Activations: " + java.util.Arrays.toString(state.getActivations()));
        System.out.println("  Excitatory: " + java.util.Arrays.toString(state.getExcitatoryInputs()));

        // Compute derivative
        var derivative = dynamics.computeDerivative(state, params, 0.0);
        System.out.println("\nDerivative:");
        System.out.println("  Values: " + java.util.Arrays.toString(derivative.toArray()));

        // Take one step
        var newState = dynamics.step(state, params, 0.0, 0.01);
        System.out.println("\nAfter 1 step (dt=0.01):");
        System.out.println("  Activations: " + java.util.Arrays.toString(newState.getActivations()));

        // Verify that activations increased
        var newActivations = newState.getActivations();
        for (int i = 0; i < newActivations.length; i++) {
            assertTrue(newActivations[i] > 0,
                "Activation " + i + " should be positive after step with excitatory input");
        }

        // Take multiple steps to see convergence
        var currentState = state;
        for (int step = 0; step < 100; step++) {
            currentState = dynamics.step(currentState, params, step * 0.01, 0.01);
        }

        System.out.println("\nAfter 100 steps:");
        System.out.println("  Activations: " + java.util.Arrays.toString(currentState.getActivations()));

        // Verify steady state activations are positive
        var steadyActivations = currentState.getActivations();
        for (int i = 0; i < steadyActivations.length; i++) {
            assertTrue(steadyActivations[i] > 0.1,
                "Steady state activation " + i + " should be substantial with strong input");
        }
    }

    @Test
    void testWorkingMemoryActivationFlow() {
        var params = WorkingMemoryParameters.builder()
            .capacity(5)
            .build();

        var workingMemory = new WorkingMemory(params);

        // Create simple patterns
        var pattern1 = new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        var pattern2 = new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        var pattern3 = new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        System.out.println("Storing 3 patterns in sequence...");

        // Store first item
        workingMemory.storeItem(pattern1, 0.1);
        var state1 = workingMemory.getDetailedState();
        System.out.println("\nAfter storing item 0:");
        System.out.println("  Activations: " + java.util.Arrays.toString(state1.shuntingState().getActivations()));
        System.out.println("  Transmitters: " + java.util.Arrays.toString(state1.transmitterState().getTransmitterLevels()));

        // Store second item
        workingMemory.storeItem(pattern2, 0.1);
        var state2 = workingMemory.getDetailedState();
        System.out.println("\nAfter storing item 1:");
        System.out.println("  Activations: " + java.util.Arrays.toString(state2.shuntingState().getActivations()));
        System.out.println("  Transmitters: " + java.util.Arrays.toString(state2.transmitterState().getTransmitterLevels()));

        // Store third item
        workingMemory.storeItem(pattern3, 0.1);
        var state3 = workingMemory.getDetailedState();
        System.out.println("\nAfter storing item 2:");
        System.out.println("  Activations: " + java.util.Arrays.toString(state3.shuntingState().getActivations()));
        System.out.println("  Transmitters: " + java.util.Arrays.toString(state3.transmitterState().getTransmitterLevels()));

        // Early items should have higher activation than late items
        var activations = state3.shuntingState().getActivations();
        System.out.println("\nPrimacy gradient check:");
        System.out.println("  Early (item 0): " + activations[0]);
        System.out.println("  Late (item 2):  " + activations[2]);

        assertTrue(activations[0] > activations[2],
            String.format("Early item (%.4f) should have more activation than late item (%.4f)",
                activations[0], activations[2]));
    }
}