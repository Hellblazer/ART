package com.hellblazer.art.laminar.canonical;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test to verify transmitter integration is working.
 */
class TransmitterIntegrationDebugTest extends CanonicalCircuitTestBase {

    @Test
    void testTransmitterStateChanges() {
        var transmitterParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.05)
            .lambda(0.5)  // Higher lambda for faster depletion
            .mu(0.1)      // Higher mu for faster depletion
            .build();

        var dynamics = new com.hellblazer.art.temporal.core.TransmitterDynamics();

        // Start with full transmitters
        var initialState = new com.hellblazer.art.temporal.core.TransmitterState(5);
        var initialLevels = initialState.getTransmitterLevels();

        System.out.println("Initial levels: " + java.util.Arrays.toString(initialLevels));

        // Apply strong signal
        var strongSignal = new double[]{1.0, 1.0, 1.0, 1.0, 1.0};
        var stateWithSignal = new com.hellblazer.art.temporal.core.TransmitterState(
            initialLevels,
            strongSignal,
            initialState.getDepletionHistory()
        );

        // Compute derivative directly
        var derivative = dynamics.computeDerivative(stateWithSignal, transmitterParams, 0.0);
        System.out.println("Derivative levels: " + java.util.Arrays.toString(derivative.toArray()));

        // Test state arithmetic
        var scaled = derivative.scale(0.1);
        System.out.println("Scaled derivative: " + java.util.Arrays.toString(scaled.toArray()));

        var updated = stateWithSignal.add(scaled);
        System.out.println("After add: " + java.util.Arrays.toString(updated.toArray()));

        // Verify change
        var updatedLevels = updated.toArray();
        for (int i = 0; i < 5; i++) {
            System.out.println("Level " + i + ": " + initialLevels[i] + " -> " + updatedLevels[i] +
                " (derivative: " + derivative.toArray()[i] + ")");
        }

        // Just verify derivative is negative (depletion)
        for (int i = 0; i < 5; i++) {
            assertTrue(derivative.toArray()[i] < 0,
                "Derivative should be negative (depletion) for transmitter " + i);
        }
    }
}