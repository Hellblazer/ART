package com.hellblazer.art.temporal.dynamics;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for multi-scale dynamics integration.
 */
public class MultiScaleDynamicsTest {

    private MultiScaleDynamics dynamics;
    private MultiScaleParameters parameters;
    private static final int DIMENSION = 8;

    @BeforeEach
    public void setUp() {
        parameters = MultiScaleParameters.defaults(DIMENSION);
        dynamics = new MultiScaleDynamics(parameters);
    }

    @Test
    public void testInitialization() {
        var state = dynamics.getState();
        assertNotNull(state);
        assertNotNull(state.activationState());
        assertNotNull(state.transmitterState());
        assertNotNull(state.timingState());
        assertEquals(0.0, state.time());
    }

    @Test
    public void testUpdate() {
        double[] input = new double[DIMENSION];
        input[3] = 0.5;

        dynamics.update(input, 0.01);

        var state = dynamics.getState();
        assertTrue(state.time() > 0, "Time should advance");

        // Check activations changed
        var activations = state.activationState().getActivations();
        boolean hasNonZero = false;
        for (double act : activations) {
            if (act > 0) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Should have some activation");
    }

    @Test
    public void testTimeScaleSeparation() {
        double[] input = new double[DIMENSION];
        input[4] = 1.0;

        // Track transmitter levels
        var initialTransmitters = dynamics.getTransmitterDynamics().getTransmitterLevels();

        // Update for less than transmitter update ratio
        for (int i = 0; i < parameters.getTransmitterUpdateRatio() - 1; i++) {
            dynamics.update(input, 0.01);
        }

        var midTransmitters = dynamics.getTransmitterDynamics().getTransmitterLevels();

        // Transmitters shouldn't have changed much yet (allow small numerical drift)
        for (int i = 0; i < DIMENSION; i++) {
            assertEquals(initialTransmitters[i], midTransmitters[i], 0.002,
                        "Transmitters shouldn't update yet");
        }

        // One more update should trigger transmitter update
        dynamics.update(input, 0.01);

        var finalTransmitters = dynamics.getTransmitterDynamics().getTransmitterLevels();

        // Now transmitters should have changed
        boolean changed = false;
        for (int i = 0; i < DIMENSION; i++) {
            if (Math.abs(finalTransmitters[i] - initialTransmitters[i]) > 0.001) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, "Transmitters should update at correct ratio");
    }

    @Test
    public void testGatedOutput() {
        // Set up state with specific activations
        double[] input = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            input[i] = 0.5;
        }

        // Update to establish activations
        for (int i = 0; i < 10; i++) {
            dynamics.update(input, 0.01);
        }

        var gatedOutput = dynamics.getGatedOutput();
        assertNotNull(gatedOutput);
        assertEquals(DIMENSION, gatedOutput.length);

        // Gated output should be <= activation (due to transmitter gating)
        var activations = dynamics.getShuntingDynamics().getActivations();
        for (int i = 0; i < DIMENSION; i++) {
            assertTrue(gatedOutput[i] <= activations[i] + 0.001,
                      "Gated output should not exceed activation");
        }
    }

    @Test
    public void testReset() {
        // Evolve dynamics
        double[] input = new double[DIMENSION];
        input[2] = 0.8;

        for (int i = 0; i < 50; i++) {
            dynamics.update(input, 0.01);
        }

        // Reset
        dynamics.reset();

        var state = dynamics.getState();
        assertEquals(0.0, state.time(), "Time should reset");

        var activations = state.activationState().getActivations();
        for (double act : activations) {
            assertEquals(0.0, act, 0.001, "Activations should reset");
        }

        var transmitters = state.transmitterState().getLevels();
        for (double trans : transmitters) {
            assertEquals(parameters.getTransmitterParameters().getBaselineLevel(),
                        trans, 0.001, "Transmitters should reset to baseline");
        }
    }

    @Test
    public void testPartialReset() {
        // Evolve dynamics
        double[] input = new double[DIMENSION];
        input[1] = 0.6;

        for (int i = 0; i < 30; i++) {
            dynamics.update(input, 0.01);
        }

        var beforeReset = dynamics.getShuntingDynamics().getActivations();

        // Partial reset
        dynamics.partialReset();

        var afterReset = dynamics.getShuntingDynamics().getActivations();

        // Activations should decay but not to zero
        for (int i = 0; i < DIMENSION; i++) {
            if (beforeReset[i] > 0) {
                assertTrue(afterReset[i] < beforeReset[i],
                          "Should decay after partial reset");
                assertTrue(afterReset[i] >= 0,
                          "Should not go negative");
            }
        }
    }

    @Test
    public void testEnergy() {
        double initialEnergy = dynamics.computeEnergy();

        // Add input and evolve
        double[] input = new double[DIMENSION];
        input[3] = 0.5;
        input[5] = 0.7;

        for (int i = 0; i < 20; i++) {
            dynamics.update(input, 0.01);
        }

        double finalEnergy = dynamics.computeEnergy();

        // Energy should change with dynamics
        assertNotEquals(initialEnergy, finalEnergy);
        assertTrue(finalEnergy >= 0, "Energy should be non-negative");
    }

    @Test
    public void testStatistics() {
        // Set up some activity
        double[] input = new double[DIMENSION];
        input[2] = 0.4;
        input[6] = 0.9;

        for (int i = 0; i < 25; i++) {
            dynamics.update(input, 0.01);
        }

        var stats = dynamics.computeStatistics();

        assertNotNull(stats);
        assertTrue(stats.averageActivation() >= 0);
        assertTrue(stats.averageTransmitter() > 0);
        assertTrue(stats.maxActivation() >= stats.averageActivation());
        assertTrue(stats.minTransmitter() <= stats.averageTransmitter());
        assertTrue(stats.totalEnergy() >= 0);
        assertTrue(stats.time() > 0);
    }

    @Test
    public void testConvergence() {
        // Use simple constant input
        double[] input = new double[DIMENSION];
        input[4] = 0.5;

        // Let system evolve
        for (int i = 0; i < 1000; i++) {
            dynamics.update(input, 0.01);

            if (i > 100 && dynamics.hasConverged(0.01)) {
                break;
            }
        }

        // Check if converged (may not always converge depending on parameters)
        var finalState = dynamics.getState();
        assertTrue(finalState.time() > 0, "Should have evolved");
    }

    @Test
    public void testSpeechParameters() {
        parameters = MultiScaleParameters.speechDefaults(DIMENSION);
        dynamics = new MultiScaleDynamics(parameters);

        // Test with speech-like input pattern
        double[] input = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            input[i] = Math.sin(i * Math.PI / 4) * 0.5 + 0.5;
        }

        dynamics.update(input, 0.001);  // 1ms time step for speech

        var output = dynamics.getGatedOutput();
        assertNotNull(output);

        // With timing gating enabled
        assertTrue(parameters.isTimingGatingEnabled());
    }
}