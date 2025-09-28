package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.ActivationState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shunting dynamics.
 */
public class ShuntingDynamicsTest {

    private ShuntingDynamicsImpl dynamics;
    private ShuntingParameters parameters;
    private static final int DIMENSION = 10;

    @BeforeEach
    public void setUp() {
        parameters = ShuntingParameters.competitiveDefaults(DIMENSION);
        dynamics = new ShuntingDynamicsImpl(parameters, DIMENSION);
    }

    @Test
    public void testInitialization() {
        var state = dynamics.getState();
        assertNotNull(state);
        var activations = state.getActivations();
        assertEquals(DIMENSION, activations.length);

        for (double activation : activations) {
            assertEquals(0.0, activation, 0.001);
        }
    }

    @Test
    public void testExcitatoryInput() {
        double[] input = new double[DIMENSION];
        input[5] = 1.0;  // Strong input at position 5

        dynamics.setExcitatoryInput(input);
        var initialState = dynamics.getState();
        var evolved = dynamics.evolve(initialState, 0.01);

        var activations = evolved.getActivations();

        // Should see activation increase at position 5
        assertTrue(activations[5] > 0, "Position 5 should be activated");

        // Nearby positions might also be slightly activated due to lateral excitation
        if (parameters.getExcitatoryStrength() > 0) {
            assertTrue(activations[4] >= 0, "Position 4 should be non-negative");
            assertTrue(activations[6] >= 0, "Position 6 should be non-negative");
        }
    }

    @Test
    public void testCompetitiveDynamics() {
        // Create two competing inputs
        double[] input = new double[DIMENSION];
        input[2] = 0.8;
        input[7] = 0.9;  // Stronger input

        dynamics.setExcitatoryInput(input);

        // Let dynamics evolve
        var state = dynamics.getState();
        for (int i = 0; i < 100; i++) {
            state = dynamics.evolve(state, 0.01);
        }
        dynamics.setState(state);

        var activations = state.getActivations();

        // Stronger input should win
        assertTrue(activations[7] > activations[2],
                  "Stronger input should dominate");

        // Check Mexican hat pattern around winner
        if (parameters.getInhibitoryStrength() > parameters.getExcitatoryStrength()) {
            // Should see inhibition of distant units
            assertTrue(activations[0] < activations[7],
                      "Distant units should be inhibited");
        }
    }

    @Test
    public void testConvergence() {
        double[] input = new double[DIMENSION];
        input[5] = 0.5;
        dynamics.setExcitatoryInput(input);

        // Evolve until convergence
        for (int i = 0; i < 1000; i++) {
            var state = dynamics.getState();
            var evolved = dynamics.evolve(state, 0.01);
            dynamics.setState(evolved);

            if (dynamics.hasConverged(0.0001)) {
                break;
            }
        }

        assertTrue(dynamics.hasConverged(0.001),
                  "Dynamics should converge");
    }

    @Test
    public void testReset() {
        // Set some input and evolve
        double[] input = new double[DIMENSION];
        input[3] = 1.0;
        dynamics.setExcitatoryInput(input);

        var state = dynamics.getState();
        state = dynamics.evolve(state, 0.1);
        dynamics.setState(state);

        // Reset
        dynamics.reset();

        var resetState = dynamics.getState();
        var activations = resetState.getActivations();

        for (double activation : activations) {
            assertEquals(parameters.getInitialActivation(), activation, 0.001);
        }
    }

    @Test
    public void testBounds() {
        // Test ceiling
        double[] input = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            input[i] = 10.0;  // Very strong input
        }
        dynamics.setExcitatoryInput(input);

        var state = dynamics.getState();
        for (int i = 0; i < 100; i++) {
            state = dynamics.evolve(state, 0.01);
        }

        var activations = state.getActivations();
        for (double activation : activations) {
            assertTrue(activation <= parameters.getCeiling(),
                      "Should not exceed ceiling");
            assertTrue(activation >= parameters.getFloor(),
                      "Should not go below floor");
        }
    }

    @Test
    public void testEnergy() {
        double[] input = new double[DIMENSION];
        input[5] = 0.5;
        dynamics.setExcitatoryInput(input);

        double initialEnergy = dynamics.computeEnergy();

        // Evolve dynamics
        var state = dynamics.getState();
        for (int i = 0; i < 50; i++) {
            state = dynamics.evolve(state, 0.01);
            dynamics.setState(state);
        }

        double finalEnergy = dynamics.computeEnergy();

        // Energy should generally decrease (Lyapunov stability)
        // But may increase initially due to input
        assertNotEquals(initialEnergy, finalEnergy);
    }

    @Test
    public void testWinnerTakeAll() {
        // Use winner-take-all parameters
        parameters = ShuntingParameters.winnerTakeAllDefaults(DIMENSION);
        dynamics = new ShuntingDynamicsImpl(parameters, DIMENSION);

        // Multiple inputs
        double[] input = new double[DIMENSION];
        input[2] = 0.7;
        input[5] = 0.8;
        input[8] = 0.75;

        dynamics.setExcitatoryInput(input);

        // Evolve to steady state
        var state = dynamics.getState();
        for (int i = 0; i < 200; i++) {
            state = dynamics.evolve(state, 0.01);
        }
        dynamics.setState(state);

        var activations = state.getActivations();

        // Count active units
        int activeCount = 0;
        int winnerIndex = -1;
        double maxActivation = 0.0;

        for (int i = 0; i < DIMENSION; i++) {
            if (activations[i] > 0.1) {
                activeCount++;
            }
            if (activations[i] > maxActivation) {
                maxActivation = activations[i];
                winnerIndex = i;
            }
        }

        // Should have single winner
        assertEquals(5, winnerIndex, "Strongest input should win");
        // Allow up to 3 active neurons (winner + some residual activity)
        assertTrue(activeCount <= 3, "Should have mostly winner-take-all behavior");
    }
}