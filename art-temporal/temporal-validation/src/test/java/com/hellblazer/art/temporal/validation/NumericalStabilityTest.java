package com.hellblazer.art.temporal.validation;

import com.hellblazer.art.temporal.dynamics.*;
import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.core.TransmitterState;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests numerical stability and convergence properties.
 * Validates that dynamics remain stable and bounded.
 */
@Tag("validation")
@Tag("mathematical")
public class NumericalStabilityTest {

    private static final double EPSILON = 1e-14;
    private static final int MAX_ITERATIONS = 10000;

    @Test
    public void testShuntingStability() {
        var params = ShuntingParameters.competitiveDefaults(10);
        var dynamics = new ShuntingDynamicsImpl(params, 10);

        // Test with extreme inputs
        double[] extremeInput = new double[10];
        for (int i = 0; i < 10; i++) {
            extremeInput[i] = 1000.0;  // Very large input
        }
        dynamics.setExcitatoryInput(extremeInput);

        // Evolve for many steps
        var state = dynamics.getState();
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            state = dynamics.evolve(state, 0.001);

            // Check bounds at each step
            var activations = state.getActivations();
            for (double act : activations) {
                assertTrue(act >= params.getFloor() - EPSILON,
                          "Activation below floor");
                assertTrue(act <= params.getCeiling() + EPSILON,
                          "Activation above ceiling");
                assertTrue(Double.isFinite(act), "Activation not finite");
            }
        }

        // Should converge to a stable state
        dynamics.setState(state);
        assertTrue(dynamics.hasConverged(1e-6), "Should converge");
    }

    @Test
    public void testTransmitterStability() {
        var params = TransmitterParameters.paperDefaults();
        var dynamics = new TransmitterDynamicsImpl(params, 5);

        // Test with oscillating signals
        for (int cycle = 0; cycle < 100; cycle++) {
            double[] signals = new double[5];
            for (int i = 0; i < 5; i++) {
                signals[i] = (cycle % 2 == 0) ? 1.0 : 0.0;
            }
            dynamics.setSignals(signals);

            var state = dynamics.getState();
            state = dynamics.evolve(state, 0.01);
            dynamics.setState(state);

            // Check bounds
            var levels = state.getLevels();
            for (double level : levels) {
                assertTrue(level >= 0.0 - EPSILON, "Level below 0");
                assertTrue(level <= 1.0 + EPSILON, "Level above 1");
                assertTrue(Double.isFinite(level), "Level not finite");
            }
        }
    }

    @Test
    public void testEnergyMonotonicity() {
        var params = ShuntingParameters.competitiveDefaults(8);
        var dynamics = new ShuntingDynamicsImpl(params, 8);

        // Random initial state
        double[] initial = new double[8];
        for (int i = 0; i < 8; i++) {
            initial[i] = Math.random() * 0.5;
        }
        dynamics.setState(new ActivationState(initial));

        // No external input (autonomous dynamics)
        dynamics.clearInputs();

        // Track energy over time
        double previousEnergy = dynamics.computeEnergy();
        int increasesCount = 0;

        for (int step = 0; step < 1000; step++) {
            var state = dynamics.getState();
            state = dynamics.evolve(state, 0.01);
            dynamics.setState(state);

            double currentEnergy = dynamics.computeEnergy();

            // Small increases allowed due to numerical errors
            if (currentEnergy > previousEnergy + 1e-10) {
                increasesCount++;
            }

            previousEnergy = currentEnergy;
        }

        // Energy should generally decrease (Lyapunov stability)
        assertTrue(increasesCount < 50, "Energy should mostly decrease");
    }

    @Test
    public void testFixedPointStability() {
        var params = ShuntingParameters.defaults(5);
        var dynamics = new ShuntingDynamicsImpl(params, 5);

        // Find equilibrium with no input
        dynamics.clearInputs();
        var state = dynamics.getState();

        // Evolve to equilibrium
        for (int i = 0; i < 5000; i++) {
            state = dynamics.evolve(state, 0.01);
        }
        dynamics.setState(state);

        var equilibrium = state.getActivations();

        // Perturb slightly
        double[] perturbed = equilibrium.clone();
        perturbed[2] += 0.01;  // Small perturbation
        dynamics.setState(new ActivationState(perturbed));

        // Evolve and check return to equilibrium
        for (int i = 0; i < 1000; i++) {
            state = dynamics.evolve(dynamics.getState(), 0.01);
            dynamics.setState(state);
        }

        var finalState = dynamics.getState().getActivations();

        // Should return close to equilibrium
        for (int i = 0; i < 5; i++) {
            assertEquals(equilibrium[i], finalState[i], 0.01,
                        "Should return to equilibrium");
        }
    }

    @Test
    public void testTimeStepStability() {
        var params = MultiScaleParameters.defaults(10);
        var dynamics = new MultiScaleDynamics(params);

        double[] input = new double[10];
        for (int i = 0; i < 10; i++) {
            input[i] = 0.5;
        }

        // Test with different time steps
        double[] timeSteps = {0.0001, 0.001, 0.01, 0.1};

        for (double dt : timeSteps) {
            dynamics.reset();

            boolean stable = true;
            for (int step = 0; step < 100; step++) {
                dynamics.update(input, dt);

                var output = dynamics.getGatedOutput();
                for (double out : output) {
                    if (!Double.isFinite(out) || out < 0 || out > 1) {
                        stable = false;
                        break;
                    }
                }
                if (!stable) break;
            }

            assertTrue(stable, "Should be stable with dt=" + dt);
        }
    }

    @Test
    public void testTransmitterConservation() {
        var params = TransmitterParameters.paperDefaults();
        var dynamics = new TransmitterDynamicsImpl(params, 3);

        // Total transmitter "mass" should be conserved in closed system
        double[] signals = {0.5, 0.5, 0.5};
        dynamics.setSignals(signals);

        double initialSum = 0.0;
        for (double level : dynamics.getTransmitterLevels()) {
            initialSum += level;
        }

        // Evolve
        for (int i = 0; i < 1000; i++) {
            dynamics.update(0.01);
        }

        double finalSum = 0.0;
        for (double level : dynamics.getTransmitterLevels()) {
            finalSum += level;
        }

        // With recovery, total should not increase beyond initial
        assertTrue(finalSum <= initialSum + EPSILON,
                  "Transmitter mass should not increase");
    }

    @Test
    public void testNumericalPrecision() {
        // Test for accumulation of numerical errors
        var params = ShuntingParameters.defaults(5);
        var dynamics = new ShuntingDynamicsImpl(params, 5);

        // Very small values
        double[] tiny = {1e-10, 1e-11, 1e-12, 1e-13, 1e-14};
        dynamics.setState(new ActivationState(tiny));

        // Many iterations with small time step
        var state = dynamics.getState();
        for (int i = 0; i < 10000; i++) {
            state = dynamics.evolve(state, 0.0001);
        }

        // Should remain stable
        var final_acts = state.getActivations();
        for (double act : final_acts) {
            assertTrue(Double.isFinite(act), "Should remain finite");
            assertTrue(act >= 0, "Should remain non-negative");
        }
    }

    @Test
    public void testChaosAbsence() {
        // Verify system is not chaotic (sensitive to initial conditions)
        var params = MultiScaleParameters.defaults(5);

        var dynamics1 = new MultiScaleDynamics(params);
        var dynamics2 = new MultiScaleDynamics(params);

        // Nearly identical initial conditions
        double[] input1 = {0.5, 0.5, 0.5, 0.5, 0.5};
        double[] input2 = {0.5, 0.5, 0.500001, 0.5, 0.5};  // Tiny difference

        // Evolve both
        for (int i = 0; i < 1000; i++) {
            dynamics1.update(input1, 0.01);
            dynamics2.update(input2, 0.01);
        }

        var output1 = dynamics1.getGatedOutput();
        var output2 = dynamics2.getGatedOutput();

        // Outputs should remain close (not exponentially diverge)
        for (int i = 0; i < 5; i++) {
            assertEquals(output1[i], output2[i], 0.01,
                        "Should not show sensitive dependence");
        }
    }
}