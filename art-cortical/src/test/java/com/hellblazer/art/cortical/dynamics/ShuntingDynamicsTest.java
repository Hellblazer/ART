package com.hellblazer.art.cortical.dynamics;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shunting dynamics (on-center off-surround competition).
 * Validates equation implementation from Grossberg (1973) and
 * Kazerounian & Grossberg (2014).
 *
 * @author Migrated from art-temporal/temporal-dynamics to art-cortical (Phase 1)
 */
@Tag("dynamics")
@Tag("mathematical")
public class ShuntingDynamicsTest {

    private ShuntingDynamics dynamics;
    private ShuntingParameters parameters;
    private static final int DIMENSION = 10;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    public void setUp() {
        parameters = ShuntingParameters.competitiveDefaults(DIMENSION);
        dynamics = new ShuntingDynamics(parameters);
    }

    @Test
    public void testInitialization() {
        var activations = dynamics.getActivation();
        assertNotNull(activations);
        assertEquals(DIMENSION, activations.length);
        assertEquals(DIMENSION, dynamics.size());

        for (var activation : activations) {
            assertEquals(parameters.initialActivation(), activation, TOLERANCE);
        }
    }

    @Test
    public void testExcitatoryInput() {
        var input = new double[DIMENSION];
        input[5] = 1.0;  // Strong input at position 5

        dynamics.setExcitatoryInput(input);
        var activations = dynamics.update(0.01);

        // Should see activation increase at position 5
        assertTrue(activations[5] > 0, "Position 5 should be activated");

        // Nearby positions might also be slightly activated due to lateral excitation
        if (parameters.excitatoryStrength() > 0) {
            assertTrue(activations[4] >= 0, "Position 4 should be non-negative");
            assertTrue(activations[6] >= 0, "Position 6 should be non-negative");
        }
    }

    @Test
    @Tag("equation")
    public void testCompetitiveDynamics() {
        // Create two competing inputs
        var input = new double[DIMENSION];
        input[2] = 0.8;
        input[7] = 0.9;  // Stronger input

        dynamics.setExcitatoryInput(input);

        // Let dynamics evolve to steady state
        for (var i = 0; i < 100; i++) {
            dynamics.update(0.01);
        }

        var activations = dynamics.getActivation();

        // Stronger input should win (competitive dynamics)
        assertTrue(activations[7] > activations[2],
                  "Stronger input should dominate: act[7]=" + activations[7] +
                  " vs act[2]=" + activations[2]);

        // Check Mexican hat pattern around winner (off-surround inhibition)
        if (parameters.inhibitoryStrength() > parameters.excitatoryStrength()) {
            // Should see inhibition of distant units
            assertTrue(activations[0] < activations[7],
                      "Distant units should be inhibited");
        }
    }

    @Test
    @Tag("equation")
    public void testConvergence() {
        var input = new double[DIMENSION];
        input[5] = 0.5;
        dynamics.setExcitatoryInput(input);

        // Evolve until convergence (max 1000 iterations)
        var converged = false;
        for (var i = 0; i < 1000; i++) {
            dynamics.update(0.01);

            if (dynamics.hasConverged()) {
                converged = true;
                break;
            }
        }

        assertTrue(converged, "Dynamics should converge within 1000 iterations");
    }

    @Test
    public void testReset() {
        // Set some input and evolve
        var input = new double[DIMENSION];
        input[3] = 1.0;
        dynamics.setExcitatoryInput(input);

        dynamics.update(0.1);

        // Verify activation changed
        var preResetActivations = dynamics.getActivation();
        assertTrue(preResetActivations[3] > parameters.initialActivation(),
                  "Activation should have increased");

        // Reset
        dynamics.reset();

        var activations = dynamics.getActivation();
        for (var activation : activations) {
            assertEquals(parameters.initialActivation(), activation, TOLERANCE,
                        "All activations should reset to initial value");
        }
    }

    @Test
    @Tag("equation")
    public void testBounds() {
        // Test ceiling saturation
        var input = new double[DIMENSION];
        for (var i = 0; i < DIMENSION; i++) {
            input[i] = 10.0;  // Very strong input to test ceiling
        }
        dynamics.setExcitatoryInput(input);

        // Evolve for sufficient time
        for (var i = 0; i < 100; i++) {
            dynamics.update(0.01);
        }

        var activations = dynamics.getActivation();
        for (var activation : activations) {
            assertTrue(activation <= parameters.ceiling(),
                      "Activation should not exceed ceiling: " + activation);
            assertTrue(activation >= parameters.floor(),
                      "Activation should not go below floor: " + activation);
        }
    }

    @Test
    @Tag("validation")
    public void testEnergy() {
        var input = new double[DIMENSION];
        input[5] = 0.5;
        dynamics.setExcitatoryInput(input);

        var initialEnergy = dynamics.computeEnergy();

        // Evolve dynamics
        for (var i = 0; i < 50; i++) {
            dynamics.update(0.01);
        }

        var finalEnergy = dynamics.computeEnergy();

        // Energy should change (Lyapunov stability proof)
        assertNotEquals(initialEnergy, finalEnergy,
                       "Energy should change during evolution");
    }

    @Test
    @Tag("equation")
    public void testWinnerTakeAll() {
        // Use winner-take-all parameters (strong global inhibition)
        parameters = ShuntingParameters.winnerTakeAllDefaults(DIMENSION);
        dynamics = new ShuntingDynamics(parameters);

        // Multiple inputs with different strengths
        var input = new double[DIMENSION];
        input[2] = 0.7;
        input[5] = 0.8;   // Strongest
        input[8] = 0.75;

        dynamics.setExcitatoryInput(input);

        // Evolve to steady state
        for (var i = 0; i < 200; i++) {
            dynamics.update(0.01);
        }

        var activations = dynamics.getActivation();

        // Find winner
        var activeCount = 0;
        var winnerIndex = -1;
        var maxActivation = 0.0;

        for (var i = 0; i < DIMENSION; i++) {
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
        assertTrue(activeCount <= 3,
                  "Should have mostly winner-take-all behavior, got " + activeCount + " active");
    }

    @Test
    @Tag("equation")
    @Tag("validation")
    public void testShuntingEquationPrecision() {
        // Validate equation implementation with known values
        // Test single unit with known inputs

        var singleUnitParams = ShuntingParameters.builder(1)
            .uniformDecay(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.0)  // Disable lateral interactions
            .excitatoryStrength(0.0)
            .inhibitoryStrength(0.0)
            .timeStep(0.01)
            .build();

        var singleDynamics = new ShuntingDynamics(singleUnitParams);

        // Set excitatory input
        singleDynamics.setExcitatoryInput(new double[]{0.5});

        // Evolve one step
        var result = singleDynamics.update(0.01);

        // Expected: dx/dt = -A*x + (B-x)*S+ - (x-C)*S-
        // At t=0: x=0, A=1.0, B=1.0, C=0.0, S+=0.5, S-=0.0
        // dx/dt = -1.0*0 + (1.0-0)*0.5 - (0-0)*0 = 0.5
        // x(dt) = x(0) + dt * dx/dt = 0 + 0.01 * 0.5 = 0.005

        var tolerance = Double.parseDouble(
            System.getProperty("cortical.validation.tolerance", "1e-10"));

        assertEquals(0.005, result[0], tolerance,
                    "Shunting equation should match analytical solution");
    }

    @Test
    public void testClearInputs() {
        var input = new double[DIMENSION];
        input[5] = 1.0;

        dynamics.setExcitatoryInput(input);
        dynamics.clearInputs();

        // After clearing, updates should not respond to previous input
        dynamics.update(0.01);
        var activations = dynamics.getActivation();

        // Activations should decay toward zero (no input)
        for (var i = 0; i < DIMENSION; i++) {
            assertTrue(activations[i] <= parameters.ceiling() * 0.1,
                      "Activation should be minimal without input");
        }
    }

    @Test
    public void testParameterValidation() {
        // Test invalid ceiling/floor
        assertThrows(IllegalArgumentException.class, () -> {
            ShuntingParameters.builder(10)
                .ceiling(0.5)
                .floor(1.0)  // Floor > ceiling
                .build();
        });

        // Test invalid time step
        assertThrows(IllegalArgumentException.class, () -> {
            ShuntingParameters.builder(10)
                .timeStep(0.0)  // Must be positive
                .build();
        });

        // Test negative range
        assertThrows(IllegalArgumentException.class, () -> {
            ShuntingParameters.builder(10)
                .excitatoryRange(-1.0)
                .build();
        });
    }
}
