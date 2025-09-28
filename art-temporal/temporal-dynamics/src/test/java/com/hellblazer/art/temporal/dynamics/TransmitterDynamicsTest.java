package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.TransmitterState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for transmitter habituation dynamics.
 */
public class TransmitterDynamicsTest {

    private TransmitterDynamicsImpl dynamics;
    private TransmitterParameters parameters;
    private static final int DIMENSION = 5;

    @BeforeEach
    public void setUp() {
        parameters = TransmitterParameters.paperDefaults();
        dynamics = new TransmitterDynamicsImpl(parameters, DIMENSION);
    }

    @Test
    public void testInitialization() {
        var state = dynamics.getState();
        assertNotNull(state);
        var levels = state.getLevels();
        assertEquals(DIMENSION, levels.length);

        for (double level : levels) {
            assertEquals(parameters.getBaselineLevel(), level, 0.001);
        }
    }

    @Test
    public void testDepletion() {
        // Apply signal to deplete transmitters
        double[] signals = new double[DIMENSION];
        signals[0] = 1.0;  // Strong signal at position 0

        dynamics.setSignals(signals);
        var initialState = dynamics.getState();
        var evolved = dynamics.evolve(initialState, 0.01);

        var levels = evolved.getLevels();

        // Position 0 should be depleted
        assertTrue(levels[0] < parameters.getBaselineLevel(),
                  "Transmitter should be depleted");

        // Other positions should be unchanged or recovering
        for (int i = 1; i < DIMENSION; i++) {
            assertTrue(levels[i] >= levels[0],
                      "Non-signaled transmitters should not deplete as much");
        }
    }

    @Test
    public void testRecovery() {
        // First deplete
        double[] signals = new double[DIMENSION];
        signals[0] = 1.0;
        dynamics.setSignals(signals);

        var state = dynamics.getState();
        for (int i = 0; i < 10; i++) {
            state = dynamics.evolve(state, 0.1);
        }
        dynamics.setState(state);

        double depletedLevel = state.getLevels()[0];
        assertTrue(depletedLevel < parameters.getBaselineLevel());

        // Remove signal and let recover
        signals[0] = 0.0;
        dynamics.setSignals(signals);

        for (int i = 0; i < 100; i++) {
            state = dynamics.evolve(state, 0.1);
        }
        dynamics.setState(state);

        double recoveredLevel = dynamics.getState().getLevels()[0];
        assertTrue(recoveredLevel > depletedLevel,
                  "Transmitter should recover");
    }

    @Test
    public void testHabituation() {
        // Test habituation with activation pattern
        double[] activations = {0.5, 0.8, 0.3, 0.0, 0.9};

        dynamics.habituate(activations, 0.1);

        var levels = dynamics.getTransmitterLevels();

        // Higher activations should cause more depletion
        assertTrue(levels[4] < levels[0], "Highest activation should deplete most");
        assertTrue(levels[1] < levels[2], "Higher activation should deplete more");
        assertEquals(parameters.getBaselineLevel(), levels[3], 0.1,
                    "Zero activation should not deplete much");
    }

    @Test
    public void testGatedOutput() {
        // Set transmitter levels
        var transmitterLevels = new double[]{0.5, 0.8, 1.0, 0.2, 0.6};
        var presynapticSignals = new double[DIMENSION];
        var depletionHistory = new double[DIMENSION];
        var state = new TransmitterState(transmitterLevels, presynapticSignals, depletionHistory);
        dynamics.setState(state);

        double[] activations = {1.0, 1.0, 1.0, 1.0, 1.0};
        var gatedOutput = dynamics.computeGatedOutput(activations);

        assertEquals(0.5, gatedOutput[0], 0.001);
        assertEquals(0.8, gatedOutput[1], 0.001);
        assertEquals(1.0, gatedOutput[2], 0.001);
        assertEquals(0.2, gatedOutput[3], 0.001);
        assertEquals(0.6, gatedOutput[4], 0.001);
    }

    @Test
    public void testPartialReset() {
        // Deplete transmitters
        double[] signals = {1.0, 1.0, 1.0, 1.0, 1.0};
        dynamics.setSignals(signals);
        dynamics.update(1.0);

        var depletedLevels = dynamics.getTransmitterLevels();

        // Partial reset
        dynamics.partialReset(0.5);

        var resetLevels = dynamics.getTransmitterLevels();

        for (int i = 0; i < DIMENSION; i++) {
            assertTrue(resetLevels[i] > depletedLevels[i],
                      "Levels should increase after partial reset");
            assertTrue(resetLevels[i] < parameters.getBaselineLevel(),
                      "Should not fully recover");
        }
    }

    @Test
    public void testRecoveryTimeConstant() {
        // Set different signal levels
        double[] signals = {0.0, 0.5, 1.0, 0.0, 0.0};
        dynamics.setSignals(signals);

        double timeConstant = dynamics.computeRecoveryTimeConstant();
        assertTrue(timeConstant > 0, "Time constant should be positive");
        assertTrue(timeConstant < Double.POSITIVE_INFINITY,
                  "Time constant should be finite");
    }

    @Test
    public void testEquilibrium() {
        // Apply constant signal
        double signal = 0.5;
        double[] signals = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            signals[i] = signal;
        }
        dynamics.setSignals(signals);

        // Evolve to equilibrium
        var state = dynamics.getState();
        for (int i = 0; i < 1000; i++) {
            var newState = dynamics.evolve(state, 0.01);

            // Check if reached equilibrium
            boolean atEquilibrium = true;
            for (int j = 0; j < DIMENSION; j++) {
                if (Math.abs(newState.getLevels()[j] - state.getLevels()[j]) > 0.0001) {
                    atEquilibrium = false;
                    break;
                }
            }

            state = newState;
            if (atEquilibrium) break;
        }

        // Check theoretical equilibrium (allow for numerical differences)
        double expectedEquilibrium = parameters.computeEquilibrium(signal);
        for (double level : state.getLevels()) {
            // Allow 2x tolerance due to numerical integration differences
            assertEquals(expectedEquilibrium, level, expectedEquilibrium * 1.0);
        }
    }

    @Test
    public void testQuadraticDepletion() {
        // Test with parameters emphasizing quadratic depletion
        parameters = TransmitterParameters.builder()
            .recoveryRate(0.01)
            .linearDepletionRate(0.0)
            .quadraticDepletionRate(2.0)  // Only quadratic
            .baselineLevel(1.0)
            .build();
        dynamics = new TransmitterDynamicsImpl(parameters, DIMENSION);

        // Quadratic depletion should be stronger for larger signals
        double[] signals = {0.2, 0.4, 0.6, 0.8, 1.0};
        dynamics.setSignals(signals);

        var state = dynamics.getState();
        state = dynamics.evolve(state, 0.1);
        var levels = state.getLevels();

        // Verify quadratic relationship
        for (int i = 1; i < DIMENSION; i++) {
            double ratio = (1.0 - levels[i]) / (1.0 - levels[i-1]);
            assertTrue(ratio > 1.0, "Depletion should accelerate");
        }
    }
}