package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.impl.AbstractPathway;
import com.hellblazer.art.laminar.impl.DefaultPathwayParameters;
import com.hellblazer.art.laminar.parameters.PathwayParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for transmitter dynamics integration in pathways.
 * Validates the habituative transmitter gating mechanism that creates
 * primacy gradient effects in temporal sequences.
 *
 * Based on Kazerounian & Grossberg (2014) Equation 7:
 * dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 *
 * @author Hal Hildebrand
 */
class TransmitterDynamicsTest extends CanonicalCircuitTestBase {

    private ShuntingPathwayDecorator pathway;
    private PathwayParameters pathwayParams;

    @BeforeEach
    void setUp() {
        // Use faster recovery for testing (10x faster than paper default)
        var transmitterParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.05)       // 10x faster recovery for testing
            .lambda(0.1)
            .mu(0.05)
            .depletionThreshold(0.2)
            .initialLevel(1.0)
            .enableQuadratic(true)
            .build();

        var shuntingParams = createStandardShuntingParameters();
        pathwayParams = new DefaultPathwayParameters(1.0, 0.5, true);

        var delegate = new TestPathway("test", "src", "tgt", PathwayType.BOTTOM_UP);
        pathway = new ShuntingPathwayDecorator(
            delegate,
            shuntingParams,
            transmitterParams,
            TimeScale.FAST
        );
    }

    // ============ Basic Transmitter Tests ============

    @Test
    void testTransmitterInitialization() {
        // Transmitters should start at full capacity
        var input = createTestPattern(10, 0.5);
        pathway.propagate(input, pathwayParams);

        var transmitterState = pathway.getTransmitterState();
        assertNotNull(transmitterState);

        var levels = transmitterState.getTransmitterLevels();
        for (var level : levels) {
            // Allow for small depletion from first propagation
            assertTrue(level > 0.99,
                "Initial transmitter levels should be near 1.0, got: " + level);
        }
    }

    @Test
    void testTransmitterDepletion() {
        // Repeated strong inputs should deplete transmitters
        var strongInput = createTestPattern(10, 1.0);

        // First presentation - high transmitter levels
        pathway.propagate(strongInput, pathwayParams);
        var initialState = pathway.getTransmitterState();
        var initialLevels = initialState.getTransmitterLevels();

        // Multiple presentations to cause depletion
        for (int i = 0; i < 50; i++) {
            pathway.propagate(strongInput, pathwayParams);
            pathway.updateDynamics(0.1);  // Larger time step for faster depletion
        }

        var depletedState = pathway.getTransmitterState();
        var depletedLevels = depletedState.getTransmitterLevels();

        // Transmitters should be depleted
        for (int i = 0; i < initialLevels.length; i++) {
            assertTrue(depletedLevels[i] < initialLevels[i],
                "Transmitter " + i + " should be depleted after repeated use");
        }
    }

    @Test
    void testTransmitterRecovery() {
        // Deplete transmitters
        var strongInput = createTestPattern(10, 1.0);
        for (int i = 0; i < 30; i++) {
            pathway.propagate(strongInput, pathwayParams);
            pathway.updateDynamics(0.1);
        }

        var depletedState = pathway.getTransmitterState();
        var depletedLevels = depletedState.getTransmitterLevels();

        // Now rest (no input) to allow recovery
        var restInput = createTestPattern(10, 0.0);
        for (int i = 0; i < 100; i++) {
            pathway.propagate(restInput, pathwayParams);
            pathway.updateDynamics(0.1);
        }

        var recoveredState = pathway.getTransmitterState();
        var recoveredLevels = recoveredState.getTransmitterLevels();

        // Transmitters should partially recover
        for (int i = 0; i < depletedLevels.length; i++) {
            assertTrue(recoveredLevels[i] > depletedLevels[i],
                "Transmitter " + i + " should recover during rest");
        }
    }

    // ============ Transmitter Gating Tests ============

    @Test
    void testTransmitterGatingEffect() {
        var input = createTestPattern(10, 0.8);

        // First propagation with full transmitters
        var firstOutput = pathway.propagate(input, pathwayParams);
        var firstMagnitude = computeMagnitude(firstOutput);

        // Deplete transmitters significantly
        for (int i = 0; i < 50; i++) {
            pathway.propagate(input, pathwayParams);
            pathway.updateDynamics(0.1);
        }

        // Same input with depleted transmitters
        var secondOutput = pathway.propagate(input, pathwayParams);
        var secondMagnitude = computeMagnitude(secondOutput);

        // Output should be weaker with depleted transmitters
        assertTrue(secondMagnitude < firstMagnitude,
            "Depleted transmitters should reduce output magnitude");
    }

    @Test
    void testSignalDependentDepletion() {
        var weakInput = createTestPattern(10, 0.2);
        var strongInput = createTestPattern(10, 0.9);

        // Present weak signal repeatedly
        for (int i = 0; i < 30; i++) {
            pathway.propagate(weakInput, pathwayParams);
            pathway.updateDynamics(0.1);
        }
        var weakDepletionLevels = pathway.getTransmitterState().getTransmitterLevels();

        // Reset pathway
        pathway.resetDynamics();

        // Present strong signal same number of times
        for (int i = 0; i < 30; i++) {
            pathway.propagate(strongInput, pathwayParams);
            pathway.updateDynamics(0.1);
        }
        var strongDepletionLevels = pathway.getTransmitterState().getTransmitterLevels();

        // Strong signals should cause more depletion
        for (int i = 0; i < weakDepletionLevels.length; i++) {
            assertTrue(strongDepletionLevels[i] < weakDepletionLevels[i],
                "Strong signals should deplete transmitters more than weak signals");
        }
    }

    // ============ Primacy Gradient Tests ============

    @Test
    void testPrimacyGradientFormation() {
        // Simulate sequence presentation: strong signal repeatedly
        var signal = createTestPattern(10, 0.8);

        // Record transmitter levels over time
        double[][] levelHistory = new double[20][];

        for (int t = 0; t < 20; t++) {
            pathway.propagate(signal, pathwayParams);
            pathway.updateDynamics(0.1);
            levelHistory[t] = pathway.getTransmitterState().getTransmitterLevels().clone();
        }

        // Early presentations should have higher transmitter levels than later ones
        var earlyLevels = levelHistory[2];  // After 2 presentations
        var lateLevels = levelHistory[15];  // After 15 presentations

        double earlyAvg = average(earlyLevels);
        double lateAvg = average(lateLevels);

        assertTrue(earlyAvg > lateAvg,
            "Primacy gradient: early items should have higher transmitter availability");
    }

    @Test
    void testTemporalChunkBoundaries() {
        // Present sequence with pause in middle
        var signal = createTestPattern(10, 0.8);

        // First chunk - multiple presentations
        for (int i = 0; i < 10; i++) {
            pathway.propagate(signal, pathwayParams);
            pathway.updateDynamics(0.05);
        }

        var endOfFirstChunk = pathway.getTransmitterState().getTransmitterLevels();

        // Pause (rest period)
        var restSignal = createTestPattern(10, 0.0);
        for (int i = 0; i < 30; i++) {
            pathway.propagate(restSignal, pathwayParams);
            pathway.updateDynamics(0.1);
        }

        var afterRest = pathway.getTransmitterState().getTransmitterLevels();

        // Transmitters should recover during rest
        for (int i = 0; i < endOfFirstChunk.length; i++) {
            assertTrue(afterRest[i] > endOfFirstChunk[i],
                "Transmitters should recover during pause");
        }
    }

    // ============ Equation Validation Tests ============

    @Test
    void testTransmitterEquationComponents() {
        // Test equation: dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
        var params = pathway.getTransmitterParameters();
        var signal = 0.7;
        var transmitterLevel = 0.8;

        // Recovery term: ε(1 - Z_i)
        var recovery = params.getEpsilon() * (1.0 - transmitterLevel);
        assertTrue(recovery > 0, "Recovery term should be positive when Z < 1");

        // Depletion term: Z_i(λ * S_i + μ * S_i²)
        var linearDepletion = params.getLambda() * signal;
        var quadraticDepletion = params.getMu() * signal * signal;
        var totalDepletion = transmitterLevel * (linearDepletion + quadraticDepletion);
        assertTrue(totalDepletion > 0, "Depletion should be positive with signal");

        // Net rate of change
        var netChange = recovery - totalDepletion;
        // With typical parameters and signal, depletion dominates
        assertTrue(netChange < 0, "Strong signal should cause net depletion");
    }

    @Test
    void testQuadraticDepletionEffect() {
        // Quadratic term should make strong signals deplete much faster
        var params = pathway.getTransmitterParameters();

        var weakSignal = 0.3;
        var strongSignal = 0.9;

        var weakDepletion = params.computeDepletionRate(weakSignal);
        var strongDepletion = params.computeDepletionRate(strongSignal);

        // Due to quadratic term, strong signal depletion should be more than 3x weak signal
        var ratio = strongDepletion / weakDepletion;
        assertTrue(ratio > 3.0,
            "Quadratic depletion should make ratio > linear ratio (0.9/0.3 = 3.0)");
    }

    @Test
    void testEquilibriumFormula() {
        // Test Z_eq = ε / (ε + λS + μS²)
        var params = pathway.getTransmitterParameters();

        var signals = new double[]{0.1, 0.3, 0.5, 0.7, 0.9};

        for (var signal : signals) {
            var expectedEquilibrium = params.computeEquilibrium(signal);

            // Verify it's in valid range
            assertTrue(expectedEquilibrium >= 0.0 && expectedEquilibrium <= 1.0,
                "Equilibrium should be in [0,1]");

            // Stronger signals should give lower equilibrium
            if (signal > 0.1) {
                var lowSignalEq = params.computeEquilibrium(0.1);
                assertTrue(expectedEquilibrium < lowSignalEq,
                    "Stronger signals should yield lower equilibrium");
            }
        }
    }

    // ============ Numerical Stability Tests ============

    @Test
    void testTransmitterBounds() {
        // Transmitters should always stay in [0, 1]
        var strongInput = createTestPattern(10, 1.0);

        for (int i = 0; i < 100; i++) {
            pathway.propagate(strongInput, pathwayParams);
            pathway.updateDynamics(0.1);

            var levels = pathway.getTransmitterState().getTransmitterLevels();
            for (int j = 0; j < levels.length; j++) {
                assertTrue(levels[j] >= 0.0 && levels[j] <= 1.0,
                    "Transmitter " + j + " out of bounds at step " + i + ": " + levels[j]);
            }
        }
    }

    @Test
    void testTransmitterStability() {
        // Test various input patterns for numerical stability
        var patterns = new double[][]{
            {0.0, 0.0, 0.0, 0.0, 0.0},
            {0.5, 0.5, 0.5, 0.5, 0.5},
            {1.0, 1.0, 1.0, 1.0, 1.0},
            {0.1, 0.3, 0.5, 0.7, 0.9},
            {0.9, 0.7, 0.5, 0.3, 0.1}
        };

        for (var patternData : patterns) {
            pathway.resetDynamics();
            var pattern = new DenseVector(patternData);

            for (int i = 0; i < 50; i++) {
                pathway.propagate(pattern, pathwayParams);
                pathway.updateDynamics(0.1);

                var levels = pathway.getTransmitterState().getTransmitterLevels();
                for (var level : levels) {
                    assertFalse(Double.isNaN(level), "NaN detected in transmitter levels");
                    assertFalse(Double.isInfinite(level), "Infinite value in transmitter levels");
                }
            }
        }
    }

    // ============ Helper Methods ============

    private double computeMagnitude(com.hellblazer.art.core.Pattern pattern) {
        double sum = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            var val = pattern.get(i);
            sum += val * val;
        }
        return Math.sqrt(sum);
    }

    private double average(double[] values) {
        double sum = 0.0;
        for (var val : values) {
            sum += val;
        }
        return sum / values.length;
    }

    // ============ Test Pathway ============

    private static class TestPathway extends AbstractPathway {
        public TestPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }
    }
}