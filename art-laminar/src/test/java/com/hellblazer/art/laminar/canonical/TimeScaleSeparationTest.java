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
 * Tests for time scale separation between fast shunting dynamics and slow transmitter dynamics.
 *
 * Validates that:
 * 1. Shunting dynamics operate on FAST time scale (10-100ms)
 * 2. Transmitter dynamics operate on SLOW time scale (500-5000ms)
 * 3. Fast dynamics reach steady state before slow dynamics show significant change
 * 4. Multi-scale integration properly coordinates the two time scales
 *
 * Based on Kazerounian & Grossberg (2014) Section 3.2.
 *
 * @author Hal Hildebrand
 */
class TimeScaleSeparationTest extends CanonicalCircuitTestBase {

    private ShuntingPathwayDecorator pathway;
    private PathwayParameters pathwayParams;

    @BeforeEach
    void setUp() {
        // Use paper-specified parameters with clear time scale separation
        var transmitterParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.005)      // Very slow recovery (paper default)
            .lambda(0.1)         // Moderate linear depletion
            .mu(0.05)            // Moderate quadratic depletion
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

    // ============ Time Scale Identification ============

    @Test
    void testShuntingTimeScale() {
        // Shunting dynamics should be classified as FAST
        var shuntingDynamics = new com.hellblazer.art.temporal.core.ShuntingDynamics();
        var timeScale = shuntingDynamics.getTimeScale();

        assertEquals(com.hellblazer.art.temporal.core.DynamicalSystem.TimeScale.FAST, timeScale,
            "Shunting dynamics should operate on FAST time scale");

        // FAST time scale is 10-100ms
        assertTrue(timeScale.getTypicalMillis() >= 10 && timeScale.getTypicalMillis() <= 100,
            "Fast time scale should be in 10-100ms range");
    }

    @Test
    void testTransmitterTimeScale() {
        // Transmitter dynamics should be classified as SLOW
        var transmitterDynamics = new com.hellblazer.art.temporal.core.TransmitterDynamics();
        var timeScale = transmitterDynamics.getTimeScale();

        assertEquals(com.hellblazer.art.temporal.core.DynamicalSystem.TimeScale.SLOW, timeScale,
            "Transmitter dynamics should operate on SLOW time scale");

        // SLOW time scale is 500-5000ms
        assertTrue(timeScale.getTypicalMillis() >= 500 && timeScale.getTypicalMillis() <= 5000,
            "Slow time scale should be in 500-5000ms range");
    }

    @Test
    void testTimeScaleRatioIsSignificant() {
        // The ratio between fast and slow time scales should be substantial (>10x)
        var separationFactor = TimeScale.FAST.getSeparationFactor(TimeScale.SLOW);

        assertTrue(separationFactor > 10.0,
            String.format("Time scale separation (%.1fx) should be > 10x for effective separation", separationFactor));
    }

    // ============ Convergence Rate Tests ============

    @Test
    void testShuntingConvergesQuickly() {
        // Shunting should reach ~90% of steady state within 100ms
        var input = createTestPattern(10, 0.8);

        // Initial propagation
        pathway.propagate(input, pathwayParams);

        // Evolve for 100ms (fast time scale)
        for (int i = 0; i < 10; i++) {
            pathway.updateDynamics(0.01);  // 10 steps of 10ms
        }

        var firstState = pathway.getShuntingState();
        var firstActivations = firstState.getActivations();

        // Continue for another 100ms to approach true steady state
        for (int i = 0; i < 10; i++) {
            pathway.updateDynamics(0.01);
        }

        var secondState = pathway.getShuntingState();
        var secondActivations = secondState.getActivations();

        // Check that change is small (already near steady state after first 100ms)
        double maxChange = 0.0;
        for (int i = 0; i < firstActivations.length; i++) {
            double change = Math.abs(secondActivations[i] - firstActivations[i]);
            maxChange = Math.max(maxChange, change);
        }

        assertTrue(maxChange < 0.1,
            String.format("Shunting should be near steady state after 100ms, but changed %.4f", maxChange));
    }

    @Test
    void testTransmitterDepletesSlowly() {
        // Transmitter should show minimal change over 100ms (fast time scale)
        var input = createTestPattern(10, 0.8);

        pathway.propagate(input, pathwayParams);
        var initialTransmitters = pathway.getTransmitterState().getTransmitterLevels();

        // Evolve for 100ms
        for (int i = 0; i < 10; i++) {
            pathway.updateDynamics(0.01);
        }

        var finalTransmitters = pathway.getTransmitterState().getTransmitterLevels();

        // Transmitters should change very little on fast time scale
        double maxChange = 0.0;
        for (int i = 0; i < initialTransmitters.length; i++) {
            double change = Math.abs(finalTransmitters[i] - initialTransmitters[i]);
            maxChange = Math.max(maxChange, change);
        }

        assertTrue(maxChange < 0.05,
            String.format("Transmitters should change minimally (<5%%) over fast time scale, but changed %.4f", maxChange));
    }

    @Test
    void testTransmitterDepletesOnSlowTimeScale() {
        // Over slow time scale (1-5 seconds), transmitters should show significant depletion
        var input = createTestPattern(10, 1.0);  // Strong signal

        pathway.propagate(input, pathwayParams);
        var initialTransmitters = pathway.getTransmitterState().getTransmitterLevels();

        // Evolve for 1 second with continued strong input
        for (int i = 0; i < 100; i++) {
            pathway.propagate(input, pathwayParams);  // Keep signal active
            pathway.updateDynamics(0.01);
        }

        var finalTransmitters = pathway.getTransmitterState().getTransmitterLevels();

        // Transmitters should show meaningful depletion over slow time scale
        double maxChange = 0.0;
        for (int i = 0; i < initialTransmitters.length; i++) {
            double change = initialTransmitters[i] - finalTransmitters[i];  // Should be positive (depletion)
            maxChange = Math.max(maxChange, change);
        }

        assertTrue(maxChange > 0.1,
            String.format("Transmitters should deplete significantly (>10%%) over slow time scale, but only changed %.4f", maxChange));
    }

    // ============ Independence Tests ============

    @Test
    void testShuntingActivationIndependentOfTransmitterRecoveryRate() {
        // Short-term shunting activation should not depend on transmitter recovery rate
        // (since transmitters change slowly)

        // Test with fast recovery
        var fastRecoveryParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.05)  // 10x faster recovery
            .lambda(0.1)
            .mu(0.05)
            .build();

        var pathwayFast = new ShuntingPathwayDecorator(
            new TestPathway("test1", "src", "tgt", PathwayType.BOTTOM_UP),
            createStandardShuntingParameters(),
            fastRecoveryParams,
            TimeScale.FAST
        );

        // Test with slow recovery
        var slowRecoveryParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.005)  // Paper default (slow)
            .lambda(0.1)
            .mu(0.05)
            .build();

        var pathwaySlow = new ShuntingPathwayDecorator(
            new TestPathway("test2", "src", "tgt", PathwayType.BOTTOM_UP),
            createStandardShuntingParameters(),
            slowRecoveryParams,
            TimeScale.FAST
        );

        // Apply same input to both
        var input = createTestPattern(10, 0.7);
        pathwayFast.propagate(input, pathwayParams);
        pathwaySlow.propagate(input, pathwayParams);

        // Evolve for short time (100ms - fast time scale)
        for (int i = 0; i < 10; i++) {
            pathwayFast.updateDynamics(0.01);
            pathwaySlow.updateDynamics(0.01);
        }

        // Shunting activations should be similar despite different transmitter recovery rates
        var fastActivations = pathwayFast.getShuntingState().getActivations();
        var slowActivations = pathwaySlow.getShuntingState().getActivations();

        for (int i = 0; i < fastActivations.length; i++) {
            double diff = Math.abs(fastActivations[i] - slowActivations[i]);
            assertTrue(diff < 0.1,
                String.format("Shunting activations at position %d should be similar (diff=%.4f) on fast time scale", i, diff));
        }
    }

    @Test
    void testTransmitterGatingPreservesShuntingDynamics() {
        // Transmitter gating should modulate but not fundamentally change shunting dynamics

        // Without gating (transmitters at 1.0)
        var noGatingParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.005)
            .lambda(0.0)  // No depletion
            .mu(0.0)
            .initialLevel(1.0)
            .build();

        var pathwayNoGating = new ShuntingPathwayDecorator(
            new TestPathway("test1", "src", "tgt", PathwayType.BOTTOM_UP),
            createStandardShuntingParameters(),
            noGatingParams,
            TimeScale.FAST
        );

        // With gating (transmitters can deplete)
        var input = createTestPattern(10, 0.8);

        pathwayNoGating.propagate(input, pathwayParams);
        pathway.propagate(input, pathwayParams);

        // Evolve briefly
        for (int i = 0; i < 5; i++) {
            pathwayNoGating.updateDynamics(0.01);
            pathway.updateDynamics(0.01);
        }

        // Activations with gating should be scaled version of no-gating
        var noGatingActivations = pathwayNoGating.getShuntingState().getActivations();
        var gatedActivations = pathway.getShuntingState().getActivations();

        // Check that the pattern is preserved (correlation should be high)
        double correlation = computeCorrelation(noGatingActivations, gatedActivations);
        assertTrue(correlation > 0.95,
            String.format("Gated activations should preserve shunting pattern (correlation=%.4f)", correlation));
    }

    // ============ Multi-Scale Integration Tests ============

    @Test
    void testSequentialTimeScaleIntegration() {
        // Test that fast dynamics complete before slow dynamics show effect
        // Use moderate input to avoid immediate transmitter depletion
        var input = createTestPattern(10, 0.5);  // Moderate strength

        // Phase 1: Fast shunting convergence (0-100ms)
        pathway.propagate(input, pathwayParams);
        var transmittersAtStart = pathway.getTransmitterState().getTransmitterLevels();
        var shuntingAtStart = pathway.getShuntingState().getActivations();

        // Let shunting converge while measuring transmitter change
        for (int i = 0; i < 10; i++) {
            pathway.propagate(input, pathwayParams);  // Maintain input signal
            pathway.updateDynamics(0.01);
        }

        var shuntingAt100ms = pathway.getShuntingState().getActivations();
        var transmittersAt100ms = pathway.getTransmitterState().getTransmitterLevels();

        // Shunting should have built up significant activation
        assertTrue(hasSignificantActivation(shuntingAt100ms),
            String.format("Shunting should have converged to significant activation, max=%.4f",
                getMaxActivation(shuntingAt100ms)));

        // Shunting should have changed dramatically (from ~0 to significant)
        double shuntingChange = computeMaxDifference(shuntingAtStart, shuntingAt100ms);
        assertTrue(shuntingChange > 0.05,
            String.format("Shunting should have changed significantly (%.4f) on fast time scale", shuntingChange));

        // Transmitters change slower than shunting
        double transmitterChange100ms = computeMaxDifference(transmittersAtStart, transmittersAt100ms);

        // Phase 2: Slow transmitter depletion (100ms-1000ms)
        // Continue with moderate input to observe transmitter depletion
        for (int i = 0; i < 90; i++) {  // 900ms more
            pathway.propagate(input, pathwayParams);  // Maintain signal
            pathway.updateDynamics(0.01);
        }

        var transmittersAt1s = pathway.getTransmitterState().getTransmitterLevels();

        double transmitterChange1s = computeMaxDifference(transmittersAtStart, transmittersAt1s);

        // Transmitter depletion should accelerate over time (cumulative effect)
        assertTrue(transmitterChange1s > transmitterChange100ms,
            String.format("Transmitters should deplete more over 1s (%.4f) than over 100ms (%.4f)",
                transmitterChange1s, transmitterChange100ms));

        // The key property: shunting converges quickly, transmitters change slowly and cumulatively
        System.out.printf("Time scale separation validated:%n");
        System.out.printf("  Shunting change (100ms): %.4f%n", shuntingChange);
        System.out.printf("  Transmitter change (100ms): %.4f%n", transmitterChange100ms);
        System.out.printf("  Transmitter change (1000ms): %.4f%n", transmitterChange1s);
    }

    // ============ Helper Methods ============

    private boolean hasSignificantActivation(double[] activations) {
        for (var act : activations) {
            if (act > 0.05) return true;  // Lower threshold to account for gating
        }
        return false;
    }

    private double getMaxActivation(double[] activations) {
        double max = 0.0;
        for (var act : activations) {
            max = Math.max(max, act);
        }
        return max;
    }

    private double computeMaxDifference(double[] a, double[] b) {
        double maxDiff = 0.0;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        return maxDiff;
    }

    private double computeCorrelation(double[] x, double[] y) {
        // Compute Pearson correlation coefficient
        double meanX = 0.0, meanY = 0.0;
        for (int i = 0; i < x.length; i++) {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= x.length;
        meanY /= y.length;

        double numerator = 0.0;
        double denomX = 0.0;
        double denomY = 0.0;

        for (int i = 0; i < x.length; i++) {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            numerator += dx * dy;
            denomX += dx * dx;
            denomY += dy * dy;
        }

        if (denomX == 0 || denomY == 0) return 0.0;
        return numerator / Math.sqrt(denomX * denomY);
    }

    // ============ Test Pathway ============

    private static class TestPathway extends AbstractPathway {
        public TestPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }
    }
}