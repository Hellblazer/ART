package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.temporal.core.ShuntingParameters;
import com.hellblazer.art.temporal.core.ShuntingState;
import com.hellblazer.art.temporal.core.TransmitterParameters;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Base test class for canonical laminar circuit integration testing.
 * Provides common utilities for testing the integration of temporal dynamics
 * into laminar pathways.
 *
 * @author Hal Hildebrand
 */
public abstract class CanonicalCircuitTestBase {

    protected static final double EPSILON = 1e-6;
    protected static final double DEFAULT_TIME_STEP = 0.01; // 10ms
    protected static final double DEFAULT_VIGILANCE = 0.8;
    protected static final double DEFAULT_LEARNING_RATE = 0.5;

    // Time scale constants from paper
    protected static final double FAST_TIME_SCALE = 0.01;    // 10ms - neural activation
    protected static final double MEDIUM_TIME_SCALE = 0.1;   // 100ms - attention shifts
    protected static final double SLOW_TIME_SCALE = 1.0;     // 1s - learning updates
    protected static final double VERY_SLOW_TIME_SCALE = 5.0; // 5s - memory consolidation

    @BeforeEach
    void baseSetUp() {
        // Base setup that all canonical circuit tests need
    }

    // ============ Test Utility Methods ============

    /**
     * Create a default test pattern with given dimension.
     */
    protected Pattern createTestPattern(int dimension, double value) {
        var data = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            data[i] = value;
        }
        return new DenseVector(data);
    }

    /**
     * Create a normalized random test pattern.
     */
    protected Pattern createRandomPattern(int dimension) {
        var data = new double[dimension];
        var sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            data[i] = Math.random();
            sum += data[i];
        }
        // Normalize to [0, 1]
        for (int i = 0; i < dimension; i++) {
            data[i] /= sum;
        }
        return new DenseVector(data);
    }

    /**
     * Create complement-coded pattern [x, 1-x] as used in FuzzyART.
     */
    protected Pattern createComplementCodedPattern(Pattern input) {
        var values = input.toArray();
        var complementCoded = new double[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            complementCoded[i] = values[i];
            complementCoded[i + values.length] = 1.0 - values[i];
        }
        return new DenseVector(complementCoded);
    }

    // ============ Shunting Dynamics Test Utilities ============

    /**
     * Create standard shunting parameters for testing.
     * Based on Grossberg's canonical parameters.
     */
    protected ShuntingParameters createStandardShuntingParameters() {
        return ShuntingParameters.builder()
            .decayRate(0.1)
            .upperBound(1.0)
            .lowerBound(0.0)
            .lateralInhibition(0.5)
            .selfExcitation(0.2)
            .build();
    }

    /**
     * Create shunting state from pattern.
     */
    protected ShuntingState createShuntingState(Pattern pattern) {
        return new ShuntingState(pattern.toArray(), pattern.toArray());
    }

    /**
     * Verify shunting equation compliance:
     * dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij
     */
    protected void assertShuntingEquation(
        double activation,
        double excitation,
        double inhibition,
        double decayRate,
        double upperBound,
        double expectedDerivative,
        String message
    ) {
        var decay = -decayRate * activation;
        var excitatory = (upperBound - activation) * excitation;
        var inhibitory = -activation * inhibition;
        var actualDerivative = decay + excitatory + inhibitory;

        assertEquals(expectedDerivative, actualDerivative, EPSILON,
            message + " - Shunting equation verification failed");
    }

    /**
     * Verify equilibrium stability condition.
     * At equilibrium: dX/dt = 0
     */
    protected void assertShuntingEquilibrium(
        ShuntingState state,
        ShuntingParameters params,
        String message
    ) {
        var activations = state.getActivations();
        for (int i = 0; i < activations.length; i++) {
            var xi = activations[i];
            var si = state.getExcitatoryInput(i);

            // At equilibrium: X_eq = B * S / (A + S + I)
            var decay = params.getDecayRate();
            var bound = params.getUpperBound();

            // Simplified equilibrium check (no lateral inhibition)
            if (si > 0) {
                var expectedEquilibrium = bound * si / (decay + si);
                assertEquals(expectedEquilibrium, xi, EPSILON,
                    message + " - Equilibrium condition failed at index " + i);
            }
        }
    }

    // ============ Transmitter Dynamics Test Utilities ============

    /**
     * Create standard transmitter parameters for testing.
     */
    protected TransmitterParameters createStandardTransmitterParameters() {
        return TransmitterParameters.builder()
            .epsilon(0.005)
            .lambda(0.1)
            .mu(0.05)
            .depletionThreshold(0.2)
            .initialLevel(1.0)
            .enableQuadratic(true)
            .build();
    }

    /**
     * Verify transmitter gating equation:
     * dZ_i/dt = -C_i * Z_i + D_i * (1 - Z_i) * X_i
     */
    protected void assertTransmitterEquation(
        double transmitterLevel,
        double activation,
        double decayRate,
        double releaseRate,
        double expectedDerivative,
        String message
    ) {
        var decay = -decayRate * transmitterLevel;
        var release = releaseRate * (1.0 - transmitterLevel) * activation;
        var actualDerivative = decay + release;

        assertEquals(expectedDerivative, actualDerivative, EPSILON,
            message + " - Transmitter equation verification failed");
    }

    // ============ Time Scale Separation Utilities ============

    /**
     * Verify time scale separation principle.
     * Fast dynamics should settle before slow dynamics change significantly.
     */
    protected void assertTimeScaleSeparation(
        double fastDynamicChange,
        double slowDynamicChange,
        double separationFactor,
        String message
    ) {
        assertTrue(Math.abs(fastDynamicChange) > separationFactor * Math.abs(slowDynamicChange),
            message + " - Time scale separation violated: fast change should be >> slow change");
    }

    /**
     * Verify multiple time scales follow proper hierarchy.
     */
    protected void assertTimeScaleHierarchy(
        double[] changeRates,
        String[] scaleNames,
        String message
    ) {
        for (int i = 0; i < changeRates.length - 1; i++) {
            assertTrue(changeRates[i] > changeRates[i + 1],
                message + String.format(" - Time scale hierarchy violated: %s (%f) should be faster than %s (%f)",
                    scaleNames[i], changeRates[i], scaleNames[i + 1], changeRates[i + 1]));
        }
    }

    // ============ Pattern Comparison Utilities ============

    /**
     * Assert two patterns are approximately equal within epsilon.
     */
    protected void assertPatternsEqual(Pattern expected, Pattern actual, String message) {
        assertPatternsEqual(expected, actual, EPSILON, message);
    }

    /**
     * Assert two patterns are approximately equal within given tolerance.
     */
    protected void assertPatternsEqual(Pattern expected, Pattern actual, double tolerance, String message) {
        assertNotNull(expected, message + " - Expected pattern is null");
        assertNotNull(actual, message + " - Actual pattern is null");

        var expectedArray = expected.toArray();
        var actualArray = actual.toArray();

        assertEquals(expectedArray.length, actualArray.length,
            message + " - Pattern dimensions differ");

        for (int i = 0; i < expectedArray.length; i++) {
            assertEquals(expectedArray[i], actualArray[i], tolerance,
                message + " - Pattern values differ at index " + i);
        }
    }

    /**
     * Assert pattern is normalized (sum = 1 or L2 norm = 1).
     */
    protected void assertPatternNormalized(Pattern pattern, NormType normType, String message) {
        var values = pattern.toArray();
        var sum = 0.0;

        for (var value : values) {
            sum += normType == NormType.L1 ? value : value * value;
        }

        if (normType == NormType.L2) {
            sum = Math.sqrt(sum);
        }

        assertEquals(1.0, sum, EPSILON, message + " - Pattern not normalized");
    }

    /**
     * Assert pattern values are bounded.
     */
    protected void assertPatternBounded(Pattern pattern, double lowerBound, double upperBound, String message) {
        var values = pattern.toArray();
        for (int i = 0; i < values.length; i++) {
            assertTrue(values[i] >= lowerBound && values[i] <= upperBound,
                message + String.format(" - Pattern value %f at index %d is outside bounds [%f, %f]",
                    values[i], i, lowerBound, upperBound));
        }
    }

    // ============ Numerical Stability Utilities ============

    /**
     * Assert no NaN or infinite values in pattern.
     */
    protected void assertPatternStable(Pattern pattern, String message) {
        var values = pattern.toArray();
        for (int i = 0; i < values.length; i++) {
            assertFalse(Double.isNaN(values[i]),
                message + " - NaN detected at index " + i);
            assertFalse(Double.isInfinite(values[i]),
                message + " - Infinite value detected at index " + i);
        }
    }

    /**
     * Assert convergence by comparing consecutive states.
     */
    protected void assertConvergence(
        Pattern previous,
        Pattern current,
        double threshold,
        String message
    ) {
        var prevArray = previous.toArray();
        var currArray = current.toArray();

        double maxChange = 0.0;
        for (int i = 0; i < prevArray.length; i++) {
            var change = Math.abs(currArray[i] - prevArray[i]);
            maxChange = Math.max(maxChange, change);
        }

        assertTrue(maxChange < threshold,
            message + String.format(" - Not converged: max change %f exceeds threshold %f",
                maxChange, threshold));
    }

    // ============ Equation Validation Utilities ============

    /**
     * Validate against paper equations by symbolic comparison.
     */
    protected void assertPaperEquation(
        String equationName,
        double computed,
        double expected,
        String citation
    ) {
        assertEquals(expected, computed, EPSILON,
            String.format("Equation '%s' from %s failed validation", equationName, citation));
    }

    /**
     * Assert derivative computation matches analytical derivative.
     */
    protected void assertDerivativeCorrect(
        double numericalDerivative,
        double analyticalDerivative,
        String message
    ) {
        assertEquals(analyticalDerivative, numericalDerivative, EPSILON * 10,
            message + " - Numerical derivative doesn't match analytical derivative");
    }

    // ============ Supporting Types ============

    protected enum NormType {
        L1, L2
    }

    /**
     * Configuration for regression tests.
     */
    protected static class RegressionConfig {
        public final Pattern input;
        public final Pattern expectedOutput;
        public final String description;

        public RegressionConfig(Pattern input, Pattern expectedOutput, String description) {
            this.input = input;
            this.expectedOutput = expectedOutput;
            this.description = description;
        }
    }
}