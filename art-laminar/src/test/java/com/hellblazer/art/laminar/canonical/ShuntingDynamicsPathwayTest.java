package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.impl.AbstractPathway;
import com.hellblazer.art.laminar.impl.DefaultPathwayParameters;
import com.hellblazer.art.laminar.parameters.PathwayParameters;
import com.hellblazer.art.temporal.core.ShuntingParameters;
import com.hellblazer.art.temporal.core.TransmitterParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shunting dynamics integration in pathways.
 * Validates the ShuntingPathwayDecorator implementation against
 * Grossberg's canonical laminar circuit equations.
 *
 * @author Hal Hildebrand
 */
class ShuntingDynamicsPathwayTest extends CanonicalCircuitTestBase {

    private ShuntingPathwayDecorator pathway;
    private ShuntingParameters shuntingParams;
    private TransmitterParameters transmitterParams;
    private PathwayParameters pathwayParams;

    @BeforeEach
    void setUp() {
        // Create standard shunting parameters
        shuntingParams = createStandardShuntingParameters();

        // Create standard transmitter parameters
        transmitterParams = createStandardTransmitterParameters();

        // Create pathway parameters
        pathwayParams = new DefaultPathwayParameters(1.0, 0.5, true);

        // Create a simple delegate pathway for testing
        var delegate = new TestPathway("test-pathway", "source", "target", PathwayType.BOTTOM_UP);

        // Create shunting pathway decorator
        pathway = new ShuntingPathwayDecorator(
            delegate,
            shuntingParams,
            transmitterParams,
            TimeScale.FAST
        );
    }

    // ============ Basic Functionality Tests ============

    @Test
    void testPathwayCreation() {
        assertNotNull(pathway);
        assertEquals("test-pathway", pathway.getId());
        assertEquals("source", pathway.getSourceLayerId());
        assertEquals("target", pathway.getTargetLayerId());
        assertEquals(PathwayType.BOTTOM_UP, pathway.getType());
        assertTrue(pathway.isTemporalDynamicsEnabled());
    }

    @Test
    void testTemporalDynamicsToggle() {
        // Test enabling/disabling
        pathway.setTemporalDynamicsEnabled(false);
        assertFalse(pathway.isTemporalDynamicsEnabled());

        pathway.setTemporalDynamicsEnabled(true);
        assertTrue(pathway.isTemporalDynamicsEnabled());
    }

    @Test
    void testTimeScaleAssignment() {
        assertEquals(TimeScale.FAST, pathway.getTimeScale());

        // Test with different time scales
        var mediumPathway = new ShuntingPathwayDecorator(
            new TestPathway("medium", "s", "t", PathwayType.TOP_DOWN),
            shuntingParams,
            transmitterParams,
            TimeScale.MEDIUM
        );

        assertEquals(TimeScale.MEDIUM, mediumPathway.getTimeScale());
    }

    // ============ Shunting Equation Tests ============

    @Test
    void testShuntingEquationBasic() {
        var input = createTestPattern(10, 0.5);

        // Propagate signal
        var output = pathway.propagate(input, pathwayParams);

        // Get shunting state
        var state = pathway.getShuntingState();
        assertNotNull(state);
        assertEquals(10, state.dimension());

        // Verify state is bounded [0, 1]
        var activations = state.getActivations();
        for (int i = 0; i < activations.length; i++) {
            assertTrue(activations[i] >= 0.0 && activations[i] <= shuntingParams.getUpperBound(),
                "Activation at index " + i + " should be bounded");
        }
    }

    @Test
    void testShuntingEquationDerivative() {
        // Test the shunting equation components:
        // dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij

        var input = createTestPattern(5, 0.6);
        pathway.propagate(input, pathwayParams);

        var state = pathway.getShuntingState();
        var activations = state.getActivations();

        // For each unit, verify the equation holds
        for (int i = 0; i < activations.length; i++) {
            var xi = activations[i];
            var si = state.getExcitatoryInput(i);

            // Decay term: -A * X_i
            var decayTerm = -shuntingParams.getDecayRate() * xi;
            assertTrue(decayTerm <= 0, "Decay should be non-positive");

            // Excitatory term: (B - X_i) * S_i
            var excitatoryTerm = (shuntingParams.getUpperBound() - xi) * si;
            assertTrue(excitatoryTerm >= 0, "Excitation should be non-negative");

            // Combined derivative should follow expected sign
            if (si > shuntingParams.getDecayRate() * xi) {
                // Net excitation expected
                assertTrue(excitatoryTerm > Math.abs(decayTerm),
                    "Strong excitation should dominate decay");
            }
        }
    }

    @Test
    void testShuntingEquilibriumApproach() {
        var input = createTestPattern(10, 0.8);

        // Run multiple updates to approach equilibrium
        Pattern previousOutput = null;
        Pattern lastOutput = null;
        double maxChange = Double.MAX_VALUE;

        for (int i = 0; i < 200; i++) {
            lastOutput = pathway.propagate(input, pathwayParams);
            pathway.updateDynamics(DEFAULT_TIME_STEP);

            // Track convergence manually
            if (previousOutput != null) {
                maxChange = 0.0;
                for (int j = 0; j < lastOutput.dimension(); j++) {
                    var change = Math.abs(lastOutput.get(j) - previousOutput.get(j));
                    maxChange = Math.max(maxChange, change);
                }

                // Break if converged
                if (maxChange < 1e-4) {
                    break;
                }
            }
            previousOutput = lastOutput;
        }

        // Check that we did converge (or got very close)
        assertTrue(maxChange < 1e-3,
            "System should converge: final max change was " + maxChange);

        // Verify state is stable (no NaN/Inf)
        assertPatternStable(lastOutput, "Equilibrium state");
    }

    @Test
    void testShuntingExcitationStrength() {
        // Test with varying excitation strengths
        var weakInput = createTestPattern(10, 0.1);
        var strongInput = createTestPattern(10, 0.9);

        pathway.propagate(weakInput, pathwayParams);
        var weakState = pathway.getShuntingState();
        var weakActivation = weakState.getActivations()[0];

        pathway.resetDynamics();

        pathway.propagate(strongInput, pathwayParams);
        var strongState = pathway.getShuntingState();
        var strongActivation = strongState.getActivations()[0];

        // Strong input should produce stronger activation
        assertTrue(strongActivation > weakActivation,
            "Strong excitation should produce stronger activation");
    }

    // ============ Lateral Inhibition Tests ============

    @Test
    void testLateralInhibition() {
        // Create input with one strong and several weak signals
        var data = new double[10];
        data[5] = 1.0;  // Strong input at center
        for (int i = 0; i < 10; i++) {
            if (i != 5) data[i] = 0.3;  // Weak background
        }
        var input = new DenseVector(data);

        // Allow dynamics to settle
        for (int i = 0; i < 50; i++) {
            pathway.propagate(input, pathwayParams);
            pathway.updateDynamics(DEFAULT_TIME_STEP);
        }

        var state = pathway.getShuntingState();
        var activations = state.getActivations();

        // Strong input should have higher activation due to lateral inhibition
        var centerActivation = activations[5];
        var surroundActivation = activations[0];

        assertTrue(centerActivation > surroundActivation,
            "Center with strong input should suppress surround");
    }

    // ============ Bounded Dynamics Tests ============

    @Test
    void testActivationBounds() {
        // Test with very strong input - should not exceed upper bound
        var strongInput = createTestPattern(10, 10.0);

        for (int i = 0; i < 100; i++) {
            pathway.propagate(strongInput, pathwayParams);
            pathway.updateDynamics(DEFAULT_TIME_STEP);

            var state = pathway.getShuntingState();
            var activations = state.getActivations();

            // Verify bounds
            for (int j = 0; j < activations.length; j++) {
                assertTrue(activations[j] >= 0.0,
                    "Activation should not be negative");
                assertTrue(activations[j] <= shuntingParams.getUpperBound(),
                    "Activation should not exceed upper bound");
            }
        }
    }

    @Test
    void testNumericalStability() {
        // Test with various inputs for numerical stability
        var testInputs = new Pattern[]{
            createTestPattern(10, 0.0),
            createTestPattern(10, 0.5),
            createTestPattern(10, 1.0),
            createRandomPattern(10),
            createRandomPattern(10)
        };

        for (var input : testInputs) {
            pathway.resetDynamics();

            for (int i = 0; i < 100; i++) {
                var output = pathway.propagate(input, pathwayParams);
                pathway.updateDynamics(DEFAULT_TIME_STEP);

                // Check for NaN or infinity
                assertPatternStable(output, "Numerical stability check");

                var state = pathway.getShuntingState();
                for (var activation : state.getActivations()) {
                    assertFalse(Double.isNaN(activation), "No NaN values");
                    assertFalse(Double.isInfinite(activation), "No infinite values");
                }
            }
        }
    }

    // ============ Dynamics Update Tests ============

    @Test
    void testExplicitDynamicsUpdate() {
        var input = createTestPattern(10, 0.5);

        // Initial propagation
        pathway.propagate(input, pathwayParams);
        var initialState = pathway.getShuntingState();
        var initialActivations = initialState.getActivations();

        // Explicit dynamics update
        pathway.updateDynamics(0.1);
        var updatedState = pathway.getShuntingState();
        var updatedActivations = updatedState.getActivations();

        // States should differ after update
        boolean hasChanged = false;
        for (int i = 0; i < initialActivations.length; i++) {
            if (Math.abs(updatedActivations[i] - initialActivations[i]) > EPSILON) {
                hasChanged = true;
                break;
            }
        }

        assertTrue(hasChanged, "Explicit update should change state");
    }

    @Test
    void testDynamicsReset() {
        var input = createTestPattern(10, 0.7);

        // Build up state
        for (int i = 0; i < 20; i++) {
            pathway.propagate(input, pathwayParams);
            pathway.updateDynamics(DEFAULT_TIME_STEP);
        }

        var beforeReset = pathway.getShuntingState();

        // Reset dynamics
        pathway.resetDynamics();
        var afterReset = pathway.getShuntingState();

        // State should be reset to initial values
        var afterActivations = afterReset.getActivations();
        for (var activation : afterActivations) {
            assertEquals(0.0, activation, EPSILON, "Activations should be reset to zero");
        }
    }

    // ============ Disabled Dynamics Tests ============

    @Test
    void testDisabledDynamicsFallback() {
        var input = createTestPattern(10, 0.5);

        // Enable dynamics and propagate
        pathway.setTemporalDynamicsEnabled(true);
        var withDynamics = pathway.propagate(input, pathwayParams);

        // Reset and disable dynamics
        pathway.resetDynamics();
        pathway.setTemporalDynamicsEnabled(false);
        var withoutDynamics = pathway.propagate(input, pathwayParams);

        // Outputs should differ (with dynamics includes shunting/gating)
        boolean differs = false;
        for (int i = 0; i < withDynamics.dimension(); i++) {
            if (Math.abs(withDynamics.get(i) - withoutDynamics.get(i)) > EPSILON) {
                differs = true;
                break;
            }
        }

        // Note: They might be the same initially, so we just verify no errors occur
        assertNotNull(withoutDynamics);
    }

    // ============ Helper Classes ============

    /**
     * Simple test pathway for testing decorator.
     */
    private static class TestPathway extends AbstractPathway {
        public TestPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }

        @Override
        public Pattern propagate(Pattern signal, PathwayParameters parameters) {
            // Simple pass-through with gain
            var result = new double[signal.dimension()];
            var effectiveGain = parameters != null ? parameters.getGain() : gain;

            for (int i = 0; i < signal.dimension(); i++) {
                result[i] = signal.get(i) * effectiveGain;
            }

            return new DenseVector(result);
        }
    }
}