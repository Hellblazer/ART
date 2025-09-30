package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.impl.AbstractPathway;
import com.hellblazer.art.laminar.impl.DefaultPathwayParameters;
import com.hellblazer.art.laminar.parameters.PathwayParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Backward compatibility regression tests ensuring that temporal dynamics integration
 * doesn't break existing pathway functionality.
 *
 * Tests verify:
 * 1. Undecorated pathways work exactly as before
 * 2. Decorated pathways with disabled dynamics behave like undecorated
 * 3. Basic pathway operations remain functional
 * 4. Pattern propagation semantics unchanged
 *
 * @author Hal Hildebrand
 */
class BackwardCompatibilityTest extends CanonicalCircuitTestBase {

    private TestPathway basicPathway;
    private PathwayParameters params;

    @BeforeEach
    void setUp() {
        basicPathway = new TestPathway("test", "src", "tgt", PathwayType.BOTTOM_UP);
        params = new DefaultPathwayParameters(1.0, 0.5, true);
    }

    // ============ Undecorated Pathway Tests ============

    @Test
    void testBasicPathwayPropagation() {
        // Basic pathway should propagate patterns without temporal dynamics
        var input = createTestPattern(10, 0.7);

        var output = basicPathway.propagate(input, params);

        assertNotNull(output, "Output should not be null");
        assertEquals(input.dimension(), output.dimension(),
            "Output dimension should match input dimension");
    }

    @Test
    void testBasicPathwayIdentity() {
        // Without any transformation logic, basic pathway returns input
        var input = createTestPattern(10, 0.8);

        var output1 = basicPathway.propagate(input, params);
        var output2 = basicPathway.propagate(input, params);

        // Multiple propagations should give consistent results
        assertPatternsEqual(output1, output2, 1e-10,
            "Basic pathway should be deterministic");
    }

    @Test
    void testBasicPathwayMultiplePatterns() {
        // Test with various patterns
        var patterns = new Pattern[]{
            createTestPattern(10, 0.3),
            createTestPattern(10, 0.5),
            createTestPattern(10, 0.9)
        };

        for (var pattern : patterns) {
            var output = basicPathway.propagate(pattern, params);
            assertNotNull(output, "All patterns should propagate successfully");
        }
    }

    // ============ Decorated vs Undecorated Comparison ============

    @Test
    void testDecoratedPathwayWithDisabledDynamics() {
        // Decorated pathway with dynamics disabled should behave exactly like undecorated
        var shuntingParams = createStandardShuntingParameters();
        var transmitterParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.005)
            .lambda(0.0)
            .mu(0.0)
            .build();

        var decorated = new ShuntingPathwayDecorator(
            basicPathway,
            shuntingParams,
            transmitterParams,
            TimeScale.FAST
        );

        // Disable temporal dynamics
        decorated.setTemporalDynamicsEnabled(false);

        var input = createTestPattern(10, 0.7);

        var basicOutput = basicPathway.propagate(input, params);
        var decoratedOutput = decorated.propagate(input, params);

        // With dynamics disabled, outputs should be identical
        assertPatternsEqual(basicOutput, decoratedOutput, 1e-10,
            "Decorated pathway with disabled dynamics should match basic exactly");
    }

    @Test
    void testDecoratedPathwayPreservesInterface() {
        // Decorated pathway should implement the same interface
        var decorated = createDecoratedPathway(basicPathway);

        // Should still work as a Pathway
        assertTrue(decorated instanceof com.hellblazer.art.laminar.core.Pathway,
            "Decorated pathway should implement Pathway interface");

        var input = createTestPattern(10, 0.5);
        var output = decorated.propagate(input, params);

        assertNotNull(output, "Decorated pathway should propagate patterns");
    }

    @Test
    void testDecoratedPathwayMetadata() {
        // Metadata should be preserved through decoration
        var decorated = createDecoratedPathway(basicPathway);

        assertEquals(basicPathway.getId(), decorated.getId(),
            "Pathway ID should be preserved");
        assertEquals(basicPathway.getSourceLayerId(), decorated.getSourceLayerId(),
            "Source layer ID should be preserved");
        assertEquals(basicPathway.getTargetLayerId(), decorated.getTargetLayerId(),
            "Target layer ID should be preserved");
        assertEquals(basicPathway.getType(), decorated.getType(),
            "Pathway type should be preserved");
    }

    // ============ Delegation Verification ============

    @Test
    void testDecoratorDelegatesBasicOperations() {
        // Decorator should delegate basic operations to wrapped pathway
        var callCountingPathway = new CallCountingPathway("test", "src", "tgt", PathwayType.BOTTOM_UP);
        var decorated = createDecoratedPathway(callCountingPathway);

        var input = createTestPattern(10, 0.6);

        assertEquals(0, callCountingPathway.propagateCallCount,
            "No calls before propagation");

        decorated.propagate(input, params);

        assertEquals(1, callCountingPathway.propagateCallCount,
            "Should delegate propagate call to wrapped pathway");
    }

    @Test
    void testDecoratorTransparency() {
        // Decorator should be transparent for non-temporal operations
        var customPathway = new CustomTransformPathway("test", "src", "tgt", PathwayType.BOTTOM_UP);
        var decorated = createDecoratedPathway(customPathway);

        var input = createTestPattern(10, 0.8);

        var basicOutput = customPathway.propagate(input, params);
        var decoratedOutput = decorated.propagate(input, params);

        // Custom transformation should still work through decorator
        // (though values may differ due to temporal dynamics)
        assertEquals(basicOutput.dimension(), decoratedOutput.dimension(),
            "Decorator should preserve dimensional transformations");
    }

    // ============ State Isolation Tests ============

    @Test
    void testMultipleDecoratedPathwaysIndependent() {
        // Multiple decorated pathways should have independent state
        var pathway1 = new TestPathway("p1", "src", "tgt", PathwayType.BOTTOM_UP);
        var pathway2 = new TestPathway("p2", "src", "tgt", PathwayType.BOTTOM_UP);

        var decorated1 = createDecoratedPathway(pathway1);
        var decorated2 = createDecoratedPathway(pathway2);

        var input = createTestPattern(10, 0.9);

        // Initialize both pathways
        decorated2.propagate(input, params);

        // Propagate through first pathway multiple times
        for (int i = 0; i < 10; i++) {
            decorated1.propagate(input, params);
            decorated1.updateDynamics(0.01);
        }

        // Second pathway should still have fresh state
        var state1 = decorated1.getTransmitterState().getTransmitterLevels();
        var state2 = decorated2.getTransmitterState().getTransmitterLevels();

        // First pathway transmitters should be somewhat depleted
        assertTrue(average(state1) < 1.0, "First pathway transmitters should be depleted");

        // Second pathway transmitters should be at full strength
        assertTrue(average(state2) >= 0.99, "Second pathway transmitters should be fresh");
    }

    @Test
    void testResetRestoresInitialState() {
        // Reset should restore decorator to fresh initial state (transmitters at 1.0)
        var decorated = createDecoratedPathway(basicPathway);
        var input = createTestPattern(10, 0.8);

        // Cause some depletion
        for (int i = 0; i < 20; i++) {
            decorated.propagate(input, params);
            decorated.updateDynamics(0.01);
        }

        var depletedTransmitters = decorated.getTransmitterState().getTransmitterLevels();
        assertTrue(average(depletedTransmitters) < 1.0,
            "Transmitters should be depleted after use");

        // Reset
        decorated.resetDynamics();

        var resetTransmitters = decorated.getTransmitterState().getTransmitterLevels();

        // After reset, transmitters should be back to full strength (1.0)
        for (var level : resetTransmitters) {
            assertEquals(1.0, level, 1e-10,
                "Reset should restore transmitters to full strength (1.0)");
        }
    }

    // ============ Parameter Compatibility Tests ============

    @Test
    void testVariousParameterConfigurations() {
        // Test that various parameter configurations work
        var paramConfigs = new PathwayParameters[]{
            new DefaultPathwayParameters(1.0, 0.5, true),
            new DefaultPathwayParameters(0.5, 0.3, false),
            new DefaultPathwayParameters(2.0, 0.8, true)
        };

        var decorated = createDecoratedPathway(basicPathway);
        var input = createTestPattern(10, 0.7);

        for (var paramConfig : paramConfigs) {
            var output = decorated.propagate(input, paramConfig);
            assertNotNull(output,
                "All parameter configurations should work");
        }
    }

    // ============ Helper Methods ============

    private ShuntingPathwayDecorator createDecoratedPathway(AbstractPathway pathway) {
        var shuntingParams = createStandardShuntingParameters();
        var transmitterParams = com.hellblazer.art.temporal.core.TransmitterParameters.builder()
            .epsilon(0.05)  // Faster for testing
            .lambda(0.1)
            .mu(0.05)
            .build();

        return new ShuntingPathwayDecorator(
            pathway,
            shuntingParams,
            transmitterParams,
            TimeScale.FAST
        );
    }

    private double average(double[] values) {
        double sum = 0.0;
        for (var val : values) {
            sum += val;
        }
        return sum / values.length;
    }

    // ============ Test Pathway Implementations ============

    private static class TestPathway extends AbstractPathway {
        public TestPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }
    }

    private static class CallCountingPathway extends AbstractPathway {
        public int propagateCallCount = 0;

        public CallCountingPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }

        @Override
        public Pattern propagate(Pattern signal, PathwayParameters parameters) {
            propagateCallCount++;
            return super.propagate(signal, parameters);
        }
    }

    private static class CustomTransformPathway extends AbstractPathway {
        public CustomTransformPathway(String id, String sourceLayerId, String targetLayerId, PathwayType type) {
            super(id, sourceLayerId, targetLayerId, type);
        }

        @Override
        public Pattern propagate(Pattern signal, PathwayParameters parameters) {
            // Custom transformation: scale by 2.0
            var values = new double[signal.dimension()];
            for (int i = 0; i < signal.dimension(); i++) {
                values[i] = signal.get(i) * 2.0;
            }
            return new DenseVector(values);
        }
    }
}