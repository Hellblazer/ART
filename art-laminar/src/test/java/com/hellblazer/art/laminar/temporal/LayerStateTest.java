package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for LayerState record and its utility methods.
 *
 * @author Hal Hildebrand
 */
@DisplayName("Layer State Tests")
class LayerStateTest {

    @Test
    @DisplayName("Create state from activation only")
    void testFromActivation() {
        var activation = createPattern(0.8);
        var state = LayerState.fromActivation(activation, 1.0);

        assertEquals(activation, state.currentActivation());
        assertNull(state.temporalContext());
        assertEquals(1.0, state.timestamp());
        assertFalse(state.hasTemporalContext());
    }

    @Test
    @DisplayName("Create state with both activation and context")
    void testWithContext() {
        var activation = createPattern(0.8);
        var context = createPattern(0.5);
        var state = LayerState.withContext(activation, context, 2.0);

        assertEquals(activation, state.currentActivation());
        assertEquals(context, state.temporalContext());
        assertEquals(2.0, state.timestamp());
        assertTrue(state.hasTemporalContext());
    }

    @Test
    @DisplayName("Combine activation and context with different weights")
    void testCombine() {
        var activation = new DenseVector(new double[]{1.0, 0.0});
        var context = new DenseVector(new double[]{0.0, 1.0});
        var state = LayerState.withContext(activation, context, 0.0);

        // Weight 0.0 - only activation
        var combined0 = state.combine(0.0);
        assertEquals(1.0, combined0.get(0), 1e-10);
        assertEquals(0.0, combined0.get(1), 1e-10);

        // Weight 1.0 - only context
        var combined1 = state.combine(1.0);
        assertEquals(0.0, combined1.get(0), 1e-10);
        assertEquals(1.0, combined1.get(1), 1e-10);

        // Weight 0.5 - equal blend
        var combined05 = state.combine(0.5);
        assertEquals(0.5, combined05.get(0), 1e-10);
        assertEquals(0.5, combined05.get(1), 1e-10);

        // Weight 0.3 - 70% activation, 30% context
        var combined03 = state.combine(0.3);
        assertEquals(0.7, combined03.get(0), 1e-10);
        assertEquals(0.3, combined03.get(1), 1e-10);
    }

    @Test
    @DisplayName("Combine without context returns activation")
    void testCombineWithoutContext() {
        var activation = createPattern(0.8);
        var state = LayerState.fromActivation(activation, 0.0);

        var combined = state.combine(0.5);
        assertEquals(activation, combined);
    }

    @Test
    @DisplayName("Get effective pattern uses default 0.3 weight")
    void testEffectivePattern() {
        var activation = new DenseVector(new double[]{1.0, 0.0});
        var context = new DenseVector(new double[]{0.0, 1.0});
        var state = LayerState.withContext(activation, context, 0.0);

        var effective = state.getEffectivePattern();
        assertEquals(0.7, effective.get(0), 1e-10, "Should be 70% activation");
        assertEquals(0.3, effective.get(1), 1e-10, "Should be 30% context");
    }

    @Test
    @DisplayName("Update activation preserves context")
    void testWithActivation() {
        var activation1 = createPattern(0.8);
        var context = createPattern(0.5);
        var state1 = LayerState.withContext(activation1, context, 1.0);

        var activation2 = createPattern(0.6);
        var state2 = state1.withActivation(activation2);

        assertEquals(activation2, state2.currentActivation());
        assertEquals(context, state2.temporalContext());
        assertEquals(1.0, state2.timestamp());
    }

    @Test
    @DisplayName("Update temporal context preserves activation")
    void testWithTemporalContext() {
        var activation = createPattern(0.8);
        var context1 = createPattern(0.5);
        var state1 = LayerState.withContext(activation, context1, 1.0);

        var context2 = createPattern(0.3);
        var state2 = state1.withTemporalContext(context2);

        assertEquals(activation, state2.currentActivation());
        assertEquals(context2, state2.temporalContext());
        assertEquals(1.0, state2.timestamp());
    }

    @Test
    @DisplayName("Update timestamp preserves patterns")
    void testWithTimestamp() {
        var activation = createPattern(0.8);
        var context = createPattern(0.5);
        var state1 = LayerState.withContext(activation, context, 1.0);

        var state2 = state1.withTimestamp(2.5);

        assertEquals(activation, state2.currentActivation());
        assertEquals(context, state2.temporalContext());
        assertEquals(2.5, state2.timestamp());
    }

    @Test
    @DisplayName("Immutability - updates return new instances")
    void testImmutability() {
        var activation1 = createPattern(0.8);
        var context1 = createPattern(0.5);
        var state1 = LayerState.withContext(activation1, context1, 1.0);

        var activation2 = createPattern(0.6);
        var state2 = state1.withActivation(activation2);

        // Original unchanged
        assertEquals(activation1, state1.currentActivation());
        assertEquals(context1, state1.temporalContext());
        assertEquals(1.0, state1.timestamp());

        // New state has updates
        assertEquals(activation2, state2.currentActivation());
        assertEquals(context1, state2.temporalContext());
        assertEquals(1.0, state2.timestamp());

        assertNotSame(state1, state2);
    }

    @Test
    @DisplayName("Pattern dimensions must match for combination")
    void testDimensionMatching() {
        var activation = new DenseVector(new double[]{1.0, 0.0});
        var context = new DenseVector(new double[]{0.0, 1.0});
        var state = LayerState.withContext(activation, context, 0.0);

        var combined = state.combine(0.5);
        assertEquals(2, combined.dimension());
    }

    // Helper method
    private Pattern createPattern(double value) {
        return new DenseVector(new double[]{value, value, value, value, value});
    }
}