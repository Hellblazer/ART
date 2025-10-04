package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.Layer5Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Layer 5 (Output to Higher Areas).
 * Layer 5 projects processed signals from Layer 2/3 to higher cortical areas.
 *
 * Key biological features:
 * - Medium time constants (50-200ms)
 * - Receives input from Layer 2/3 pyramidal cells
 * - Amplification/gating for salient features
 * - Output normalization for stable signaling
 * - Category signal generation
 * - Burst firing capability for important signals
 *
 * @author Hal Hildebrand
 */
public class Layer5Test {

    private Layer5Implementation layer;
    private static final int LAYER_SIZE = 10;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    public void setup() {
        layer = new Layer5Implementation("layer5-test", LAYER_SIZE);
    }

    @Test
    public void testBasicSignalAmplificationFromLayer23() {
        // Test 1: Basic signal amplification from Layer 2/3
        var input = new DenseVector(new double[]{0.5, 0.3, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .amplificationGain(2.0)
            .outputGain(1.0)
            .build();

        var output = layer.processBottomUp(input, params);

        assertNotNull(output);
        assertEquals(LAYER_SIZE, output.dimension());

        // Check amplification occurred
        assertTrue(output.get(0) > input.get(0), "Signal should be amplified");
        assertTrue(output.get(2) > input.get(2), "Strong signal should be amplified");
    }

    @Test
    public void testOutputNormalizationAccuracy() {
        // Test 2: Output normalization accuracy
        var input = new DenseVector(new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .outputNormalization(0.1)  // Strong normalization
            .build();

        var output = layer.processBottomUp(input, params);

        // Calculate sum of outputs
        double sum = 0.0;
        for (int i = 0; i < output.dimension(); i++) {
            sum += output.get(i);
        }

        // With normalization, sum should be controlled
        assertTrue(sum < 5.0, "Normalized output should have controlled sum");
        assertTrue(sum > 0.0, "Output should be non-zero");
    }

    @Test
    public void testTimeConstantValidation() {
        // Test 3: Time constant validation (50-200ms range)
        // Valid time constant
        var validParams = Layer5Parameters.builder()
            .timeConstant(100.0)  // Mid-range
            .build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), validParams));

        // Test boundary values
        var minParams = Layer5Parameters.builder().timeConstant(50.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), minParams));

        var maxParams = Layer5Parameters.builder().timeConstant(200.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), maxParams));

        // Invalid time constants should throw
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().timeConstant(30.0).build());
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().timeConstant(250.0).build());
    }

    @Test
    public void testMultipleCategorySignalHandling() {
        // Test 4: Multiple category signal handling
        var input = new DenseVector(new double[]{0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .categoryThreshold(0.6)
            .build();

        var output = layer.processBottomUp(input, params);

        // Should handle multiple active categories
        int activeCategories = 0;
        for (int i = 0; i < output.dimension(); i++) {
            if (output.get(i) > params.getCategoryThreshold()) {
                activeCategories++;
            }
        }

        assertTrue(activeCategories >= 2, "Should detect multiple active categories");
    }

    @Test
    public void testBurstFiringForSalientInputs() {
        // Test 5: Burst firing for salient inputs
        var normalInput = new DenseVector(new double[]{0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var salientInput = new DenseVector(new double[]{0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        var params = Layer5Parameters.builder()
            .burstThreshold(0.8)
            .burstAmplification(3.0)
            .build();

        var normalOutput = layer.processBottomUp(normalInput, params);
        var salientOutput = layer.processBottomUp(salientInput, params);

        // Salient inputs should trigger burst amplification
        for (int i = 0; i < 3; i++) {
            assertTrue(salientOutput.get(i) / normalOutput.get(i) > 2.0,
                "Salient inputs should show burst amplification");
        }
    }

    @Test
    public void testParameterValidationAndBoundsChecking() {
        // Test 6: Parameter validation and bounds checking
        // Valid parameters
        assertDoesNotThrow(() -> Layer5Parameters.builder()
            .timeConstant(100.0)
            .amplificationGain(2.0)
            .outputGain(1.5)
            .build());

        // Invalid amplification gain
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().amplificationGain(-1.0).build());

        // Invalid output gain
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().outputGain(-0.5).build());

        // Invalid burst threshold
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().burstThreshold(1.5).build());
    }

    @Test
    public void testIntegrationWithBottomUpSignals() {
        // Test 7: Integration with bottom-up signals
        var weakInput = new DenseVector(new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
        var strongInput = new DenseVector(new double[]{0.8, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0});

        var params = Layer5Parameters.builder().build();

        var weakOutput = layer.processBottomUp(weakInput, params);
        var strongOutput = layer.processBottomUp(strongInput, params);

        // Strong inputs should produce proportionally stronger outputs
        for (int i = 0; i < 5; i++) {
            var ratio = strongOutput.get(i) / weakOutput.get(i);
            assertTrue(ratio > 1.0, "Strong inputs should produce stronger outputs");
        }
    }

    @Test
    public void testStatePersistenceAcrossTimeSteps() {
        // Test 8: State persistence across time steps
        var input = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .timeConstant(100.0)
            .build();

        // Process input
        var output1 = layer.processBottomUp(input, params);

        // Process zero input - should show decay
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        var output2 = layer.processBottomUp(zeroInput, params);

        // Should show some persistence due to time constant
        for (int i = 0; i < 3; i++) {
            assertTrue(output2.get(i) > 0.0, "Should show state persistence");
            assertTrue(output2.get(i) < output1.get(i), "Should show decay");
        }
    }

    @Test
    public void testResetToInitialConditions() {
        // Test 9: Reset to initial conditions
        var input = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder().build();

        // Process input
        layer.processBottomUp(input, params);
        var activation = layer.getActivation();
        assertNotNull(activation);

        // Verify activation is non-zero
        boolean hasNonZero = false;
        for (int i = 0; i < activation.dimension(); i++) {
            if (activation.get(i) > 0) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Should have non-zero activation after processing");

        // Reset
        layer.reset();

        // Check all values are reset
        var resetActivation = layer.getActivation();
        for (int i = 0; i < resetActivation.dimension(); i++) {
            assertEquals(0.0, resetActivation.get(i), EPSILON, "Activation should be reset to zero");
        }
    }

    @Test
    public void testEdgeCasesZeroMaxInput() {
        // Test 10: Edge cases (zero/max input)
        var params = Layer5Parameters.builder().build();

        // Zero input
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        var zeroOutput = layer.processBottomUp(zeroInput, params);
        assertNotNull(zeroOutput);
        for (int i = 0; i < zeroOutput.dimension(); i++) {
            assertTrue(zeroOutput.get(i) >= 0.0, "Output should be non-negative for zero input");
        }

        // Max input
        var maxArray = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            maxArray[i] = 1.0;
        }
        var maxInput = new DenseVector(maxArray);
        var maxOutput = layer.processBottomUp(maxInput, params);
        assertNotNull(maxOutput);
        for (int i = 0; i < maxOutput.dimension(); i++) {
            assertTrue(maxOutput.get(i) <= params.getCeiling(), "Output should respect ceiling");
        }
    }

    @Test
    public void testPerformanceBenchmark() {
        // Test 11: Performance benchmark (<1ms)
        var input = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5});
        var params = Layer5Parameters.builder().build();

        // Warm up
        for (int i = 0; i < 100; i++) {
            layer.processBottomUp(input, params);
        }

        // Measure
        long startTime = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            layer.processBottomUp(input, params);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / 1000.0;
        assertTrue(avgTimeMs < 1.0, "Average processing time should be < 1ms, got: " + avgTimeMs);
    }

    @Test
    public void testBiologicalConstraintsFiringRates() {
        // Test 12: Biological constraints (firing rates)
        var input = new DenseVector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0});
        var params = Layer5Parameters.builder()
            .maxFiringRate(100.0)  // 100Hz max
            .build();

        var output = layer.processBottomUp(input, params);

        // Check firing rates are within biological limits
        for (int i = 0; i < output.dimension(); i++) {
            // Convert activation to firing rate (assuming linear mapping)
            var firingRate = output.get(i) * params.getMaxFiringRate();
            assertTrue(firingRate <= 100.0, "Firing rate should be <= 100Hz");
            assertTrue(firingRate >= 0.0, "Firing rate should be >= 0Hz");
        }
    }

    @Test
    public void testIntegrationWithExistingPathways() {
        // Test 13: Integration with existing pathways
        var layer23Input = new DenseVector(new double[]{0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder().build();

        // Test bottom-up processing from Layer 2/3
        var output = layer.processBottomUp(layer23Input, params);
        assertNotNull(output);

        // Test that it properly amplifies and projects
        for (int i = 0; i < 6; i++) {
            if (layer23Input.get(i) > 0) {
                assertTrue(output.get(i) > 0, "Non-zero input should produce non-zero output");
            }
        }

        // Test top-down modulation (should be minimal for Layer 5)
        var topDown = new DenseVector(new double[]{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3});
        var modulated = layer.processTopDown(topDown, params);
        assertNotNull(modulated);
    }

    @Test
    public void testOutputStabilityOverTime() {
        // Test 14: Output stability over time
        var input = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .timeConstant(100.0)
            .build();

        // Process multiple time steps with same input
        Pattern[] outputs = new Pattern[10];
        for (int t = 0; t < 10; t++) {
            outputs[t] = layer.processBottomUp(input, params);
        }

        // Check convergence to stable state
        for (int i = 0; i < 5; i++) {
            var diff = Math.abs(outputs[9].get(i) - outputs[8].get(i));
            assertTrue(diff < 0.01, "Output should stabilize over time");
        }
    }

    @Test
    public void testBackwardCompatibilityWithPhase1() {
        // Test 15: Backward compatibility with Phase 1
        // Test basic layer interface methods
        assertEquals("layer5-test", layer.getId());
        assertEquals(LAYER_SIZE, layer.size());
        assertEquals(LayerType.CUSTOM, layer.getType());

        // Test activation listener compatibility
        var listenerCalled = new AtomicInteger(0);
        var oldActivation = new AtomicReference<Pattern>();
        var newActivation = new AtomicReference<Pattern>();

        LayerActivationListener listener = (layerId, oldAct, newAct) -> {
            listenerCalled.incrementAndGet();
            oldActivation.set(oldAct);
            newActivation.set(newAct);
        };

        layer.addActivationListener(listener);

        // Process input
        var input = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder().build();
        layer.processBottomUp(input, params);

        // Verify listener was called
        assertTrue(listenerCalled.get() > 0, "Listener should be called");

        // Test weight matrix compatibility
        var weights = layer.getWeights();
        assertNotNull(weights);
        assertEquals(LAYER_SIZE, weights.getRows());
        assertEquals(LAYER_SIZE, weights.getCols());
    }
}