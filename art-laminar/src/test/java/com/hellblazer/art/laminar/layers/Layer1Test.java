package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.Layer1Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Layer 1 (Top-Down Attentional Priming).
 * Layer 1 contains apical dendrites that receive top-down attentional signals
 * from higher cortical areas.
 *
 * Key biological features:
 * - Very slow time constants (200-1000ms)
 * - Receives top-down signals from higher cortical areas
 * - Sustained attention effects (persists after input ends)
 * - Priming without driving cells
 * - Integrates with Layer 2/3 apical dendrites
 * - Long-duration memory traces
 *
 * @author Hal Hildebrand
 */
public class Layer1Test {

    private Layer1Implementation layer;
    private static final int LAYER_SIZE = 10;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    public void setup() {
        layer = new Layer1Implementation("layer1-test", LAYER_SIZE);
    }

    @Test
    public void testVerySlowTimeConstantValidation() {
        // Test 1: Very slow time constant validation (200-1000ms)
        // Valid time constant
        var validParams = Layer1Parameters.builder()
            .timeConstant(500.0)  // Mid-range
            .build();
        assertDoesNotThrow(() -> layer.processTopDown(new DenseVector(new double[LAYER_SIZE]), validParams));

        // Test boundary values
        var minParams = Layer1Parameters.builder().timeConstant(200.0).build();
        assertDoesNotThrow(() -> layer.processTopDown(new DenseVector(new double[LAYER_SIZE]), minParams));

        var maxParams = Layer1Parameters.builder().timeConstant(1000.0).build();
        assertDoesNotThrow(() -> layer.processTopDown(new DenseVector(new double[LAYER_SIZE]), maxParams));

        // Invalid time constants should throw
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().timeConstant(100.0).build());
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().timeConstant(1200.0).build());
    }

    @Test
    public void testTopDownSignalReception() {
        // Test 2: Top-down signal reception
        var topDownSignal = new DenseVector(new double[]{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0});
        var params = Layer1Parameters.builder().build();

        var output = layer.processTopDown(topDownSignal, params);

        assertNotNull(output);
        assertEquals(LAYER_SIZE, output.dimension());

        // Should receive and process top-down signals
        for (int i = 0; i < 8; i++) {
            assertTrue(output.get(i) > 0.0, "Should process top-down signal at index " + i);
        }

        // Signal strength should correlate with input
        assertTrue(output.get(0) > output.get(7), "Stronger input should produce stronger priming");
    }

    @Test
    public void testSustainedAttentionAfterInputRemoval() {
        // Test 3: Sustained attention after input removal
        var attentionSignal = new DenseVector(new double[]{1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .timeConstant(500.0)  // 500ms for noticeable persistence
            .sustainedDecayRate(0.001)  // Very slow decay
            .build();

        // Apply attention
        var initialOutput = layer.processTopDown(attentionSignal, params);

        // Remove attention (zero input)
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);

        // Check persistence over multiple time steps
        Pattern[] outputs = new Pattern[5];
        for (int t = 0; t < 5; t++) {
            outputs[t] = layer.processTopDown(zeroInput, params);
        }

        // Should show gradual decay, not immediate drop
        for (int i = 0; i < 5; i++) {
            assertTrue(outputs[0].get(i) > 0.0, "Should maintain attention after removal");
            if (i < 4) {
                assertTrue(outputs[0].get(i) > outputs[4].get(i),
                    "Should show gradual decay, not sudden drop");
            }
        }

        // But should still be decaying
        assertTrue(outputs[4].get(0) < initialOutput.get(0),
            "Should eventually decay from initial level");
    }

    @Test
    public void testPrimingBehaviorEnhancesButDoesNotDrive() {
        // Test 4: Priming behavior (enhances but doesn't drive)
        var primingSignal = new DenseVector(new double[]{0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .primingStrength(0.3)  // 30% enhancement
            .build();

        // Apply priming
        var primingOutput = layer.processTopDown(primingSignal, params);

        // Priming should not drive strong activation by itself
        assertTrue(primingOutput.get(0) < 0.5,
            "Priming alone should not drive strong activation");

        // Get priming effect for integration with other layers
        var primingEffect = layer.getPrimingEffect();
        assertNotNull(primingEffect);

        // Priming should provide moderate enhancement
        assertTrue(primingEffect.get(0) > 0.0 && primingEffect.get(0) < 0.5,
            "Priming should provide moderate enhancement, not drive");
    }

    @Test
    public void testIntegrationWithLayer23ApicalDendrites() {
        // Test 5: Integration with Layer 2/3 apical dendrites
        var attentionSignal = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .apicalIntegration(0.5)  // 50% integration strength
            .build();

        // Process attention signal
        layer.processTopDown(attentionSignal, params);

        // Get signal for Layer 2/3 apical dendrites
        var apicalSignal = layer.getApicalDendriteSignal();
        assertNotNull(apicalSignal);
        assertEquals(LAYER_SIZE, apicalSignal.dimension());

        // Should provide modulated signal to apical dendrites
        for (int i = 0; i < 5; i++) {
            assertTrue(apicalSignal.get(i) > 0.0,
                "Should provide signal to apical dendrites");
            assertTrue(apicalSignal.get(i) <= attentionSignal.get(i),
                "Apical signal should be modulated, not amplified");
        }
    }

    @Test
    public void testMultipleAttentionSources() {
        // Test 6: Multiple attention sources
        var attention1 = new DenseVector(new double[]{0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var attention2 = new DenseVector(new double[]{0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var attention3 = new DenseVector(new double[]{0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder().build();

        // Apply multiple attention sources sequentially
        layer.processTopDown(attention1, params);
        layer.processTopDown(attention2, params);
        var output = layer.processTopDown(attention3, params);

        // Should maintain attention at multiple locations
        assertTrue(output.get(0) > 0.0, "Should maintain attention at first location");
        assertTrue(output.get(2) > 0.0, "Should maintain attention at second location");
        assertTrue(output.get(4) > 0.0, "Should maintain attention at third location");
    }

    @Test
    public void testLongDurationPersistenceSeconds() {
        // Test 7: Long-duration persistence (seconds)
        var strongAttention = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .timeConstant(1000.0)  // 1 second
            .sustainedDecayRate(0.0005)  // Very slow decay for seconds of persistence
            .build();

        // Apply strong attention
        var initialOutput = layer.processTopDown(strongAttention, params);
        assertTrue(initialOutput.get(0) > 0.8, "Should have strong initial attention");

        // Simulate 100 time steps (roughly 1 second at 10ms steps)
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        Pattern lastOutput = initialOutput;
        for (int t = 0; t < 100; t++) {
            lastOutput = layer.processTopDown(zeroInput, params);
        }

        // Should still maintain some attention after ~1 second
        // With very slow decay, should retain at least 20% of original
        assertTrue(lastOutput.get(0) > 0.15,
            "Should maintain some attention after 1 second, got: " + lastOutput.get(0));
    }

    @Test
    public void testParameterValidation() {
        // Test 8: Parameter validation
        // Valid parameters
        assertDoesNotThrow(() -> Layer1Parameters.builder()
            .timeConstant(500.0)
            .primingStrength(0.3)
            .sustainedDecayRate(0.001)
            .build());

        // Invalid priming strength
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().primingStrength(-0.1).build());
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().primingStrength(1.5).build());

        // Invalid decay rate
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().sustainedDecayRate(-0.001).build());

        // Invalid apical integration
        assertThrows(IllegalArgumentException.class, () ->
            Layer1Parameters.builder().apicalIntegration(-0.5).build());
    }

    @Test
    public void testStateManagement() {
        // Test 9: State management
        var attention = new DenseVector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0});
        var params = Layer1Parameters.builder().build();

        // Build up attention state
        for (int i = 0; i < 5; i++) {
            layer.processTopDown(attention, params);
        }

        // Get current attention state
        var attentionState = layer.getAttentionState();
        assertNotNull(attentionState);
        assertEquals(LAYER_SIZE, attentionState.dimension());

        // Should have built up attention state
        for (int i = 0; i < 9; i++) {
            assertTrue(attentionState.get(i) > 0.0, "Should have attention state at index " + i);
        }
    }

    @Test
    public void testResetFunctionality() {
        // Test 10: Reset functionality
        var attention = new DenseVector(new double[]{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1});
        var params = Layer1Parameters.builder().build();

        // Process attention
        layer.processTopDown(attention, params);
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

        // Attention state should also be reset
        var resetAttention = layer.getAttentionState();
        for (int i = 0; i < resetAttention.dimension(); i++) {
            assertEquals(0.0, resetAttention.get(i), EPSILON, "Attention state should be reset");
        }
    }

    @Test
    public void testEdgeCasesSuddenAttentionShifts() {
        // Test 11: Edge cases (sudden attention shifts)
        var attention1 = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var attention2 = new DenseVector(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .attentionShiftRate(0.5)  // Allow moderate shift rate
            .build();

        // Apply first attention
        layer.processTopDown(attention1, params);

        // Sudden shift to different location
        var shiftedOutput = layer.processTopDown(attention2, params);

        // Should maintain some attention at old location (persistence)
        assertTrue(shiftedOutput.get(0) > 0.0,
            "Should maintain some attention at old location");

        // But should also respond to new location
        assertTrue(shiftedOutput.get(5) > 0.0,
            "Should respond to new attention location");
    }

    @Test
    public void testPerformanceBenchmark() {
        // Test 12: Performance benchmark (<1ms)
        var attention = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5});
        var params = Layer1Parameters.builder().build();

        // Warm up
        for (int i = 0; i < 100; i++) {
            layer.processTopDown(attention, params);
        }

        // Measure
        long startTime = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            layer.processTopDown(attention, params);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / 1000.0;
        assertTrue(avgTimeMs < 1.0, "Average processing time should be < 1ms, got: " + avgTimeMs);
    }

    @Test
    public void testBiologicalConstraints() {
        // Test 13: Biological constraints
        var attention = new DenseVector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0});
        var params = Layer1Parameters.builder()
            .maxFiringRate(30.0)  // Layer 1 has low firing rates
            .build();

        var output = layer.processTopDown(attention, params);

        // Check firing rates are within biological limits
        for (int i = 0; i < output.dimension(); i++) {
            // Convert activation to firing rate
            var firingRate = output.get(i) * params.getMaxFiringRate();
            assertTrue(firingRate <= 30.0, "Layer 1 firing rate should be <= 30Hz");
            assertTrue(firingRate >= 0.0, "Firing rate should be >= 0Hz");
        }
    }

    @Test
    public void testMemoryTraceDecayCurves() {
        // Test 14: Memory trace decay curves
        var attention = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder()
            .timeConstant(400.0)
            .sustainedDecayRate(0.002)
            .build();

        // Apply attention
        layer.processTopDown(attention, params);

        // Track decay over time
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        double[] decayValues = new double[20];

        for (int t = 0; t < 20; t++) {
            var output = layer.processTopDown(zeroInput, params);
            decayValues[t] = output.get(0);
        }

        // Verify exponential-like decay
        for (int t = 1; t < 20; t++) {
            assertTrue(decayValues[t] < decayValues[t-1],
                "Should show monotonic decay");

            // Decay rate should slow down (larger time constant = slower decay)
            if (t > 10) {
                var earlyDecayRate = decayValues[1] - decayValues[2];
                var lateDecayRate = decayValues[t-1] - decayValues[t];
                assertTrue(lateDecayRate < earlyDecayRate,
                    "Decay should follow exponential curve");
            }
        }
    }

    @Test
    public void testBackwardCompatibility() {
        // Test 15: Backward compatibility
        // Test basic layer interface methods
        assertEquals("layer1-test", layer.getId());
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
        var attention = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer1Parameters.builder().build();
        layer.processTopDown(attention, params);

        // Verify listener was called
        assertTrue(listenerCalled.get() > 0, "Listener should be called");

        // Test weight matrix compatibility
        var weights = layer.getWeights();
        assertNotNull(weights);
        assertEquals(LAYER_SIZE, weights.getRows());
        assertEquals(LAYER_SIZE, weights.getCols());

        // Test processBottomUp method (Layer 1 doesn't use it much but should not crash)
        var bottomUp = new DenseVector(new double[]{0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        assertDoesNotThrow(() -> layer.processBottomUp(bottomUp, params));
    }
}