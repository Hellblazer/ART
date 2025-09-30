package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.Layer6Parameters;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Layer 6 (Feedback Modulation).
 * Layer 6 provides modulatory feedback to Layer 4 and thalamus.
 *
 * CRITICAL: Implements ART matching rule - modulatory only!
 * - Cannot fire cells alone
 * - Requires bottom-up + top-down to fire (ART matching rule)
 * - On-center, off-surround dynamics
 * - Top-down expectation generation
 * - Attentional gain control
 *
 * Biological constraints:
 * - Slow time constants (100-500ms)
 * - Modulatory only - CANNOT fire cells alone
 * - Requires coincidence of bottom-up and top-down signals
 *
 * @author Hal Hildebrand
 */
public class Layer6Test {

    private Layer6Implementation layer;
    private static final int LAYER_SIZE = 10;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    public void setup() {
        layer = new Layer6Implementation("layer6-test", LAYER_SIZE);
    }

    @Test
    public void testModulatoryBehaviorCannotFireAlone() {
        // Test 1: Modulatory behavior - CANNOT fire alone (MOST CRITICAL!)
        // Only top-down input, no bottom-up
        var topDown = new DenseVector(new double[]{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        // Set only top-down expectation
        layer.setTopDownExpectation(topDown);

        // Process with NO bottom-up input
        var zeroBottomUp = new DenseVector(new double[LAYER_SIZE]);
        var output = layer.processBottomUp(zeroBottomUp, params);

        // CRITICAL: Output must be zero when only top-down is present
        for (int i = 0; i < output.dimension(); i++) {
            assertEquals(0.0, output.get(i), EPSILON,
                "Layer 6 MUST NOT fire with only top-down (modulatory) input - violates ART matching rule!");
        }
    }

    @Test
    public void testRequiresBothBottomUpAndTopDownToFire() {
        // Test 2: Requires both bottom-up + top-down to fire
        var bottomUp = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        // First, test with only bottom-up (no top-down)
        layer.setTopDownExpectation(new DenseVector(new double[LAYER_SIZE]));
        var bottomOnlyOutput = layer.processBottomUp(bottomUp, params);

        // Should pass through bottom-up but with minimal activation
        boolean hasNonZeroBottomOnly = false;
        for (int i = 0; i < 3; i++) {
            if (bottomOnlyOutput.get(i) > 0.0) {
                hasNonZeroBottomOnly = true;
            }
        }
        assertTrue(hasNonZeroBottomOnly, "Bottom-up alone should produce some output");

        // Now test with both bottom-up and top-down
        layer.setTopDownExpectation(topDown);
        var bothOutput = layer.processBottomUp(bottomUp, params);

        // With both signals, output should be enhanced
        for (int i = 0; i < 3; i++) {
            assertTrue(bothOutput.get(i) > bottomOnlyOutput.get(i),
                "Bottom-up + top-down should enhance activation (ART matching)");
        }
    }

    @Test
    public void testOnCenterExcitationDynamics() {
        // Test 3: On-center excitation dynamics
        var bottomUp = new DenseVector(new double[]{0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder()
            .onCenterWeight(2.0)  // Strong on-center
            .build();

        layer.setTopDownExpectation(topDown);
        var output = layer.processBottomUp(bottomUp, params);

        // On-center (index 2) should be strongly enhanced
        assertTrue(output.get(2) > bottomUp.get(2) * 1.5,
            "On-center should show strong enhancement");

        // Nearby cells should receive less enhancement
        assertTrue(output.get(1) < output.get(2), "Adjacent cells should receive less enhancement");
        assertTrue(output.get(3) < output.get(2), "Adjacent cells should receive less enhancement");
    }

    @Test
    public void testOffSurroundInhibitionDynamics() {
        // Test 4: Off-surround inhibition dynamics
        var bottomUp = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder()
            .offSurroundStrength(0.3)  // Moderate surround inhibition
            .build();

        layer.setTopDownExpectation(topDown);
        var output = layer.processBottomUp(bottomUp, params);

        // Center should be enhanced
        assertTrue(output.get(2) > bottomUp.get(2), "Center should be enhanced");

        // Surround should be suppressed relative to no top-down case
        layer.setTopDownExpectation(new DenseVector(new double[LAYER_SIZE]));
        var noTopDownOutput = layer.processBottomUp(bottomUp, params);

        assertTrue(output.get(0) < noTopDownOutput.get(0), "Surround should be suppressed");
        assertTrue(output.get(4) < noTopDownOutput.get(4), "Surround should be suppressed");
    }

    @Test
    public void testTimeConstantValidation() {
        // Test 5: Time constant validation (100-500ms)
        // Valid time constant
        var validParams = Layer6Parameters.builder()
            .timeConstant(300.0)  // Mid-range
            .build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), validParams));

        // Test boundary values
        var minParams = Layer6Parameters.builder().timeConstant(100.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), minParams));

        var maxParams = Layer6Parameters.builder().timeConstant(500.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), maxParams));

        // Invalid time constants should throw
        assertThrows(IllegalArgumentException.class, () ->
            Layer6Parameters.builder().timeConstant(50.0).build());
        assertThrows(IllegalArgumentException.class, () ->
            Layer6Parameters.builder().timeConstant(600.0).build());
    }

    @Test
    public void testTopDownExpectationGeneration() {
        // Test 6: Top-down expectation generation
        var expectation = new DenseVector(new double[]{0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        // Set expectation
        layer.setTopDownExpectation(expectation);

        // Verify expectation is stored and used
        var bottomUp = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0});
        var output = layer.processBottomUp(bottomUp, params);

        // Output should be modulated by expectation pattern
        assertTrue(output.get(0) > output.get(3),
            "Higher expectation should produce stronger modulation");
        assertTrue(output.get(1) > output.get(3),
            "Higher expectation should produce stronger modulation");
    }

    @Test
    public void testAttentionalGainModulation() {
        // Test 7: Attentional gain modulation
        var bottomUp = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var attention = new DenseVector(new double[]{1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        var lowGainParams = Layer6Parameters.builder()
            .attentionalGain(0.5)
            .build();
        var highGainParams = Layer6Parameters.builder()
            .attentionalGain(2.0)
            .build();

        layer.setTopDownExpectation(attention);

        var lowGainOutput = layer.processBottomUp(bottomUp, lowGainParams);
        var highGainOutput = layer.processBottomUp(bottomUp, highGainParams);

        // High gain should produce stronger modulation
        assertTrue(highGainOutput.get(0) > lowGainOutput.get(0),
            "Higher attentional gain should produce stronger modulation");
    }

    @Test
    public void testParameterValidation() {
        // Test 8: Parameter validation
        // Valid parameters
        assertDoesNotThrow(() -> Layer6Parameters.builder()
            .timeConstant(200.0)
            .onCenterWeight(1.5)
            .offSurroundStrength(0.2)
            .build());

        // Invalid on-center weight
        assertThrows(IllegalArgumentException.class, () ->
            Layer6Parameters.builder().onCenterWeight(-1.0).build());

        // Invalid off-surround strength
        assertThrows(IllegalArgumentException.class, () ->
            Layer6Parameters.builder().offSurroundStrength(-0.5).build());

        // Invalid modulation threshold
        assertThrows(IllegalArgumentException.class, () ->
            Layer6Parameters.builder().modulationThreshold(1.5).build());
    }

    @Test
    public void testIntegrationWithLayer4Feedback() {
        // Test 9: Integration with Layer 4 feedback
        var layer4Signal = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        layer.setTopDownExpectation(topDown);

        // Process Layer 4 input
        var modulatedOutput = layer.processBottomUp(layer4Signal, params);

        // Should provide feedback modulation
        assertNotNull(modulatedOutput);
        for (int i = 0; i < 5; i++) {
            assertTrue(modulatedOutput.get(i) > 0.0, "Should provide modulation to Layer 4");
        }

        // Test feedback path
        var feedback = layer.generateFeedbackToLayer4(modulatedOutput, params);
        assertNotNull(feedback);
        assertEquals(LAYER_SIZE, feedback.dimension());
    }

    @Test
    public void testStatePersistence() {
        // Test 10: State persistence
        var bottomUp = new DenseVector(new double[]{0.6, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder()
            .timeConstant(200.0)
            .build();

        // Set expectation and process
        layer.setTopDownExpectation(topDown);
        var output1 = layer.processBottomUp(bottomUp, params);

        // Process with zero input - should show slow decay
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        var output2 = layer.processBottomUp(zeroInput, params);

        // Should maintain modulation state (but zero due to no bottom-up)
        for (int i = 0; i < output2.dimension(); i++) {
            assertEquals(0.0, output2.get(i), EPSILON,
                "Without bottom-up, output must be zero (modulatory only)");
        }

        // But internal state should persist - test with new bottom-up
        var newBottomUp = new DenseVector(new double[]{0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var output3 = layer.processBottomUp(newBottomUp, params);

        // Should still show modulation from persistent top-down
        assertTrue(output3.get(0) > newBottomUp.get(0) * 0.9,
            "Should maintain top-down modulation state");
    }

    @Test
    public void testResetFunctionality() {
        // Test 11: Reset functionality
        var bottomUp = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        // Process input
        layer.setTopDownExpectation(topDown);
        layer.processBottomUp(bottomUp, params);
        var activation = layer.getActivation();
        assertNotNull(activation);

        // Reset
        layer.reset();

        // Check all values are reset
        var resetActivation = layer.getActivation();
        for (int i = 0; i < resetActivation.dimension(); i++) {
            assertEquals(0.0, resetActivation.get(i), EPSILON, "Activation should be reset to zero");
        }

        // Top-down expectation should also be reset
        var output = layer.processBottomUp(bottomUp, params);
        // Should act as if no top-down (just pass through bottom-up with minimal processing)
        for (int i = 0; i < 7; i++) {
            assertTrue(output.get(i) <= bottomUp.get(i) * 1.1,
                "After reset, should have minimal modulation");
        }
    }

    @Test
    public void testEdgeCases() {
        // Test 12: Edge cases
        var params = Layer6Parameters.builder().build();

        // Test with all zeros
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        var zeroTopDown = new DenseVector(new double[LAYER_SIZE]);
        layer.setTopDownExpectation(zeroTopDown);
        var zeroOutput = layer.processBottomUp(zeroInput, params);

        for (int i = 0; i < zeroOutput.dimension(); i++) {
            assertEquals(0.0, zeroOutput.get(i), EPSILON, "All-zero input should produce zero output");
        }

        // Test with max values
        var maxArray = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            maxArray[i] = 1.0;
        }
        var maxInput = new DenseVector(maxArray);
        var maxTopDown = new DenseVector(maxArray);
        layer.setTopDownExpectation(maxTopDown);
        var maxOutput = layer.processBottomUp(maxInput, params);

        for (int i = 0; i < maxOutput.dimension(); i++) {
            assertTrue(maxOutput.get(i) <= params.getCeiling(),
                "Output should respect ceiling constraint");
        }
    }

    @Test
    public void testPerformanceBenchmark() {
        // Test 13: Performance benchmark (<1ms)
        var bottomUp = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5});
        var topDown = new DenseVector(new double[]{0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6});
        var params = Layer6Parameters.builder().build();

        layer.setTopDownExpectation(topDown);

        // Warm up
        for (int i = 0; i < 100; i++) {
            layer.processBottomUp(bottomUp, params);
        }

        // Measure
        long startTime = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            layer.processBottomUp(bottomUp, params);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / 1000.0;
        assertTrue(avgTimeMs < 1.0, "Average processing time should be < 1ms, got: " + avgTimeMs);
    }

    @Test
    public void testBiologicalConstraints() {
        // Test 14: Biological constraints
        var bottomUp = new DenseVector(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0});
        var topDown = new DenseVector(new double[]{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1});
        var params = Layer6Parameters.builder()
            .maxFiringRate(50.0)  // Layer 6 has lower firing rates
            .build();

        layer.setTopDownExpectation(topDown);
        var output = layer.processBottomUp(bottomUp, params);

        // Check firing rates are within biological limits
        for (int i = 0; i < output.dimension(); i++) {
            // Convert activation to firing rate
            var firingRate = output.get(i) * params.getMaxFiringRate();
            assertTrue(firingRate <= 50.0, "Layer 6 firing rate should be <= 50Hz");
            assertTrue(firingRate >= 0.0, "Firing rate should be >= 0Hz");
        }
    }

    @Test
    public void testBackwardCompatibility() {
        // Test 15: Backward compatibility
        // Test basic layer interface methods
        assertEquals("layer6-test", layer.getId());
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
        var bottomUp = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
        var topDown = new DenseVector(new double[]{0.6, 0.5, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer6Parameters.builder().build();

        layer.setTopDownExpectation(topDown);
        layer.processBottomUp(bottomUp, params);

        // Verify listener was called
        assertTrue(listenerCalled.get() > 0, "Listener should be called");

        // Test weight matrix compatibility
        var weights = layer.getWeights();
        assertNotNull(weights);
        assertEquals(LAYER_SIZE, weights.getRows());
        assertEquals(LAYER_SIZE, weights.getCols());
    }
}