package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 5 (Motor Output & Action Selection).
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3)
 */
class Layer5Test {

    private Layer5 layer;
    private static final int LAYER_SIZE = 10;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    void setup() {
        layer = new Layer5("layer5-test", LAYER_SIZE);
    }

    @Test
    void testBasicAmplification() {
        var input = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .amplificationGain(2.0)
            .build();

        var output = layer.processBottomUp(input, params);

        // Should amplify input
        for (var i = 0; i < 5; i++) {
            assertTrue(output.get(i) > input.get(i),
                "Layer 5 should amplify input at index " + i);
        }
    }

    @Test
    void testBurstFiring() {
        var input = new DenseVector(new double[]{0.7, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .burstThreshold(0.6)
            .burstAmplification(1.5)
            .amplificationGain(1.2)
            .build();

        var output = layer.processBottomUp(input, params);

        // Strong input should show increased output (burst effect visible before ceiling clamp)
        assertTrue(output.get(0) > input.get(0),
            "Should show burst amplification effect, got " + output.get(0) + " vs input " + input.get(0));
        assertTrue(output.get(0) > output.get(1),
            "Burst unit should have higher activation than non-burst");
    }

    @Test
    void testOutputNormalization() {
        var input = new DenseVector(new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .outputNormalization(0.1)
            .build();

        var output = layer.processBottomUp(input, params);

        // Normalization should prevent runaway activation
        var sum = 0.0;
        for (var i = 0; i < output.dimension(); i++) {
            sum += output.get(i);
        }
        assertTrue(sum < 10.0, "Normalization should limit total activation");
    }

    @Test
    void testCategoryFormation() {
        var input = new DenseVector(new double[]{0.9, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder()
            .categoryThreshold(0.5)
            .build();

        layer.processBottomUp(input, params);

        assertTrue(layer.isCategoryFormed(), "Should detect category formation");
    }

    @Test
    void testStatePersistence() {
        var input1 = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder().build();

        layer.processBottomUp(input1, params);
        var activation1 = layer.getActivation();

        // Process with different input
        var input2 = new DenseVector(new double[]{0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        layer.processBottomUp(input2, params);
        var activation2 = layer.getActivation();

        // Should show some persistence from previous state
        assertNotEquals(activation1, activation2);
    }

    @Test
    void testTimeConstantValidation() {
        var params = Layer5Parameters.builder()
            .timeConstant(100.0)  // Mid-range
            .build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), params));

        // Test boundary values
        var minParams = Layer5Parameters.builder().timeConstant(50.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), minParams));

        var maxParams = Layer5Parameters.builder().timeConstant(200.0).build();
        assertDoesNotThrow(() -> layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), maxParams));

        // Invalid time constants should throw
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().timeConstant(30.0).build());
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().timeConstant(300.0).build());
    }

    @Test
    void testOutputGainControl() {
        var input = new DenseVector(new double[]{0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        var lowGainParams = Layer5Parameters.builder()
            .outputGain(0.5)
            .build();
        var highGainParams = Layer5Parameters.builder()
            .outputGain(2.0)
            .build();

        var lowGainLayer = new Layer5("layer5-lowgain", LAYER_SIZE);
        var lowGainOutput = lowGainLayer.processBottomUp(input, lowGainParams);

        var highGainLayer = new Layer5("layer5-highgain", LAYER_SIZE);
        var highGainOutput = highGainLayer.processBottomUp(input, highGainParams);

        // High gain should produce stronger output
        assertTrue(highGainOutput.get(0) > lowGainOutput.get(0),
            "Higher gain should produce stronger output");
    }

    @Test
    void testResetFunctionality() {
        var input = new DenseVector(new double[]{0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0});
        var params = Layer5Parameters.builder().build();

        layer.processBottomUp(input, params);
        var activation = layer.getActivation();
        assertNotNull(activation);

        layer.reset();

        var resetActivation = layer.getActivation();
        for (var i = 0; i < resetActivation.dimension(); i++) {
            assertEquals(0.0, resetActivation.get(i), EPSILON,
                "Activation should be reset to zero");
        }
    }

    @Test
    void testParameterValidation() {
        // Valid parameters
        assertDoesNotThrow(() -> Layer5Parameters.builder()
            .amplificationGain(1.5)
            .outputGain(1.0)
            .categoryThreshold(0.5)
            .build());

        // Invalid amplification gain
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().amplificationGain(-1.0).build());

        // Invalid output gain
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().outputGain(-0.5).build());

        // Invalid category threshold
        assertThrows(IllegalArgumentException.class, () ->
            Layer5Parameters.builder().categoryThreshold(1.5).build());
    }

    @Test
    void testLayerInterfaceCompliance() {
        assertEquals("layer5-test", layer.getId());
        assertEquals(LAYER_SIZE, layer.size());
        assertEquals(LayerType.LAYER_5, layer.getType());

        var weights = layer.getWeights();
        assertNotNull(weights);
        assertEquals(LAYER_SIZE, weights.getRows());
        assertEquals(LAYER_SIZE, weights.getCols());
    }
}
