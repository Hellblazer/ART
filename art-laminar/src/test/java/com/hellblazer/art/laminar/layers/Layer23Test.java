package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer23Parameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 2/3 implementation.
 * Validates horizontal grouping, complex cells, and signal integration.
 *
 * @author Hal Hildebrand
 */
public class Layer23Test {

    private Layer23Implementation layer23;
    private static final int LAYER_SIZE = 100;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    void setUp() {
        layer23 = new Layer23Implementation("layer23", LAYER_SIZE);
    }

    @Test
    void testBottomUpSignalReception() {
        // Test receiving bottom-up input from Layer 4
        var bottomUpData = new double[LAYER_SIZE];
        for (int i = 40; i < 60; i++) {
            bottomUpData[i] = 0.8;
        }
        var bottomUpInput = new DenseVector(bottomUpData);

        layer23.receiveBottomUpInput(bottomUpInput);
        layer23.process(bottomUpInput, 0.001);

        var activation = layer23.getActivation();

        // Should have activation in the input region
        for (int i = 40; i < 60; i++) {
            assertTrue(activation.get(i) > 0.5, "Should have activation at position " + i);
        }

        // Should be quiet outside input region
        assertTrue(activation.get(30) < 0.1, "Should be quiet outside input region");
        assertTrue(activation.get(70) < 0.1, "Should be quiet outside input region");
    }

    @Test
    void testTopDownPrimingFromLayer1() {
        // Test top-down priming influence
        var topDownData = new double[LAYER_SIZE];
        for (int i = 30; i < 40; i++) {
            topDownData[i] = 0.5;  // Priming signal
        }
        var topDownPriming = new DenseVector(topDownData);

        // Weak bottom-up input
        var bottomUpData = new double[LAYER_SIZE];
        for (int i = 30; i < 40; i++) {
            bottomUpData[i] = 0.3;  // Weak input
        }
        var bottomUpInput = new DenseVector(bottomUpData);

        // Without priming
        layer23.process(bottomUpInput, 0.001);
        var withoutPriming = layer23.getActivation();

        // With priming
        layer23.reset();
        layer23.receiveTopDownPriming(topDownPriming);
        layer23.process(bottomUpInput, 0.001);
        var withPriming = layer23.getActivation();

        // Priming should enhance activation
        for (int i = 30; i < 40; i++) {
            assertTrue(withPriming.get(i) > withoutPriming.get(i),
                "Priming should enhance activation at " + i);
        }
    }

    @Test
    void testHorizontalGroupingIntegration() {
        // Test that horizontal grouping via bipole cells works
        var inputData = new double[LAYER_SIZE];

        // Create pattern with gaps for horizontal grouping to fill
        inputData[30] = 0.8;
        inputData[35] = 0.8;
        inputData[40] = 0.8;
        // Gaps at 31-34, 36-39

        var input = new DenseVector(inputData);

        // Process multiple times to allow horizontal propagation
        for (int i = 0; i < 5; i++) {
            layer23.process(input, 0.001);
        }

        var grouping = layer23.getHorizontalGrouping();

        // Should have some horizontal grouping activity
        assertTrue(layer23.isHorizontalGroupingActive(), "Horizontal grouping should be active");

        // Check that grouping has spread
        double totalGrouping = 0.0;
        for (int i = 25; i < 45; i++) {
            totalGrouping += ((DenseVector) grouping).get(i);
        }
        assertTrue(totalGrouping > 1.0, "Should have horizontal grouping activity");
    }

    @Test
    void testCombinedSignalIntegration() {
        // Test integration of bottom-up, top-down, and horizontal signals
        var bottomUpData = new double[LAYER_SIZE];
        var topDownData = new double[LAYER_SIZE];

        // Different patterns for each input
        for (int i = 40; i < 50; i++) {
            bottomUpData[i] = 0.5;  // Bottom-up
            topDownData[i + 5] = 0.3;  // Top-down (shifted)
        }

        layer23.receiveBottomUpInput(new DenseVector(bottomUpData));
        layer23.receiveTopDownPriming(new DenseVector(topDownData));
        layer23.process(new DenseVector(bottomUpData), 0.001);

        var activation = layer23.getActivation();

        // Should have activation from bottom-up
        assertTrue(activation.get(45) > 0.3, "Should have bottom-up activation");

        // Should have some activation from top-down priming
        assertTrue(activation.get(47) > 0.1, "Should have some top-down influence");
    }

    @Test
    void testTimeConstantValidation() {
        // Test that Layer 2/3 has appropriate time constants (30-150ms)
        var params = Layer23Parameters.builder()
            .size(LAYER_SIZE)
            .timeConstant(0.05)  // 50ms - valid
            .build();

        var layer = new Layer23Implementation("test", params);
        assertEquals(0.05, layer.getLayer23Parameters().timeConstant(), TOLERANCE);

        // Test invalid time constants
        assertThrows(IllegalArgumentException.class, () -> {
            Layer23Parameters.builder()
                .size(LAYER_SIZE)
                .timeConstant(0.02)  // 20ms - too fast
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Layer23Parameters.builder()
                .size(LAYER_SIZE)
                .timeConstant(0.2)  // 200ms - too slow
                .build();
        });
    }

    @Test
    void testComplexCellPooling() {
        // Test complex cell pooling of opposite contrasts
        var params = Layer23Parameters.builder()
            .size(LAYER_SIZE)
            .enableComplexCells(true)
            .complexCellThreshold(0.4)
            .build();

        var layer = new Layer23Implementation("complex", params);

        // Create alternating contrast pattern
        var inputData = new double[LAYER_SIZE];
        for (int i = 40; i < 50; i++) {
            inputData[i] = (i % 2 == 0) ? 0.8 : 0.0;  // Alternating
        }

        layer.process(new DenseVector(inputData), 0.001);
        var complexActivation = layer.getComplexCellActivation();

        // Complex cells should pool nearby opposite contrasts
        for (int i = 41; i < 49; i++) {
            assertTrue(((DenseVector) complexActivation).get(i) > 0.2,
                "Complex cell should pool at " + i);
        }
    }

    @Test
    void testPerceptualGroupingPatterns() {
        // Test that perceptual grouping emerges from horizontal connections
        var inputData = new double[LAYER_SIZE];

        // Create two groups
        for (int i = 20; i < 30; i++) {
            inputData[i] = 0.7;
        }
        for (int i = 60; i < 70; i++) {
            inputData[i] = 0.7;
        }

        layer23.process(new DenseVector(inputData), 0.001);

        // Process multiple times for grouping
        for (int i = 0; i < 3; i++) {
            layer23.process(new DenseVector(inputData), 0.001);
        }

        var activation = layer23.getActivation();

        // Each group should maintain coherence
        double group1Sum = 0.0, group2Sum = 0.0;
        for (int i = 20; i < 30; i++) {
            group1Sum += activation.get(i);
        }
        for (int i = 60; i < 70; i++) {
            group2Sum += activation.get(i);
        }

        assertTrue(group1Sum > 5.0, "Group 1 should maintain activity");
        assertTrue(group2Sum > 5.0, "Group 2 should maintain activity");

        // Gap between groups will show some activity due to bipole cell boundary completion
        // This is biologically correct - bipole cells create illusory contours between nearby features
        double gapSum = 0.0;
        for (int i = 35; i < 55; i++) {
            gapSum += activation.get(i);
        }
        // Allow for boundary completion while ensuring gap doesn't become overly active
        assertTrue(gapSum < 12.0, "Gap activity should be limited (boundary completion is expected)");
    }

    @Test
    void testStateManagementAndReset() {
        // Test that state is properly managed and can be reset
        var inputData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            inputData[i] = Math.random() * 0.5;
        }

        layer23.process(new DenseVector(inputData), 0.001);

        // Should have some activation
        var beforeReset = layer23.getActivation();
        double sumBefore = 0.0;
        for (int i = 0; i < LAYER_SIZE; i++) {
            sumBefore += beforeReset.get(i);
        }
        assertTrue(sumBefore > 0, "Should have activation before reset");

        // Reset
        layer23.reset();

        // Should be zero after reset
        var afterReset = layer23.getActivation();
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertEquals(0.0, afterReset.get(i), TOLERANCE, "Should be zero after reset");
        }

        // Should respond to new input after reset
        layer23.process(new DenseVector(inputData), 0.001);
        var afterNewInput = layer23.getActivation();
        double sumAfter = 0.0;
        for (int i = 0; i < LAYER_SIZE; i++) {
            sumAfter += afterNewInput.get(i);
        }
        assertTrue(sumAfter > 0, "Should respond to input after reset");
    }

    @Test
    void testIntegrationWithOtherLayers() {
        // Test that Layer 2/3 can integrate with other layer types
        var layer4 = new Layer4Implementation("layer4", LAYER_SIZE);
        var layer1 = new Layer1Implementation("layer1", LAYER_SIZE);

        // Simulate Layer 4 bottom-up input
        var bottomUpData = new double[LAYER_SIZE];
        for (int i = 45; i < 55; i++) {
            bottomUpData[i] = 0.8;
        }
        layer4.setActivation(new DenseVector(bottomUpData));

        // Simulate Layer 1 top-down
        var topDownData = new double[LAYER_SIZE];
        for (int i = 43; i < 57; i++) {
            topDownData[i] = 0.3;
        }
        layer1.setActivation(new DenseVector(topDownData));

        // Layer 2/3 receives from both
        layer23.receiveBottomUpInput(layer4.getActivation());
        layer23.receiveTopDownPriming(layer1.getActivation());
        layer23.process(layer4.getActivation(), 0.001);

        // Should integrate signals
        var activation = layer23.getActivation();
        assertTrue(activation.get(50) > 0.5, "Should integrate signals from both layers");

        // Check that horizontal grouping is computed
        var grouping = layer23.getHorizontalGrouping();
        assertNotNull(grouping, "Should produce horizontal grouping output");
    }
}