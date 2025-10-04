package com.hellblazer.art.laminar.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.core.WeightMatrix;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.Layer4Parameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Layer 4 (Thalamic Driving Input).
 * Layer 4 receives driving input from thalamus (LGN) and initiates cortical processing.
 *
 * Key characteristics:
 * - Strong driving signals that can fire cells
 * - Fast time constants (10-50ms)
 * - Simple feedforward processing
 * - No lateral inhibition initially
 * - Integration with shunting dynamics
 *
 * @author Hal Hildebrand
 */
public class Layer4Test {

    private Layer4Implementation layer4;
    private Layer4Parameters parameters;
    private static final int LAYER_SIZE = 100;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    void setUp() {
        layer4 = new Layer4Implementation("L4", LAYER_SIZE);
        parameters = Layer4Parameters.builder()
            .timeConstant(25.0) // Mid-range: 10-50ms
            .drivingStrength(0.8)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0) // No lateral inhibition initially
            .build();
    }

    @Test
    void testBasicActivationWithDrivingInput() {
        // Test that Layer 4 can be driven by thalamic input
        var inputData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            inputData[i] = Math.random() * 0.5; // Moderate input
        }
        var input = new DenseVector(inputData);

        var output = layer4.processBottomUp(input, parameters);

        assertNotNull(output, "Output should not be null");
        assertEquals(LAYER_SIZE, output.dimension(), "Output dimension should match layer size");

        // Verify activation is driven by input
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertTrue(output.get(i) >= 0, "Activation should be non-negative");
            assertTrue(output.get(i) <= parameters.getCeiling(),
                "Activation should not exceed ceiling");
        }
    }

    @Test
    void testTimeConstantValidation() {
        // Test fast time constants (10-50ms range)
        var fastParams = Layer4Parameters.builder()
            .timeConstant(10.0) // Minimum
            .drivingStrength(0.8)
            .build();

        var slowParams = Layer4Parameters.builder()
            .timeConstant(50.0) // Maximum
            .drivingStrength(0.8)
            .build();

        var input = createTestInput(0.5);

        // Fast dynamics should reach steady state quickly
        var fastOutput = layer4.processBottomUp(input, fastParams);
        var slowOutput = layer4.processBottomUp(input, slowParams);

        // Both should be valid but potentially different
        assertNotNull(fastOutput);
        assertNotNull(slowOutput);

        // Test invalid time constant
        assertThrows(IllegalArgumentException.class, () -> {
            Layer4Parameters.builder()
                .timeConstant(100.0) // Too slow for Layer 4
                .drivingStrength(0.8)
                .build();
        });
    }

    @Test
    void testSignalPropagationAccuracy() {
        // Test that signals propagate accurately through the layer
        var sparseInputData = new double[LAYER_SIZE];
        sparseInputData[10] = 1.0; // Single active input
        sparseInputData[50] = 0.5;
        sparseInputData[90] = 0.75;
        var sparseInput = new DenseVector(sparseInputData);

        var output = layer4.processBottomUp(sparseInput, parameters);

        // Active inputs should generate stronger responses
        assertTrue(output.get(10) > 0, "Active input should generate response");
        assertTrue(output.get(50) > 0, "Active input should generate response");
        assertTrue(output.get(90) > 0, "Active input should generate response");

        // Response should be proportional to input strength
        assertTrue(output.get(10) > output.get(50),
            "Stronger input should generate stronger response");
    }

    @Test
    void testParameterValidation() {
        // Test parameter bounds and validation
        assertThrows(IllegalArgumentException.class, () -> {
            Layer4Parameters.builder()
                .timeConstant(-1.0) // Invalid negative
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Layer4Parameters.builder()
                .drivingStrength(-0.1) // Invalid negative
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Layer4Parameters.builder()
                .drivingStrength(1.1) // Too strong
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Layer4Parameters.builder()
                .ceiling(0.5)
                .floor(0.6) // Floor > ceiling
                .build();
        });
    }

    @Test
    void testIntegrationWithShuntingDynamics() {
        // Test that Layer 4 properly integrates with shunting dynamics
        var input = createTestInput(0.7);

        // Process multiple time steps
        Pattern output = input;
        for (int t = 0; t < 10; t++) {
            output = layer4.processBottomUp(output, parameters);
        }

        // Should reach steady state with shunting dynamics
        var steadyState = layer4.processBottomUp(output, parameters);

        // Check convergence (steady state)
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertEquals(output.get(i), steadyState.get(i), 0.01,
                "Should reach steady state");
        }
    }

    @Test
    void testBottomUpProcessing() {
        // Test pure bottom-up processing without top-down influence
        var input = createTestInput(0.6);

        // Bottom-up should work independently
        var output = layer4.processBottomUp(input, parameters);

        // Should be driven purely by input
        assertNotNull(output);
        assertTrue(averageActivation(output) > 0,
            "Bottom-up should generate activation");

        // No top-down influence should be needed
        var topDown = new DenseVector(new double[LAYER_SIZE]); // Zero top-down
        var modulated = layer4.processTopDown(topDown, parameters);

        // Top-down with zero input should not enhance
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertTrue(modulated.get(i) <= output.get(i) * 1.1,
                "Zero top-down should not significantly enhance");
        }
    }

    @Test
    void testActivationBoundsChecking() {
        // Test that activations stay within biological bounds
        var maxInputData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            maxInputData[i] = 1.0; // Maximum input
        }
        var maxInput = new DenseVector(maxInputData);

        var output = layer4.processBottomUp(maxInput, parameters);

        for (int i = 0; i < LAYER_SIZE; i++) {
            assertTrue(output.get(i) >= parameters.getFloor(),
                "Activation should not go below floor");
            assertTrue(output.get(i) <= parameters.getCeiling(),
                "Activation should not exceed ceiling");
        }
    }

    @Test
    void testMultipleInputPatterns() {
        // Test processing of multiple different patterns
        Pattern[] patterns = {
            createTestInput(0.2), // Weak
            createTestInput(0.5), // Medium
            createTestInput(0.8), // Strong
            createSparseInput(10), // Sparse
            createSparseInput(50)  // Dense sparse
        };

        for (var pattern : patterns) {
            var output = layer4.processBottomUp(pattern, parameters);
            assertNotNull(output, "Should process all patterns");
            assertEquals(LAYER_SIZE, output.dimension(),
                "Output dimension should be consistent");

            // Verify reasonable activation levels
            var avgAct = averageActivation(output);
            assertTrue(avgAct >= 0 && avgAct <= 1.0,
                "Average activation should be in valid range");
        }
    }

    @Test
    void testStatePersistence() {
        // Test that layer maintains state across processing steps
        var input1 = createTestInput(0.5);
        var output1 = layer4.processBottomUp(input1, parameters);

        var currentActivation = layer4.getActivation();
        assertNotNull(currentActivation);

        // State should persist
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertEquals(output1.get(i), currentActivation.get(i), EPSILON,
                "Activation state should persist");
        }

        // Process new input
        var input2 = createTestInput(0.3);
        var output2 = layer4.processBottomUp(input2, parameters);

        // State should update
        currentActivation = layer4.getActivation();
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertEquals(output2.get(i), currentActivation.get(i), EPSILON,
                "Activation state should update");
        }
    }

    @Test
    void testResetFunctionality() {
        // Test reset clears all state
        var input = createTestInput(0.7);
        layer4.processBottomUp(input, parameters);

        // Verify activation exists
        var beforeReset = layer4.getActivation();
        assertTrue(averageActivation(beforeReset) > 0,
            "Should have activation before reset");

        // Reset
        layer4.reset();

        // Verify reset
        var afterReset = layer4.getActivation();
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertEquals(0.0, afterReset.get(i), EPSILON,
                "Activation should be zero after reset");
        }
    }

    @Test
    void testEdgeCasesZeroInput() {
        // Test with zero input
        var zeroInput = new DenseVector(new double[LAYER_SIZE]);
        var output = layer4.processBottomUp(zeroInput, parameters);

        // Should handle gracefully
        assertNotNull(output);
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertTrue(output.get(i) >= 0, "Should handle zero input");
            assertTrue(output.get(i) <= 0.1,
                "Zero input should produce minimal activation");
        }
    }

    @Test
    void testEdgeCasesMaxInput() {
        // Test with maximum input
        var maxInputData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            maxInputData[i] = 1.0;
        }
        var maxInput = new DenseVector(maxInputData);

        var output = layer4.processBottomUp(maxInput, parameters);

        // Should saturate but not exceed bounds
        assertNotNull(output);
        for (int i = 0; i < LAYER_SIZE; i++) {
            assertTrue(output.get(i) <= parameters.getCeiling(),
                "Should not exceed ceiling with max input");
            // Sigmoid saturation means max input gives ~0.44
            assertTrue(output.get(i) > 0.4,
                "Max input should produce strong activation");
        }
    }

    @Test
    void testPerformanceBenchmark() {
        // Test performance meets <1ms requirement for typical patterns
        var input = createTestInput(0.5);

        // Warm up
        for (int i = 0; i < 100; i++) {
            layer4.processBottomUp(input, parameters);
        }

        // Measure
        long startTime = System.nanoTime();
        int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            layer4.processBottomUp(input, parameters);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / (iterations * 1_000_000.0);
        assertTrue(avgTimeMs < 1.0,
            "Processing should be faster than 1ms per pattern, was: " + avgTimeMs);
    }

    @Test
    void testBiologicalFiringRateConstraints() {
        // Test that firing rates stay within biological range (0-100Hz)
        var input = createTestInput(0.6);
        var timeStepMs = parameters.getTimeConstant() / 10; // Simulation time step

        // Process for 1 second (1000ms)
        int steps = (int)(1000.0 / timeStepMs);
        Pattern current = input;

        double[] firingRates = new double[LAYER_SIZE];
        for (int step = 0; step < steps; step++) {
            current = layer4.processBottomUp(current, parameters);

            // Count spikes (activation > threshold)
            for (int i = 0; i < LAYER_SIZE; i++) {
                if (current.get(i) > 0.5) { // Spike threshold
                    firingRates[i]++;
                }
            }
        }

        // Convert to Hz
        for (int i = 0; i < LAYER_SIZE; i++) {
            firingRates[i] = firingRates[i]; // Already in Hz (1 second simulation)
            assertTrue(firingRates[i] >= 0 && firingRates[i] <= 100,
                "Firing rate should be in biological range 0-100Hz, was: " + firingRates[i]);
        }
    }

    @Test
    void testBackwardCompatibility() {
        // Test that Layer 4 integrates with existing circuit infrastructure
        assertEquals("L4", layer4.getId(), "ID should be set correctly");
        assertEquals(LAYER_SIZE, layer4.size(), "Size should be set correctly");
        assertEquals(LayerType.CUSTOM, layer4.getType(), "Type should be CUSTOM for now");

        // Test weight matrix
        var weights = layer4.getWeights();
        assertNotNull(weights, "Should have weight matrix");
        assertEquals(LAYER_SIZE, weights.getRows(), "Weight matrix rows should match size");
        assertEquals(LAYER_SIZE, weights.getCols(), "Weight matrix cols should match size");

        // Test listener support
        var listenerCalled = new boolean[1];
        layer4.addActivationListener(new LayerActivationListener() {
            @Override
            public void onActivationChanged(String layerId, Pattern oldActivation, Pattern newActivation) {
                listenerCalled[0] = true;
                assertEquals("L4", layerId);
                assertNotNull(newActivation);
            }
        });

        var input = createTestInput(0.5);
        layer4.setActivation(input);
        assertTrue(listenerCalled[0], "Listener should be called on activation change");
    }

    // Helper methods

    private Pattern createTestInput(double strength) {
        var inputData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            inputData[i] = Math.random() * strength;
        }
        return new DenseVector(inputData);
    }

    private Pattern createSparseInput(int activeCount) {
        var inputData = new double[LAYER_SIZE];
        for (int i = 0; i < activeCount; i++) {
            int idx = (int)(Math.random() * LAYER_SIZE);
            inputData[idx] = Math.random();
        }
        return new DenseVector(inputData);
    }

    private double averageActivation(Pattern pattern) {
        double sum = 0;
        for (int i = 0; i < pattern.dimension(); i++) {
            sum += pattern.get(i);
        }
        return sum / pattern.dimension();
    }
}