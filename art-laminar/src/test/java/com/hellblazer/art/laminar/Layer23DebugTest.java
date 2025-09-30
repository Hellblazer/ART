package com.hellblazer.art.laminar;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.layers.Layer23Implementation;
import org.junit.jupiter.api.Test;

/**
 * Debug test for Layer 2/3 implementation issues.
 */
public class Layer23DebugTest {

    @Test
    void debugBottomUpSignalReception() {
        System.out.println("\n=== Debug Bottom-Up Signal Reception ===");

        var layer23 = new Layer23Implementation("debug", 100);

        // Create bottom-up input
        var bottomUpData = new double[100];
        for (int i = 40; i < 60; i++) {
            bottomUpData[i] = 0.8;
        }
        var bottomUpInput = new DenseVector(bottomUpData);

        System.out.println("Bottom-up input created (cells 40-59 = 0.8)");

        // Receive bottom-up input
        layer23.receiveBottomUpInput(bottomUpInput);
        System.out.println("Bottom-up input received");

        // Process with the bottom-up input (using same as direct input for debugging)
        layer23.process(bottomUpInput, 0.001);
        System.out.println("Processing completed");

        var activation = layer23.getActivation();

        // Check activations
        System.out.println("\nActivations after processing:");
        for (int i = 35; i < 65; i++) {
            if (i == 40 || i == 50 || i == 59) {
                System.out.printf("  Cell %d: %.4f%s\n", i, activation.get(i),
                    (i >= 40 && i < 60) ? " (should be > 0.5)" : "");
            }
        }

        // Check specific positions
        boolean hasActivation40 = activation.get(40) > 0.5;
        boolean hasActivation50 = activation.get(50) > 0.5;
        boolean hasActivation59 = activation.get(59) > 0.5;

        System.out.println("\nTest results:");
        System.out.println("  Cell 40: " + (hasActivation40 ? "✓ PASS" : "✗ FAIL"));
        System.out.println("  Cell 50: " + (hasActivation50 ? "✓ PASS" : "✗ FAIL"));
        System.out.println("  Cell 59: " + (hasActivation59 ? "✓ PASS" : "✗ FAIL"));

        // Check if any cells have activation
        double totalActivation = 0.0;
        for (int i = 0; i < 100; i++) {
            totalActivation += activation.get(i);
        }
        System.out.printf("\nTotal activation across layer: %.4f\n", totalActivation);

        if (totalActivation < 0.01) {
            System.out.println("WARNING: Layer has almost no activation!");
        }
    }

    @Test
    void debugProcessingPipeline() {
        System.out.println("\n=== Debug Processing Pipeline ===");

        var layer23 = new Layer23Implementation("debug", 10);

        // Simple input
        var inputData = new double[10];
        inputData[5] = 1.0;
        var input = new DenseVector(inputData);

        System.out.println("Input: cell 5 = 1.0");

        // Process directly
        System.out.println("\n1. Direct processing:");
        layer23.reset();
        layer23.process(input, 0.001);
        var directActivation = layer23.getActivation();
        System.out.printf("  Cell 5 activation: %.4f\n", directActivation.get(5));

        // Process with bottom-up
        System.out.println("\n2. With bottom-up input:");
        layer23.reset();
        layer23.receiveBottomUpInput(input);
        layer23.process(new DenseVector(new double[10]), 0.001);  // Empty direct input
        var bottomUpActivation = layer23.getActivation();
        System.out.printf("  Cell 5 activation: %.4f\n", bottomUpActivation.get(5));

        // Process with both
        System.out.println("\n3. With both bottom-up and direct:");
        layer23.reset();
        layer23.receiveBottomUpInput(input);
        layer23.process(input, 0.001);
        var bothActivation = layer23.getActivation();
        System.out.printf("  Cell 5 activation: %.4f\n", bothActivation.get(5));
    }

    @Test
    void debugComplexCellPooling() {
        System.out.println("\n=== Debug Complex Cell Pooling ===");

        var params = com.hellblazer.art.laminar.parameters.Layer23Parameters.builder()
            .size(100)
            .enableComplexCells(true)
            .complexCellThreshold(0.4)
            .build();

        var layer = new com.hellblazer.art.laminar.layers.Layer23Implementation("complex", params);

        // Create alternating contrast pattern
        var inputData = new double[100];
        for (int i = 40; i < 50; i++) {
            inputData[i] = (i % 2 == 0) ? 0.8 : 0.0;  // Alternating
        }

        System.out.println("Input pattern (alternating 0.8/0.0 from cells 40-49):");
        for (int i = 40; i < 50; i++) {
            System.out.printf("  Cell %d: %.1f\n", i, inputData[i]);
        }

        layer.process(new com.hellblazer.art.core.DenseVector(inputData), 0.001);
        var complexActivation = layer.getComplexCellActivation();

        System.out.println("\nComplex cell activation:");
        for (int i = 40; i < 50; i++) {
            double value = ((com.hellblazer.art.core.DenseVector) complexActivation).get(i);
            System.out.printf("  Cell %d: %.4f%s\n", i, value,
                value > 0.2 ? " ✓" : " ✗");
        }

        // Check specific cell
        double cell41 = ((com.hellblazer.art.core.DenseVector) complexActivation).get(41);
        System.out.printf("\nCell 41 activation: %.4f (expected > 0.2)\n", cell41);
        if (cell41 > 0.2) {
            System.out.println("✓ TEST WOULD PASS");
        } else {
            System.out.println("✗ TEST FAILS");
        }
    }
}