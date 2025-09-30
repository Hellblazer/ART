package com.hellblazer.art.laminar;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.network.BipoleCellNetwork;
import com.hellblazer.art.laminar.parameters.BipoleCellParameters;
import org.junit.jupiter.api.Test;

/**
 * Debug test for understanding bipole cell behavior
 */
public class BipoleCellDebugTest {

    @Test
    void debugBilateralActivation() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(false) // Disable for simpler testing
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Create a gap with active cells on both sides
        var inputData = new double[100];
        inputData[45] = 0.6; // Left side active
        inputData[55] = 0.6; // Right side active
        // Cell 50 has no direct input (gap)

        System.out.println("\n=== Debug Bilateral Activation Test ===");
        System.out.println("Parameters:");
        System.out.println("  Max horizontal range: " + parameters.maxHorizontalRange());
        System.out.println("  Distance sigma: " + parameters.distanceSigma());
        System.out.println("  Max weight: " + parameters.maxWeight());

        // Check connection weights before processing
        var cell45 = network.getCell(45);
        var cell50 = network.getCell(50);
        var cell55 = network.getCell(55);

        System.out.println("\nConnection weights (before processing):");
        System.out.println("  Weight from 45 to 50: " + cell45.computeConnectionWeight(50, 0.0));
        System.out.println("  Weight from 55 to 50: " + cell55.computeConnectionWeight(50, 0.0));

        var input = new DenseVector(inputData);
        var output = network.process(input);

        System.out.println("\nInputs:");
        System.out.println("  Input at 45: " + inputData[45]);
        System.out.println("  Input at 50: " + inputData[50] + " (gap)");
        System.out.println("  Input at 55: " + inputData[55]);

        System.out.println("\nOutputs:");
        System.out.println("  Output at 45: " + output.get(45));
        System.out.println("  Output at 50: " + output.get(50) + " (expected > 0.3)");
        System.out.println("  Output at 55: " + output.get(55));

        // Check intermediate cells
        System.out.println("\nIntermediate cells (46-54):");
        for (int i = 46; i <= 54; i++) {
            System.out.printf("  Cell %d: %.4f\n", i, output.get(i));
        }

        // Check cell 50's inputs
        System.out.println("\nCell 50 internal state:");
        System.out.println("  Direct input: " + cell50.getDirectInput());
        System.out.println("  Left horizontal: " + cell50.getLeftHorizontalInput());
        System.out.println("  Right horizontal: " + cell50.getRightHorizontalInput());
        System.out.println("  Activation: " + cell50.getActivation());

        // Test assertion
        if (output.get(50) > 0.3) {
            System.out.println("\n✓ TEST WOULD PASS: Cell 50 fires with bilateral input");
        } else {
            System.out.println("\n✗ TEST FAILS: Cell 50 does not fire with bilateral input");
        }
    }

    @Test
    void debugSingleCellDirectActivation() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(true)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        System.out.println("\n=== Debug Single Cell Direct Activation Test ===");

        // Test Condition 1: Strong direct input alone fires cell
        var inputData = new double[100];
        inputData[50] = 0.9; // Strong direct input above threshold

        var input = new DenseVector(inputData);
        var output = network.process(input);

        System.out.println("Input at cell 50: " + inputData[50]);
        System.out.println("Strong direct threshold: " + parameters.strongDirectThreshold());
        System.out.println("Output at cell 50: " + output.get(50) + " (expected > 0.5)");

        // Check neighboring cells
        System.out.println("\nNeighboring cells:");
        System.out.println("  Cell 49: " + output.get(49) + " (expected < 0.000001)");
        System.out.println("  Cell 51: " + output.get(51) + " (expected < 0.000001)");

        var cell50 = network.getCell(50);
        System.out.println("\nCell 50 internal state:");
        System.out.println("  Direct input: " + cell50.getDirectInput());
        System.out.println("  Left horizontal: " + cell50.getLeftHorizontalInput());
        System.out.println("  Right horizontal: " + cell50.getRightHorizontalInput());
        System.out.println("  Activation: " + cell50.getActivation());

        // Test assertions
        if (output.get(50) > 0.5) {
            System.out.println("\n✓ Cell 50 fires with strong direct input");
        } else {
            System.out.println("\n✗ Cell 50 does not fire with strong direct input");
        }

        if (output.get(49) < 0.000001 && output.get(51) < 0.000001) {
            System.out.println("✓ Neighbors remain quiet");
        } else {
            System.out.println("✗ Neighbors are incorrectly activated");
        }
    }

    @Test
    void debugDistanceWeightedConnections() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(true)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        System.out.println("\n=== Debug Distance Weighted Connections Test ===");

        // Central strong input
        var inputData = new double[100];
        inputData[50] = 0.8;

        // Set network to propagate based on distance
        network.enablePropagation(true);
        var input = new DenseVector(inputData);
        var output = network.process(input);

        System.out.println("Input at cell 50: " + inputData[50]);
        System.out.println("Propagation enabled: true");
        System.out.println("\nActivation decay with distance:");

        // Check distance-based decay
        double prev = output.get(50);
        System.out.printf("  Distance 0: %.4f (cell 50)\n", prev);

        boolean decayCorrect = true;
        for (int dist = 1; dist <= 5; dist++) {
            double curr = output.get(50 + dist);
            System.out.printf("  Distance %d: %.4f (cell %d)", dist, curr, 50 + dist);

            if (curr < prev) {
                System.out.println(" ✓ decays");
            } else {
                System.out.println(" ✗ does not decay");
                decayCorrect = false;
            }
            prev = curr;
        }

        if (decayCorrect) {
            System.out.println("\n✓ TEST WOULD PASS: Activation decays with distance");
        } else {
            System.out.println("\n✗ TEST FAILS: Activation does not decay with distance");
        }
    }
}