package com.hellblazer.art.laminar.network;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.BipoleCellParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BipoleCellNetwork - validates three-way firing logic,
 * boundary completion, and horizontal grouping mechanisms.
 *
 * @author Hal Hildebrand
 */
public class BipoleCellNetworkTest {

    private BipoleCellNetwork network;
    private BipoleCellParameters parameters;
    private static final int NETWORK_SIZE = 100;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    void setUp() {
        parameters = BipoleCellParameters.builder()
            .networkSize(NETWORK_SIZE)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(true)
            .build();
        network = new BipoleCellNetwork(parameters);
    }

    @Test
    void testSingleCellDirectActivation() {
        // Test Condition 1: Strong direct input alone fires cell
        var inputData = new double[NETWORK_SIZE];
        inputData[50] = 0.9; // Strong direct input above threshold

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cell 50 should fire due to strong direct input
        assertTrue(output.get(50) > 0.5, "Cell with strong direct input should fire");

        // Neighboring cells should not fire without horizontal support
        assertEquals(0.0, output.get(49), TOLERANCE, "Neighbor without input should not fire");
        assertEquals(0.0, output.get(51), TOLERANCE, "Neighbor without input should not fire");
    }

    @Test
    void testBilateralHorizontalActivation() {
        // Test Condition 2: Both horizontal sides active (boundary completion)
        var inputData = new double[NETWORK_SIZE];

        // Create a gap with active cells on both sides
        inputData[45] = 0.6; // Left side active
        inputData[55] = 0.6; // Right side active
        // Cell 50 has no direct input (gap)

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cell 50 should fire due to bilateral horizontal activation
        assertTrue(output.get(50) > 0.3, "Cell with bilateral horizontal input should fire for boundary completion");
    }

    @Test
    void testCollinearFacilitation() {
        // Test collinear facilitation along a straight line
        var inputData = new double[NETWORK_SIZE];

        // Create a collinear pattern
        for (int i = 40; i <= 60; i += 2) {
            inputData[i] = 0.7;
        }

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cells along the line should be facilitated
        for (int i = 40; i <= 60; i++) {
            assertTrue(output.get(i) > 0.4, "Collinear cells should be facilitated at position " + i);
        }
    }

    @Test
    void testBoundaryCompletion() {
        // Test boundary completion across gaps (illusory contours)
        var inputData = new double[NETWORK_SIZE];

        // Create pattern with gaps
        inputData[30] = 0.8;
        inputData[31] = 0.8;
        inputData[32] = 0.8;
        // Gap at 33-34
        inputData[35] = 0.8;
        inputData[36] = 0.8;
        inputData[37] = 0.8;

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Gap should be filled by boundary completion
        assertTrue(output.get(33) > 0.3, "Gap position 33 should be filled");
        assertTrue(output.get(34) > 0.3, "Gap position 34 should be filled");
    }

    @Test
    void testThreeWayFiringLogic() {
        // Comprehensive test of all three firing conditions
        var inputData = new double[NETWORK_SIZE];

        // Condition 1: Strong direct (cell 10)
        inputData[10] = 0.85;

        // Condition 2: Bilateral (cell 20 with neighbors at 15 and 25)
        inputData[15] = 0.6;
        inputData[25] = 0.6;

        // Condition 3: Weak direct + one side (cell 40)
        inputData[40] = 0.35; // Weak direct
        inputData[45] = 0.7;  // One side active

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Verify all three conditions trigger firing
        assertTrue(output.get(10) > 0.5, "Condition 1: Strong direct should fire");
        assertTrue(output.get(20) > 0.3, "Condition 2: Bilateral should fire");
        assertTrue(output.get(40) > 0.3, "Condition 3: Weak+side should fire");
    }

    @Test
    void testDistanceWeightedConnections() {
        // Test that connection strength decreases with distance
        var inputData = new double[NETWORK_SIZE];

        // Central strong input
        inputData[50] = 0.8;

        // Set network to propagate based on distance
        network.enablePropagation(true);
        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Check distance-based decay
        double prev = output.get(50);
        for (int dist = 1; dist <= 5; dist++) {
            double curr = output.get(50 + dist);
            assertTrue(curr < prev, "Activation should decay with distance at " + dist);
            prev = curr;
        }
    }

    @Test
    void testOrientationSelectivity() {
        // Test orientation-selective connections for boundary completion
        var inputData = new double[NETWORK_SIZE];

        // Create oriented edge segments
        // Horizontal orientation
        inputData[40] = 0.7;
        inputData[41] = 0.7;
        inputData[42] = 0.7;

        // Gap

        // Matching orientation
        inputData[45] = 0.7;
        inputData[46] = 0.7;
        inputData[47] = 0.7;

        network.setOrientation(40, 0.0); // Horizontal
        network.setOrientation(41, 0.0);
        network.setOrientation(42, 0.0);
        network.setOrientation(45, 0.0);
        network.setOrientation(46, 0.0);
        network.setOrientation(47, 0.0);

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Gap should be completed due to matching orientations
        assertTrue(output.get(43) > 0.3, "Gap with matching orientations should be completed");
        assertTrue(output.get(44) > 0.3, "Gap with matching orientations should be completed");
    }

    @Test
    void testPopulationDynamics() {
        // Test network-wide population dynamics
        var inputData = new double[NETWORK_SIZE];

        // Create multiple active regions
        for (int center : new int[]{20, 50, 80}) {
            for (int i = -2; i <= 2; i++) {
                inputData[center + i] = 0.6;
            }
        }

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Each region should maintain activity
        for (int center : new int[]{20, 50, 80}) {
            double regionSum = 0.0;
            for (int i = -2; i <= 2; i++) {
                regionSum += output.get(center + i);
            }
            assertTrue(regionSum > 2.0, "Region around " + center + " should maintain population activity");
        }
    }

    @Test
    void testIllusoryContourFormation() {
        // Test formation of illusory contours (Kanizsa triangle-like)
        var inputData = new double[NETWORK_SIZE];

        // Create pac-man like inducers
        // Left inducer
        for (int i = 20; i <= 25; i++) {
            if (i != 23) inputData[i] = 0.8; // Gap at 23
        }

        // Right inducer
        for (int i = 35; i <= 40; i++) {
            if (i != 37) inputData[i] = 0.8; // Gap at 37
        }

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Illusory contour should form between gaps
        double contourStrength = 0.0;
        for (int i = 26; i <= 34; i++) {
            contourStrength += output.get(i);
        }
        assertTrue(contourStrength > 1.0, "Illusory contour should form between inducers");
    }

    @Test
    void testEdgeCasesNoInput() {
        // Test network behavior with no input
        var inputData = new double[NETWORK_SIZE];
        // All zeros

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // No cells should fire without input
        for (int i = 0; i < NETWORK_SIZE; i++) {
            assertEquals(0.0, output.get(i), TOLERANCE, "No cells should fire without input");
        }
    }

    @Test
    void testEdgeCasesMaxInput() {
        // Test network behavior with maximum input
        var inputData = new double[NETWORK_SIZE];
        for (int i = 0; i < NETWORK_SIZE; i++) {
            inputData[i] = 1.0;
        }

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // All cells should fire strongly
        for (int i = 0; i < NETWORK_SIZE; i++) {
            assertTrue(output.get(i) > 0.8, "All cells should fire with max input at " + i);
        }
    }

    @Test
    void testPerformance() {
        // Test that processing 100 cells completes within 2ms
        var inputData = new double[NETWORK_SIZE];

        // Create realistic input pattern
        for (int i = 0; i < NETWORK_SIZE; i += 3) {
            inputData[i] = Math.random() * 0.8;
        }

        // Create input vector
        var input = new DenseVector(inputData);

        // Warm up
        for (int i = 0; i < 10; i++) {
            network.process(input);
        }

        // Measure performance
        long startTime = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            network.process(input);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / (100.0 * 1_000_000);
        assertTrue(avgTimeMs < 2.0, "Processing should complete within 2ms, took: " + avgTimeMs + "ms");
    }
}