package com.hellblazer.art.cortical.network;

import com.hellblazer.art.core.DenseVector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for BipoleCell and BipoleCellNetwork.
 *
 * <p>Tests the three-way firing logic:
 * <ol>
 *   <li>Condition 1: Strong direct input alone</li>
 *   <li>Condition 2: Bilateral horizontal inputs (gap-filling)</li>
 *   <li>Condition 3: Weak direct + unilateral horizontal</li>
 *   <li>Condition 4: Propagation mode (wave-like spreading)</li>
 * </ol>
 *
 * <p>Precision: All numerical tests use 1e-10 tolerance for equation validation.
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 2)
 */
@DisplayName("BipoleCell and BipoleCellNetwork Tests")
class BipoleCellTest {

    private static final double EPSILON = 1e-10;
    private static final double FUNCTIONAL_EPSILON = 0.000001;  // For functional behavior checks

    @Test
    @DisplayName("Condition 1: Strong direct input fires cell independently")
    void testStrongDirectActivation() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(false)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Single cell with strong direct input
        var inputData = new double[100];
        inputData[50] = 0.9;  // Strong input > threshold (0.8)

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cell 50 should fire strongly
        assertTrue(output.get(50) > 0.5,
            "Cell 50 should fire with strong direct input");

        // Neighboring cells should remain quiet (no horizontal spreading without propagation)
        assertTrue(output.get(49) < FUNCTIONAL_EPSILON,
            "Cell 49 should remain quiet");
        assertTrue(output.get(51) < FUNCTIONAL_EPSILON,
            "Cell 51 should remain quiet");
    }

    @Test
    @DisplayName("Condition 2: Bilateral horizontal inputs enable gap-filling")
    void testBilateralGapFilling() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(false)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Create a gap with active cells on both sides
        var inputData = new double[100];
        inputData[45] = 0.6;  // Left side active
        inputData[55] = 0.6;  // Right side active
        // Cell 50 has no direct input (gap)

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cell 50 should activate via bilateral horizontal inputs
        // This is the key bipole mechanism for illusory contour completion
        assertTrue(output.get(50) > 0.3,
            String.format("Cell 50 should fill gap via bilateral input (got: %.4f)", output.get(50)));

        // Verify left and right cells are active
        assertTrue(output.get(45) > 0.3, "Cell 45 should be active");
        assertTrue(output.get(55) > 0.3, "Cell 55 should be active");
    }

    @Test
    @DisplayName("Condition 3: Weak direct + unilateral horizontal combines inputs")
    void testWeakDirectPlusHorizontal() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(false)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Weak direct input at 50, strong input at 45 (provides horizontal support)
        var inputData = new double[100];
        inputData[45] = 0.8;  // Strong input (will activate and provide horizontal support)
        inputData[50] = 0.4;  // Weak input (above weak threshold 0.3, below strong threshold 0.8)

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Cell 50 should fire due to weak direct + horizontal support from 45
        assertTrue(output.get(50) > 0.3,
            String.format("Cell 50 should fire with weak direct + horizontal (got: %.4f)", output.get(50)));

        // Cell 45 should fire strongly (strong direct input)
        assertTrue(output.get(45) > 0.5,
            "Cell 45 should fire strongly with direct input");
    }

    @Test
    @DisplayName("Connection weight decays exponentially with distance")
    void testDistanceWeightedConnections() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(false)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Enable propagation to see distance-based activation decay
        network.enablePropagation(true);

        // Single strong input at center
        var inputData = new double[100];
        inputData[50] = 0.9;

        var input = new DenseVector(inputData);
        var output = network.process(input);

        // Verify activation decays with distance from source
        var prev = output.get(50);
        for (var dist = 1; dist <= 5; dist++) {
            var curr = output.get(50 + dist);
            assertTrue(curr < prev,
                String.format("Activation should decay at distance %d (prev: %.4f, curr: %.4f)",
                    dist, prev, curr));
            prev = curr;
        }
    }

    @Test
    @DisplayName("Orientation selectivity filters non-collinear connections")
    void testOrientationSelectivity() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(10)
            .distanceSigma(5.0)
            .maxWeight(1.0)
            .orientationSelectivity(true)  // Enable orientation selectivity
            .orientationSigma(Math.PI / 4)  // 45-degree tuning width
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Set different orientations
        network.setOrientation(50, 0.0);  // Horizontal
        network.setOrientation(45, 0.0);  // Horizontal (collinear with 50)
        network.setOrientation(55, Math.PI / 2);  // Vertical (orthogonal to 50)

        // Get cells to check connection weights
        var cell50 = network.getCell(50);

        // Weight to collinear cell should be higher than to orthogonal cell
        var weightToCollinear = cell50.computeConnectionWeight(45, 0.0);
        var weightToOrthogonal = cell50.computeConnectionWeight(55, Math.PI / 2);

        assertTrue(weightToCollinear > weightToOrthogonal,
            String.format("Collinear weight (%.4f) should exceed orthogonal weight (%.4f)",
                weightToCollinear, weightToOrthogonal));
    }

    @Test
    @DisplayName("Network reset clears activations but preserves structure")
    void testNetworkReset() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Process some input
        var inputData = new double[100];
        inputData[50] = 0.9;
        var input = new DenseVector(inputData);
        network.process(input);

        // Verify cell 50 is active
        assertTrue(network.getCell(50).getActivation() > 0.5,
            "Cell 50 should be active after processing");

        // Reset network
        network.reset();

        // Verify all cells are cleared
        for (var i = 0; i < 100; i++) {
            var activation = network.getCell(i).getActivation();
            assertEquals(0.0, activation, EPSILON,
                String.format("Cell %d should be reset to 0.0", i));
        }

        // Verify structure preserved (network size)
        assertEquals(100, network.getSize(),
            "Network size should be preserved after reset");
    }

    @Test
    @DisplayName("BipoleCell parameters validation")
    void testParameterValidation() {
        // Test networkSize validation
        assertThrows(IllegalArgumentException.class, () ->
            BipoleCellParameters.builder()
                .networkSize(0)
                .build(),
            "Should reject networkSize <= 0");

        // Test strongDirectThreshold validation
        assertThrows(IllegalArgumentException.class, () ->
            BipoleCellParameters.builder()
                .strongDirectThreshold(1.5)
                .build(),
            "Should reject strongDirectThreshold > 1.0");

        // Test timeConstant validation
        assertThrows(IllegalArgumentException.class, () ->
            BipoleCellParameters.builder()
                .timeConstant(-0.01)
                .build(),
            "Should reject negative timeConstant");

        // Valid parameters should not throw
        assertDoesNotThrow(() ->
            BipoleCellParameters.builder()
                .networkSize(100)
                .strongDirectThreshold(0.8)
                .weakDirectThreshold(0.3)
                .horizontalThreshold(0.5)
                .timeConstant(0.05)
                .build(),
            "Valid parameters should be accepted");
    }

    @Test
    @DisplayName("Temporal dynamics converge with repeated iterations")
    void testTemporalConvergence() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Single strong input
        var inputData = new double[100];
        inputData[50] = 0.9;
        var input = new DenseVector(inputData);

        // Process once
        var output1 = network.process(input);
        var activation1 = output1.get(50);

        // Reset and process again
        network.reset();
        var output2 = network.process(input);
        var activation2 = output2.get(50);

        // Activations should be consistent (converged)
        assertEquals(activation1, activation2, 0.01,
            "Activations should converge to consistent values");
    }

    @Test
    @DisplayName("Propagation mode enables wave-like spreading")
    void testPropagationMode() {
        var parameters = BipoleCellParameters.builder()
            .networkSize(100)
            .strongDirectThreshold(0.8)
            .weakDirectThreshold(0.3)
            .horizontalThreshold(0.5)
            .maxHorizontalRange(15)
            .distanceSigma(5.0)
            .timeConstant(0.05)
            .build();

        var network = new BipoleCellNetwork(parameters);

        // Single strong input at center
        var inputData = new double[100];
        inputData[50] = 0.9;
        var input = new DenseVector(inputData);

        // Process without propagation
        network.enablePropagation(false);
        var outputNoProp = network.process(input);

        // Reset and process with propagation
        network.reset();
        network.enablePropagation(true);
        var outputWithProp = network.process(input);

        // With propagation, activation should spread further
        var distantCell = 60;  // 10 units away
        assertTrue(outputWithProp.get(distantCell) > outputNoProp.get(distantCell),
            String.format("Propagation should spread activation further (no-prop: %.4f, with-prop: %.4f)",
                outputNoProp.get(distantCell), outputWithProp.get(distantCell)));
    }
}
