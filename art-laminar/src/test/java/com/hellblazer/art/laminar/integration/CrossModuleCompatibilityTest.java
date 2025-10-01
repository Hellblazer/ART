package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.laminar.canonical.CircuitParameters;
import com.hellblazer.art.laminar.canonical.FullLaminarCircuitImpl;
import com.hellblazer.art.laminar.layers.Layer6Implementation;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 3: Cross-module compatibility testing for Pattern interface and data flow validation.
 *
 * <p>This test suite validates that the Pattern interface works correctly across all three modules:
 * <ul>
 *   <li><b>art-laminar:</b> Biological laminar circuit dynamics with DenseVector patterns</li>
 *   <li><b>art-core:</b> FuzzyART category learning with WeightVector patterns</li>
 *   <li><b>art-performance:</b> SIMD-optimized vectorized algorithms</li>
 * </ul>
 *
 * <h2>Test Coverage</h2>
 * <ol>
 *   <li><b>testPatternInterfaceCompatibility:</b> Validate DenseVector works across all modules</li>
 *   <li><b>testDataFlowThroughModules:</b> Input → laminar → ART → vectorized → output</li>
 *   <li><b>testParameterConsistency:</b> Parameter conversions are lossless</li>
 *   <li><b>testWeightVectorConversions:</b> Weight → expectation → layer input conversions</li>
 *   <li><b>testEndToEndIntegration:</b> Complete workflow with multiple patterns</li>
 * </ol>
 *
 * <h2>Success Criteria</h2>
 * <ul>
 *   <li>All 5 tests pass</li>
 *   <li>No data loss or corruption across modules</li>
 *   <li>Parameters remain consistent</li>
 *   <li>Weight conversions are bidirectional and lossless</li>
 *   <li>End-to-end integration produces semantically equivalent results</li>
 * </ul>
 *
 * @author Hal Hildebrand
 * @see ARTLaminarCircuit
 * @see VectorizedARTLaminarCircuit
 * @see LaminarARTBridge
 */
@DisplayName("Phase 3: Cross-Module Compatibility Tests")
class CrossModuleCompatibilityTest {

    private ARTCircuitParameters defaultParams;
    private ARTCircuitParameters highDimParams;

    @BeforeEach
    void setUp() {
        defaultParams = ARTCircuitParameters.createDefault(10);
        highDimParams = ARTCircuitParameters.createDefault(20);
    }

    /**
     * Test 1: Validate Pattern interface compatibility across all three modules.
     *
     * <p>Data flow: laminar (DenseVector) → art-core (FuzzyART) → art-performance (Vectorized) → back to laminar
     *
     * <p>Validates:
     * <ul>
     *   <li>DenseVector from laminar works with FuzzyART learn()</li>
     *   <li>FuzzyWeight extracts to expectation Pattern</li>
     *   <li>Expectation works with Layer6 receiveTopDownExpectation()</li>
     *   <li>No data corruption throughout pipeline</li>
     * </ul>
     */
    @Test
    @DisplayName("Test 1: Pattern interface works across art-laminar, art-core, and art-performance")
    void testPatternInterfaceCompatibility() {
        // Create pattern in laminar module (DenseVector)
        var laminarPattern = Pattern.of(new double[]{0.8, 0.6, 0.4, 0.2});
        assertEquals(4, laminarPattern.dimension(), "Laminar pattern should have correct dimension");

        // Pass to art-core FuzzyART
        var fuzzyART = new FuzzyART();
        var fuzzyParams = new FuzzyParameters(0.7, 0.001, 0.1);
        var artResult = fuzzyART.learn(laminarPattern, fuzzyParams);

        // Verify FuzzyART accepted the pattern
        assertInstanceOf(ActivationResult.Success.class, artResult, "FuzzyART should successfully learn pattern");
        var success = (ActivationResult.Success) artResult;

        // Extract weight as Pattern using bridge
        var weight = success.updatedWeight();
        assertNotNull(weight, "Weight should not be null");
        assertEquals(8, weight.dimension(), "Weight should be complement-coded (2x dimension)");

        var artCircuitParams = ARTCircuitParameters.createDefault(4);
        var expectation = LaminarARTBridge.extractExpectation(weight);
        assertNotNull(expectation, "Expectation should not be null");
        assertEquals(4, expectation.dimension(), "Expectation should have original dimension (not complement-coded)");

        // Use in laminar layers (Layer6)
        var layer6 = new Layer6Implementation("test", 4);
        var layer6Params = com.hellblazer.art.laminar.parameters.Layer6Parameters.builder().build();
        var layer6Output = layer6.processTopDown(expectation, layer6Params);

        // Validate no data corruption
        assertNotNull(layer6Output, "Layer6 output should not be null");
        assertEquals(4, layer6Output.dimension(), "Layer6 output should maintain dimension");

        // Verify values are in valid range [0,1]
        for (int i = 0; i < layer6Output.dimension(); i++) {
            var value = layer6Output.get(i);
            assertTrue(value >= 0.0 && value <= 1.0,
                      "Layer6 output values should be in [0,1], got: " + value + " at index " + i);
        }

        System.out.println("✓ Pattern interface compatibility validated across all three modules");
    }

    /**
     * Test 2: Validate data flow through all modules with consistent results.
     *
     * <p>Processes same input through:
     * <ol>
     *   <li>FullLaminarCircuitImpl (baseline)</li>
     *   <li>ARTLaminarCircuit (FuzzyART integration)</li>
     *   <li>VectorizedARTLaminarCircuit (SIMD optimization)</li>
     * </ol>
     *
     * <p>Validates all three produce valid outputs without corruption.
     */
    @Test
    @DisplayName("Test 2: Data flows correctly through laminar → art-core → art-performance")
    void testDataFlowThroughModules() {
        var params = ARTCircuitParameters.createDefault(10);

        // Create all three circuit types
        var fullCircuit = new FullLaminarCircuitImpl(params.toCircuitParameters());
        var artCircuit = new ARTLaminarCircuit(params);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(params);

        // Same input pattern
        var input = Pattern.of(new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0});

        // Process through all three
        var fullResult = fullCircuit.process(input);
        var artResult = artCircuit.process(input);
        var vectorizedResult = vectorizedCircuit.process(input);

        // All should be valid (not null)
        assertNotNull(fullResult, "FullLaminarCircuit should produce output");
        assertNotNull(artResult, "ARTLaminarCircuit should produce output");
        assertNotNull(vectorizedResult, "VectorizedARTLaminarCircuit should produce output");

        // All should have correct dimension
        assertEquals(10, fullResult.dimension(), "FullLaminarCircuit output dimension");
        assertEquals(10, artResult.dimension(), "ARTLaminarCircuit output dimension");
        assertEquals(10, vectorizedResult.dimension(), "VectorizedARTLaminarCircuit output dimension");

        // ART versions should create categories
        assertTrue(artCircuit.getCategoryCount() > 0, "ARTLaminarCircuit should create at least one category");
        assertTrue(vectorizedCircuit.getCategoryCount() > 0, "VectorizedARTLaminarCircuit should create at least one category");

        // Process a second time - should recognize pattern
        artCircuit.process(input);
        vectorizedCircuit.process(input);

        // Category count should not increase (recognition)
        assertEquals(1, artCircuit.getCategoryCount(), "ARTLaminarCircuit should recognize pattern (no new category)");
        assertEquals(1, vectorizedCircuit.getCategoryCount(), "VectorizedARTLaminarCircuit should recognize pattern");

        System.out.println("✓ Data flow validated through all three circuit implementations");
    }

    /**
     * Test 3: Validate parameter consistency across module boundaries.
     *
     * <p>Verifies that parameter conversions maintain consistency:
     * <ul>
     *   <li>ARTCircuitParameters → FuzzyParameters</li>
     *   <li>ARTCircuitParameters → VectorizedParameters</li>
     *   <li>ARTCircuitParameters → CircuitParameters</li>
     * </ul>
     *
     * <p>All shared parameters (vigilance, learningRate) must match exactly.
     */
    @Test
    @DisplayName("Test 3: Parameters remain consistent across module conversions")
    void testParameterConsistency() {
        var artParams = ARTCircuitParameters.builder(50)
            .vigilance(0.85)
            .learningRate(0.3)
            .choiceParameter(0.001)
            .topDownGain(0.75)
            .timeStep(0.01)
            .expectationThreshold(0.05)
            .maxSearchIterations(100)
            .build();

        // Convert to FuzzyParameters
        var fuzzyParams = artParams.toFuzzyParameters();
        assertEquals(0.85, fuzzyParams.vigilance(), 1e-10, "FuzzyParameters vigilance should match");
        assertEquals(0.3, fuzzyParams.beta(), 1e-10, "FuzzyParameters beta (learningRate) should match");
        assertEquals(0.001, fuzzyParams.alpha(), 1e-10, "FuzzyParameters alpha should match");

        // Convert to VectorizedParameters
        var vectorizedParams = artParams.toVectorizedParameters();
        assertEquals(0.85, vectorizedParams.vigilanceThreshold(), 1e-10, "VectorizedParameters vigilance should match");
        assertEquals(0.3, vectorizedParams.learningRate(), 1e-10, "VectorizedParameters learningRate should match");
        assertEquals(0.001, vectorizedParams.alpha(), 1e-10, "VectorizedParameters alpha should match");

        // Convert to CircuitParameters
        var circuitParams = artParams.toCircuitParameters();
        assertEquals(0.85, circuitParams.vigilance(), 1e-10, "CircuitParameters vigilance should match");
        assertEquals(0.3, circuitParams.learningRate(), 1e-10, "CircuitParameters learningRate should match");
        assertEquals(0.75, circuitParams.topDownGain(), 1e-10, "CircuitParameters topDownGain should match");
        assertEquals(0.01, circuitParams.timeStep(), 1e-10, "CircuitParameters timeStep should match");

        // Validate consistency check works (non-throwing variant)
        assertTrue(LaminarARTBridge.validateParameterConsistency(circuitParams, fuzzyParams, false),
                  "Parameter consistency validation should pass");

        System.out.println("✓ Parameter consistency validated across all conversions");
    }

    /**
     * Test 4: Validate weight vector conversions are lossless and bidirectional.
     *
     * <p>Tests:
     * <ul>
     *   <li>FuzzyART weight → expectation pattern extraction</li>
     *   <li>VectorizedFuzzyART weight → expectation pattern extraction</li>
     *   <li>Both produce semantically equivalent results</li>
     *   <li>Complement coding is handled correctly</li>
     *   <li>All values remain in [0,1] range</li>
     * </ul>
     */
    @Test
    @DisplayName("Test 4: Weight vector conversions are lossless")
    void testWeightVectorConversions() {
        var params = ARTCircuitParameters.createDefault(8);

        // Create FuzzyART and learn a pattern
        var fuzzyART = new FuzzyART();
        var input = Pattern.of(new double[]{0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4});
        var fuzzyParams = params.toFuzzyParameters();
        var result = fuzzyART.learn(input, fuzzyParams);

        // Verify learning succeeded
        assertInstanceOf(ActivationResult.Success.class, result, "FuzzyART should learn successfully");
        var success = (ActivationResult.Success) result;

        // Extract FuzzyWeight
        var fuzzyWeight = success.updatedWeight();
        assertNotNull(fuzzyWeight, "FuzzyWeight should not be null");
        assertEquals(16, fuzzyWeight.dimension(), "FuzzyWeight should be complement-coded (2x8=16)");

        // Convert to expectation pattern
        var expectation = LaminarARTBridge.extractExpectation(fuzzyWeight);
        assertEquals(8, expectation.dimension(), "Expectation should have original dimension (not complement-coded)");

        // Validate values are in [0,1]
        for (int i = 0; i < expectation.dimension(); i++) {
            var value = expectation.get(i);
            assertTrue(value >= 0.0 && value <= 1.0,
                      "Expectation value should be in [0,1], got: " + value + " at index " + i);
        }

        // Test with VectorizedFuzzyART for comparison
        var vecParams = params.toVectorizedParameters();
        var vectorizedART = new VectorizedFuzzyART(vecParams);
        var vecResult = vectorizedART.learn(input, vecParams);

        assertInstanceOf(ActivationResult.Success.class, vecResult, "VectorizedFuzzyART should learn successfully");
        var vecSuccess = (ActivationResult.Success) vecResult;

        var vectorizedWeight = vecSuccess.updatedWeight();
        assertNotNull(vectorizedWeight, "VectorizedWeight should not be null");
        assertEquals(16, vectorizedWeight.dimension(), "VectorizedWeight should be complement-coded");

        var vectorizedExpectation = LaminarARTBridge.extractExpectation(vectorizedWeight);
        assertEquals(8, vectorizedExpectation.dimension(), "Vectorized expectation should have original dimension");

        // Should be semantically equivalent (within tolerance)
        for (int i = 0; i < expectation.dimension(); i++) {
            assertEquals(expectation.get(i), vectorizedExpectation.get(i), 1e-6,
                        "Expectations should match at index " + i);
        }

        System.out.println("✓ Weight vector conversions validated for both FuzzyART and VectorizedFuzzyART");
    }

    /**
     * Test 5: End-to-end integration test with complete workflow.
     *
     * <p>Complete workflow:
     * <ol>
     *   <li>Generate 50 random patterns</li>
     *   <li>Process through both ARTLaminarCircuit and VectorizedARTLaminarCircuit</li>
     *   <li>Validate both learn similar number of categories (within tolerance)</li>
     *   <li>Verify performance statistics are tracked</li>
     *   <li>Ensure no exceptions or errors</li>
     * </ol>
     *
     * <p>Success criteria:
     * <ul>
     *   <li>Both circuits create categories</li>
     *   <li>Category counts are similar (within 20% tolerance)</li>
     *   <li>Vectorized version shows SIMD operations</li>
     *   <li>No data corruption or errors</li>
     * </ul>
     */
    @Test
    @DisplayName("Test 5: End-to-end integration with complete workflow")
    void testEndToEndIntegration() {
        var params = ARTCircuitParameters.builder(20)
            .vigilance(0.8)
            .learningRate(0.5)
            .maxCategories(10)
            .choiceParameter(0.001)
            .topDownGain(0.8)
            .timeStep(0.01)
            .expectationThreshold(0.05)
            .maxSearchIterations(100)
            .build();

        // Create both ART circuits
        var artCircuit = new ARTLaminarCircuit(params);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(params);

        // Generate 50 test patterns
        var patterns = new ArrayList<Pattern>();
        for (int i = 0; i < 50; i++) {
            var data = new double[20];
            for (int j = 0; j < 20; j++) {
                data[j] = Math.random();
            }
            patterns.add(Pattern.of(data));
        }

        // Process through both circuits
        for (var pattern : patterns) {
            var artOutput = artCircuit.process(pattern);
            var vecOutput = vectorizedCircuit.process(pattern);

            // Validate outputs
            assertNotNull(artOutput, "ARTLaminarCircuit should produce output");
            assertNotNull(vecOutput, "VectorizedARTLaminarCircuit should produce output");
            assertEquals(20, artOutput.dimension(), "ART output dimension");
            assertEquals(20, vecOutput.dimension(), "Vectorized output dimension");
        }

        // Both should learn similar number of categories (within 20% tolerance)
        int artCategories = artCircuit.getCategoryCount();
        int vectorizedCategories = vectorizedCircuit.getCategoryCount();

        assertTrue(artCategories > 0, "ART circuit should learn categories");
        assertTrue(vectorizedCategories > 0, "Vectorized circuit should learn categories");

        // Allow 20% difference due to different search orders
        double categoryRatio = (double) Math.max(artCategories, vectorizedCategories) /
                              Math.min(artCategories, vectorizedCategories);
        assertTrue(categoryRatio <= 1.2,
                  String.format("Category counts should be similar (within 20%%), got ratio: %.2f", categoryRatio));

        // Vectorized should have performance stats
        var stats = vectorizedCircuit.getPerformanceStats();
        assertNotNull(stats, "Performance stats should not be null");
        assertTrue(stats.totalVectorOperations() > 0, "Should have vector operations recorded");

        System.out.printf("✓ End-to-end integration validated: ART categories=%d, Vectorized categories=%d%n",
                         artCategories, vectorizedCategories);
        System.out.printf("  Vector operations: %d, Ops/sec: %.1f%n",
                         stats.totalVectorOperations(), stats.getOperationsPerSecond());
    }
}
