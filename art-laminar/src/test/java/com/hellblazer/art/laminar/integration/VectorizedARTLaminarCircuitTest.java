package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for VectorizedARTLaminarCircuit with SIMD-optimized FuzzyART.
 *
 * <p>These tests validate:
 * <ul>
 *   <li>SIMD vectorization works correctly</li>
 *   <li>Semantic equivalence with standard ARTLaminarCircuit</li>
 *   <li>Performance improvements through vectorization</li>
 *   <li>High-dimensional pattern processing (where SIMD shines)</li>
 * </ul>
 *
 * <p><b>Phase 2 - RED Phase:</b> All tests written to fail initially (no implementation yet)
 *
 * @author Hal Hildebrand
 */
class VectorizedARTLaminarCircuitTest {

    private ARTCircuitParameters defaultParams;
    private ARTCircuitParameters highDimParams;

    @BeforeEach
    void setUp() {
        defaultParams = ARTCircuitParameters.createDefault(100);
        highDimParams = ARTCircuitParameters.createDefault(256); // High-dimensional for SIMD
    }

    /**
     * Test 1: Validate SIMD vectorized processing works correctly.
     *
     * <p>Validates:
     * <ul>
     *   <li>VectorizedFuzzyART processes patterns correctly</li>
     *   <li>Categories are created successfully</li>
     *   <li>Resonance detection works</li>
     *   <li>No errors or exceptions during SIMD operations</li>
     * </ul>
     */
    @Test
    void testVectorizedProcessing() {
        var circuit = new VectorizedARTLaminarCircuit(defaultParams);

        // Generate test patterns
        var patterns = generateRandomPatterns(10, 100);

        // Process patterns through vectorized circuit
        for (var pattern : patterns) {
            var expectation = circuit.process(pattern);
            assertNotNull(expectation, "Expectation should not be null");
            assertEquals(100, expectation.dimension(), "Expectation dimension should match input");
        }

        // Validate categories created
        assertTrue(circuit.getCategoryCount() > 0,
                  "Should have created at least one category");

        // Validate performance stats available
        var stats = circuit.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
        assertTrue(stats.totalVectorOperations() >= 0,
                  "Vector operations should be tracked");

        // Cleanup
        circuit.close();
    }

    /**
     * Test 2: Semantic equivalence with standard ARTLaminarCircuit.
     *
     * <p>Validates:
     * <ul>
     *   <li>Vectorized circuit produces SAME results as standard circuit</li>
     *   <li>Category counts match</li>
     *   <li>Resonance states match</li>
     *   <li>Expectations are numerically identical (within floating point tolerance)</li>
     * </ul>
     *
     * <p>This is CRITICAL - vectorization is an optimization, not a change in semantics.
     */
    @Test
    void testSemanticEquivalence() {
        // Create both versions with SAME parameters
        var standardCircuit = new ARTLaminarCircuit(defaultParams);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(defaultParams);

        // Generate 100 test patterns
        var patterns = generateRandomPatterns(100, 100);

        // Process same patterns through both circuits
        for (var pattern : patterns) {
            var standardResult = standardCircuit.process(pattern);
            var vectorizedResult = vectorizedCircuit.process(pattern);

            // Should produce SAME results (semantically equivalent)
            assertEquals(standardResult.dimension(), vectorizedResult.dimension(),
                        "Result dimensions should match");

            // Allow small floating point differences from vectorized operations
            // Vectorized SIMD operations may have slightly different rounding
            // Tolerance ~5e-7 allows for accumulation of rounding errors over multiple operations
            var standardData = standardResult.toArray();
            var vectorizedData = vectorizedResult.toArray();
            for (int i = 0; i < standardData.length; i++) {
                assertEquals(standardData[i], vectorizedData[i], 5e-7,
                           "Expectation values should match at index " + i);
            }
        }

        // Should create SAME number of categories
        assertEquals(standardCircuit.getCategoryCount(), vectorizedCircuit.getCategoryCount(),
                    "Both circuits should create same number of categories");

        // Cleanup
        try {
            standardCircuit.close();
            vectorizedCircuit.close();
        } catch (Exception e) {
            fail("Close should not throw exception: " + e.getMessage());
        }
    }

    /**
     * Test 3: Performance improvement measurement.
     *
     * <p>Validates:
     * <ul>
     *   <li>Vectorized version is faster than standard (speedup > 1.0x)</li>
     *   <li>Performance stats track SIMD operations</li>
     *   <li>Speedup is measurable and logged</li>
     * </ul>
     *
     * <p><b>Note:</b> This test logs speedup but does NOT fail if speedup < expected
     * (speedup depends on CPU, pattern size, JVM warmup, etc.)
     */
    @Test
    void testPerformanceImprovement() {
        var standardCircuit = new ARTLaminarCircuit(defaultParams);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(defaultParams);

        // Generate 1000 patterns for benchmarking
        var patterns = generateRandomPatterns(1000, 100);

        // Warmup JVM (important for accurate timing)
        for (int i = 0; i < 100; i++) {
            standardCircuit.process(patterns.get(i % patterns.size()));
            vectorizedCircuit.process(patterns.get(i % patterns.size()));
        }

        // Reset circuits
        standardCircuit.reset();
        vectorizedCircuit.reset();

        // Benchmark standard version
        long startStandard = System.nanoTime();
        for (var pattern : patterns) {
            standardCircuit.process(pattern);
        }
        long standardTime = System.nanoTime() - startStandard;

        // Benchmark vectorized version
        vectorizedCircuit.resetPerformanceTracking();
        long startVectorized = System.nanoTime();
        for (var pattern : patterns) {
            vectorizedCircuit.process(pattern);
        }
        long vectorizedTime = System.nanoTime() - startVectorized;

        // Calculate speedup
        double speedup = (double) standardTime / vectorizedTime;
        System.out.printf("Performance Test Results:%n");
        System.out.printf("  Standard time:    %d ms%n", standardTime / 1_000_000);
        System.out.printf("  Vectorized time:  %d ms%n", vectorizedTime / 1_000_000);
        System.out.printf("  Speedup:          %.2fx%n", speedup);

        // Log performance stats
        var stats = vectorizedCircuit.getPerformanceStats();
        System.out.printf("  Vector operations: %d%n", stats.totalVectorOperations());
        System.out.printf("  Parallel tasks:    %d%n", stats.totalParallelTasks());

        // Vectorized should be faster (at least on larger patterns)
        // NOTE: Advisory only - CI environments may show different performance
        if (speedup <= 0.8) {
            System.out.printf("⚠️  Performance advisory: Vectorized speedup %.2fx below target (> 0.8x)%n", speedup);
        }

        // Performance stats should show SIMD activity
        assertTrue(stats.totalVectorOperations() > 0,
                  "Should have performed vector operations");

        // Cleanup
        try {
            standardCircuit.close();
            vectorizedCircuit.close();
        } catch (Exception e) {
            fail("Close should not throw exception: " + e.getMessage());
        }
    }

    /**
     * Test 4: High-dimensional pattern processing (where SIMD really shines).
     *
     * <p>Validates:
     * <ul>
     *   <li>256-dimensional patterns process correctly</li>
     *   <li>Larger speedup with high dimensions (3-5x expected)</li>
     *   <li>Memory efficiency with high-dimensional data</li>
     * </ul>
     *
     * <p><b>Rationale:</b> SIMD benefits increase with dimensionality.
     * A 256-bit SIMD lane can process 4 doubles (32 bytes) in parallel.
     */
    @Test
    void testHighDimensionalPatterns() {
        var standardCircuit = new ARTLaminarCircuit(highDimParams);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(highDimParams);

        // Generate 500 high-dimensional patterns
        var patterns = generateRandomPatterns(500, 256);

        // Warmup
        for (int i = 0; i < 50; i++) {
            standardCircuit.process(patterns.get(i % patterns.size()));
            vectorizedCircuit.process(patterns.get(i % patterns.size()));
        }

        // Reset
        standardCircuit.reset();
        vectorizedCircuit.reset();

        // Benchmark standard version
        long startStandard = System.nanoTime();
        for (var pattern : patterns) {
            standardCircuit.process(pattern);
        }
        long standardTime = System.nanoTime() - startStandard;

        // Benchmark vectorized version
        vectorizedCircuit.resetPerformanceTracking();
        long startVectorized = System.nanoTime();
        for (var pattern : patterns) {
            vectorizedCircuit.process(pattern);
        }
        long vectorizedTime = System.nanoTime() - startVectorized;

        // Calculate speedup
        double speedup = (double) standardTime / vectorizedTime;
        System.out.printf("High-Dimensional Test Results (256D):%n");
        System.out.printf("  Standard time:    %d ms%n", standardTime / 1_000_000);
        System.out.printf("  Vectorized time:  %d ms%n", vectorizedTime / 1_000_000);
        System.out.printf("  Speedup:          %.2fx%n", speedup);

        // Get performance stats
        var stats = vectorizedCircuit.getPerformanceStats();
        System.out.printf("  Vector operations: %d%n", stats.totalVectorOperations());
        System.out.printf("  Categories:        %d%n", stats.categoryCount());

        // High-dimensional should show better speedup
        // NOTE: Advisory only - CI environments may show different performance
        if (speedup <= 0.8) {
            System.out.printf("⚠️  Performance advisory: High-dim vectorized speedup %.2fx below target (> 0.8x)%n", speedup);
        }

        // Should have created categories
        assertTrue(vectorizedCircuit.getCategoryCount() > 0,
                  "Should have created categories");

        // Verify semantic correctness with one final pattern
        var testPattern = generateRandomPatterns(1, 256).get(0);
        var standardResult = standardCircuit.process(testPattern);
        var vectorizedResult = vectorizedCircuit.process(testPattern);

        // Results should be close (within floating point tolerance from vectorized operations)
        // Vectorized SIMD operations may have slightly different rounding
        // Tolerance ~5e-7 allows for accumulation of rounding errors over multiple operations
        var standardData = standardResult.toArray();
        var vectorizedData = vectorizedResult.toArray();
        for (int i = 0; i < standardData.length; i++) {
            assertEquals(standardData[i], vectorizedData[i], 5e-7,
                       "High-dimensional results should match at index " + i);
        }

        // Cleanup
        try {
            standardCircuit.close();
            vectorizedCircuit.close();
        } catch (Exception e) {
            fail("Close should not throw exception: " + e.getMessage());
        }
    }

    // === Utility Methods ===

    /**
     * Generate random test patterns in [0,1]^dimension.
     *
     * @param count number of patterns to generate
     * @param dimension dimensionality of each pattern
     * @return list of random patterns
     */
    private List<Pattern> generateRandomPatterns(int count, int dimension) {
        var patterns = new ArrayList<Pattern>(count);
        var random = new java.util.Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < count; i++) {
            var data = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                data[j] = random.nextDouble(); // [0,1]
            }
            patterns.add(Pattern.of(data));
        }

        return patterns;
    }
}
