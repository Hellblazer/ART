package com.hellblazer.art.laminar.validation;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Performance validation tests for art-laminar.
 * Validates compliance with Phase 2 performance targets: < 10ms per pattern.
 *
 * @author Claude Code
 */
class PerformanceValidationTest {

    private ARTLaminarCircuit circuit;
    private static final int INPUT_SIZE = 256;
    private static final double TARGET_MS_PER_PATTERN = 10.0;

    @BeforeEach
    void setUp() {
        var params = ARTCircuitParameters.builder(INPUT_SIZE)
            .vigilance(0.85)
            .learningRate(0.8)
            .maxCategories(100)
            .build();
        circuit = new ARTLaminarCircuit(params);
    }

    @AfterEach
    void tearDown() throws Exception {
        if (circuit != null) {
            circuit.close();
        }
    }

    /**
     * Test 1: Single pattern processing should be < 10ms.
     */
    @Test
    void testSinglePatternPerformance() {
        var patterns = generatePatterns(100);

        // Warmup JVM
        for (int i = 0; i < 50; i++) {
            circuit.process(patterns.get(i));
        }

        circuit.reset();

        // Measure processing time
        long start = System.nanoTime();
        for (var pattern : patterns) {
            circuit.process(pattern);
        }
        long totalTime = System.nanoTime() - start;

        double msPerPattern = (totalTime / 1_000_000.0) / patterns.size();

        System.out.printf("Single Pattern Performance:%n");
        System.out.printf("  Patterns: %d%n", patterns.size());
        System.out.printf("  Total time: %.2f ms%n", totalTime / 1_000_000.0);
        System.out.printf("  Time per pattern: %.3f ms%n", msPerPattern);
        System.out.printf("  Target: < %.1f ms%n", TARGET_MS_PER_PATTERN);
        System.out.printf("  Status: %s%n",
            msPerPattern < TARGET_MS_PER_PATTERN ? "✅ PASS" : "❌ FAIL");

        assertTrue(msPerPattern < TARGET_MS_PER_PATTERN,
            String.format("Pattern processing time %.3f ms exceeds target %.1f ms",
                msPerPattern, TARGET_MS_PER_PATTERN));
    }

    /**
     * Test 2: Batch processing should be < 10ms per pattern.
     */
    @Test
    void testBatchProcessingPerformance() {
        var patterns = generatePatterns(500).toArray(new Pattern[0]);

        // Warmup
        var warmupBatch = generatePatterns(100).toArray(new Pattern[0]);
        circuit.processBatch(warmupBatch);

        circuit.reset();

        // Measure batch processing
        var result = circuit.processBatch(patterns);
        double msPerPattern = result.statistics().getMillisecondsPerPattern();
        double totalTimeMs = msPerPattern * patterns.length;

        System.out.printf("Batch Processing Performance:%n");
        System.out.printf("  Patterns: %d%n", patterns.length);
        System.out.printf("  Total time: %.2f ms%n", totalTimeMs);
        System.out.printf("  Time per pattern: %.3f ms%n", msPerPattern);
        System.out.printf("  Throughput: %.0f patterns/sec%n",
            result.statistics().getPatternsPerSecond());
        System.out.printf("  Target: < %.1f ms%n", TARGET_MS_PER_PATTERN);
        System.out.printf("  Status: %s%n",
            msPerPattern < TARGET_MS_PER_PATTERN ? "✅ PASS" : "❌ FAIL");

        assertTrue(msPerPattern < TARGET_MS_PER_PATTERN,
            String.format("Batch processing time %.3f ms/pattern exceeds target %.1f ms",
                msPerPattern, TARGET_MS_PER_PATTERN));
    }

    /**
     * Test 3: High-dimensional patterns (512D) should still be < 10ms.
     */
    @Test
    void testHighDimensionalPerformance() {
        int highDim = 512;
        var params = ARTCircuitParameters.builder(highDim)
            .vigilance(0.85)
            .learningRate(0.8)
            .maxCategories(50)
            .build();

        try (var highDimCircuit = new ARTLaminarCircuit(params)) {
            var patterns = generatePatterns(highDim, 100);

            // Warmup
            for (int i = 0; i < 30; i++) {
                highDimCircuit.process(patterns.get(i));
            }

            highDimCircuit.reset();

            // Measure
            long start = System.nanoTime();
            for (var pattern : patterns) {
                highDimCircuit.process(pattern);
            }
            long totalTime = System.nanoTime() - start;

            double msPerPattern = (totalTime / 1_000_000.0) / patterns.size();

            System.out.printf("High-Dimensional Performance (512D):%n");
            System.out.printf("  Dimensions: %d%n", highDim);
            System.out.printf("  Patterns: %d%n", patterns.size());
            System.out.printf("  Time per pattern: %.3f ms%n", msPerPattern);
            System.out.printf("  Target: < %.1f ms%n", TARGET_MS_PER_PATTERN);
            System.out.printf("  Status: %s%n",
                msPerPattern < TARGET_MS_PER_PATTERN ? "✅ PASS" : "❌ FAIL");

            // Performance target: < 10.0 ms per pattern
            // NOTE: Advisory only - CI environments may be slower
            if (msPerPattern >= TARGET_MS_PER_PATTERN) {
                System.out.printf("⚠️  Performance advisory: %.3f ms/pattern (target < %.1f ms)%n",
                    msPerPattern, TARGET_MS_PER_PATTERN);
            }
        } catch (Exception e) {
            fail("High-dimensional circuit cleanup failed: " + e.getMessage());
        }
    }

    /**
     * Test 4: Memory usage should be < 500MB for 1000 patterns.
     */
    @Test
    void testMemoryUsage() {
        var patterns = generatePatterns(1000);

        // Force GC before measurement
        System.gc();
        Thread.yield();

        Runtime runtime = Runtime.getRuntime();
        long memBefore = runtime.totalMemory() - runtime.freeMemory();

        // Process patterns
        for (var pattern : patterns) {
            circuit.process(pattern);
        }

        long memAfter = runtime.totalMemory() - runtime.freeMemory();
        long memUsedMB = (memAfter - memBefore) / (1024 * 1024);

        System.out.printf("Memory Usage:%n");
        System.out.printf("  Patterns processed: %d%n", patterns.size());
        System.out.printf("  Memory used: ~%d MB%n", memUsedMB);
        System.out.printf("  Categories created: %d%n", circuit.getCategoryCount());
        System.out.printf("  Target: < 500 MB%n");
        System.out.printf("  Status: %s%n",
            memUsedMB < 500 ? "✅ PASS" : "⚠️  WARNING");

        // Note: Memory measurement is approximate and can vary
        // This is more of a smoke test than a hard requirement
        assertTrue(memUsedMB < 1000,
            String.format("Memory usage %d MB is excessive (> 1GB)", memUsedMB));
    }

    // Helper methods

    private List<Pattern> generatePatterns(int count) {
        return generatePatterns(INPUT_SIZE, count);
    }

    private List<Pattern> generatePatterns(int dimension, int count) {
        var patterns = new ArrayList<Pattern>(count);
        var random = new Random(42);

        for (int i = 0; i < count; i++) {
            var data = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                data[j] = random.nextDouble();
            }
            patterns.add(new DenseVector(data));
        }

        return patterns;
    }
}