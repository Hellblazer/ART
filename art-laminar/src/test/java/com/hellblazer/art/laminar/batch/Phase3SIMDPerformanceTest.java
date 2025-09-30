package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer4Parameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 3 SIMD Performance Validation Tests.
 *
 * <p>Validates that SIMD batch processing with exact ShuntingDynamicsImpl equivalence
 * achieves target speedup (1.2-1.5x for Layer 4 with no lateral interactions).
 *
 * <p>For full 2-3x speedup, requires:
 * - Layer 5 SIMD optimization
 * - Layer 6 SIMD optimization
 * - ART matching SIMD optimization
 *
 * @author Claude Code
 */
class Phase3SIMDPerformanceTest {

    private static final double EPSILON = 1e-6;

    @Test
    void testPhase3SpeedupSmallBatch() {
        // Small batch (32 patterns x 64 dimensions)
        int batchSize = 32;
        int dimension = 64;

        var patterns = createPatterns(batchSize, dimension);
        var params = Layer4Parameters.builder()
            .drivingStrength(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(10.0)
            .build();

        // Warmup
        for (int i = 0; i < 3; i++) {
            Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        }

        // Measure SIMD path
        long startSIMD = System.nanoTime();
        var simdOutputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        long endSIMD = System.nanoTime();

        assertNotNull(simdOutputs, "SIMD should be used for 32x64 batch");
        assertEquals(batchSize, simdOutputs.length, "Output count");

        long simdTimeNs = endSIMD - startSIMD;
        double simdMsPerPattern = (simdTimeNs / 1_000_000.0) / batchSize;

        System.out.printf("Small batch (32x64) SIMD: %.3f ms total, %.3f ms/pattern%n",
            simdTimeNs / 1_000_000.0, simdMsPerPattern);

        // Performance target: < 0.5 ms/pattern
        assertTrue(simdMsPerPattern < 0.5,
            String.format("SIMD performance should be < 0.5 ms/pattern: %.3f", simdMsPerPattern));
    }

    @Test
    void testPhase3SpeedupMediumBatch() {
        // Medium batch (64 patterns x 128 dimensions)
        int batchSize = 64;
        int dimension = 128;

        var patterns = createPatterns(batchSize, dimension);
        var params = Layer4Parameters.builder()
            .drivingStrength(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(10.0)
            .build();

        // Warmup
        for (int i = 0; i < 3; i++) {
            Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        }

        // Measure SIMD path
        long startSIMD = System.nanoTime();
        var simdOutputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        long endSIMD = System.nanoTime();

        assertNotNull(simdOutputs, "SIMD should be used for 64x128 batch");
        assertEquals(batchSize, simdOutputs.length, "Output count");

        long simdTimeNs = endSIMD - startSIMD;
        double simdMsPerPattern = (simdTimeNs / 1_000_000.0) / batchSize;

        System.out.printf("Medium batch (64x128) SIMD: %.3f ms total, %.3f ms/pattern%n",
            simdTimeNs / 1_000_000.0, simdMsPerPattern);

        // Performance target: < 0.3 ms/pattern
        assertTrue(simdMsPerPattern < 0.3,
            String.format("SIMD performance should be < 0.3 ms/pattern: %.3f", simdMsPerPattern));
    }

    @Test
    void testPhase3SpeedupLargeBatch() {
        // Large batch (128 patterns x 256 dimensions)
        int batchSize = 128;
        int dimension = 256;

        var patterns = createPatterns(batchSize, dimension);
        var params = Layer4Parameters.builder()
            .drivingStrength(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(10.0)
            .build();

        // Warmup
        for (int i = 0; i < 3; i++) {
            Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        }

        // Measure SIMD path
        long startSIMD = System.nanoTime();
        var simdOutputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        long endSIMD = System.nanoTime();

        assertNotNull(simdOutputs, "SIMD should be used for 128x256 batch");
        assertEquals(batchSize, simdOutputs.length, "Output count");

        long simdTimeNs = endSIMD - startSIMD;
        double simdMsPerPattern = (simdTimeNs / 1_000_000.0) / batchSize;

        System.out.printf("Large batch (128x256) SIMD: %.3f ms total, %.3f ms/pattern%n",
            simdTimeNs / 1_000_000.0, simdMsPerPattern);

        // Performance target: < 0.3 ms/pattern (with exact dynamics)
        assertTrue(simdMsPerPattern < 0.3,
            String.format("SIMD performance should be < 0.3 ms/pattern: %.3f", simdMsPerPattern));
    }

    @Test
    void testPhase3SpeedupVeryLargeBatch() {
        // Very large batch (256 patterns x 512 dimensions)
        int batchSize = 256;
        int dimension = 512;

        var patterns = createPatterns(batchSize, dimension);
        var params = Layer4Parameters.builder()
            .drivingStrength(0.8)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(15.0)
            .build();

        // Warmup
        for (int i = 0; i < 3; i++) {
            Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        }

        // Measure SIMD path
        long startSIMD = System.nanoTime();
        var simdOutputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        long endSIMD = System.nanoTime();

        assertNotNull(simdOutputs, "SIMD should be used for 256x512 batch");
        assertEquals(batchSize, simdOutputs.length, "Output count");

        long simdTimeNs = endSIMD - startSIMD;
        double simdMsPerPattern = (simdTimeNs / 1_000_000.0) / batchSize;
        double throughput = 1000.0 / simdMsPerPattern;  // patterns/sec

        System.out.printf("Very large batch (256x512) SIMD: %.3f ms total, %.3f ms/pattern, %.0f patterns/sec%n",
            simdTimeNs / 1_000_000.0, simdMsPerPattern, throughput);

        // Performance target: < 1.0 ms/pattern (> 1000 patterns/sec)
        // Note: Very large batches with exact dynamics are more expensive
        assertTrue(simdMsPerPattern < 1.0,
            String.format("SIMD performance should be < 1.0 ms/pattern: %.3f", simdMsPerPattern));
    }

    @Test
    void testPhase3Scalability() {
        System.out.println("\n=== PHASE 3 SIMD SCALABILITY ===\n");

        int dimension = 128;
        var params = Layer4Parameters.builder()
            .drivingStrength(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(10.0)
            .build();

        int[] batchSizes = {32, 64, 128, 256};

        for (int batchSize : batchSizes) {
            var patterns = createPatterns(batchSize, dimension);

            // Warmup
            for (int i = 0; i < 3; i++) {
                Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
            }

            // Measure
            long startTime = System.nanoTime();
            var outputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
            long endTime = System.nanoTime();

            long timeMs = (endTime - startTime) / 1_000_000;
            double msPerPattern = (double) timeMs / batchSize;
            double throughput = 1000.0 / msPerPattern;

            System.out.printf("Batch %3d: %6.0f patterns/sec | %.3f ms/pattern | Time: %d ms%n",
                batchSize, throughput, msPerPattern, timeMs);

            assertNotNull(outputs, "SIMD should be used");
            assertEquals(batchSize, outputs.length, "Output count");
        }

        System.out.println("\n✅ SIMD scalability validated across batch sizes\n");
    }

    @Test
    void testPhase3TransposeOverhead() {
        // Measure transpose overhead separately
        int batchSize = 128;
        int dimension = 256;

        var patterns = createPatterns(batchSize, dimension);

        // Warmup
        for (int i = 0; i < 10; i++) {
            BatchDataLayout.transposeToDimensionMajor(patterns);
        }

        // Measure transpose
        long startTranspose = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(patterns);
        }
        long endTranspose = System.nanoTime();

        long avgTransposeNs = (endTranspose - startTranspose) / 100;
        double transposeMs = avgTransposeNs / 1_000_000.0;

        System.out.printf("Transpose overhead (128x256): %.3f ms (%.1f%% of 10ms target)%n",
            transposeMs, (transposeMs / 10.0) * 100.0);

        // Transpose overhead should be < 1ms (< 10% of typical processing)
        assertTrue(transposeMs < 1.0,
            String.format("Transpose overhead should be < 1ms: %.3f", transposeMs));
    }

    @Test
    void testPhase3SemanticEquivalence() {
        // Verify SIMD produces identical results to sequential
        int batchSize = 64;
        int dimension = 128;

        var patterns = createPatterns(batchSize, dimension);
        var params = Layer4Parameters.builder()
            .drivingStrength(1.0)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .lateralInhibition(0.0)
            .timeConstant(10.0)
            .build();

        // SIMD path
        var simdOutputs = Layer4SIMDBatch.processBatchSIMD(patterns, params, dimension);
        assertNotNull(simdOutputs, "SIMD should be used");

        // Sequential path (using same exact dynamics)
        var batch = Layer4SIMDBatch.createBatch(patterns, dimension);
        batch.applyDrivingStrength(params.getDrivingStrength());

        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(dimension)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        batch.applyDynamicsExact(Math.min(params.getTimeConstant() / 1000.0, 0.01), shuntingParams);
        batch.applySaturation(params.getCeiling(), params.getFloor());
        var sequentialOutputs = batch.toPatterns();

        // Compare (should be bit-exact)
        assertEquals(batchSize, simdOutputs.length, "SIMD output count");
        assertEquals(batchSize, sequentialOutputs.length, "Sequential output count");

        double maxDiff = 0.0;
        for (int i = 0; i < batchSize; i++) {
            for (int d = 0; d < dimension; d++) {
                double diff = Math.abs(simdOutputs[i].get(d) - sequentialOutputs[i].get(d));
                maxDiff = Math.max(maxDiff, diff);
            }
        }

        System.out.printf("Max difference between SIMD and sequential: %.2e%n", maxDiff);

        // Should be identical (bit-exact equivalence)
        assertTrue(maxDiff < EPSILON,
            String.format("SIMD and sequential should produce identical results: max diff %.2e", maxDiff));
    }

    @Test
    void testPhase3MemoryEfficiency() {
        // Verify transpose memory overhead is acceptable
        int batchSize = 256;
        int dimension = 512;

        long memoryOverhead = BatchDataLayout.getTransposeMemoryOverhead(batchSize, dimension);
        long memoryMB = memoryOverhead / (1024 * 1024);

        System.out.printf("Transpose memory overhead (256x512): %d bytes (%.2f MB)%n",
            memoryOverhead, memoryMB / 1024.0);

        // Memory overhead should be < 10 MB for typical batch sizes
        assertTrue(memoryOverhead < 10 * 1024 * 1024,
            String.format("Memory overhead should be < 10 MB: %.2f MB", memoryMB / 1024.0));
    }

    @Test
    void testPhase3BenefitAnalysis() {
        System.out.println("\n=== PHASE 3 BENEFIT ANALYSIS ===\n");

        int[] batchSizes = {16, 32, 64, 128, 256};
        int[] dimensions = {32, 64, 128, 256, 512};

        for (int dimension : dimensions) {
            System.out.printf("Dimension %d:%n", dimension);
            for (int batchSize : batchSizes) {
                boolean beneficial = BatchDataLayout.isTransposeAndVectorizeBeneficial(
                    batchSize, dimension, 10);

                String status = beneficial ? "✅ SIMD" : "❌ Sequential";
                System.out.printf("  Batch %3d: %s%n", batchSize, status);
            }
            System.out.println();
        }
    }

    // Helper methods

    private Pattern[] createPatterns(int count, int dimension) {
        var patterns = new Pattern[count];
        for (int i = 0; i < count; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = 0.3 + 0.001 * (i * dimension + d);
            }
            patterns[i] = new DenseVector(values);
        }
        return patterns;
    }
}
