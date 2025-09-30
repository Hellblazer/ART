package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive validation tests for transpose-and-vectorize pattern.
 *
 * Tests data layout transformation utilities that enable SIMD vectorization
 * across batch dimension for 2-3x performance improvement.
 *
 * @author Claude Code
 */
class BatchDataLayoutTest {

    private static final double EPSILON = 1e-9;

    @Test
    void testTransposeToDimensionMajor() {
        // Create pattern-major data: patterns[i][d]
        var patterns = new Pattern[] {
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0}),
            new DenseVector(new double[]{7.0, 8.0, 9.0})
        };

        // Transpose to dimension-major: dimensions[d][i]
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        // Validate structure
        assertEquals(3, dimensionMajor.length, "Should have 3 dimensions");
        assertEquals(3, dimensionMajor[0].length, "Should have 3 patterns per dimension");

        // Validate data correctness
        // Dimension 0: [1.0, 4.0, 7.0]
        assertArrayEquals(new double[]{1.0, 4.0, 7.0}, dimensionMajor[0], EPSILON);
        // Dimension 1: [2.0, 5.0, 8.0]
        assertArrayEquals(new double[]{2.0, 5.0, 8.0}, dimensionMajor[1], EPSILON);
        // Dimension 2: [3.0, 6.0, 9.0]
        assertArrayEquals(new double[]{3.0, 6.0, 9.0}, dimensionMajor[2], EPSILON);
    }

    @Test
    void testTransposeToPatternMajor() {
        // Create dimension-major data: dimensions[d][i]
        var dimensionMajor = new double[][] {
            {1.0, 4.0, 7.0},  // All patterns at dimension 0
            {2.0, 5.0, 8.0},  // All patterns at dimension 1
            {3.0, 6.0, 9.0}   // All patterns at dimension 2
        };

        // Transpose to pattern-major: patterns[i][d]
        var patternMajor = BatchDataLayout.transposeToPatternMajor(dimensionMajor);

        // Validate structure
        assertEquals(3, patternMajor.length, "Should have 3 patterns");
        assertEquals(3, patternMajor[0].length, "Should have 3 dimensions per pattern");

        // Validate data correctness
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, patternMajor[0], EPSILON);
        assertArrayEquals(new double[]{4.0, 5.0, 6.0}, patternMajor[1], EPSILON);
        assertArrayEquals(new double[]{7.0, 8.0, 9.0}, patternMajor[2], EPSILON);
    }

    @Test
    void testRoundTripTranspose() {
        // Original patterns
        var original = new Pattern[] {
            new DenseVector(new double[]{0.1, 0.2, 0.3, 0.4}),
            new DenseVector(new double[]{0.5, 0.6, 0.7, 0.8}),
            new DenseVector(new double[]{0.9, 1.0, 1.1, 1.2})
        };

        // Round trip: pattern-major → dimension-major → pattern-major
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(original);
        var patternMajor = BatchDataLayout.transposeToPatternMajor(dimensionMajor);

        // Validate round trip preserves data
        assertEquals(original.length, patternMajor.length, "Pattern count preserved");
        for (int i = 0; i < original.length; i++) {
            for (int d = 0; d < original[i].dimension(); d++) {
                assertEquals(original[i].get(d), patternMajor[i][d], EPSILON,
                    String.format("Round trip preserved data at pattern[%d][%d]", i, d));
            }
        }
    }

    @Test
    void testTransposeEmptyPatternsThrows() {
        var exception = assertThrows(IllegalArgumentException.class,
            () -> BatchDataLayout.transposeToDimensionMajor(new Pattern[0]));

        assertTrue(exception.getMessage().contains("cannot be null or empty"),
            "Should indicate empty patterns array");
    }

    @Test
    void testTransposeNullPatternsThrows() {
        var exception = assertThrows(IllegalArgumentException.class,
            () -> BatchDataLayout.transposeToDimensionMajor(null));

        assertTrue(exception.getMessage().contains("cannot be null or empty"),
            "Should indicate null patterns array");
    }

    @Test
    void testTransposeInconsistentDimensionsThrows() {
        var patterns = new Pattern[] {
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0}),  // Different dimension!
            new DenseVector(new double[]{7.0, 8.0, 9.0})
        };

        var exception = assertThrows(IllegalArgumentException.class,
            () -> BatchDataLayout.transposeToDimensionMajor(patterns));

        assertTrue(exception.getMessage().contains("Inconsistent dimensions"),
            "Should indicate inconsistent dimensions");
    }

    @Test
    void testTransposeNullPatternThrows() {
        var patterns = new Pattern[] {
            new DenseVector(new double[]{1.0, 2.0}),
            null,  // Null pattern!
            new DenseVector(new double[]{7.0, 8.0})
        };

        var exception = assertThrows(NullPointerException.class,
            () -> BatchDataLayout.transposeToDimensionMajor(patterns));

        assertTrue(exception.getMessage().contains("is null"),
            "Should indicate null pattern");
    }

    @Test
    void testVectorizedTransposeAligned() {
        int laneSize = BatchDataLayout.getVectorLaneSize();
        int batchSize = laneSize * 4;  // Aligned batch size
        int dimension = 8;

        // Create aligned batch
        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = i * dimension + d;
            }
            patterns[i] = new DenseVector(values);
        }

        // Vectorized transpose
        var dimensionMajor = BatchDataLayout.transposeVectorized(patterns);

        // Validate correctness
        assertEquals(dimension, dimensionMajor.length, "Dimension count correct");
        assertEquals(batchSize, dimensionMajor[0].length, "Batch size correct");

        // Spot check values
        for (int d = 0; d < dimension; d++) {
            for (int i = 0; i < batchSize; i++) {
                double expected = i * dimension + d;
                assertEquals(expected, dimensionMajor[d][i], EPSILON,
                    String.format("Vectorized transpose correct at [%d][%d]", d, i));
            }
        }
    }

    @Test
    void testVectorizedTransposeUnaligned() {
        int laneSize = BatchDataLayout.getVectorLaneSize();
        int batchSize = laneSize + 1;  // Unaligned batch size
        int dimension = 4;

        // Create unaligned batch
        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            patterns[i] = new DenseVector(new double[]{i * 1.0, i * 2.0, i * 3.0, i * 4.0});
        }

        // Should fall back to scalar transpose
        var dimensionMajor = BatchDataLayout.transposeVectorized(patterns);

        // Validate correctness (falls back to scalar)
        assertEquals(dimension, dimensionMajor.length, "Dimension count correct");
        assertEquals(batchSize, dimensionMajor[0].length, "Batch size correct");
    }

    @Test
    void testTransposeMemoryOverhead() {
        int batchSize = 128;
        int dimension = 256;

        long overhead = BatchDataLayout.getTransposeMemoryOverhead(batchSize, dimension);

        // Expected: dimension * 8 (array refs) + dimension * batchSize * 8 (doubles)
        long expectedArrayOverhead = dimension * 8L;
        long expectedDataSize = (long) dimension * batchSize * 8L;
        long expectedTotal = expectedArrayOverhead + expectedDataSize;

        assertEquals(expectedTotal, overhead, "Memory overhead calculation correct");
    }

    @Test
    void testTransposeAndVectorizeBeneficial() {
        // Case 1: Small batch - NOT beneficial
        assertFalse(BatchDataLayout.isTransposeAndVectorizeBeneficial(16, 128, 10),
            "Small batch (16) should not be beneficial");

        // Case 2: Small dimension - NOT beneficial
        assertFalse(BatchDataLayout.isTransposeAndVectorizeBeneficial(64, 32, 10),
            "Small dimension (32) should not be beneficial");

        // Case 3: Too few operations - NOT beneficial
        assertFalse(BatchDataLayout.isTransposeAndVectorizeBeneficial(64, 128, 2),
            "Too few operations (2) should not be beneficial");

        // Case 4: Good parameters - beneficial
        assertTrue(BatchDataLayout.isTransposeAndVectorizeBeneficial(64, 128, 10),
            "Good parameters should be beneficial");

        // Case 5: Large batch + dimension - beneficial
        assertTrue(BatchDataLayout.isTransposeAndVectorizeBeneficial(256, 512, 20),
            "Large batch+dimension should be beneficial");
    }

    @Test
    void testTransposeStatistics() {
        int batchSize = 128;
        int dimension = 256;
        int operations = 15;

        var stats = BatchDataLayout.TransposeStatistics.estimate(batchSize, dimension, operations);

        // Validate statistics
        assertEquals(batchSize, stats.batchSize(), "Batch size recorded");
        assertEquals(dimension, stats.dimension(), "Dimension recorded");
        assertTrue(stats.memoryOverhead() > 0, "Memory overhead computed");
        assertEquals(5.0, stats.estimatedOverheadPercent(), EPSILON, "5% overhead typical");
        assertTrue(stats.estimatedSpeedup() >= 1.0, "Speedup >= 1.0");
        assertTrue(stats.beneficial(), "Should be beneficial for these parameters");
    }

    @Test
    void testTransposeStatisticsNotBeneficial() {
        int batchSize = 16;  // Too small
        int dimension = 32;   // Too small
        int operations = 2;   // Too few

        var stats = BatchDataLayout.TransposeStatistics.estimate(batchSize, dimension, operations);

        assertFalse(stats.beneficial(), "Should not be beneficial for small parameters");
        assertEquals(1.0, stats.estimatedSpeedup(), EPSILON,
            "No speedup when not beneficial");
    }

    @Test
    void testGetVectorSpecies() {
        var species = BatchDataLayout.getVectorSpecies();
        assertNotNull(species, "Vector species should not be null");
        assertTrue(species.length() >= 2, "Lane size should be >= 2");
    }

    @Test
    void testGetVectorLaneSize() {
        int laneSize = BatchDataLayout.getVectorLaneSize();
        assertTrue(laneSize >= 2, "Lane size should be >= 2");
        assertTrue(laneSize <= 16, "Lane size should be <= 16 (reasonable for AVX512)");

        // Lane size should be power of 2
        assertTrue((laneSize & (laneSize - 1)) == 0,
            "Lane size should be power of 2");
    }

    @Test
    void testLargePatternBatch() {
        // Test with realistic large batch
        int batchSize = 256;
        int dimension = 512;

        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = Math.sin(i * 0.1) * Math.cos(d * 0.1);
            }
            patterns[i] = new DenseVector(values);
        }

        // Transpose
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        // Validate structure
        assertEquals(dimension, dimensionMajor.length, "Dimension count");
        assertEquals(batchSize, dimensionMajor[0].length, "Batch size");

        // Spot check some values
        for (int d = 0; d < 10; d++) {
            for (int i = 0; i < 10; i++) {
                double expected = Math.sin(i * 0.1) * Math.cos(d * 0.1);
                assertEquals(expected, dimensionMajor[d][i], EPSILON,
                    "Large batch transpose correct");
            }
        }
    }
}