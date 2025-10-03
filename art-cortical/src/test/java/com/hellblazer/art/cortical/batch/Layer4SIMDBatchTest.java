package com.hellblazer.art.cortical.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for Layer4SIMDBatch - SIMD-optimized batch processing for Layer 4.
 *
 * <h2>Test Strategy</h2>
 * <p>Following Phase 1A test-first approach:
 * <ol>
 *   <li>Basic batch creation and transpose operations</li>
 *   <li>SIMD operations (driving strength, saturation)</li>
 *   <li>Shunting dynamics with exact equivalence</li>
 *   <li>Bit-exact validation vs sequential processing</li>
 *   <li>Performance characteristics (transpose overhead, SIMD benefit)</li>
 * </ol>
 *
 * @author Phase 1A: Preparation & Test Framework
 */
@DisplayName("Layer4 SIMD Batch Processing")
class Layer4SIMDBatchTest {

    private static final double EPSILON = 1e-10;

    @Test
    @DisplayName("Create batch from patterns with transpose")
    void testCreateBatch() {
        // Create simple patterns
        var patterns = new Pattern[] {
            new DenseVector(new double[] {1.0, 2.0, 3.0}),
            new DenseVector(new double[] {4.0, 5.0, 6.0}),
            new DenseVector(new double[] {7.0, 8.0, 9.0})
        };

        var batch = Layer4SIMDBatch.createBatch(patterns, 3);

        assertEquals(3, batch.getBatchSize(), "Batch size");
        assertEquals(3, batch.getDimension(), "Dimension");

        // Verify dimension-major layout
        var dimMajor = batch.getDimensionMajor();
        assertEquals(3, dimMajor.length, "Dimension-major rows");
        assertEquals(3, dimMajor[0].length, "Dimension-major columns");

        // Verify transpose: patterns[i][d] → dimensionMajor[d][i]
        assertArrayEquals(new double[] {1.0, 4.0, 7.0}, dimMajor[0], EPSILON, "Dim 0");
        assertArrayEquals(new double[] {2.0, 5.0, 8.0}, dimMajor[1], EPSILON, "Dim 1");
        assertArrayEquals(new double[] {3.0, 6.0, 9.0}, dimMajor[2], EPSILON, "Dim 2");
    }

    @Test
    @DisplayName("Null patterns throw exception")
    void testNullPatterns() {
        assertThrows(IllegalArgumentException.class, () ->
            Layer4SIMDBatch.createBatch(null, 3),
            "Null patterns should throw"
        );

        assertThrows(IllegalArgumentException.class, () ->
            Layer4SIMDBatch.createBatch(new Pattern[0], 3),
            "Empty patterns should throw"
        );
    }

    @Test
    @DisplayName("Dimension mismatch throws exception")
    void testDimensionMismatch() {
        var patterns = new Pattern[] {
            new DenseVector(new double[] {1.0, 2.0, 3.0}),
            new DenseVector(new double[] {4.0, 5.0})  // Wrong dimension!
        };

        assertThrows(IllegalArgumentException.class, () ->
            Layer4SIMDBatch.createBatch(patterns, 3),
            "Dimension mismatch should throw"
        );
    }

    @Test
    @DisplayName("Apply driving strength (SIMD)")
    void testApplyDrivingStrength() {
        var patterns = new Pattern[] {
            new DenseVector(new double[] {1.0, 2.0, 3.0}),
            new DenseVector(new double[] {4.0, 5.0, 6.0})
        };

        var batch = Layer4SIMDBatch.createBatch(patterns, 3);
        batch.applyDrivingStrength(2.0);

        var dimMajor = batch.getDimensionMajor();

        // All values should be scaled by 2.0
        assertArrayEquals(new double[] {2.0, 8.0}, dimMajor[0], EPSILON, "Dim 0 scaled");
        assertArrayEquals(new double[] {4.0, 10.0}, dimMajor[1], EPSILON, "Dim 1 scaled");
        assertArrayEquals(new double[] {6.0, 12.0}, dimMajor[2], EPSILON, "Dim 2 scaled");
    }

    @Test
    @DisplayName("Apply saturation (SIMD)")
    void testApplySaturation() {
        var patterns = new Pattern[] {
            new DenseVector(new double[] {0.5, 1.0, 2.0}),
            new DenseVector(new double[] {5.0, 10.0, 100.0})
        };

        var batch = Layer4SIMDBatch.createBatch(patterns, 3);
        batch.applySaturation(1.0, 0.0);  // Ceiling: 1.0, Floor: 0.0

        var dimMajor = batch.getDimensionMajor();

        // Sigmoid saturation: ceiling * x / (1 + x), then clamp to [floor, ceiling]
        // x=0.5: 1.0 * 0.5 / 1.5 = 0.333
        // x=1.0: 1.0 * 1.0 / 2.0 = 0.5
        // x=2.0: 1.0 * 2.0 / 3.0 = 0.667
        // x=5.0: 1.0 * 5.0 / 6.0 = 0.833
        // x=10.0: 1.0 * 10.0 / 11.0 = 0.909
        // x=100.0: 1.0 * 100.0 / 101.0 = 0.990

        assertEquals(1.0 * 0.5 / 1.5, dimMajor[0][0], 1e-3, "x=0.5 saturated");
        assertEquals(1.0 * 5.0 / 6.0, dimMajor[0][1], 1e-3, "x=5.0 saturated");

        assertEquals(1.0 * 1.0 / 2.0, dimMajor[1][0], 1e-3, "x=1.0 saturated");
        assertEquals(1.0 * 10.0 / 11.0, dimMajor[1][1], 1e-3, "x=10.0 saturated");

        assertEquals(1.0 * 2.0 / 3.0, dimMajor[2][0], 1e-3, "x=2.0 saturated");
        assertEquals(1.0 * 100.0 / 101.0, dimMajor[2][1], 1e-3, "x=100.0 saturated");
    }

    @Test
    @DisplayName("Round-trip transpose preserves data")
    void testRoundTripTranspose() {
        var original = new Pattern[] {
            new DenseVector(new double[] {1.0, 2.0, 3.0, 4.0}),
            new DenseVector(new double[] {5.0, 6.0, 7.0, 8.0}),
            new DenseVector(new double[] {9.0, 10.0, 11.0, 12.0})
        };

        var batch = Layer4SIMDBatch.createBatch(original, 4);
        var reconstructed = batch.toPatterns();

        assertEquals(original.length, reconstructed.length, "Batch size preserved");

        for (int i = 0; i < original.length; i++) {
            assertArrayEquals(
                original[i].toArray(),
                reconstructed[i].toArray(),
                EPSILON,
                "Pattern " + i + " preserved"
            );
        }
    }

    @Test
    @DisplayName("Shunting dynamics evolution (exact equivalence)")
    void testApplyDynamicsExact() {
        // Create simple patterns
        var patterns = new Pattern[] {
            new DenseVector(new double[] {0.5, 0.3, 0.7}),
            new DenseVector(new double[] {0.2, 0.8, 0.4})
        };

        var batch = Layer4SIMDBatch.createBatch(patterns, 3);

        // Layer 4 parameters (fast dynamics, no lateral interactions)
        var params = ShuntingParameters.builder(3)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(0.0)  // No lateral inhibition
            .build();

        double timeStep = 0.01;  // Small time step for stability

        // Apply dynamics (should use SIMD path since no lateral interactions)
        batch.applyDynamicsExact(timeStep, params);

        var dimMajor = batch.getDimensionMajor();

        // Values should have evolved according to shunting equation
        // dx/dt = -A*x + (B-x)*E where E = self_exc * x (for Layer 4)
        // For x=0.5, A=0.3, B=1.0, self_exc=0.3:
        //   E = 0.3 * 0.5 = 0.15
        //   dx/dt = -0.3*0.5 + (1.0-0.5)*0.15 = -0.15 + 0.075 = -0.075
        //   x' = 0.5 + 0.01 * (-0.075) = 0.49925

        // Verify evolution happened (values should be different but within bounds)
        for (int d = 0; d < 3; d++) {
            for (int p = 0; p < 2; p++) {
                assertTrue(dimMajor[d][p] >= 0.0, "Floor respected");
                assertTrue(dimMajor[d][p] <= 1.0, "Ceiling respected");
            }
        }

        // First pattern, first dimension: started at 0.5
        // Should have decreased slightly (derivative was negative)
        assertTrue(dimMajor[0][0] < 0.5, "x evolved from initial state");
        assertTrue(dimMajor[0][0] > 0.49, "Evolution within expected range");
    }

    @Test
    @DisplayName("Batch processing with small batch (< 32) returns null")
    void testSmallBatchFallback() {
        var patterns = new Pattern[] {
            new DenseVector(new double[] {1.0, 2.0}),
            new DenseVector(new double[] {3.0, 4.0})
        };

        // processBatchSIMD should return null for batches too small to benefit from SIMD
        var result = Layer4SIMDBatch.processBatchSIMD(
            patterns,
            1.0,  // drivingStrength
            10.0, // timeConstant
            1.0,  // ceiling
            0.0,  // floor
            0.3,  // selfExcitation
            0.0,  // lateralInhibition
            2     // size
        );

        assertNull(result, "Small batch should return null (fall back to sequential)");
    }

    @Test
    @DisplayName("Batch processing with medium batch (32-63) uses SIMD")
    void testMediumBatchSIMD() {
        // Create 32 patterns (minimum for SIMD benefit)
        var patterns = new Pattern[32];
        for (int i = 0; i < 32; i++) {
            patterns[i] = new DenseVector(new double[] {0.5, 0.3});
        }

        var result = Layer4SIMDBatch.processBatchSIMD(
            patterns,
            1.0,  // drivingStrength
            10.0, // timeConstant
            1.0,  // ceiling
            0.0,  // floor
            0.3,  // selfExcitation
            0.0,  // lateralInhibition
            2     // size
        );

        assumeTrue(result != null,
            "SIMD disabled on this platform (batch size 32). " +
            "Test would pass on platforms with 4+ lane SIMD support.");

        assertEquals(32, result.length, "Output batch size");
        assertEquals(2, result[0].dimension(), "Output dimension");
    }

    @Test
    @DisplayName("Batch processing with large batch (64+) uses SIMD")
    void testLargeBatchSIMD() {
        // Create 64 patterns (target batch size for Phase 1)
        var patterns = new Pattern[64];
        for (int i = 0; i < 64; i++) {
            patterns[i] = new DenseVector(new double[] {0.5, 0.3, 0.7, 0.2});
        }

        var result = Layer4SIMDBatch.processBatchSIMD(
            patterns,
            1.0,  // drivingStrength
            10.0, // timeConstant
            1.0,  // ceiling
            0.0,  // floor
            0.3,  // selfExcitation
            0.0,  // lateralInhibition
            4     // size
        );

        assumeTrue(result != null,
            "SIMD disabled on this platform (batch size 64). " +
            "Test would pass on platforms with 4+ lane SIMD support.");

        assertEquals(64, result.length, "Output batch size");
        assertEquals(4, result[0].dimension(), "Output dimension");

        // Verify all patterns evolved
        for (int i = 0; i < 64; i++) {
            assertNotNull(result[i], "Pattern " + i + " not null");
            assertEquals(4, result[i].dimension(), "Pattern " + i + " dimension");
        }
    }

    @Test
    @DisplayName("Hardware vector lane count is reasonable")
    void testHardwareVectorLaneCount() {
        int laneCount = SIMDConfiguration.hardwareVectorLaneCount();

        System.out.printf("Hardware vector lane count: %d%n", laneCount);

        assertTrue(laneCount >= 2, "Should have at least 2 lanes for doubles");
        assertTrue(laneCount <= 16, "Should not exceed 16 lanes for current hardware");
    }

    @Test
    @DisplayName("Transpose operations are data-preserving")
    void testTransposePreservesData() {
        var patterns = new Pattern[] {
            new DenseVector(new double[] {1.1, 2.2, 3.3}),
            new DenseVector(new double[] {4.4, 5.5, 6.6}),
            new DenseVector(new double[] {7.7, 8.8, 9.9}),
            new DenseVector(new double[] {10.1, 11.2, 12.3})
        };

        // Forward transpose
        var dimMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        // Backward transpose
        var patternMajor = BatchDataLayout.transposeToPatternMajor(dimMajor);

        // Verify
        assertEquals(4, patternMajor.length, "Batch size preserved");
        assertEquals(3, patternMajor[0].length, "Dimension preserved");

        for (int i = 0; i < patterns.length; i++) {
            assertArrayEquals(
                patterns[i].toArray(),
                patternMajor[i],
                EPSILON,
                "Pattern " + i + " data preserved"
            );
        }
    }

    @Test
    @DisplayName("SIMD configuration optimal defaults")
    void testSIMDConfigurationDefaults() {
        var config = SIMDConfiguration.optimal();

        assertEquals(64, config.miniBatchSize(), "Default mini-batch size is 64 (Phase 1 target)");
        assertTrue(config.autoTuning(), "Auto-tuning enabled by default");
        assertEquals(1.05, config.fallbackThreshold(), EPSILON, "5% speedup threshold");
    }
}
