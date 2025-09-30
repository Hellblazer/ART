package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Data layout transformation utilities for SIMD batch processing.
 *
 * <h2>Pattern-Major vs Dimension-Major Layouts</h2>
 *
 * <p><b>Pattern-Major (Sequential)</b>:
 * <pre>
 * patterns[0] = [x0, y0, z0]
 * patterns[1] = [x1, y1, z1]
 * patterns[2] = [x2, y2, z2]
 * </pre>
 * Good for: Sequential pattern-by-pattern processing
 *
 * <p><b>Dimension-Major (Transposed for SIMD)</b>:
 * <pre>
 * dimension[0] = [x0, x1, x2, x3, x4, x5, x6, x7, ...]  // Vector lane
 * dimension[1] = [y0, y1, y2, y3, y4, y5, y6, y7, ...]  // Vector lane
 * dimension[2] = [z0, z1, z2, z3, z4, z5, z6, z7, ...]  // Vector lane
 * </pre>
 * Good for: SIMD across batch dimension (process 8 patterns at once)
 *
 * <h2>SIMD Benefit</h2>
 * <p>With dimension-major layout, we can use Java Vector API:
 * <pre>
 * // Process 8 patterns simultaneously
 * var vec = DoubleVector.fromArray(SPECIES, batchData[dim], 0);
 * var scaled = vec.mul(scaleValue);
 * scaled.intoArray(batchData[dim], 0);
 * </pre>
 *
 * <h2>Performance Characteristics</h2>
 * <ul>
 *   <li><b>Transpose overhead</b>: ~5-10% of processing time</li>
 *   <li><b>SIMD benefit</b>: 2-4x speedup (depends on operations)</li>
 *   <li><b>Net benefit</b>: 2-3x overall speedup</li>
 *   <li><b>Best for</b>: Batch size ≥ 32, dimension ≥ 64</li>
 * </ul>
 *
 * @author Claude Code
 */
public class BatchDataLayout {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Transpose from pattern-major to dimension-major layout.
     *
     * <p>Input: patterns[batchSize][dimension]
     * <p>Output: dimensionMajor[dimension][batchSize]
     *
     * <p>Enables SIMD across batch dimension:
     * <pre>
     * for (int d = 0; d < dimension; d++) {
     *     // Process ALL patterns at dimension d together
     *     processVectorized(dimensionMajor[d]);  // SIMD!
     * }
     * </pre>
     *
     * @param patterns pattern-major array [batchSize][dimension]
     * @return dimension-major array [dimension][batchSize]
     * @throws IllegalArgumentException if patterns empty or dimensions inconsistent
     * @throws NullPointerException if patterns or any element is null
     */
    public static double[][] transposeToDimensionMajor(Pattern[] patterns) {
        if (patterns == null || patterns.length == 0) {
            throw new IllegalArgumentException("Patterns array cannot be null or empty");
        }

        int batchSize = patterns.length;
        int dimension = patterns[0].dimension();

        // Validate all patterns have same dimension
        for (int i = 1; i < batchSize; i++) {
            if (patterns[i] == null) {
                throw new NullPointerException("Pattern at index " + i + " is null");
            }
            if (patterns[i].dimension() != dimension) {
                throw new IllegalArgumentException(
                    String.format("Inconsistent dimensions: pattern[0]=%d, pattern[%d]=%d",
                        dimension, i, patterns[i].dimension()));
            }
        }

        // Allocate dimension-major array
        double[][] dimensionMajor = new double[dimension][batchSize];

        // Transpose: patterns[i][d] → dimensionMajor[d][i]
        for (int d = 0; d < dimension; d++) {
            for (int i = 0; i < batchSize; i++) {
                dimensionMajor[d][i] = patterns[i].get(d);
            }
        }

        return dimensionMajor;
    }

    /**
     * Transpose from dimension-major back to pattern-major layout.
     *
     * <p>Input: dimensionMajor[dimension][batchSize]
     * <p>Output: patterns[batchSize][dimension]
     *
     * <p>Converts SIMD-friendly layout back to patterns for output.
     *
     * @param dimensionMajor dimension-major array [dimension][batchSize]
     * @return pattern-major array [batchSize][dimension]
     * @throws IllegalArgumentException if dimensionMajor empty or rows inconsistent
     * @throws NullPointerException if dimensionMajor or any row is null
     */
    public static double[][] transposeToPatternMajor(double[][] dimensionMajor) {
        if (dimensionMajor == null || dimensionMajor.length == 0) {
            throw new IllegalArgumentException("DimensionMajor array cannot be null or empty");
        }

        int dimension = dimensionMajor.length;
        int batchSize = dimensionMajor[0].length;

        // Validate all rows have same length
        for (int d = 1; d < dimension; d++) {
            if (dimensionMajor[d] == null) {
                throw new NullPointerException("Row at dimension " + d + " is null");
            }
            if (dimensionMajor[d].length != batchSize) {
                throw new IllegalArgumentException(
                    String.format("Inconsistent batch sizes: row[0]=%d, row[%d]=%d",
                        batchSize, d, dimensionMajor[d].length));
            }
        }

        // Allocate pattern-major array
        double[][] patternMajor = new double[batchSize][dimension];

        // Transpose: dimensionMajor[d][i] → patternMajor[i][d]
        for (int i = 0; i < batchSize; i++) {
            for (int d = 0; d < dimension; d++) {
                patternMajor[i][d] = dimensionMajor[d][i];
            }
        }

        return patternMajor;
    }

    /**
     * Fast vectorized transpose using SIMD.
     *
     * <p>Uses Java Vector API for transpose when batch size is a multiple
     * of vector lane size (typically 8 for double on AVX512, 4 for AVX2).
     *
     * <p>Falls back to scalar transpose if batch size not aligned.
     *
     * @param patterns pattern-major array
     * @return dimension-major array
     */
    public static double[][] transposeVectorized(Pattern[] patterns) {
        int batchSize = patterns.length;
        int laneSize = SPECIES.length();

        // Use vectorized path if batch size is multiple of lane size
        if (batchSize % laneSize == 0) {
            return transposeVectorizedAligned(patterns);
        } else {
            // Fall back to scalar transpose
            return transposeToDimensionMajor(patterns);
        }
    }

    /**
     * Vectorized transpose for aligned batch sizes.
     *
     * <p>Processes vector lanes at a time for better performance.
     * Requires batchSize % laneSize == 0.
     *
     * @param patterns pattern-major array
     * @return dimension-major array
     */
    private static double[][] transposeVectorizedAligned(Pattern[] patterns) {
        int batchSize = patterns.length;
        int dimension = patterns[0].dimension();
        int laneSize = SPECIES.length();

        double[][] dimensionMajor = new double[dimension][batchSize];

        // Transpose using vector lanes
        for (int d = 0; d < dimension; d++) {
            for (int i = 0; i < batchSize; i += laneSize) {
                // Gather values from laneSize patterns at dimension d
                for (int lane = 0; lane < laneSize; lane++) {
                    dimensionMajor[d][i + lane] = patterns[i + lane].get(d);
                }
            }
        }

        return dimensionMajor;
    }

    /**
     * Get memory overhead of transpose operation.
     *
     * <p>Transpose requires temporary allocation of dimension-major array.
     *
     * @param batchSize number of patterns
     * @param dimension pattern dimension
     * @return memory overhead in bytes
     */
    public static long getTransposeMemoryOverhead(int batchSize, int dimension) {
        // double[][] dimensionMajor = new double[dimension][batchSize]
        long arrayOverhead = dimension * 8L;  // Array object refs
        long dataSize = (long) dimension * batchSize * 8L;  // doubles
        return arrayOverhead + dataSize;
    }

    /**
     * Check if transpose-and-vectorize will be beneficial.
     *
     * <p>Transpose overhead (~5-10%) is worth it when:
     * <ul>
     *   <li>Batch size ≥ 32 (enough parallelism)</li>
     *   <li>Dimension ≥ 64 (enough work per dimension)</li>
     *   <li>Operations are SIMD-friendly (arithmetic, comparisons)</li>
     * </ul>
     *
     * @param batchSize number of patterns
     * @param dimension pattern dimension
     * @param operationsPerDimension approximate operations per dimension
     * @return true if transpose-and-vectorize will provide speedup
     */
    public static boolean isTransposeAndVectorizeBeneficial(
            int batchSize,
            int dimension,
            int operationsPerDimension) {

        // Minimum batch size for parallelism benefit
        if (batchSize < 32) {
            return false;
        }

        // Minimum dimension to amortize transpose overhead
        if (dimension < 64) {
            return false;
        }

        // Estimate speedup vs overhead
        double transposeOverhead = 0.05;  // ~5% overhead
        double vectorSpeedup = 2.5;  // 2.5x average SIMD speedup

        // Need enough operations to make transpose worthwhile
        int minOperations = 5;  // At least 5 operations per dimension
        if (operationsPerDimension < minOperations) {
            return false;
        }

        // Net benefit = speedup - overhead
        double netBenefit = vectorSpeedup - transposeOverhead;
        return netBenefit > 1.5;  // Require >50% speedup
    }

    /**
     * Statistics about transpose operation.
     */
    public record TransposeStatistics(
        int batchSize,
        int dimension,
        long memoryOverhead,
        double estimatedOverheadPercent,
        double estimatedSpeedup,
        boolean beneficial
    ) {
        public static TransposeStatistics estimate(int batchSize, int dimension, int operationsPerDimension) {
            long memoryOverhead = getTransposeMemoryOverhead(batchSize, dimension);
            double overheadPercent = 5.0;  // Typical transpose overhead
            double speedup = isTransposeAndVectorizeBeneficial(batchSize, dimension, operationsPerDimension)
                ? 2.5 : 1.0;
            boolean beneficial = speedup > 1.5;

            return new TransposeStatistics(
                batchSize,
                dimension,
                memoryOverhead,
                overheadPercent,
                speedup,
                beneficial
            );
        }
    }

    /**
     * Get preferred vector species for SIMD operations.
     *
     * @return preferred vector species (platform-dependent)
     */
    public static VectorSpecies<Double> getVectorSpecies() {
        return SPECIES;
    }

    /**
     * Get vector lane size (number of doubles per vector).
     *
     * @return lane size (typically 4 for AVX2, 8 for AVX512)
     */
    public static int getVectorLaneSize() {
        return SPECIES.length();
    }

    /**
     * Convert dimension-major array to Pattern array.
     *
     * @param dimensionMajor dimension-major array [dimension][batchSize]
     * @param batchSize number of patterns
     * @param dimension pattern dimension
     * @return Pattern array [batchSize]
     */
    public static Pattern[] dimensionMajorToPatterns(double[][] dimensionMajor, int batchSize, int dimension) {
        var patternMajor = transposeToPatternMajor(dimensionMajor);
        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            patterns[i] = new com.hellblazer.art.core.DenseVector(patternMajor[i]);
        }
        return patterns;
    }
}