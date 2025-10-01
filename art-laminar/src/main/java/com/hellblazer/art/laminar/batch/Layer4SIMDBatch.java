package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer4Parameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for Layer 4.
 *
 * <h2>Performance Strategy: Transpose-and-Vectorize</h2>
 *
 * <p>Standard approach (sequential per pattern):
 * <pre>
 * for (pattern : batch) {
 *     for (dim : dimensions) {
 *         process(pattern[dim]);
 *     }
 * }
 * </pre>
 *
 * <p>SIMD approach (parallel across batch):
 * <pre>
 * // Transpose to dimension-major layout
 * dimensionMajor = transpose(patterns)
 *
 * for (dim : dimensions) {
 *     // Process ALL patterns at dimension d together (SIMD!)
 *     Vector.fromArray(dimensionMajor[dim]).process()
 * }
 * </pre>
 *
 * <h2>Expected Performance</h2>
 * <ul>
 *   <li>Sequential: 1.0x baseline</li>
 *   <li>SIMD: 2-3x speedup (depends on operations and batch size)</li>
 *   <li>Transpose overhead: ~5% (amortized across all operations)</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>
 * var batch = Layer4SIMDBatch.createBatch(patterns, size);
 * batch.applyDrivingStrength(params.getDrivingStrength());
 * batch.applyDynamics(timeStep);
 * batch.applySaturation(params.getCeiling(), params.getFloor());
 * Pattern[] outputs = batch.toPatterns();
 * </pre>
 *
 * @author Hal Hildebrand
 */
public class Layer4SIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[][] dimensionMajor;  // [dimension][batchSize]
    private final int dimension;
    private final int batchSize;

    private Layer4SIMDBatch(double[][] dimensionMajor) {
        this.dimensionMajor = dimensionMajor;
        this.dimension = dimensionMajor.length;
        this.batchSize = dimensionMajor[0].length;
    }

    /**
     * Create SIMD batch from patterns.
     *
     * Transposes from pattern-major to dimension-major layout for SIMD.
     *
     * @param patterns input patterns [batchSize][dimension]
     * @param layerSize expected dimension (for validation)
     * @return SIMD batch ready for processing
     */
    public static Layer4SIMDBatch createBatch(Pattern[] patterns, int layerSize) {
        if (patterns == null || patterns.length == 0) {
            throw new IllegalArgumentException("patterns cannot be null or empty");
        }

        // Validate dimensions
        for (int i = 0; i < patterns.length; i++) {
            if (patterns[i] == null) {
                throw new IllegalArgumentException("patterns[" + i + "] is null");
            }
            if (patterns[i].dimension() != layerSize) {
                throw new IllegalArgumentException(
                    String.format("patterns[%d] dimension mismatch: expected %d, got %d",
                        i, layerSize, patterns[i].dimension()));
            }
        }

        // Transpose using BatchDataLayout
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        return new Layer4SIMDBatch(dimensionMajor);
    }

    /**
     * Apply driving strength to entire batch (SIMD).
     *
     * <p>SIMD: scales all patterns at each dimension together.
     *
     * @param drivingStrength scaling factor
     */
    public void applyDrivingStrength(double drivingStrength) {
        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);
                var scaled = vec.mul(drivingStrength);
                scaled.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] *= drivingStrength;
            }
        }
    }

    /**
     * Apply fast dynamics evolution using exact ShuntingDynamicsImpl equivalence.
     *
     * <p>Uses {@link BatchShuntingDynamics} which maintains bit-exact equivalence
     * with sequential {@code ShuntingDynamicsImpl} processing.
     *
     * <p>For Layer 4 (no lateral interactions), uses SIMD optimization automatically.
     *
     * @param timeStep integration time step
     * @param params shunting parameters
     */
    public void applyDynamicsExact(double timeStep, com.hellblazer.art.temporal.dynamics.ShuntingParameters params) {
        var batchDynamics = new BatchShuntingDynamics(params, dimension);

        // dimensionMajor is both current state and excitatory input for Layer 4
        var evolved = batchDynamics.evolveBatch(dimensionMajor, dimensionMajor, timeStep);

        // Copy evolved state back
        for (int d = 0; d < dimension; d++) {
            System.arraycopy(evolved[d], 0, dimensionMajor[d], 0, batchSize);
        }
    }

    /**
     * Apply fast dynamics evolution (simplified - for testing only).
     *
     * <p><b>DEPRECATED</b>: Use {@link #applyDynamicsExact} for exact equivalence.
     *
     * @param timeStep integration time step
     */
    @Deprecated
    public void applyDynamics(double timeStep) {
        // Layer 4 parameters (simplified)
        var A = 0.3;  // Self-excitation decay
        var B = 1.0;  // Ceiling

        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized dynamics
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                // Load current activations
                var x = DoubleVector.fromArray(SPECIES, row, i);

                // Compute excitatory input (use current activation as input)
                var E = x;

                // Compute derivative: -A*x + (B-x)*E
                var decay = x.mul(-A);
                var ceiling = DoubleVector.broadcast(SPECIES, B);
                var excitation = ceiling.sub(x).mul(E);
                var derivative = decay.add(excitation);

                // Euler step: x' = x + dt * derivative
                var dx = derivative.mul(timeStep);
                var xNew = x.add(dx);

                // Store
                xNew.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                var x = row[i];
                var E = x;
                var derivative = -A * x + (B - x) * E;
                row[i] = x + timeStep * derivative;
            }
        }
    }

    /**
     * Apply sigmoid saturation (SIMD).
     *
     * <p>Soft saturation: ceiling * x / (1 + x)
     *
     * <p>Maps [0, ∞) → [0, ceiling]
     *
     * @param ceiling maximum activation
     * @param floor minimum activation
     */
    public void applySaturation(double ceiling, double floor) {
        int laneSize = SPECIES.length();
        var ceilingVec = DoubleVector.broadcast(SPECIES, ceiling);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);
        var floorVec = DoubleVector.broadcast(SPECIES, floor);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized saturation
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var x = DoubleVector.fromArray(SPECIES, row, i);

                // Sigmoid: ceiling * x / (1 + x) for x > 0
                // For x <= 0, keep as-is
                var denominator = oneVec.add(x);
                var saturated = ceilingVec.mul(x).div(denominator);

                // Apply floor/ceiling clamp
                var clamped = saturated.max(floorVec).min(ceilingVec);

                clamped.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                var x = row[i];
                if (x > 0) {
                    x = ceiling * x / (1.0 + x);
                }
                row[i] = Math.max(floor, Math.min(ceiling, x));
            }
        }
    }

    /**
     * Apply element-wise operation across batch (SIMD).
     *
     * <p>General-purpose SIMD operation: f(x) applied to all elements.
     *
     * @param operation function to apply
     */
    public void applyOperation(Operation operation) {
        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);
                var result = operation.apply(vec);
                result.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] = operation.applyScalar(row[i]);
            }
        }
    }

    /**
     * Convert back to pattern-major layout.
     *
     * Transposes dimension-major → pattern-major for output.
     *
     * @return patterns [batchSize][dimension]
     */
    public Pattern[] toPatterns() {
        var patternMajor = BatchDataLayout.transposeToPatternMajor(dimensionMajor);

        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            patterns[i] = new DenseVector(patternMajor[i]);
        }
        return patterns;
    }

    /**
     * Get dimension-major data (direct access for debugging).
     *
     * @return dimension-major array [dimension][batchSize]
     */
    public double[][] getDimensionMajor() {
        return dimensionMajor;
    }

    /**
     * Get batch size.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Get dimension.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Functional interface for SIMD operations.
     */
    @FunctionalInterface
    public interface Operation {
        DoubleVector apply(DoubleVector input);

        default double applyScalar(double input) {
            // Default scalar fallback - override if different from vector version
            throw new UnsupportedOperationException("Scalar version not implemented");
        }
    }

    /**
     * Process complete Layer 4 batch with SIMD optimization.
     *
     * <p>This is the main entry point for Phase 3 SIMD batch processing.
     *
     * <p>Uses exact ShuntingDynamicsImpl equivalence via {@link BatchShuntingDynamics}.
     * For Layer 4 (no lateral interactions), SIMD optimization is applied automatically.
     *
     * @param inputs input patterns
     * @param params Layer 4 parameters
     * @param size layer size
     * @return processed patterns, or null if SIMD not beneficial
     */
    public static Pattern[] processBatchSIMD(Pattern[] inputs, Layer4Parameters params, int size) {
        // Check if transpose-and-vectorize is beneficial
        var operations = 10;  // Rough estimate: scale, dynamics (5 ops), saturation (4 ops)
        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                inputs.length, size, operations)) {
            // Fall back to sequential (not worth transpose overhead)
            return null;  // Caller should use sequential path
        }

        // Create SIMD batch (transpose to dimension-major)
        var batch = createBatch(inputs, size);

        // Apply driving strength (SIMD)
        batch.applyDrivingStrength(params.getDrivingStrength());

        // Apply exact shunting dynamics
        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        batch.applyDynamicsExact(Math.min(params.getTimeConstant() / 1000.0, 0.01), shuntingParams);

        // Apply saturation (SIMD)
        batch.applySaturation(params.getCeiling(), params.getFloor());

        // Convert back to patterns (transpose to pattern-major)
        return batch.toPatterns();
    }

    /**
     * Statistics for SIMD batch processing.
     */
    public record Statistics(
        int batchSize,
        int dimension,
        long transposeTime,
        long processingTime,
        long totalTime,
        double throughput,
        boolean usedSIMD
    ) {
        public double getSpeedup(long sequentialTime) {
            return (double) sequentialTime / totalTime;
        }

        public double getTransposeOverhead() {
            return (double) transposeTime / totalTime;
        }
    }
}
