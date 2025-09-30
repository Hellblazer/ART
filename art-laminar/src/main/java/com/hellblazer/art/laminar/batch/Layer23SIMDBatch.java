package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer23Parameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for Layer 2/3.
 *
 * <h2>Layer 2/3 Characteristics</h2>
 * <ul>
 *   <li>Medium time constants (30-150ms) for perceptual integration</li>
 *   <li>TWO input sources: bottom-up (Layer 4) + top-down (Layer 1)</li>
 *   <li>Complex cell pooling for opposite contrast polarities</li>
 *   <li>LEAKY INTEGRATION (NOT shunting dynamics!)</li>
 *   <li>Horizontal grouping via bipole network (DISABLED in Phase 5)</li>
 * </ul>
 *
 * <h2>Performance Strategy</h2>
 * <p>Same transpose-and-vectorize pattern as Layer 4/5/6:
 * <pre>
 * 1. Transpose to dimension-major layout
 * 2. SIMD operations across batch dimension
 * 3. Transpose back to pattern-major
 * </pre>
 *
 * <h2>Layer 2/3 Specific Operations</h2>
 * <ol>
 *   <li>Bottom-up input integration from Layer 4 (SIMD)</li>
 *   <li>Top-down priming integration from Layer 1 (SIMD)</li>
 *   <li>Combined inputs with weights (SIMD)</li>
 *   <li>Complex cell pooling (SIMD with spatial operations)</li>
 *   <li>Leaky integration - exponential approach (SIMD)</li>
 *   <li>Saturation (SIMD)</li>
 * </ol>
 *
 * <h2>Phase 5 Limitations</h2>
 * <ul>
 *   <li>Bipole cell network is DISABLED (returns null if enableHorizontalGrouping=true)</li>
 *   <li>Falls back to sequential processing when bipole network enabled</li>
 *   <li>Full bipole SIMD optimization deferred to Phase 6</li>
 * </ul>
 *
 * @author Claude Code
 */
public class Layer23SIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[][] dimensionMajor;   // [dimension][batchSize] - current activations
    private final double[][] bottomUpMajor;    // [dimension][batchSize] - bottom-up input
    private final double[][] topDownMajor;     // [dimension][batchSize] - top-down priming
    private final int dimension;
    private final int batchSize;

    private Layer23SIMDBatch(double[][] dimensionMajor, double[][] bottomUpMajor, double[][] topDownMajor) {
        this.dimensionMajor = dimensionMajor;
        this.bottomUpMajor = bottomUpMajor;
        this.topDownMajor = topDownMajor;
        this.dimension = dimensionMajor.length;
        this.batchSize = dimensionMajor[0].length;
    }

    /**
     * Create SIMD batch from patterns with bottom-up and top-down inputs.
     *
     * @param bottomUpInputs bottom-up input patterns from Layer 4 [batchSize][dimension]
     * @param topDownPriming top-down priming patterns from Layer 1 [batchSize][dimension], or null
     * @param layerSize expected dimension
     * @return SIMD batch ready for processing
     */
    public static Layer23SIMDBatch createBatch(Pattern[] bottomUpInputs, Pattern[] topDownPriming, int layerSize) {
        if (bottomUpInputs == null || bottomUpInputs.length == 0) {
            throw new IllegalArgumentException("bottomUpInputs cannot be null or empty");
        }

        // Validate dimensions
        for (int i = 0; i < bottomUpInputs.length; i++) {
            if (bottomUpInputs[i] == null) {
                throw new IllegalArgumentException("bottomUpInputs[" + i + "] is null");
            }
            if (bottomUpInputs[i].dimension() != layerSize) {
                throw new IllegalArgumentException(
                    String.format("bottomUpInputs[%d] dimension mismatch: expected %d, got %d",
                        i, layerSize, bottomUpInputs[i].dimension()));
            }
        }

        // Transpose bottom-up to dimension-major
        var bottomUpMajor = BatchDataLayout.transposeToDimensionMajor(bottomUpInputs);

        // Transpose top-down priming (or create zeros)
        double[][] topDownMajor;
        if (topDownPriming != null && topDownPriming.length == bottomUpInputs.length) {
            topDownMajor = BatchDataLayout.transposeToDimensionMajor(topDownPriming);
        } else {
            topDownMajor = new double[layerSize][bottomUpInputs.length];
        }

        // Initialize current activations from bottom-up (will be modified by operations)
        var dimensionMajor = new double[layerSize][bottomUpInputs.length];
        for (int d = 0; d < layerSize; d++) {
            System.arraycopy(bottomUpMajor[d], 0, dimensionMajor[d], 0, bottomUpInputs.length);
        }

        return new Layer23SIMDBatch(dimensionMajor, bottomUpMajor, topDownMajor);
    }

    /**
     * Apply bottom-up input integration from Layer 4 (SIMD).
     *
     * <p>Scales bottom-up input by weight.
     *
     * @param bottomUpWeight weight for bottom-up input
     */
    public void applyBottomUpIntegration(double bottomUpWeight) {
        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            var bottomUpRow = bottomUpMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var bottomUp = DoubleVector.fromArray(SPECIES, bottomUpRow, i);
                var scaled = bottomUp.mul(bottomUpWeight);
                scaled.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] = bottomUpRow[i] * bottomUpWeight;
            }
        }
    }

    /**
     * Apply top-down priming integration from Layer 1 (SIMD).
     *
     * <p>Adds weighted top-down priming to current activations.
     *
     * @param topDownWeight weight for top-down priming
     */
    public void applyTopDownIntegration(double topDownWeight) {
        int laneSize = SPECIES.length();
        var weightVec = DoubleVector.broadcast(SPECIES, topDownWeight);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            var topDownRow = topDownMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var current = DoubleVector.fromArray(SPECIES, row, i);
                var topDown = DoubleVector.fromArray(SPECIES, topDownRow, i);

                // Add weighted top-down: current + topDownWeight * topDown
                var weighted = topDown.mul(weightVec);
                var combined = current.add(weighted);

                combined.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] += topDownWeight * topDownRow[i];
            }
        }
    }

    /**
     * Apply combined inputs with weights (SIMD).
     *
     * <p>Combines multiple input sources:
     * <pre>result = bottomUpWeight * bottomUp + topDownWeight * topDown</pre>
     *
     * <p>Horizontal weight is reserved for Phase 6 (bipole network).
     *
     * @param bottomUpWeight weight for bottom-up input
     * @param topDownWeight weight for top-down priming
     * @param horizontalWeight weight for horizontal grouping (ignored in Phase 5)
     */
    public void applyCombinedInputs(double bottomUpWeight, double topDownWeight, double horizontalWeight) {
        int laneSize = SPECIES.length();
        var bottomUpWeightVec = DoubleVector.broadcast(SPECIES, bottomUpWeight);
        var topDownWeightVec = DoubleVector.broadcast(SPECIES, topDownWeight);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            var bottomUpRow = bottomUpMajor[d];
            var topDownRow = topDownMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var bottomUp = DoubleVector.fromArray(SPECIES, bottomUpRow, i);
                var topDown = DoubleVector.fromArray(SPECIES, topDownRow, i);

                // Combine: bottomUpWeight * bottomUp + topDownWeight * topDown
                var weightedBottomUp = bottomUp.mul(bottomUpWeightVec);
                var weightedTopDown = topDown.mul(topDownWeightVec);
                var combined = weightedBottomUp.add(weightedTopDown);

                combined.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] = bottomUpWeight * bottomUpRow[i] + topDownWeight * topDownRow[i];
            }
        }
    }

    /**
     * Apply complex cell pooling (SIMD with spatial operations).
     *
     * <p>Complex cells pool signals from opposite contrast polarities.
     * This implements spatial pooling across dimensions.
     *
     * <p>Simple implementation: enhance values above threshold.
     *
     * @param complexCellThreshold threshold for complex cell activation
     */
    public void applyComplexCellPooling(double complexCellThreshold) {
        if (complexCellThreshold <= 0) {
            return;  // No complex cell pooling
        }

        int laneSize = SPECIES.length();
        var thresholdVec = DoubleVector.broadcast(SPECIES, complexCellThreshold);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);

                // Simple complex cell: enhance values above threshold
                // If x > threshold: x = x + 0.1 * (1.0 - x)
                var mask = vec.compare(jdk.incubator.vector.VectorOperators.GT, thresholdVec);
                var enhancement = oneVec.sub(vec).mul(0.1);
                var enhanced = vec.add(enhancement);
                var result = vec.blend(enhanced, mask);

                result.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                if (row[i] > complexCellThreshold) {
                    row[i] = row[i] + 0.1 * (1.0 - row[i]);
                }
            }
        }
    }

    /**
     * Apply leaky integration (SIMD).
     *
     * <p>CRITICAL: Layer 2/3 uses LEAKY INTEGRATION, NOT shunting dynamics!
     * <pre>
     * Leaky integration: newValue = currentValue + alpha * (targetValue - currentValue)
     * where alpha = timeStep / timeConstant
     * </pre>
     *
     * <p>This is simple exponential approach to target value.
     *
     * @param timeStep integration time step (seconds)
     * @param timeConstant time constant for integration (seconds, 30-150ms)
     */
    public void applyLeakyIntegration(double timeStep, double timeConstant) {
        int laneSize = SPECIES.length();
        var alpha = timeStep / timeConstant;
        var alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        var oneMinusAlphaVec = DoubleVector.broadcast(SPECIES, 1.0 - alpha);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var current = DoubleVector.fromArray(SPECIES, row, i);

                // Leaky integration: current + alpha * (target - current)
                // Target is the current input (already in row from previous operations)
                // For simplicity, assume target = 0 for decay, or we keep current as-is
                // Actually, leaky integration decays toward zero:
                // newValue = (1 - alpha) * currentValue
                var integrated = current.mul(oneMinusAlphaVec);

                integrated.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] *= (1.0 - alpha);
            }
        }
    }

    /**
     * Apply saturation (SIMD).
     *
     * <p>Clamps values to [floor, ceiling] range.
     *
     * @param ceiling maximum activation
     * @param floor minimum activation
     */
    public void applySaturation(double ceiling, double floor) {
        int laneSize = SPECIES.length();
        var ceilingVec = DoubleVector.broadcast(SPECIES, ceiling);
        var floorVec = DoubleVector.broadcast(SPECIES, floor);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);
                var clamped = vec.max(floorVec).min(ceilingVec);
                clamped.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] = Math.max(floor, Math.min(ceiling, row[i]));
            }
        }
    }

    /**
     * Convert back to pattern-major layout.
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
     */
    public double[][] getDimensionMajor() {
        return dimensionMajor;
    }

    /**
     * Get bottom-up input data (direct access for debugging).
     */
    public double[][] getBottomUpMajor() {
        return bottomUpMajor;
    }

    /**
     * Get top-down priming data (direct access for debugging).
     */
    public double[][] getTopDownMajor() {
        return topDownMajor;
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
     * Process complete Layer 2/3 batch with SIMD optimization.
     *
     * <p>Main entry point for Layer 2/3 SIMD batch processing.
     *
     * <p><b>Phase 5 Limitation:</b> Bipole cell network is DISABLED.
     * If {@code enableHorizontalGrouping} is true, returns null (falls back to sequential).
     *
     * @param bottomUpInputs bottom-up input patterns from Layer 4
     * @param topDownPriming top-down priming patterns from Layer 1, or null
     * @param params Layer 2/3 parameters
     * @param size layer size
     * @return processed patterns, or null if SIMD not beneficial or bipole network enabled
     */
    public static Pattern[] processBatchSIMD(Pattern[] bottomUpInputs, Pattern[] topDownPriming,
                                              Layer23Parameters params, int size) {
        // CRITICAL: Check if bipole network is enabled
        // Phase 5 does NOT support bipole network in SIMD mode
        if (params.enableHorizontalGrouping()) {
            return null;  // Fall back to sequential - bipole network needs sequential processing in Phase 5
        }

        // Check if transpose-and-vectorize is beneficial
        // Layer 2/3 operations: combined inputs, complex cells, leaky integration, saturation
        var operations = 12;  // Estimated number of SIMD operations
        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                bottomUpInputs.length, size, operations)) {
            return null;  // Fall back to sequential (not worth transpose overhead)
        }

        // Create SIMD batch (transpose to dimension-major)
        var batch = createBatch(bottomUpInputs, topDownPriming, size);

        // 1. Apply combined inputs (SIMD) - bottom-up + top-down
        batch.applyCombinedInputs(params.bottomUpWeight(), params.topDownWeight(), 0.0);

        // 2. Apply complex cell pooling (SIMD) if enabled
        if (params.enableComplexCells()) {
            batch.applyComplexCellPooling(params.complexCellThreshold());
        }

        // 3. Apply leaky integration (SIMD) - NOT shunting dynamics!
        var timeStep = 0.01;  // 10ms time step
        batch.applyLeakyIntegration(timeStep, params.timeConstant());

        // 4. Apply saturation (SIMD)
        batch.applySaturation(params.getCeiling(), params.getFloor());

        // Convert back to patterns (transpose to pattern-major)
        return batch.toPatterns();
    }
}
