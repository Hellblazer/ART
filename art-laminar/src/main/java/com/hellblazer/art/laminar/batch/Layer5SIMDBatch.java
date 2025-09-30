package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer5Parameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for Layer 5.
 *
 * <h2>Layer 5 Characteristics</h2>
 * <ul>
 *   <li>Medium time constants (50-200ms) for sustained output</li>
 *   <li>Amplification gain for salient features</li>
 *   <li>Burst firing capability for important signals</li>
 *   <li>Output normalization for stable signaling</li>
 *   <li>Moderate lateral inhibition (0.1)</li>
 * </ul>
 *
 * <h2>Performance Strategy</h2>
 * <p>Same transpose-and-vectorize pattern as Layer 4:
 * <pre>
 * 1. Transpose to dimension-major layout
 * 2. SIMD operations across batch dimension
 * 3. Transpose back to pattern-major
 * </pre>
 *
 * <h2>Layer 5 Specific Operations</h2>
 * <ol>
 *   <li>Amplification gain (SIMD)</li>
 *   <li>Burst detection and amplification (SIMD with masking)</li>
 *   <li>State persistence blending (SIMD)</li>
 *   <li>Shunting dynamics (exact equivalence via BatchShuntingDynamics)</li>
 *   <li>Output gain (SIMD)</li>
 *   <li>Output normalization (SIMD with reduction)</li>
 * </ol>
 *
 * @author Claude Code
 */
public class Layer5SIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[][] dimensionMajor;  // [dimension][batchSize]
    private final double[][] previousActivation;  // State persistence
    private final int dimension;
    private final int batchSize;

    private Layer5SIMDBatch(double[][] dimensionMajor, double[][] previousActivation) {
        this.dimensionMajor = dimensionMajor;
        this.previousActivation = previousActivation;
        this.dimension = dimensionMajor.length;
        this.batchSize = dimensionMajor[0].length;
    }

    /**
     * Create SIMD batch from patterns with state persistence.
     *
     * @param patterns input patterns [batchSize][dimension]
     * @param previousStates previous activation states [batchSize][dimension], or null
     * @param layerSize expected dimension
     * @return SIMD batch ready for processing
     */
    public static Layer5SIMDBatch createBatch(Pattern[] patterns, Pattern[] previousStates, int layerSize) {
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

        // Transpose patterns to dimension-major
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        // Transpose previous states (or create zeros)
        double[][] prevDimMajor;
        if (previousStates != null && previousStates.length == patterns.length) {
            prevDimMajor = BatchDataLayout.transposeToDimensionMajor(previousStates);
        } else {
            prevDimMajor = new double[layerSize][patterns.length];
        }

        return new Layer5SIMDBatch(dimensionMajor, prevDimMajor);
    }

    /**
     * Apply amplification gain to entire batch (SIMD).
     *
     * @param amplificationGain scaling factor
     */
    public void applyAmplificationGain(double amplificationGain) {
        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);
                var scaled = vec.mul(amplificationGain);
                scaled.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] *= amplificationGain;
            }
        }
    }

    /**
     * Detect and apply burst firing amplification (SIMD with masking).
     *
     * <p>Burst firing occurs when activations exceed threshold.
     * Uses SIMD masking for efficient conditional processing.
     *
     * @param burstThreshold threshold for burst detection
     * @param burstAmplification amplification factor during burst
     */
    public void applyBurstFiring(double burstThreshold, double burstAmplification) {
        int laneSize = SPECIES.length();
        var thresholdVec = DoubleVector.broadcast(SPECIES, burstThreshold);
        var amplificationVec = DoubleVector.broadcast(SPECIES, burstAmplification);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop with masking
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);

                // Create mask for values > threshold
                var mask = vec.compare(VectorOperators.GT, thresholdVec);

                // Apply amplification only to bursting neurons
                var amplified = vec.mul(amplificationVec);
                var result = vec.blend(amplified, mask);

                result.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                if (row[i] > burstThreshold) {
                    row[i] *= burstAmplification;
                }
            }
        }
    }

    /**
     * Apply state persistence blending with decay (SIMD).
     *
     * <p>Blends current input with decayed previous activation:
     * <pre>result = input + persistence * previousActivation</pre>
     *
     * @param persistence persistence factor (1.0 - decay)
     */
    public void applyStatePersistence(double persistence) {
        int laneSize = SPECIES.length();
        var persistenceVec = DoubleVector.broadcast(SPECIES, persistence);

        for (int d = 0; d < dimension; d++) {
            var currentRow = dimensionMajor[d];
            var prevRow = previousActivation[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var current = DoubleVector.fromArray(SPECIES, currentRow, i);
                var prev = DoubleVector.fromArray(SPECIES, prevRow, i);

                // Blend: current + persistence * prev
                var decayedPrev = prev.mul(persistenceVec);
                var blended = current.add(decayedPrev);

                blended.intoArray(currentRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                currentRow[i] += persistence * prevRow[i];
            }
        }
    }

    /**
     * Apply exact shunting dynamics evolution.
     *
     * <p>Uses {@link BatchShuntingDynamics} for bit-exact equivalence.
     * Layer 5 has weak lateral inhibition (0.1).
     *
     * @param timeStep integration time step
     * @param params shunting parameters
     */
    public void applyDynamicsExact(double timeStep, com.hellblazer.art.temporal.dynamics.ShuntingParameters params) {
        var batchDynamics = new BatchShuntingDynamics(params, dimension);

        // dimensionMajor is both current state and excitatory input
        var evolved = batchDynamics.evolveBatch(dimensionMajor, dimensionMajor, timeStep);

        // Copy evolved state back
        for (int d = 0; d < dimension; d++) {
            System.arraycopy(evolved[d], 0, dimensionMajor[d], 0, batchSize);
        }
    }

    /**
     * Apply output gain (SIMD).
     *
     * @param outputGain output scaling factor
     */
    public void applyOutputGain(double outputGain) {
        int laneSize = SPECIES.length();

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var vec = DoubleVector.fromArray(SPECIES, row, i);
                var scaled = vec.mul(outputGain);
                scaled.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                row[i] *= outputGain;
            }
        }
    }

    /**
     * Apply output normalization (SIMD with reduction).
     *
     * <p>Normalizes output: scale = 1 / (1 + normalization * sum)
     *
     * @param normalization normalization strength
     */
    public void applyOutputNormalization(double normalization) {
        if (normalization <= 0) {
            return;  // No normalization
        }

        int laneSize = SPECIES.length();

        // For each pattern in batch, compute sum and normalize
        for (int p = 0; p < batchSize; p++) {
            // Compute sum across all dimensions for pattern p (vertical slice)
            double sum = 0.0;
            for (int d = 0; d < dimension; d++) {
                sum += dimensionMajor[d][p];
            }

            // Apply normalization if sum is significant
            if (sum > 0.01) {
                double normalizer = 1.0 / (1.0 + normalization * sum);

                // Scale all dimensions for this pattern
                for (int d = 0; d < dimension; d++) {
                    dimensionMajor[d][p] *= normalizer;
                }
            }
        }
    }

    /**
     * Apply saturation (SIMD).
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
     * Update previous activation state for next iteration.
     */
    public void updatePreviousActivation() {
        for (int d = 0; d < dimension; d++) {
            System.arraycopy(dimensionMajor[d], 0, previousActivation[d], 0, batchSize);
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
     * Process complete Layer 5 batch with SIMD optimization.
     *
     * <p>Main entry point for Layer 5 SIMD batch processing.
     *
     * @param inputs input patterns
     * @param previousStates previous activation states (for persistence), or null
     * @param params Layer 5 parameters
     * @param size layer size
     * @return processed patterns, or null if SIMD not beneficial
     */
    public static Pattern[] processBatchSIMD(Pattern[] inputs, Pattern[] previousStates,
                                              Layer5Parameters params, int size) {
        // Check if transpose-and-vectorize is beneficial
        // Layer 5 has more operations than Layer 4: amplification, burst, persistence, dynamics, normalization
        var operations = 15;  // More operations than Layer 4
        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                inputs.length, size, operations)) {
            return null;  // Fall back to sequential
        }

        // Create SIMD batch (transpose to dimension-major)
        var batch = createBatch(inputs, previousStates, size);

        // 1. Apply amplification gain (SIMD)
        batch.applyAmplificationGain(params.getAmplificationGain());

        // 2. Detect and apply burst firing (SIMD with masking)
        batch.applyBurstFiring(params.getBurstThreshold(), params.getBurstAmplification());

        // 3. Apply state persistence (SIMD)
        var persistence = 1.0 - params.getDecayRate() * 0.01;
        batch.applyStatePersistence(persistence);

        // 4. Apply exact shunting dynamics
        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        var timeStep = Math.min(params.getTimeConstant() / 10000.0, 0.01);
        batch.applyDynamicsExact(timeStep, shuntingParams);

        // 5. Apply output gain (SIMD)
        batch.applyOutputGain(params.getOutputGain());

        // 6. Apply output normalization (SIMD with reduction)
        batch.applyOutputNormalization(params.getOutputNormalization());

        // 7. Apply saturation (SIMD)
        batch.applySaturation(params.getCeiling(), params.getFloor());

        // 8. Update previous activation state for next iteration
        batch.updatePreviousActivation();

        // Convert back to patterns (transpose to pattern-major)
        return batch.toPatterns();
    }
}
