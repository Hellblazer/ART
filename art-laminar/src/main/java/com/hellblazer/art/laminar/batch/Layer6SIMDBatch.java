package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer6Parameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for Layer 6.
 *
 * <h2>Layer 6 Characteristics</h2>
 * <ul>
 *   <li>CRITICAL: ART matching rule - modulatory only (cannot fire without bottom-up!)</li>
 *   <li>Slow time constants (100-500ms) for sustained modulation</li>
 *   <li>On-center, off-surround dynamics</li>
 *   <li>Attentional gain modulation</li>
 *   <li>Lower firing rates than other layers</li>
 *   <li>Moderate lateral inhibition (0.3)</li>
 * </ul>
 *
 * <h2>Performance Strategy</h2>
 * <p>Same transpose-and-vectorize pattern as Layer 4/5:
 * <pre>
 * 1. Transpose to dimension-major layout
 * 2. SIMD operations across batch dimension
 * 3. Transpose back to pattern-major
 * </pre>
 *
 * <h2>Layer 6 Specific Operations</h2>
 * <ol>
 *   <li>ART matching rule: bottomUp * (1 + onCenterWeight * topDown) - CRITICAL: zero if no bottom-up!</li>
 *   <li>Off-surround inhibition (SIMD with reduction)</li>
 *   <li>Attentional gain modulation (SIMD)</li>
 *   <li>Shunting dynamics (exact equivalence via BatchShuntingDynamics)</li>
 *   <li>Saturation (SIMD)</li>
 * </ol>
 *
 * @author Hal Hildebrand
 */
public class Layer6SIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[][] dimensionMajor;  // [dimension][batchSize]
    private final double[][] topDownMajor;     // Top-down expectations [dimension][batchSize]
    private final int dimension;
    private final int batchSize;

    private Layer6SIMDBatch(double[][] dimensionMajor, double[][] topDownMajor) {
        this.dimensionMajor = dimensionMajor;
        this.topDownMajor = topDownMajor;
        this.dimension = dimensionMajor.length;
        this.batchSize = dimensionMajor[0].length;
    }

    /**
     * Create SIMD batch from patterns with top-down expectations.
     *
     * @param bottomUpInputs bottom-up input patterns [batchSize][dimension]
     * @param topDownExpectations top-down expectation patterns [batchSize][dimension], or null
     * @param layerSize expected dimension
     * @return SIMD batch ready for processing
     */
    public static Layer6SIMDBatch createBatch(Pattern[] bottomUpInputs, Pattern[] topDownExpectations, int layerSize) {
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
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(bottomUpInputs);

        // Transpose top-down expectations (or create zeros)
        double[][] topDownMajor;
        if (topDownExpectations != null && topDownExpectations.length == bottomUpInputs.length) {
            topDownMajor = BatchDataLayout.transposeToDimensionMajor(topDownExpectations);
        } else {
            topDownMajor = new double[layerSize][bottomUpInputs.length];
        }

        return new Layer6SIMDBatch(dimensionMajor, topDownMajor);
    }

    /**
     * Apply ART matching rule (SIMD).
     *
     * <p>CRITICAL: Layer 6 is modulatory only!
     * <pre>
     * output = bottomUp * (1 + onCenterWeight * topDown)
     * IF bottomUp == 0, THEN output MUST BE 0 (no bottom-up = no output!)
     * </pre>
     *
     * @param onCenterWeight on-center modulation strength
     * @param modulationThreshold minimum bottom-up for modulation
     */
    public void applyARTMatchingRule(double onCenterWeight, double modulationThreshold) {
        int laneSize = SPECIES.length();
        var onCenterVec = DoubleVector.broadcast(SPECIES, onCenterWeight);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);
        var thresholdVec = DoubleVector.broadcast(SPECIES, modulationThreshold);

        for (int d = 0; d < dimension; d++) {
            var bottomUpRow = dimensionMajor[d];
            var topDownRow = topDownMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var bottomUp = DoubleVector.fromArray(SPECIES, bottomUpRow, i);
                var topDown = DoubleVector.fromArray(SPECIES, topDownRow, i);

                // modulation = bottomUp * (1 + onCenterWeight * topDown)
                var modulation = bottomUp.mul(
                    oneVec.add(topDown.mul(onCenterVec))
                );

                // CRITICAL: Zero output where bottom-up is <= threshold (use LE comparison)
                var mask = bottomUp.compare(VectorOperators.LE, thresholdVec);
                var result = modulation.blend(DoubleVector.zero(SPECIES), mask);

                result.intoArray(bottomUpRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                if (bottomUpRow[i] <= modulationThreshold) {
                    // NO BOTTOM-UP = NO OUTPUT (CRITICAL!)
                    bottomUpRow[i] = 0.0;
                } else {
                    // Apply modulation
                    bottomUpRow[i] *= (1.0 + onCenterWeight * topDownRow[i]);
                }
            }
        }
    }

    /**
     * Apply off-surround inhibition (SIMD with spatial reduction).
     *
     * <p>Each position is inhibited by its spatial neighbors.
     * This implements the center-surround organization of Layer 6.
     *
     * @param offSurroundStrength inhibition strength
     * @param surroundSize neighborhood size (e.g., 2 = ±2 neighbors)
     */
    public void applyOffSurroundInhibition(double offSurroundStrength, int surroundSize) {
        if (offSurroundStrength <= 0 || surroundSize <= 0) {
            return;  // No inhibition
        }

        int laneSize = SPECIES.length();
        var strengthVec = DoubleVector.broadcast(SPECIES, offSurroundStrength);

        // Process each dimension independently
        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];

            // For each position, compute surround inhibition
            // We'll do this pattern-wise (not vectorized across space, but across batch)
            for (int p = 0; p < batchSize; p++) {
                double surround = 0.0;

                // Compute surround sum (spatial neighbors in dimension space)
                for (int offset = 1; offset <= surroundSize; offset++) {
                    // Left neighbor in dimension space
                    if (d - offset >= 0) {
                        surround += dimensionMajor[d - offset][p];
                    }
                    // Right neighbor in dimension space
                    if (d + offset < dimension) {
                        surround += dimensionMajor[d + offset][p];
                    }
                }

                // Apply inhibition
                row[p] = Math.max(0.0, row[p] - offSurroundStrength * surround);
            }
        }
    }

    /**
     * Apply attentional gain modulation (SIMD).
     *
     * <p>Scales activations by attentional gain when top-down signal is present.
     *
     * @param attentionalGain gain factor
     */
    public void applyAttentionalGain(double attentionalGain) {
        if (attentionalGain == 1.0) {
            return;  // No gain modulation
        }

        int laneSize = SPECIES.length();
        var gainVec = DoubleVector.broadcast(SPECIES, attentionalGain);

        for (int d = 0; d < dimension; d++) {
            var row = dimensionMajor[d];
            var topDownRow = topDownMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var activation = DoubleVector.fromArray(SPECIES, row, i);
                var topDown = DoubleVector.fromArray(SPECIES, topDownRow, i);

                // Gain: activation * (1 + (gain - 1) * topDown)
                // When topDown = 0, gain = 1 (no modulation)
                // When topDown = 1, gain = attentionalGain (full modulation)
                var modulatedGain = DoubleVector.broadcast(SPECIES, 1.0).add(
                    topDown.mul(gainVec.sub(DoubleVector.broadcast(SPECIES, 1.0)))
                );

                var modulated = activation.mul(modulatedGain);
                modulated.intoArray(row, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                var modulatedGain = 1.0 + (attentionalGain - 1.0) * topDownRow[i];
                row[i] *= modulatedGain;
            }
        }
    }

    /**
     * Apply exact shunting dynamics evolution.
     *
     * <p>Uses {@link BatchShuntingDynamics} for bit-exact equivalence.
     * Layer 6 has moderate lateral inhibition (0.3).
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
     * Get top-down expectation data (direct access for debugging).
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
     * Process complete Layer 6 batch with SIMD optimization.
     *
     * <p>Main entry point for Layer 6 SIMD batch processing.
     *
     * @param bottomUpInputs bottom-up input patterns
     * @param topDownExpectations top-down expectation patterns, or null
     * @param params Layer 6 parameters
     * @param size layer size
     * @return processed patterns, or null if SIMD not beneficial
     */
    public static Pattern[] processBatchSIMD(Pattern[] bottomUpInputs, Pattern[] topDownExpectations,
                                              Layer6Parameters params, int size) {
        // Check if transpose-and-vectorize is beneficial
        // Layer 6 operations: matching rule, off-surround, gain, dynamics, saturation
        var operations = 15;  // Similar complexity to Layer 4/5
        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                bottomUpInputs.length, size, operations)) {
            return null;  // Fall back to sequential
        }

        // Create SIMD batch (transpose to dimension-major)
        var batch = createBatch(bottomUpInputs, topDownExpectations, size);

        // 1. Apply ART matching rule (SIMD) - CRITICAL: modulatory only!
        batch.applyARTMatchingRule(params.getOnCenterWeight(), params.getModulationThreshold());

        // 2. Apply off-surround inhibition (SIMD with reduction)
        batch.applyOffSurroundInhibition(params.getOffSurroundStrength(), 2);  // surroundSize = 2

        // 3. Apply attentional gain modulation (SIMD)
        batch.applyAttentionalGain(params.getAttentionalGain());

        // 4. Apply exact shunting dynamics
        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        var timeStep = params.getTimeConstant() / 5000.0;  // Convert ms to seconds
        batch.applyDynamicsExact(timeStep, shuntingParams);

        // 5. Apply saturation (SIMD)
        batch.applySaturation(params.getCeiling(), params.getFloor());

        // Convert back to patterns (transpose to pattern-major)
        return batch.toPatterns();
    }
}
