package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer1Parameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for Layer 1 (Top-Down Attentional Priming).
 *
 * <h2>Layer 1 Characteristics</h2>
 * <ul>
 *   <li>Very slow time constants (200-1000ms) - slowest of all layers</li>
 *   <li>Top-down processing (processes expectations, NOT bottom-up input)</li>
 *   <li>Priming only - does NOT drive cells directly (capped at 0.5)</li>
 *   <li>Sustained persistence - effects last long after input ends</li>
 *   <li>Three state systems:
 *     <ul>
 *       <li>attentionState - current attention focus</li>
 *       <li>primingEffect - active priming signal</li>
 *       <li>memoryTrace - long-term memory trace (seconds)</li>
 *     </ul>
 *   </li>
 *   <li>Provides apical dendrite signals for Layer 2/3 integration</li>
 * </ul>
 *
 * <h2>Performance Strategy</h2>
 * <p>Same transpose-and-vectorize pattern as Layers 4/5/6:
 * <pre>
 * 1. Transpose to dimension-major layout
 * 2. SIMD operations across batch dimension
 * 3. Transpose back to pattern-major
 * </pre>
 *
 * <h2>Layer 1 Specific Operations</h2>
 * <ol>
 *   <li>Attention state update with very slow decay (SIMD)</li>
 *   <li>Memory trace integration for long-term persistence (SIMD)</li>
 *   <li>Priming effect calculation: attention + memory (SIMD)</li>
 *   <li>Apical dendrite signal generation for Layer 2/3 (SIMD)</li>
 *   <li>Shunting dynamics with very slow time constants (exact equivalence)</li>
 *   <li>Saturation (SIMD)</li>
 * </ol>
 *
 * <h2>Key Differences from Layers 4/5/6</h2>
 * <ul>
 *   <li>Processes TOP-DOWN signals (expectations), not bottom-up input</li>
 *   <li>THREE state arrays instead of one</li>
 *   <li>VERY slow time constants (10x slower than other layers)</li>
 *   <li>Priming ONLY - capped at 0.5 to prevent driving responses</li>
 * </ul>
 *
 * @author Hal Hildebrand
 */
public class Layer1SIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[][] dimensionMajor;     // [dimension][batchSize] - current activation
    private final double[][] attentionMajor;     // [dimension][batchSize] - attention state
    private final double[][] primingMajor;       // [dimension][batchSize] - priming effect
    private final double[][] memoryMajor;        // [dimension][batchSize] - memory trace
    private final int dimension;
    private final int batchSize;

    private Layer1SIMDBatch(double[][] dimensionMajor, double[][] attentionMajor,
                            double[][] primingMajor, double[][] memoryMajor) {
        this.dimensionMajor = dimensionMajor;
        this.attentionMajor = attentionMajor;
        this.primingMajor = primingMajor;
        this.memoryMajor = memoryMajor;
        this.dimension = dimensionMajor.length;
        this.batchSize = dimensionMajor[0].length;
    }

    /**
     * Create SIMD batch from top-down expectations with state persistence.
     *
     * <p>Layer 1 maintains three separate state arrays for attention, priming, and memory.
     *
     * @param expectations top-down expectation patterns [batchSize][dimension]
     * @param previousAttention previous attention state [batchSize][dimension], or null
     * @param previousPriming previous priming effect [batchSize][dimension], or null
     * @param previousMemory previous memory trace [batchSize][dimension], or null
     * @param layerSize expected dimension
     * @return SIMD batch ready for processing
     */
    public static Layer1SIMDBatch createBatch(Pattern[] expectations,
                                               Pattern[] previousAttention,
                                               Pattern[] previousPriming,
                                               Pattern[] previousMemory,
                                               int layerSize) {
        if (expectations == null || expectations.length == 0) {
            throw new IllegalArgumentException("expectations cannot be null or empty");
        }

        // Validate dimensions
        for (int i = 0; i < expectations.length; i++) {
            if (expectations[i] == null) {
                throw new IllegalArgumentException("expectations[" + i + "] is null");
            }
            if (expectations[i].dimension() != layerSize) {
                throw new IllegalArgumentException(
                    String.format("expectations[%d] dimension mismatch: expected %d, got %d",
                        i, layerSize, expectations[i].dimension()));
            }
        }

        // Transpose expectations to dimension-major
        var dimensionMajor = BatchDataLayout.transposeToDimensionMajor(expectations);

        // Transpose or create attention state
        double[][] attentionDimMajor;
        if (previousAttention != null && previousAttention.length == expectations.length) {
            attentionDimMajor = BatchDataLayout.transposeToDimensionMajor(previousAttention);
        } else {
            attentionDimMajor = new double[layerSize][expectations.length];
        }

        // Transpose or create priming effect
        double[][] primingDimMajor;
        if (previousPriming != null && previousPriming.length == expectations.length) {
            primingDimMajor = BatchDataLayout.transposeToDimensionMajor(previousPriming);
        } else {
            primingDimMajor = new double[layerSize][expectations.length];
        }

        // Transpose or create memory trace
        double[][] memoryDimMajor;
        if (previousMemory != null && previousMemory.length == expectations.length) {
            memoryDimMajor = BatchDataLayout.transposeToDimensionMajor(previousMemory);
        } else {
            memoryDimMajor = new double[layerSize][expectations.length];
        }

        return new Layer1SIMDBatch(dimensionMajor, attentionDimMajor, primingDimMajor, memoryDimMajor);
    }

    /**
     * Apply attention state update with very slow decay (SIMD).
     *
     * <p>Attention decays VERY slowly (200-1000ms time constants) and shifts
     * gradually to new locations.
     *
     * <p>Update rule:
     * <pre>
     * attentionState *= (1 - sustainedDecayRate * 0.01)  // Very slow decay
     * attentionState += input * attentionShiftRate        // Gradual shift
     * attentionState = min(attentionState, 1.0)           // Cap at 1.0
     * </pre>
     *
     * @param sustainedDecayRate very slow decay rate (0-0.01)
     * @param attentionShiftRate rate of attention shifting (0-1)
     */
    public void applyAttentionStateUpdate(double sustainedDecayRate, double attentionShiftRate) {
        int laneSize = SPECIES.length();
        var decayFactor = 1.0 - sustainedDecayRate * 0.01;  // Much slower decay
        var decayVec = DoubleVector.broadcast(SPECIES, decayFactor);
        var shiftVec = DoubleVector.broadcast(SPECIES, attentionShiftRate);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);
        var zeroVec = DoubleVector.broadcast(SPECIES, 0.0);

        for (int d = 0; d < dimension; d++) {
            var attentionRow = attentionMajor[d];
            var inputRow = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var attention = DoubleVector.fromArray(SPECIES, attentionRow, i);
                var input = DoubleVector.fromArray(SPECIES, inputRow, i);

                // Very slow decay
                var decayed = attention.mul(decayVec);

                // Shift attention gradually to new location (only if input > 0)
                var shift = input.mul(shiftVec);
                var updated = decayed.add(shift);

                // Cap at 1.0
                var capped = updated.min(oneVec);

                capped.intoArray(attentionRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                // Very slow decay
                attentionRow[i] *= decayFactor;

                // Integrate new input with shift rate (only if input > 0)
                if (inputRow[i] > 0) {
                    attentionRow[i] += inputRow[i] * attentionShiftRate;
                }

                // Cap attention state
                attentionRow[i] = Math.min(attentionRow[i], 1.0);
            }
        }
    }

    /**
     * Apply memory trace integration with very slow dynamics (SIMD).
     *
     * <p>Memory traces persist for seconds (much slower than attention).
     *
     * <p>Update rule:
     * <pre>
     * memoryTrace *= (1 - traceDecayRate)             // Very slow decay
     * if (attentionState > threshold) {                // Memory formation threshold
     *     memoryTrace += attentionState * 0.1          // Slow accumulation
     * }
     * memoryTrace = min(memoryTrace, 0.8)              // Cap at 0.8
     * </pre>
     *
     * @param traceDecayRate very slow decay rate for memory (typically sustainedDecayRate * 0.05)
     * @param memoryThreshold threshold for memory formation (typically 0.3)
     */
    public void applyMemoryTraceIntegration(double traceDecayRate, double memoryThreshold) {
        int laneSize = SPECIES.length();
        var decayFactor = 1.0 - traceDecayRate;
        var decayVec = DoubleVector.broadcast(SPECIES, decayFactor);
        var thresholdVec = DoubleVector.broadcast(SPECIES, memoryThreshold);
        var accumRate = 0.1;  // Slow accumulation
        var accumVec = DoubleVector.broadcast(SPECIES, accumRate);
        var capVec = DoubleVector.broadcast(SPECIES, 0.8);

        for (int d = 0; d < dimension; d++) {
            var memoryRow = memoryMajor[d];
            var attentionRow = attentionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var memory = DoubleVector.fromArray(SPECIES, memoryRow, i);
                var attention = DoubleVector.fromArray(SPECIES, attentionRow, i);

                // Very slow decay
                var decayed = memory.mul(decayVec);

                // Build up memory trace from sustained attention (if above threshold)
                var mask = attention.compare(jdk.incubator.vector.VectorOperators.GT, thresholdVec);
                var accumulation = attention.mul(accumVec);
                var updated = decayed.add(accumulation, mask);

                // Cap at 0.8
                var capped = updated.min(capVec);

                capped.intoArray(memoryRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                // Very slow decay
                memoryRow[i] *= decayFactor;

                // Build up memory trace from sustained attention
                if (attentionRow[i] > memoryThreshold) {
                    memoryRow[i] += attentionRow[i] * accumRate;
                }

                // Cap memory trace
                memoryRow[i] = Math.min(memoryRow[i], 0.8);
            }
        }
    }

    /**
     * Calculate priming effect combining attention and memory (SIMD).
     *
     * <p>Priming is a combination of current attention and long-term memory.
     * CRITICAL: Priming is CAPPED at 0.5 to prevent driving responses directly.
     *
     * <p>Calculation:
     * <pre>
     * combinedAttention = attentionState + memoryTrace * 0.5
     * // For strong initial attention, allow higher values
     * if (input > 0.8) {
     *     combinedAttention = max(combinedAttention, input * 0.9)
     * }
     * primingEffect = combinedAttention * primingStrength
     * primingEffect = min(primingEffect, 0.5)  // CRITICAL: cap at 0.5
     * </pre>
     *
     * @param primingStrength strength of priming effect (0-1)
     */
    public void applyPrimingEffect(double primingStrength) {
        int laneSize = SPECIES.length();
        var strengthVec = DoubleVector.broadcast(SPECIES, primingStrength);
        var memoryWeightVec = DoubleVector.broadcast(SPECIES, 0.5);
        var strongThresholdVec = DoubleVector.broadcast(SPECIES, 0.8);
        var strongWeightVec = DoubleVector.broadcast(SPECIES, 0.9);
        var primingCapVec = DoubleVector.broadcast(SPECIES, 0.5);  // CRITICAL: priming cap

        for (int d = 0; d < dimension; d++) {
            var primingRow = primingMajor[d];
            var attentionRow = attentionMajor[d];
            var memoryRow = memoryMajor[d];
            var inputRow = dimensionMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var attention = DoubleVector.fromArray(SPECIES, attentionRow, i);
                var memory = DoubleVector.fromArray(SPECIES, memoryRow, i);
                var input = DoubleVector.fromArray(SPECIES, inputRow, i);

                // Combine attention state and memory trace
                var combined = attention.add(memory.mul(memoryWeightVec));

                // For initial strong attention, allow higher values
                var strongMask = input.compare(jdk.incubator.vector.VectorOperators.GT, strongThresholdVec);
                var strongValue = input.mul(strongWeightVec);
                // Use max to take the larger of combined or strongValue, then blend based on mask
                var maxValue = combined.max(strongValue);
                combined = combined.blend(maxValue, strongMask);

                // Apply priming strength
                var priming = combined.mul(strengthVec);

                // CRITICAL: Cap at 0.5 to prevent driving
                var capped = priming.min(primingCapVec);

                capped.intoArray(primingRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                // Combine attention state and memory trace
                var combined = attentionRow[i] + memoryRow[i] * 0.5;

                // For initial strong attention, allow higher values
                if (inputRow[i] > 0.8) {
                    combined = Math.max(combined, inputRow[i] * 0.9);
                }

                // Apply priming strength
                primingRow[i] = combined * primingStrength;

                // CRITICAL: Ensure priming doesn't exceed 0.5
                primingRow[i] = Math.min(primingRow[i], 0.5);
            }
        }
    }

    /**
     * Generate apical dendrite signal for Layer 2/3 integration (SIMD).
     *
     * <p>Apical dendrite signal combines attention and priming effects
     * for integration with Layer 2/3 apical dendrites.
     *
     * <p>Signal calculation:
     * <pre>
     * apicalSignal = (attentionState + primingEffect) * apicalIntegration
     * </pre>
     *
     * <p>Result is stored in dimensionMajor for output.
     *
     * @param apicalIntegration integration strength with Layer 2/3 (0-1)
     */
    public void applyApicalDendriteSignal(double apicalIntegration) {
        int laneSize = SPECIES.length();
        var integrationVec = DoubleVector.broadcast(SPECIES, apicalIntegration);

        for (int d = 0; d < dimension; d++) {
            var outputRow = dimensionMajor[d];
            var attentionRow = attentionMajor[d];
            var primingRow = primingMajor[d];
            int i = 0;

            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var attention = DoubleVector.fromArray(SPECIES, attentionRow, i);
                var priming = DoubleVector.fromArray(SPECIES, primingRow, i);

                // Modulated signal for apical dendrites
                var signal = attention.add(priming).mul(integrationVec);

                signal.intoArray(outputRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                // Modulated signal for apical dendrites
                outputRow[i] = (attentionRow[i] + primingRow[i]) * apicalIntegration;
            }
        }
    }

    /**
     * Apply exact shunting dynamics evolution with very slow time constants.
     *
     * <p>Uses {@link BatchShuntingDynamics} for bit-exact equivalence.
     * Layer 1 has very weak self-excitation (0.05) and lateral inhibition (0.05).
     *
     * @param timeStep integration time step (very slow: timeConstant / 20000.0)
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
     * Get attention state data (direct access for debugging).
     */
    public double[][] getAttentionMajor() {
        return attentionMajor;
    }

    /**
     * Get priming effect data (direct access for debugging).
     */
    public double[][] getPrimingMajor() {
        return primingMajor;
    }

    /**
     * Get memory trace data (direct access for debugging).
     */
    public double[][] getMemoryMajor() {
        return memoryMajor;
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
     * Process complete Layer 1 batch with SIMD optimization.
     *
     * <p>Main entry point for Layer 1 SIMD batch processing.
     *
     * <p>IMPORTANT: Layer 1 processes TOP-DOWN expectations, not bottom-up input!
     *
     * @param expectations top-down expectation patterns
     * @param params Layer 1 parameters
     * @param size layer size
     * @return processed patterns, or null if SIMD not beneficial
     */
    public static Pattern[] processTopDownBatchSIMD(Pattern[] expectations,
                                                     Layer1Parameters params,
                                                     int size) {
        // Check if transpose-and-vectorize is beneficial
        // Layer 1 has ~10 operations (fewer than Layer 5)
        var operations = 10;
        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                expectations.length, size, operations)) {
            return null;  // Fall back to sequential
        }

        // Create SIMD batch (transpose to dimension-major)
        // No previous states for this simple version
        var batch = createBatch(expectations, null, null, null, size);

        // 1. Apply attention state update (SIMD - very slow decay)
        batch.applyAttentionStateUpdate(params.getSustainedDecayRate(), params.getAttentionShiftRate());

        // 2. Apply memory trace integration (SIMD - long-term persistence)
        batch.applyMemoryTraceIntegration(params.getSustainedDecayRate() * 0.05, 0.3);

        // 3. Calculate priming effect (SIMD - attention + memory)
        batch.applyPrimingEffect(params.getPrimingStrength());

        // 4. Generate apical dendrite signals for Layer 2/3 (SIMD)
        batch.applyApicalDendriteSignal(params.getApicalIntegration());

        // 5. Apply exact shunting dynamics (very slow time constant)
        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(size)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        var timeStep = params.getTimeConstant() / 20000.0;  // Convert ms to seconds
        batch.applyDynamicsExact(timeStep, shuntingParams);

        // 6. Apply saturation (SIMD)
        batch.applySaturation(params.getCeiling(), params.getFloor());

        // Convert back to patterns (transpose to pattern-major)
        return batch.toPatterns();
    }
}
