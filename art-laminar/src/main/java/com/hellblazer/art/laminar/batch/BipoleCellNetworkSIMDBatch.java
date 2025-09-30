package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.BipoleCellParameters;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-optimized batch processing for BipoleCellNetwork.
 *
 * The bipole cell network implements horizontal connections for boundary completion
 * and contour integration. Each cell receives:
 * 1. Direct bottom-up input
 * 2. Horizontal input from neighboring cells with similar orientations
 *
 * Cells fire under three conditions:
 * 1. Strong direct input alone (threshold1)
 * 2. Bilateral horizontal support (both sides above threshold2)
 * 3. Weak direct input + any horizontal support
 *
 * Key SIMD insights:
 * - Horizontal connections are SPATIAL (within pattern), not TEMPORAL
 * - Each pattern can be processed independently through the network
 * - Dimension-major layout enables SIMD across batch
 * - All patterns converge in parallel (10-15 iterations)
 *
 * Cost-benefit analysis:
 * - I (iterations) = 10-15
 * - R (reuse per dimension) = batchSize
 * - L (layout conversion) = 8 operations
 * - I × R > L always true for batch size > 1
 *
 * Memory layout:
 * - Input: Pattern-major [batchSize][dimension]
 * - Internal: Dimension-major [dimension][batchSize]
 * - Output: Pattern-major [batchSize][dimension]
 *
 * @author Hal Hildebrand
 */
public class BipoleCellNetworkSIMDBatch {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final int dimension;
    private final int batchSize;
    private final double[][] directInput;           // Immutable direct input [dimension][batchSize]
    private final double[][] activation;            // Current activations [dimension][batchSize]
    private final double[][] horizontalInput;       // Computed horizontal input [dimension][batchSize]
    private final double[][] connectionWeights;     // Precomputed weights [dimension][dimension]
    private final BipoleCellParameters params;

    /**
     * Create batch processor with precomputed connection weights.
     *
     * @param patterns Input patterns (pattern-major layout)
     * @param connectionWeights Precomputed connection weight matrix [dimension][dimension]
     * @param params Bipole cell parameters
     */
    private BipoleCellNetworkSIMDBatch(Pattern[] patterns, double[][] connectionWeights,
                                       BipoleCellParameters params) {
        this.batchSize = patterns.length;
        this.dimension = patterns[0].dimension();
        this.connectionWeights = connectionWeights;
        this.params = params;

        // Allocate dimension-major storage
        this.directInput = new double[dimension][batchSize];
        this.activation = new double[dimension][batchSize];
        this.horizontalInput = new double[dimension][batchSize];

        // Convert to dimension-major layout and initialize both direct input and activation
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dimension; d++) {
                directInput[d][b] = patterns[b].get(d);
                activation[d][b] = patterns[b].get(d);  // Initialize activation to input
            }
        }
    }

    /**
     * Process batch of patterns through bipole cell network.
     *
     * @param patterns Input patterns
     * @param connectionWeights Precomputed connection weights [dimension][dimension]
     * @param params Bipole cell parameters
     * @return Processed patterns (null if sequential is more efficient)
     */
    public static Pattern[] processBatch(Pattern[] patterns, double[][] connectionWeights,
                                         BipoleCellParameters params) {
        if (patterns == null || patterns.length == 0) {
            throw new IllegalArgumentException("patterns cannot be null or empty");
        }
        if (connectionWeights == null) {
            throw new NullPointerException("connectionWeights cannot be null");
        }
        if (params == null) {
            throw new NullPointerException("params cannot be null");
        }

        // Cost-benefit analysis
        // Operations per pattern: 10-15 iterations × dimension × connections
        // Layout conversion: 8 operations per element
        var avgIterations = 12;  // Typical convergence
        var operations = avgIterations * patterns[0].dimension();

        if (!BatchDataLayout.isTransposeAndVectorizeBeneficial(
                patterns.length, patterns[0].dimension(), operations)) {
            return null;  // Fall back to sequential
        }

        var batch = new BipoleCellNetworkSIMDBatch(patterns, connectionWeights, params);
        batch.iterateToConvergence();
        return batch.toPatterns();
    }

    /**
     * Iterate bipole network until convergence.
     * Uses fixed iteration count to match BipoleCellNetwork behavior.
     */
    private void iterateToConvergence() {
        // Match BipoleCellNetwork: 10-15 iterations for convergence
        var iterations = 12;  // Middle ground for typical convergence

        for (int iter = 0; iter < iterations; iter++) {
            // Compute horizontal input from current activations
            computeHorizontalInputSIMD();

            // Apply three-way firing logic and temporal dynamics
            applyFiringLogicAndDynamicsSIMD();
        }
    }

    /**
     * Compute horizontal input for all dimensions and patterns (SIMD).
     *
     * For each dimension d and pattern b:
     *   horizontalInput[d][b] = Σ(connectionWeights[d][j] × activation[j][b])
     *
     * This is a matrix-vector product for each pattern in the batch.
     * SIMD across batch: Process all patterns together for each dimension.
     */
    private void computeHorizontalInputSIMD() {
        int laneSize = SPECIES.length();

        // For each target dimension
        for (int d = 0; d < dimension; d++) {
            var targetRow = horizontalInput[d];
            var weights = connectionWeights[d];

            // Zero out target row
            for (int i = 0; i < batchSize; i++) {
                targetRow[i] = 0.0;
            }

            // Accumulate weighted contributions from all source dimensions
            for (int j = 0; j < dimension; j++) {
                var weight = weights[j];
                if (weight == 0.0) continue;  // Skip zero weights

                var sourceRow = activation[j];  // Use current activation
                var weightVec = DoubleVector.broadcast(SPECIES, weight);

                int i = 0;
                // Vectorized loop
                for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                    var sourceVec = DoubleVector.fromArray(SPECIES, sourceRow, i);
                    var targetVec = DoubleVector.fromArray(SPECIES, targetRow, i);
                    var contribution = sourceVec.mul(weightVec);
                    var newTarget = targetVec.add(contribution);
                    newTarget.intoArray(targetRow, i);
                }

                // Scalar tail
                for (; i < batchSize; i++) {
                    targetRow[i] += sourceRow[i] * weight;
                }
            }
        }
    }

    /**
     * Apply three-way firing logic and temporal dynamics (SIMD).
     *
     * Three conditions for firing (matching BipoleCell.computeActivation):
     * 1. Strong direct input: directInput > strongDirectThreshold
     * 2. Bilateral horizontal: leftInput > 0.1 && rightInput > 0.1
     * 3. Weak direct + horizontal: directInput > weakDirectThreshold && (leftInput > horizontalThreshold || rightInput > horizontalThreshold)
     *
     * Temporal dynamics: exponential approach
     *   activation = activation + alpha × (targetActivation - activation)
     *   where alpha = timeStep / timeConstant
     *
     * Note: This is a simplified version that combines left/right horizontal into total horizontal.
     * For exact equivalence, BipoleCellNetwork would need to track left/right separately.
     */
    private void applyFiringLogicAndDynamicsSIMD() {
        int laneSize = SPECIES.length();
        var strongThreshold = params.strongDirectThreshold();
        var weakThreshold = params.weakDirectThreshold();
        var horizontalThreshold = params.horizontalThreshold();
        var timeConstant = params.timeConstant();
        var timeStep = 0.01;  // Match BipoleCellNetwork TIME_STEP
        var alpha = timeStep / timeConstant;

        var strongThresholdVec = DoubleVector.broadcast(SPECIES, strongThreshold);
        var weakThresholdVec = DoubleVector.broadcast(SPECIES, weakThreshold);
        var horizontalThresholdVec = DoubleVector.broadcast(SPECIES, horizontalThreshold);
        var bilateralThresholdVec = DoubleVector.broadcast(SPECIES, 0.1);
        var alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        var decayAlphaVec = DoubleVector.broadcast(SPECIES, alpha / 2.0);  // Slower decay
        var zeroVec = DoubleVector.broadcast(SPECIES, 0.0);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);

        for (int d = 0; d < dimension; d++) {
            var directRow = directInput[d];       // Immutable direct input
            var activationRow = activation[d];    // Current activation
            var horizontalRow = horizontalInput[d];

            int i = 0;
            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var directVec = DoubleVector.fromArray(SPECIES, directRow, i);
                var horizontalVec = DoubleVector.fromArray(SPECIES, horizontalRow, i);

                // Condition 1: Strong direct input
                var condition1 = directVec.compare(VectorOperators.GT, strongThresholdVec);
                var target1 = directVec;  // Use direct input as target

                // Condition 2: Bilateral horizontal (simplified: total horizontal > bilateral threshold)
                var condition2 = horizontalVec.compare(VectorOperators.GT, bilateralThresholdVec);
                var target2 = horizontalVec.mul(0.8);  // Match BipoleCell bilateral activation

                // Condition 3: Weak direct + horizontal
                var weakDirect = directVec.compare(VectorOperators.GT, weakThresholdVec);
                var anyHorizontal = horizontalVec.compare(VectorOperators.GT, horizontalThresholdVec);
                var condition3 = weakDirect.and(anyHorizontal);
                var target3 = directVec.add(horizontalVec).mul(0.5);  // Combined activation

                // Combined firing condition
                var shouldFire = condition1.or(condition2).or(condition3);

                // Select appropriate target activation
                var targetVec = target1.blend(zeroVec, condition1);
                targetVec = target2.blend(targetVec, condition2);
                targetVec = target3.blend(targetVec, condition3);

                // Clamp target to [0, 1]
                targetVec = targetVec.max(zeroVec).min(oneVec);

                // Get current activation
                var activationVec = DoubleVector.fromArray(SPECIES, activationRow, i);

                // Temporal dynamics: exponential approach if firing, decay if not
                var delta = targetVec.sub(activationVec);
                var effectiveAlpha = alphaVec.blend(decayAlphaVec, shouldFire);
                var change = delta.mul(effectiveAlpha);
                var newActivation = activationVec.add(change);

                // Clamp to [0, 1]
                newActivation = newActivation.max(zeroVec).min(oneVec);

                // Store new activation
                newActivation.intoArray(activationRow, i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                var direct = directRow[i];
                var horizontal = horizontalRow[i];
                var currentActivation = activationRow[i];

                // Three-way firing logic
                var shouldFire = false;
                var target = 0.0;

                // Condition 1: Strong direct
                if (direct > strongThreshold) {
                    shouldFire = true;
                    target = Math.max(target, direct);
                }

                // Condition 2: Bilateral horizontal
                if (horizontal > 0.1) {
                    shouldFire = true;
                    target = Math.max(target, horizontal * 0.8);
                }

                // Condition 3: Weak direct + horizontal
                if (direct > weakThreshold && horizontal > horizontalThreshold) {
                    shouldFire = true;
                    target = Math.max(target, (direct + horizontal) / 2.0);
                }

                // Clamp target
                target = Math.max(0.0, Math.min(1.0, target));

                // Temporal dynamics
                var effectiveAlpha = shouldFire ? alpha : (alpha / 2.0);
                var newActivation = currentActivation + effectiveAlpha * (target - currentActivation);

                // Clamp and store activation
                activationRow[i] = Math.max(0.0, Math.min(1.0, newActivation));
            }
        }
    }

    /**
     * Convert dimension-major back to pattern-major for output.
     */
    private Pattern[] toPatterns() {
        return BatchDataLayout.dimensionMajorToPatterns(activation, batchSize, dimension);
    }

    /**
     * Get current activations in dimension-major layout (for testing).
     */
    double[][] getActivation() {
        return activation;
    }

    /**
     * Get direct input in dimension-major layout (for testing).
     */
    double[][] getDirectInput() {
        return directInput;
    }

    /**
     * Get horizontal input in dimension-major layout (for testing).
     */
    double[][] getHorizontalInput() {
        return horizontalInput;
    }
}
