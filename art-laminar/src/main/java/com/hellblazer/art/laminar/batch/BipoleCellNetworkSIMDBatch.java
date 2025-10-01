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
 * 2. Left horizontal input from neighboring cells
 * 3. Right horizontal input from neighboring cells
 *
 * Cells fire under three conditions:
 * 1. Strong direct input alone (threshold1)
 * 2. Bilateral horizontal support (BOTH left AND right above threshold)
 * 3. Weak direct input + any horizontal support
 *
 * Key SIMD insights:
 * - Horizontal connections are SPATIAL (within pattern), not TEMPORAL
 * - Each pattern can be processed independently through the network
 * - Dimension-major layout enables SIMD across batch
 * - All patterns converge in parallel (10-15 iterations)
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
    private double[][] activation;                  // Current activations [dimension][batchSize]
    private double[][] nextActivation;              // Next iteration activations [dimension][batchSize]
    private final double[][] leftHorizontalInput;   // Left horizontal input [dimension][batchSize]
    private final double[][] rightHorizontalInput;  // Right horizontal input [dimension][batchSize]
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
        this.nextActivation = new double[dimension][batchSize];
        this.leftHorizontalInput = new double[dimension][batchSize];
        this.rightHorizontalInput = new double[dimension][batchSize];

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
     * @return Processed patterns (always processes, no fallback to null)
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

        // Always process with SIMD for BipoleCellNetwork
        // The cost-benefit analysis was too restrictive for this use case
        var batch = new BipoleCellNetworkSIMDBatch(patterns, connectionWeights, params);
        batch.iterateToConvergence();
        return batch.toPatterns();
    }

    /**
     * Iterate bipole network until convergence.
     * Uses fixed iteration count to match BipoleCellNetwork behavior.
     */
    private void iterateToConvergence() {
        // Match BipoleCellNetwork: 10 iterations (default, propagationEnabled=false)
        var iterations = 10;  // Match BipoleCellNetwork exactly

        for (int iter = 0; iter < iterations; iter++) {
            // Compute left and right horizontal inputs from current activation
            computeHorizontalInputSIMD();

            // Apply three-way firing logic and temporal dynamics
            // Writes to nextActivation for synchronous update
            applyFiringLogicAndDynamicsSIMD();

            // Swap buffers for next iteration (synchronous update)
            var temp = activation;
            activation = nextActivation;
            nextActivation = temp;
        }
    }


    /**
     * Compute left and right horizontal inputs for all dimensions and patterns (SIMD).
     *
     * For each dimension d and pattern b:
     *   leftHorizontalInput[d][b] = Σ(connectionWeights[d][j] × activation[j][b]) for j < d
     *   rightHorizontalInput[d][b] = Σ(connectionWeights[d][j] × activation[j][b]) for j > d
     *
     * This matches BipoleCellNetwork's separate left/right computation.
     */
    private void computeHorizontalInputSIMD() {
        int laneSize = SPECIES.length();
        var maxRange = params.maxHorizontalRange();


        // For each target dimension
        for (int d = 0; d < dimension; d++) {
            var leftTargetRow = leftHorizontalInput[d];
            var rightTargetRow = rightHorizontalInput[d];

            // Zero out target rows
            for (int i = 0; i < batchSize; i++) {
                leftTargetRow[i] = 0.0;
                rightTargetRow[i] = 0.0;
            }


            // Accumulate LEFT horizontal input (from cells to the left: j < d)
            for (int j = Math.max(0, d - maxRange); j < d; j++) {
                // Match BipoleCellNetwork: use weight FROM d TO j
                var weight = connectionWeights[d][j];
                if (weight == 0.0) continue;  // Skip zero weights


                var sourceRow = activation[j];
                var weightVec = DoubleVector.broadcast(SPECIES, weight);

                int i = 0;
                // Vectorized loop
                for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                    var sourceVec = DoubleVector.fromArray(SPECIES, sourceRow, i);
                    var targetVec = DoubleVector.fromArray(SPECIES, leftTargetRow, i);
                    var contribution = sourceVec.mul(weightVec);
                    var newTarget = targetVec.add(contribution);
                    newTarget.intoArray(leftTargetRow, i);
                }

                // Scalar tail
                for (; i < batchSize; i++) {
                    leftTargetRow[i] += sourceRow[i] * weight;
                }
            }

            // Accumulate RIGHT horizontal input (from cells to the right: j > d)
            for (int j = d + 1; j < Math.min(dimension, d + maxRange + 1); j++) {
                // Match BipoleCellNetwork: use weight FROM d TO j
                var weight = connectionWeights[d][j];
                if (weight == 0.0) continue;  // Skip zero weights


                var sourceRow = activation[j];
                var weightVec = DoubleVector.broadcast(SPECIES, weight);

                int i = 0;
                // Vectorized loop
                for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                    var sourceVec = DoubleVector.fromArray(SPECIES, sourceRow, i);
                    var targetVec = DoubleVector.fromArray(SPECIES, rightTargetRow, i);
                    var contribution = sourceVec.mul(weightVec);
                    var newTarget = targetVec.add(contribution);
                    newTarget.intoArray(rightTargetRow, i);
                }

                // Scalar tail
                for (; i < batchSize; i++) {
                    rightTargetRow[i] += sourceRow[i] * weight;
                }
            }
        }
    }

    /**
     * Apply three-way firing logic and temporal dynamics (SIMD).
     *
     * Three conditions for firing (matching BipoleCell.computeActivation):
     * 1. Strong direct input: directInput > strongDirectThreshold
     * 2. Bilateral horizontal: leftInput > 0.1 AND rightInput > 0.1
     * 3. Weak direct + horizontal: directInput > weakDirectThreshold && (leftInput > horizontalThreshold || rightInput > horizontalThreshold)
     *
     * Temporal dynamics: exponential approach
     *   activation = activation + alpha × (targetActivation - activation)
     *   where alpha = timeStep / timeConstant
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
            var leftRow = leftHorizontalInput[d];
            var rightRow = rightHorizontalInput[d];

            int i = 0;
            // Vectorized loop
            for (; i < SPECIES.loopBound(batchSize); i += laneSize) {
                var directVec = DoubleVector.fromArray(SPECIES, directRow, i);
                var leftVec = DoubleVector.fromArray(SPECIES, leftRow, i);
                var rightVec = DoubleVector.fromArray(SPECIES, rightRow, i);


                // Condition 1: Strong direct input
                var condition1 = directVec.compare(VectorOperators.GT, strongThresholdVec);
                var target1 = directVec;  // Use direct input as target

                // Condition 2: Bilateral horizontal (BOTH left AND right above threshold)
                var leftAboveThreshold = leftVec.compare(VectorOperators.GT, bilateralThresholdVec);
                var rightAboveThreshold = rightVec.compare(VectorOperators.GT, bilateralThresholdVec);
                var condition2 = leftAboveThreshold.and(rightAboveThreshold);
                var target2 = leftVec.add(rightVec).mul(0.8).min(oneVec);  // Clamp bilateral to [0,1]

                // Condition 3: Weak direct + any horizontal
                var weakDirect = directVec.compare(VectorOperators.GT, weakThresholdVec);
                var leftHorizontal = leftVec.compare(VectorOperators.GT, horizontalThresholdVec);
                var rightHorizontal = rightVec.compare(VectorOperators.GT, horizontalThresholdVec);
                var anyHorizontal = leftHorizontal.or(rightHorizontal);
                var condition3 = weakDirect.and(anyHorizontal);
                var maxHorizontal = leftVec.max(rightVec);  // Use max, not sum
                var target3 = directVec.add(maxHorizontal).mul(0.5);  // Match BipoleCell: (direct + max(left,right)) / 2

                // Combined firing condition (only 3 conditions match BipoleCell)
                var shouldFire = condition1.or(condition2).or(condition3);

                // Select target using Math.max logic like BipoleCell
                // Start with zero, then take max of each condition's target if that condition is true
                var targetVec = zeroVec;

                // If condition1: target = max(target, directInput)
                // blend(a, b, mask) returns a where mask is false, b where mask is true
                // We want target1 where condition1 is true, zero where false
                targetVec = targetVec.max(zeroVec.blend(target1, condition1));

                // If condition2: target = max(target, bilateralActivation)
                targetVec = targetVec.max(zeroVec.blend(target2, condition2));

                // If condition3: target = max(target, combinedActivation)
                targetVec = targetVec.max(zeroVec.blend(target3, condition3));

                // DO NOT clamp target here - BipoleCell doesn't clamp until after dynamics
                // targetVec = targetVec.max(zeroVec).min(oneVec);

                // Get current activation
                var activationVec = DoubleVector.fromArray(SPECIES, activationRow, i);

                // Temporal dynamics: match BipoleCell exactly
                // If firing: activation = activation + alpha * (target - activation)
                // If not firing: activation = activation * (1 - alpha/2)
                var delta = targetVec.sub(activationVec);
                var approachChange = delta.mul(alphaVec);
                var decayFactor = oneVec.sub(decayAlphaVec);  // (1 - alpha/2)

                // Select between approach and decay based on shouldFire
                // blend(a, b, mask) returns a where mask is false, b where mask is true
                // We want approach where shouldFire is true, decay where false
                var approach = activationVec.add(approachChange);
                var decay = activationVec.mul(decayFactor);
                var newActivation = decay.blend(approach, shouldFire);

                // Clamp to [0, 1]
                newActivation = newActivation.max(zeroVec).min(oneVec);

                // Store new activation in next buffer (synchronous update)
                newActivation.intoArray(nextActivation[d], i);
            }

            // Scalar tail
            for (; i < batchSize; i++) {
                var direct = directRow[i];
                var left = leftRow[i];
                var right = rightRow[i];
                var currentActivation = activationRow[i];


                // Three-way firing logic with proper left/right separation
                var shouldFire = false;
                var target = 0.0;

                // Condition 1: Strong direct
                if (direct > strongThreshold) {
                    shouldFire = true;
                    target = direct;
                }

                // Condition 2: Bilateral horizontal (BOTH left AND right)
                if (left > 0.1 && right > 0.1) {
                    shouldFire = true;
                    // Match BipoleCell: (left + right) * 0.8, clamped to [0,1]
                    target = Math.max(target, Math.min(1.0, (left + right) * 0.8));
                }

                // Condition 3: Weak direct + any horizontal
                if (direct > weakThreshold && (left > horizontalThreshold || right > horizontalThreshold)) {
                    shouldFire = true;
                    // Match BipoleCell: (direct + max(left, right)) / 2.0
                    double horizontalSupport = Math.max(left, right);
                    target = Math.max(target, (direct + horizontalSupport) / 2.0);
                }

                // DO NOT clamp target here - BipoleCell doesn't clamp until after dynamics
                // target = Math.max(0.0, Math.min(1.0, target));

                // Temporal dynamics: match BipoleCell exactly
                var newActivation = 0.0;
                if (shouldFire) {
                    // Exponential approach to target
                    newActivation = currentActivation + alpha * (target - currentActivation);
                } else {
                    // Multiplicative decay
                    var decayAlpha = alpha / 2.0;  // Slower decay
                    newActivation = currentActivation * (1.0 - decayAlpha);
                }

                // Clamp and store activation in next buffer (synchronous update)
                nextActivation[d][i] = Math.max(0.0, Math.min(1.0, newActivation));
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
     * Get left horizontal input in dimension-major layout (for testing).
     */
    double[][] getLeftHorizontalInput() {
        return leftHorizontalInput;
    }

    /**
     * Get right horizontal input in dimension-major layout (for testing).
     */
    double[][] getRightHorizontalInput() {
        return rightHorizontalInput;
    }
}