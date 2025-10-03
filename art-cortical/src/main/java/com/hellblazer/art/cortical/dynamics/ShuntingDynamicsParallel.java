package com.hellblazer.art.cortical.dynamics;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

/**
 * Parallel implementation of shunting dynamics using ForkJoinPool.
 *
 * <h2>Parallelization Strategy</h2>
 *
 * <p>The shunting dynamics equation is computed independently for each neuron:
 * <pre>
 * dx_i/dt = -A_i * x_i + (B - x_i) * S_i^+ - (x_i - C) * S_i^-
 * </pre>
 *
 * <p>Since each neuron's update only depends on the <i>previous</i> state
 * of all neurons, updates can be parallelized without race conditions.
 *
 * <h2>Performance Strategy</h2>
 * <ul>
 *   <li><b>Work Stealing</b>: ForkJoinPool balances load across cores</li>
 *   <li><b>Cache Locality</b>: Each thread processes contiguous neuron chunks</li>
 *   <li><b>No Synchronization</b>: Read-only access to previous state, write to separate result array</li>
 *   <li><b>Adaptive Granularity</b>: Minimum chunk size prevents excessive task overhead</li>
 * </ul>
 *
 * <h2>Expected Speedup</h2>
 * <ul>
 *   <li>1 core: 1.0x (baseline)</li>
 *   <li>4 cores: 3.0-3.5x (75-87% efficiency)</li>
 *   <li>8 cores: 5.0-6.0x (62-75% efficiency)</li>
 * </ul>
 *
 * <p>Efficiency decreases with more cores due to memory bandwidth limits
 * and cache coherency overhead.
 *
 * <h2>When to Use</h2>
 * <ul>
 *   <li><b>Large networks</b>: dimension ≥ 256 (overhead amortized)</li>
 *   <li><b>Multiple cores available</b>: Runtime.availableProcessors() ≥ 4</li>
 *   <li><b>Convergence iterations</b>: Many updates justify thread pool overhead</li>
 * </ul>
 *
 * <p>For small networks (dimension < 256) or single-threaded environments,
 * use {@link ShuntingDynamics} instead.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Create with default ForkJoinPool
 * var dynamics = new ShuntingDynamicsParallel(parameters);
 *
 * // Or with custom pool
 * var pool = new ForkJoinPool(4);
 * var dynamics = new ShuntingDynamicsParallel(parameters, pool);
 *
 * // Use exactly like ShuntingDynamics
 * dynamics.setExcitatoryInput(input);
 * var output = dynamics.update(timeStep);
 * }</pre>
 *
 * @author Phase 4C: Shunting Dynamics Parallelization
 * @see ShuntingDynamics
 */
public class ShuntingDynamicsParallel implements NeuralDynamics {

    private static final int MIN_CHUNK_SIZE = 64;  // Minimum neurons per thread

    private final ShuntingParameters parameters;
    private final int dimension;
    private final double[] activations;
    private final double[] excitatoryInput;
    private final double[] inhibitoryInput;
    private final ForkJoinPool pool;
    private final boolean ownsPool;

    /**
     * Create parallel shunting dynamics with default ForkJoinPool.
     *
     * @param parameters shunting dynamics parameters
     */
    public ShuntingDynamicsParallel(ShuntingParameters parameters) {
        this(parameters, ForkJoinPool.commonPool());
        // Note: We don't own the common pool, so don't shut it down
    }

    /**
     * Create parallel shunting dynamics with custom ForkJoinPool.
     *
     * @param parameters shunting dynamics parameters
     * @param pool custom ForkJoinPool for parallel execution
     */
    public ShuntingDynamicsParallel(ShuntingParameters parameters, ForkJoinPool pool) {
        this(parameters, pool, false);
    }

    /**
     * Full constructor for testing (allows pool ownership specification).
     */
    ShuntingDynamicsParallel(ShuntingParameters parameters, ForkJoinPool pool, boolean ownsPool) {
        this.parameters = parameters;
        this.dimension = parameters.getDimension();
        this.activations = new double[dimension];
        this.excitatoryInput = new double[dimension];
        this.inhibitoryInput = new double[dimension];
        this.pool = pool;
        this.ownsPool = ownsPool;

        // Initialize activations
        for (var i = 0; i < dimension; i++) {
            activations[i] = parameters.initialActivation();
        }
    }

    @Override
    public double[] update(double timeStep) {
        if (timeStep <= 0) {
            throw new IllegalArgumentException("Time step must be positive: " + timeStep);
        }

        var result = new double[dimension];

        // Determine if parallel execution is beneficial
        var numCores = pool.getParallelism();
        var shouldParallelize = dimension >= MIN_CHUNK_SIZE && numCores > 1;

        if (shouldParallelize) {
            // Parallel execution using ForkJoinPool
            var task = new UpdateTask(0, dimension, result, timeStep);
            pool.invoke(task);
        } else {
            // Sequential execution for small networks
            updateRange(0, dimension, result, timeStep);
        }

        // Update internal state
        System.arraycopy(result, 0, activations, 0, dimension);

        return result.clone();
    }

    /**
     * Update a range of neurons sequentially.
     * This is the core computation that can be parallelized.
     */
    private void updateRange(int start, int end, double[] result, double timeStep) {
        for (var i = start; i < end; i++) {
            var decay = parameters.getDecayRate(i);
            var ceiling = parameters.ceiling();
            var floor = parameters.floor();

            var excitation = computeExcitation(i);
            var inhibition = computeInhibition(i);

            // Shunting equation (Grossberg 1973)
            var derivative = -decay * activations[i] +
                            (ceiling - activations[i]) * excitation -
                            (activations[i] - floor) * inhibition;

            // Euler integration
            result[i] = activations[i] + timeStep * derivative;

            // Enforce bounds
            result[i] = Math.max(floor, Math.min(ceiling, result[i]));
        }
    }

    @Override
    public void reset() {
        for (var i = 0; i < dimension; i++) {
            activations[i] = parameters.initialActivation();
            excitatoryInput[i] = 0.0;
            inhibitoryInput[i] = 0.0;
        }
    }

    @Override
    public double[] getActivation() {
        return activations.clone();
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public boolean hasConverged() {
        // Check if activation changes are below tolerance
        var tolerance = 1e-6;
        var tempActivations = activations.clone();

        update(parameters.timeStep());

        var maxChange = 0.0;
        for (var i = 0; i < dimension; i++) {
            var change = Math.abs(activations[i] - tempActivations[i]);
            maxChange = Math.max(maxChange, change);
        }

        // Restore previous state
        System.arraycopy(tempActivations, 0, activations, 0, dimension);

        return maxChange < tolerance;
    }

    /**
     * Set external excitatory input for all units.
     */
    public void setExcitatoryInput(double[] input) {
        System.arraycopy(input, 0, excitatoryInput, 0, Math.min(input.length, dimension));
    }

    /**
     * Set external inhibitory input for all units.
     */
    public void setInhibitoryInput(double[] input) {
        System.arraycopy(input, 0, inhibitoryInput, 0, Math.min(input.length, dimension));
    }

    /**
     * Clear all external inputs.
     */
    public void clearInputs() {
        for (var i = 0; i < dimension; i++) {
            excitatoryInput[i] = 0.0;
            inhibitoryInput[i] = 0.0;
        }
    }

    /**
     * Get the ForkJoinPool used for parallel execution.
     */
    public ForkJoinPool getPool() {
        return pool;
    }

    /**
     * Shutdown the pool if we own it.
     * Only call this if you created the dynamics with a custom pool.
     */
    public void shutdown() {
        if (ownsPool && pool != null && !pool.isShutdown()) {
            pool.shutdown();
        }
    }

    /**
     * Compute total excitatory input for unit i.
     */
    private double computeExcitation(int i) {
        var total = 0.0;

        // Self-excitation
        total += parameters.selfExcitation() * activations[i];

        // Lateral excitation (on-center)
        for (var j = 0; j < dimension; j++) {
            if (i != j) {
                var weight = computeExcitatoryWeight(i, j);
                total += weight * activations[j];
            }
        }

        // External input
        total += excitatoryInput[i];

        return Math.max(0, total);
    }

    /**
     * Compute total inhibitory input for unit i.
     */
    private double computeInhibition(int i) {
        var total = 0.0;

        // Lateral inhibition (off-surround)
        for (var j = 0; j < dimension; j++) {
            if (i != j) {
                var weight = computeInhibitoryWeight(i, j);
                total += weight * activations[j];
            }
        }

        // External input
        total += inhibitoryInput[i];

        return Math.max(0, total);
    }

    /**
     * Compute excitatory weight from unit j to unit i (on-center Gaussian).
     */
    private double computeExcitatoryWeight(int i, int j) {
        if (parameters.excitatoryStrength() == 0.0) {
            return 0.0;
        }

        var distance = Math.abs(i - j);
        var sigma = parameters.excitatoryRange();

        // Narrow Gaussian kernel
        return parameters.excitatoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Compute inhibitory weight from unit j to unit i (off-surround Gaussian).
     */
    private double computeInhibitoryWeight(int i, int j) {
        if (parameters.inhibitoryStrength() == 0.0) {
            return 0.0;
        }

        var distance = Math.abs(i - j);
        var sigma = parameters.inhibitoryRange();

        // Broad Gaussian kernel
        return parameters.inhibitoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Recursive task for parallel neuron updates using Fork/Join framework.
     */
    private class UpdateTask extends RecursiveAction {
        private final int start;
        private final int end;
        private final double[] result;
        private final double timeStep;

        UpdateTask(int start, int end, double[] result, double timeStep) {
            this.start = start;
            this.end = end;
            this.result = result;
            this.timeStep = timeStep;
        }

        @Override
        protected void compute() {
            var length = end - start;

            // If chunk is small enough, compute directly
            if (length <= MIN_CHUNK_SIZE) {
                updateRange(start, end, result, timeStep);
            } else {
                // Split task in half and process in parallel
                var mid = start + length / 2;
                var left = new UpdateTask(start, mid, result, timeStep);
                var right = new UpdateTask(mid, end, result, timeStep);

                // Fork both subtasks
                invokeAll(left, right);
            }
        }
    }
}
