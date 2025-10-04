package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.temporal.TemporalProcessor;

import java.util.concurrent.*;

/**
 * Performance-optimized cortical circuit with parallel layer processing.
 *
 * <h2>Optimization Strategy</h2>
 *
 * <p>Standard circuit processes layers sequentially:
 * <pre>
 * Temporal → L4 → L2/3 → L1 → L6 → L2/3(top-down) → L4(top-down) → L5
 * Total time: sum of all layer latencies
 * </pre>
 *
 * <p>Optimized circuit exploits independence:
 * <pre>
 * Temporal → L4 → L2/3 → ┌─ L1 (parallel)
 *                         └─ L6 (parallel)
 *                         Wait for both
 *                         → L2/3(top-down) → L4(top-down) → L5
 *
 * Time saved: max(L1, L6) instead of L1 + L6
 * </pre>
 *
 * <h2>Expected Performance</h2>
 *
 * <table border="1">
 *   <tr>
 *     <th>Metric</th>
 *     <th>Sequential</th>
 *     <th>Parallel</th>
 *     <th>Speedup</th>
 *   </tr>
 *   <tr>
 *     <td>Processing time</td>
 *     <td>100%</td>
 *     <td>70-85%</td>
 *     <td>1.2-1.4x</td>
 *   </tr>
 *   <tr>
 *     <td>Throughput (patterns/sec)</td>
 *     <td>Baseline</td>
 *     <td>1.2-1.4x</td>
 *     <td>20-40% more</td>
 *   </tr>
 * </table>
 *
 * <h2>Biological Validity</h2>
 *
 * <p>Parallel processing is biologically realistic:
 * <ul>
 *   <li>Layer 1 and Layer 6 operate independently in biological cortex</li>
 *   <li>Different cell types process in parallel (pyramidal, stellate, basket)</li>
 *   <li>Neural processing is massively parallel, not sequential</li>
 *   <li>Oscillatory synchronization coordinates parallel streams</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // Drop-in replacement for CorticalCircuit
 * try (var circuit = new CorticalCircuitOptimized(
 *     256,
 *     layer1Params,
 *     layer23Params,
 *     layer4Params,
 *     layer5Params,
 *     layer6Params,
 *     temporalProcessor,
 *     4  // thread pool size
 * )) {
 *     var output = circuit.process(input);
 * }  // Automatic thread pool shutdown
 * }</pre>
 *
 * <h2>Thread Pool Management</h2>
 *
 * <p>Uses fixed thread pool for predictable performance:
 * <ul>
 *   <li>Pool size = 2-4 threads (for L1/L6 parallelism)</li>
 *   <li>Reuses threads across pattern processing</li>
 *   <li>AutoCloseable for proper cleanup</li>
 * </ul>
 *
 * @author Phase 4F: Circuit-Level Optimization
 * @see CorticalCircuit
 */
public class CorticalCircuitOptimized implements AutoCloseable {

    private final CorticalCircuit delegate;
    private final ExecutorService executor;
    private final boolean ownsExecutor;

    // Layers for direct access (avoid delegation overhead)
    private final Layer1 layer1;
    private final Layer23 layer23;
    private final Layer4 layer4;
    private final Layer5 layer5;
    private final Layer6 layer6;
    private final TemporalProcessor temporalProcessor;

    // Parameters
    private final Layer1Parameters layer1Params;
    private final Layer23Parameters layer23Params;
    private final Layer4Parameters layer4Params;
    private final Layer5Parameters layer5Params;
    private final Layer6Parameters layer6Params;

    /**
     * Create optimized circuit with default thread pool (2 threads).
     *
     * @param size number of units per layer
     * @param layer1Params Layer 1 parameters
     * @param layer23Params Layer 2/3 parameters
     * @param layer4Params Layer 4 parameters
     * @param layer5Params Layer 5 parameters
     * @param layer6Params Layer 6 parameters
     * @param temporalProcessor temporal processing pipeline
     */
    public CorticalCircuitOptimized(
            int size,
            Layer1Parameters layer1Params,
            Layer23Parameters layer23Params,
            Layer4Parameters layer4Params,
            Layer5Parameters layer5Params,
            Layer6Parameters layer6Params,
            TemporalProcessor temporalProcessor) {
        this(size, layer1Params, layer23Params, layer4Params, layer5Params, layer6Params,
             temporalProcessor, 2);
    }

    /**
     * Create optimized circuit with custom thread pool size.
     *
     * @param size number of units per layer
     * @param layer1Params Layer 1 parameters
     * @param layer23Params Layer 2/3 parameters
     * @param layer4Params Layer 4 parameters
     * @param layer5Params Layer 5 parameters
     * @param layer6Params Layer 6 parameters
     * @param temporalProcessor temporal processing pipeline
     * @param poolSize thread pool size (typically 2-4)
     */
    public CorticalCircuitOptimized(
            int size,
            Layer1Parameters layer1Params,
            Layer23Parameters layer23Params,
            Layer4Parameters layer4Params,
            Layer5Parameters layer5Params,
            Layer6Parameters layer6Params,
            TemporalProcessor temporalProcessor,
            int poolSize) {
        this(size, layer1Params, layer23Params, layer4Params, layer5Params, layer6Params,
             temporalProcessor,
             Executors.newFixedThreadPool(poolSize,
                 new ThreadFactory() {
                     private int count = 0;
                     public Thread newThread(Runnable r) {
                         var t = new Thread(r, "cortical-circuit-" + count++);
                         t.setDaemon(true);
                         return t;
                     }
                 }),
             true);
    }

    /**
     * Create optimized circuit with custom executor.
     *
     * @param size number of units per layer
     * @param layer1Params Layer 1 parameters
     * @param layer23Params Layer 2/3 parameters
     * @param layer4Params Layer 4 parameters
     * @param layer5Params Layer 5 parameters
     * @param layer6Params Layer 6 parameters
     * @param temporalProcessor temporal processing pipeline
     * @param executor custom executor service
     * @param ownsExecutor whether to shutdown executor on close
     */
    public CorticalCircuitOptimized(
            int size,
            Layer1Parameters layer1Params,
            Layer23Parameters layer23Params,
            Layer4Parameters layer4Params,
            Layer5Parameters layer5Params,
            Layer6Parameters layer6Params,
            TemporalProcessor temporalProcessor,
            ExecutorService executor,
            boolean ownsExecutor) {

        // Create delegate circuit for standard processing fallback
        this.delegate = new CorticalCircuit(size, layer1Params, layer23Params, layer4Params,
                                           layer5Params, layer6Params, temporalProcessor);

        this.executor = executor;
        this.ownsExecutor = ownsExecutor;

        // Direct layer access
        this.layer1 = delegate.getLayer1();
        this.layer23 = delegate.getLayer23();
        this.layer4 = delegate.getLayer4();
        this.layer5 = delegate.getLayer5();
        this.layer6 = delegate.getLayer6();
        this.temporalProcessor = delegate.getTemporalProcessor();

        // Parameters
        this.layer1Params = layer1Params;
        this.layer23Params = layer23Params;
        this.layer4Params = layer4Params;
        this.layer5Params = layer5Params;
        this.layer6Params = layer6Params;
    }

    /**
     * Process input through circuit with parallel layer optimization.
     *
     * <p>Optimization: L1 and L6 process in parallel after L2/3.
     *
     * @param input input pattern
     * @return processed output from Layer 5
     */
    public Pattern process(Pattern input) {
        try {
            // Step 1: Temporal chunking
            var inputArray = ((DenseVector) input).data();
            var temporalResult = temporalProcessor.processItem(inputArray);
            var wmState = temporalResult.workingMemoryState();
            var chunkedPattern = new DenseVector(wmState.getCombinedPattern());

            // Step 2: Bottom-up pathway (sequential: L4 → L2/3)
            var l4Output = layer4.processBottomUp(chunkedPattern, layer4Params);
            var l23Output = layer23.processBottomUp(l4Output, layer23Params);

            // Step 3: Parallel processing of L1 and L6 (both depend on L2/3)
            var futureL1 = executor.submit(() ->
                layer1.processBottomUp(l23Output, layer1Params)
            );
            var futureL6 = executor.submit(() ->
                layer6.processBottomUp(l23Output, layer6Params)
            );

            // Wait for both to complete
            var l1Output = futureL1.get();
            var l6Output = futureL6.get();

            // Step 4: Top-down pathway (sequential: L6 → L2/3 → L4)
            var l23Modulated = layer23.processTopDown(l6Output, layer23Params);
            var l4Modulated = layer4.processTopDown(l23Modulated, layer4Params);

            // Step 5: Layer 1 top-down priming to L2/3
            var l23WithL1 = layer23.processTopDown(l1Output, layer23Params);

            // Step 6: Output pathway (L2/3 → L5)
            var l5Output = layer5.processBottomUp(l23WithL1, layer5Params);

            return l5Output;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Circuit processing interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Circuit processing failed", e.getCause());
        }
    }

    /**
     * Process with detailed pathway results (uses delegate for full tracking).
     *
     * @param input input pattern
     * @return detailed circuit result
     */
    public CorticalCircuit.CircuitResult processDetailed(Pattern input) {
        // For detailed tracking, use delegate (parallel optimization less critical)
        return delegate.processDetailed(input);
    }

    /**
     * Get the underlying delegate circuit.
     *
     * @return delegate circuit instance
     */
    public CorticalCircuit getDelegate() {
        return delegate;
    }

    /**
     * Get the executor service being used.
     *
     * @return executor service
     */
    public ExecutorService getExecutor() {
        return executor;
    }

    /**
     * Get Layer 1.
     */
    public Layer1 getLayer1() {
        return layer1;
    }

    /**
     * Get Layer 2/3.
     */
    public Layer23 getLayer23() {
        return layer23;
    }

    /**
     * Get Layer 4.
     */
    public Layer4 getLayer4() {
        return layer4;
    }

    /**
     * Get Layer 5.
     */
    public Layer5 getLayer5() {
        return layer5;
    }

    /**
     * Get Layer 6.
     */
    public Layer6 getLayer6() {
        return layer6;
    }

    /**
     * Get temporal processor.
     */
    public TemporalProcessor getTemporalProcessor() {
        return temporalProcessor;
    }

    /**
     * Enable learning (delegates to underlying circuit).
     */
    public void enableLearning(com.hellblazer.art.cortical.learning.LearningRule learningRule) {
        delegate.enableLearning(learningRule);
    }

    /**
     * Disable learning (delegates to underlying circuit).
     */
    public void disableLearning() {
        delegate.disableLearning();
    }

    /**
     * Reset all layers (delegates to underlying circuit).
     */
    public void reset() {
        delegate.reset();
    }

    @Override
    public void close() {
        // Close delegate first
        delegate.close();

        // Shutdown executor if we own it
        if (ownsExecutor && executor != null && !executor.isShutdown()) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    @Override
    public String toString() {
        return "CorticalCircuitOptimized[" +
               "executor=" + executor +
               ", ownsExecutor=" + ownsExecutor +
               ", delegate=" + delegate +
               "]";
    }
}
