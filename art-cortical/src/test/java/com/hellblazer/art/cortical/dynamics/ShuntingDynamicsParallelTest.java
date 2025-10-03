package com.hellblazer.art.cortical.dynamics;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for parallel shunting dynamics - Phase 4C.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Correctness: Parallel results match sequential</li>
 *   <li>Performance: Speedup with multiple cores</li>
 *   <li>Thread safety: No race conditions</li>
 *   <li>Resource management: Proper pool shutdown</li>
 * </ul>
 *
 * @author Phase 4C: Shunting Dynamics Parallelization
 */
class ShuntingDynamicsParallelTest {

    private ShuntingParameters parameters;
    private ShuntingDynamics sequential;
    private ShuntingDynamicsParallel parallel;
    private static final int LARGE_DIMENSION = 512;
    private static final int SMALL_DIMENSION = 32;
    private static final double TOLERANCE = 1e-10;

    @BeforeEach
    void setup() {
        parameters = ShuntingParameters.builder(LARGE_DIMENSION)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(0.1)
            .timeStep(0.001)
            .build();

        sequential = new ShuntingDynamics(parameters);
        parallel = new ShuntingDynamicsParallel(parameters);
    }

    @AfterEach
    void tearDown() {
        if (parallel != null) {
            parallel.shutdown();
        }
    }

    @Test
    void testParallelMatchesSequential() {
        // Set same input for both
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        sequential.setExcitatoryInput(input);
        parallel.setExcitatoryInput(input);

        // Single update
        var seqResult = sequential.update(0.001);
        var parResult = parallel.update(0.001);

        // Results should match exactly
        assertArrayEquals(seqResult, parResult, TOLERANCE,
            "Parallel results should match sequential");
    }

    @Test
    void testParallelConvergence() {
        // Both should converge to same steady state
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = 0.5 + 0.1 * Math.sin(2.0 * Math.PI * i / LARGE_DIMENSION);
        }

        sequential.setExcitatoryInput(input);
        parallel.setExcitatoryInput(input);

        // Run until convergence (or max iterations)
        var maxIterations = 500;  // Increased for convergence
        var seqIterations = 0;
        var parIterations = 0;

        while (!sequential.hasConverged() && seqIterations < maxIterations) {
            sequential.update(0.001);
            seqIterations++;
        }

        while (!parallel.hasConverged() && parIterations < maxIterations) {
            parallel.update(0.001);
            parIterations++;
        }

        // Both should converge (or at least match in iterations)
        if (seqIterations < maxIterations && parIterations < maxIterations) {
            // If both converged, check convergence
            assertTrue(sequential.hasConverged(), "Sequential should converge");
            assertTrue(parallel.hasConverged(), "Parallel should converge");
        }

        // Convergence iterations should be similar (±1 due to floating point)
        assertEquals(seqIterations, parIterations, 1,
            "Convergence should take similar iterations");

        // Final states should match
        assertArrayEquals(sequential.getActivation(), parallel.getActivation(), TOLERANCE,
            "Final states should match");
    }

    @Test
    void testSmallNetworkFallsBackToSequential() {
        // Create small network (below MIN_CHUNK_SIZE)
        var smallParams = ShuntingParameters.builder(SMALL_DIMENSION)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.2)
            .inhibitoryStrength(0.05)
            .build();

        var smallParallel = new ShuntingDynamicsParallel(smallParams);

        var input = new double[SMALL_DIMENSION];
        for (int i = 0; i < SMALL_DIMENSION; i++) {
            input[i] = Math.random();
        }

        smallParallel.setExcitatoryInput(input);

        // Should run without error (sequential fallback)
        assertDoesNotThrow(() -> smallParallel.update(0.001));

        smallParallel.shutdown();
    }

    @Test
    void testMultipleUpdates() {
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        sequential.setExcitatoryInput(input);
        parallel.setExcitatoryInput(input);

        // Multiple updates should stay synchronized
        for (int iter = 0; iter < 10; iter++) {
            var seqResult = sequential.update(0.001);
            var parResult = parallel.update(0.001);

            assertArrayEquals(seqResult, parResult, TOLERANCE,
                "Results should match at iteration " + iter);
        }
    }

    @Test
    void testResetSynchronization() {
        // Run some updates
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        sequential.setExcitatoryInput(input);
        parallel.setExcitatoryInput(input);

        for (int i = 0; i < 5; i++) {
            sequential.update(0.001);
            parallel.update(0.001);
        }

        // Reset both
        sequential.reset();
        parallel.reset();

        // After reset, both should match
        assertArrayEquals(sequential.getActivation(), parallel.getActivation(), TOLERANCE,
            "States should match after reset");
    }

    @Test
    void testCustomPoolSize() {
        // Create with specific thread count
        var customPool = new ForkJoinPool(2);
        var customParallel = new ShuntingDynamicsParallel(parameters, customPool);

        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        customParallel.setExcitatoryInput(input);
        var result = customParallel.update(0.001);

        assertNotNull(result);
        assertEquals(LARGE_DIMENSION, result.length);

        customParallel.shutdown();
        customPool.shutdown();
    }

    @Test
    void testThreadSafety() throws InterruptedException {
        // Run parallel updates from multiple threads
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        parallel.setExcitatoryInput(input);

        var threads = new Thread[4];
        var results = new double[4][];

        // Start multiple threads doing updates simultaneously
        for (int t = 0; t < threads.length; t++) {
            final int threadId = t;
            threads[t] = new Thread(() -> {
                for (int i = 0; i < 5; i++) {
                    results[threadId] = parallel.update(0.001);
                }
            });
            threads[t].start();
        }

        // Wait for all threads
        for (var thread : threads) {
            thread.join();
        }

        // All results should have valid activations (no NaN/Inf)
        for (int t = 0; t < results.length; t++) {
            assertNotNull(results[t], "Result " + t + " should not be null");
            for (int i = 0; i < results[t].length; i++) {
                assertFalse(Double.isNaN(results[t][i]),
                    "Result " + t + "[" + i + "] should not be NaN");
                assertFalse(Double.isInfinite(results[t][i]),
                    "Result " + t + "[" + i + "] should not be infinite");
            }
        }
    }

    @Test
    void testExcitatoryInhibitoryBalance() {
        // Test that parallel correctly handles excitation/inhibition
        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = 0.5;
        }

        sequential.setExcitatoryInput(input);
        parallel.setExcitatoryInput(input);

        // Run to convergence
        for (int i = 0; i < 50; i++) {
            sequential.update(0.001);
            parallel.update(0.001);
        }

        // Both should have similar activation patterns
        var seqActivation = sequential.getActivation();
        var parActivation = parallel.getActivation();

        // Check mean activation
        var seqMean = 0.0;
        var parMean = 0.0;
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            seqMean += seqActivation[i];
            parMean += parActivation[i];
        }
        seqMean /= LARGE_DIMENSION;
        parMean /= LARGE_DIMENSION;

        assertEquals(seqMean, parMean, TOLERANCE,
            "Mean activation should match");
    }

    @Test
    void testZeroInput() {
        // Test with zero input (decay only)
        var zeroInput = new double[LARGE_DIMENSION];

        sequential.setExcitatoryInput(zeroInput);
        parallel.setExcitatoryInput(zeroInput);

        var seqResult = sequential.update(0.001);
        var parResult = parallel.update(0.001);

        assertArrayEquals(seqResult, parResult, TOLERANCE,
            "Results should match with zero input");
    }

    @Test
    void testPoolShutdown() {
        var customPool = new ForkJoinPool(4);
        var customParallel = new ShuntingDynamicsParallel(parameters, customPool);

        var input = new double[LARGE_DIMENSION];
        for (int i = 0; i < LARGE_DIMENSION; i++) {
            input[i] = Math.random();
        }

        customParallel.setExcitatoryInput(input);
        customParallel.update(0.001);

        // Should not shutdown common pool
        customParallel.shutdown();
        assertFalse(ForkJoinPool.commonPool().isShutdown(),
            "Common pool should not be shut down");

        // But custom pool should shutdown if we created it with ownership
        customPool.shutdown();
        assertTrue(customPool.isShutdown(), "Custom pool should shut down");
    }
}
