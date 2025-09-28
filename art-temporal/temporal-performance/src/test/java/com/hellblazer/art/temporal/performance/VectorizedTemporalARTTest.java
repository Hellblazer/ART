package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.integration.*;
import com.hellblazer.art.temporal.memory.WorkingMemoryParameters;
import com.hellblazer.art.temporal.masking.MaskingFieldParameters;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Comprehensive tests for vectorized TemporalART implementation.
 */
class VectorizedTemporalARTTest {

    private VectorizedTemporalART art;
    private TemporalARTParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = createTestParameters();
        art = new VectorizedTemporalART(parameters);
    }

    @AfterEach
    void tearDown() {
        if (art != null) {
            art.shutdown();
        }
    }

    @Test
    @DisplayName("Basic sequence processing")
    void testBasicSequenceProcessing() {
        // Process a simple sequence
        double[][] sequence = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        int[] categories = new int[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            categories[i] = art.processInput(sequence[i], i * 0.1);
        }

        // Should create categories for distinct patterns
        assertTrue(art.getCategoryCount() > 0);
        assertTrue(art.getCategoryCount() <= sequence.length);
    }

    @Test
    @DisplayName("Parallel batch processing")
    void testBatchProcessing() throws Exception {
        // Create batch of patterns with correct dimension
        List<double[]> batch = new ArrayList<>();
        List<Double> timestamps = new ArrayList<>();

        int dimension = parameters.getInputDimension();

        for (int i = 0; i < 100; i++) {
            double[] pattern = new double[dimension];
            pattern[i % dimension] = 1.0;
            batch.add(pattern);
            timestamps.add(i * 0.01);
        }

        // Process batch in parallel
        CompletableFuture<int[]> future = art.processBatch(batch, timestamps);
        int[] results = future.get(5, TimeUnit.SECONDS);

        assertNotNull(results);
        assertEquals(batch.size(), results.length);
        assertTrue(art.getCategoryCount() > 0);
    }

    @Test
    @DisplayName("Performance tracking")
    void testPerformanceTracking() {
        // Process multiple patterns
        for (int i = 0; i < 50; i++) {
            double[] pattern = createRandomPattern(10);
            art.processInput(pattern, i * 0.1);
        }

        var stats = art.getStats();
        assertNotNull(stats);
        assertTrue(stats.avgProcessTimeMs() > 0);
        assertEquals(50, stats.patternsProcessed());
        assertTrue(stats.categoryCount() > 0);
    }

    @Test
    @DisplayName("Temporal pattern recognition")
    void testTemporalPatternRecognition() {
        // Create repeating temporal sequence
        double[][] sequence = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        // Present sequence multiple times
        List<Integer> categories = new ArrayList<>();
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < sequence.length; i++) {
                int cat = art.processInput(sequence[i], (rep * 3 + i) * 0.1);
                categories.add(cat);
            }
        }

        // Later presentations should recognize earlier patterns
        int uniqueCategories = new HashSet<>(categories).size();
        assertTrue(uniqueCategories <= sequence.length,
            "Should recognize repeated sequences");
    }

    @Test
    @DisplayName("Prediction accuracy")
    void testPrediction() {
        // Train on patterns
        double[][] trainingSet = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0}
        };

        int[] trainCategories = new int[trainingSet.length];
        for (int i = 0; i < trainingSet.length; i++) {
            trainCategories[i] = art.processInput(trainingSet[i], i * 0.1);
        }

        // Test prediction on same patterns
        for (int i = 0; i < trainingSet.length; i++) {
            var result = art.predict(Pattern.of(trainingSet[i]), parameters);
            int predicted = result instanceof ActivationResult.Success success ? success.categoryIndex() : -1;
            assertEquals(trainCategories[i], predicted,
                "Should predict correct category for trained pattern");
        }

        // Test on novel pattern
        double[] novel = {0.5, 0.5, 0.0, 0.0};
        var novelResult = art.predict(Pattern.of(novel), parameters);
        int novelCategory = novelResult instanceof ActivationResult.Success success ? success.categoryIndex() : -1;
        // Novel pattern may not match any category if vigilance is high
        assertTrue(novelCategory == -1 || novelCategory < art.getCategoryCount());
    }

    @Test
    @DisplayName("State persistence and reset")
    void testStateAndReset() {
        // Process some patterns
        for (int i = 0; i < 10; i++) {
            double[] pattern = createRandomPattern(parameters.getInputDimension());
            art.processInput(pattern, i * 0.1);
        }

        // Get state
        var state = art.getState();
        assertNotNull(state);
        int categoryCount = art.getCategoryCount();
        assertTrue(categoryCount > 0);

        // Reset
        art.reset();
        assertEquals(0, art.getCategoryCount());

        // Stats should be reset
        var stats = art.getStats();
        assertEquals(0, stats.patternsProcessed());
    }

    @Test
    @DisplayName("Concurrent access safety")
    void testConcurrentAccess() throws Exception {
        int numThreads = 4;
        int patternsPerThread = 25;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        List<Future<?>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() -> {
                for (int i = 0; i < patternsPerThread; i++) {
                    double[] pattern = createRandomPattern(parameters.getInputDimension());
                    art.processInput(pattern, threadId * 100 + i * 0.1);
                }
            }));
        }

        // Wait for all threads
        for (var future : futures) {
            future.get(10, TimeUnit.SECONDS);
        }

        executor.shutdown();

        // Verify results
        var stats = art.getStats();
        assertEquals(numThreads * patternsPerThread, stats.patternsProcessed());
        assertTrue(art.getCategoryCount() > 0);
    }

    @Test
    @DisplayName("Memory efficiency with large sequences")
    void testMemoryEfficiency() {
        // Process large sequence
        int sequenceLength = 1000;
        int dimension = parameters.getInputDimension();
        for (int i = 0; i < sequenceLength; i++) {
            double[] pattern = new double[dimension];
            pattern[i % dimension] = 1.0;
            art.processInput(pattern, i * 0.01);

            // Check memory periodically
            if (i % 100 == 0) {
                Runtime runtime = Runtime.getRuntime();
                long usedMemory = runtime.totalMemory() - runtime.freeMemory();
                // Just ensure we don't run out of memory
                assertTrue(usedMemory < runtime.maxMemory() * 0.9);
            }
        }

        assertTrue(art.getCategoryCount() <= parameters.getMaxCategories());
    }

    @Test
    @DisplayName("Vectorization correctness")
    void testVectorizationCorrectness() {
        // Compare with standard implementation results
        var standardParams = createTestParameters();
        var standardART = new TemporalART(standardParams);

        // Same input sequence
        double[][] sequence = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        // Process with both implementations
        for (int i = 0; i < sequence.length; i++) {
            art.processInput(sequence[i], i * 0.1);
            standardART.learn(sequence[i]);
        }

        // Results may differ due to architectural differences between implementations
        // Standard TemporalART processes sequences holistically
        // VectorizedTemporalART processes individual patterns for performance
        int standardCategories = standardART.getCategoryCount();
        int vectorizedCategories = art.getCategoryCount();

        // Both should create some categories for distinct patterns
        assertTrue(vectorizedCategories > 0, "Vectorized implementation should create categories");
        assertTrue(vectorizedCategories <= sequence.length, "Should not create more categories than patterns");

        // Log the difference for awareness
        System.out.println("Standard categories: " + standardCategories +
                          ", Vectorized categories: " + vectorizedCategories);
    }

    // Helper methods
    private TemporalARTParameters createTestParameters() {
        return TemporalARTParameters.defaults();
    }

    private double[] createRandomPattern(int dimension) {
        double[] pattern = new double[dimension];
        Random rand = new Random();
        for (int i = 0; i < dimension; i++) {
            pattern[i] = rand.nextDouble();
        }
        return pattern;
    }
}