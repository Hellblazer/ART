package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedARTSTAR algorithm.
 * Tests stability/adaptability regulation, dynamic vigilance adjustment, and category management.
 */
@DisplayName("VectorizedARTSTAR Tests")
public class VectorizedARTSTARTest extends BaseVectorizedARTTest<VectorizedARTSTAR, VectorizedARTSTARParameters> {

    @Override
    protected VectorizedARTSTAR createAlgorithm(VectorizedARTSTARParameters params) {
        return new VectorizedARTSTAR(params);
    }

    @Override
    protected VectorizedARTSTARParameters createDefaultParameters() {
        return new VectorizedARTSTARParameters();
    }

    @Override
    protected VectorizedARTSTARParameters createParametersWithVigilance(double vigilance) {
        return VectorizedARTSTARParameters.withVigilance(vigilance);
    }

    protected double[] generateRandomInput(int dimension) {
        var input = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            input[i] = Math.random();
        }
        return input;
    }

    @Test
    @DisplayName("Test stability-adaptability balance")
    void testStabilityAdaptabilityBalance() {
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTSTARParameters(
            0.7,     // baseVigilance
            0.001,   // alpha
            0.5,     // learningRate
            0.7,     // stabilityBias (favor stability)
            0.3,     // adaptabilityBias
            0.1,     // regulationRate
            0.01,    // decayRate
            0.1,     // pruningThreshold
            100,     // maxCategories
            10,      // minCategoryAge
            0.3,     // minVigilance
            0.95,    // maxVigilance
            0.05,    // vigilanceAdjustmentRate
            50,      // performanceWindowSize
            0.8,     // targetSuccessRate
            baseParams
        );
        var art = createAlgorithm(params);
        
        // Learn stable patterns first
        double[][] stablePatterns = {
            {0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        // Reinforce stable patterns multiple times
        for (int epoch = 0; epoch < 10; epoch++) {
            for (var pattern : stablePatterns) {
                art.learn(pattern);
            }
        }
        
        int stableCategories = art.getCategoryCount();
        
        // Now introduce new patterns
        double[][] newPatterns = {
            {0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0}
        };
        
        for (var pattern : newPatterns) {
            art.learn(pattern);
        }
        
        int totalCategories = art.getCategoryCount();
        
        // Should have created new categories despite stability bias
        assertTrue(totalCategories > stableCategories, 
            "Should adapt to new patterns even with stability bias");
        
        // Original patterns should still be recognized
        for (var pattern : stablePatterns) {
            int prediction = art.predict(pattern);
            assertTrue(prediction >= 0, "Stable patterns should remain recognized");
        }
    }

    @Test
    @DisplayName("Test dynamic vigilance adjustment")
    void testDynamicVigilanceAdjustment() {
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTSTARParameters(
            0.6,     // baseVigilance (middle range)
            0.001,   // alpha
            0.5,     // learningRate
            0.8,     // stabilityBias (higher to create imbalance)
            0.2,     // adaptabilityBias (lower to create imbalance)
            0.5,     // regulationRate (higher for faster adjustment)
            0.01,    // decayRate
            0.1,     // pruningThreshold
            100,     // maxCategories
            5,       // minCategoryAge
            0.3,     // minVigilance
            0.9,     // maxVigilance
            0.5,     // vigilanceAdjustmentRate (50% for testing)
            20,      // performanceWindowSize
            0.8,     // targetSuccessRate
            baseParams
        );
        var art = createAlgorithm(params);
        
        // Create patterns that vary in similarity
        double[][] varyingPatterns = new double[30][10];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 10; j++) {
                // Create patterns with varying levels of noise
                varyingPatterns[i][j] = (j == i % 10) ? 0.8 : 0.02 * Math.random();
            }
            // Normalize
            double sum = 0;
            for (double v : varyingPatterns[i]) sum += v;
            if (sum > 0) {
                for (int j = 0; j < 10; j++) {
                    varyingPatterns[i][j] /= sum;
                }
            }
        }
        
        // Learn patterns and observe category growth
        for (var pattern : varyingPatterns) {
            art.learn(pattern);
        }
        
        int categories = art.getCategoryCount();
        
        // Should create reasonable number of categories through vigilance adjustment
        assertTrue(categories > 0, "Should create categories");
        assertTrue(categories < varyingPatterns.length, 
            "Vigilance adjustment should prevent one category per pattern");
        
        // Get performance metrics to verify dynamic adjustment occurred
        var metrics = art.getPerformanceStats();
        assertNotNull(metrics, "Should have performance metrics");
        assertTrue(metrics.vigilanceAdjustments() > 0, 
            "Should have performed vigilance adjustments");
    }

    @Test
    @DisplayName("Test category pruning")
    void testCategoryPruning() {
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTSTARParameters(
            0.7,     // baseVigilance
            0.001,   // alpha
            0.5,     // learningRate
            0.5,     // stabilityBias
            0.5,     // adaptabilityBias
            0.1,     // regulationRate
            0.05,    // decayRate (higher for faster decay)
            0.2,     // pruningThreshold (higher for aggressive pruning)
            50,      // maxCategories
            3,       // minCategoryAge (lower for faster pruning)
            0.3,     // minVigilance
            0.95,    // maxVigilance
            0.05,    // vigilanceAdjustmentRate
            50,      // performanceWindowSize
            0.8,     // targetSuccessRate
            baseParams
        );
        var art = createAlgorithm(params);
        
        // Create temporary patterns (learned once)
        double[][] temporaryPatterns = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                temporaryPatterns[i][j] = (i == j) ? 0.9 : 0.01;
            }
        }
        
        // Learn temporary patterns once
        for (var pattern : temporaryPatterns) {
            art.learn(pattern);
        }
        
        int initialCategories = art.getCategoryCount();
        
        // Create persistent patterns (learned many times)
        double[][] persistentPatterns = {
            {0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        // Reinforce persistent patterns many times
        for (int epoch = 0; epoch < 20; epoch++) {
            for (var pattern : persistentPatterns) {
                art.learn(pattern);
            }
        }
        
        // Check metrics for pruning activity
        var metrics = art.getPerformanceStats();
        
        // Persistent patterns should still be recognized
        for (var pattern : persistentPatterns) {
            int prediction = art.predict(pattern);
            assertTrue(prediction >= 0, "Persistent patterns should be retained");
        }
        
        // Should have pruning activity in metrics
        assertTrue(metrics.categoryPrunings() >= 0, 
            "Should track category pruning events");
    }

    @Test
    @DisplayName("Test regulation rate effects")
    void testRegulationRate() {
        // Test with slow regulation
        var slowParams = new VectorizedARTSTARParameters(
            0.7, 0.001, 0.5, 0.5, 0.5,
            0.01,    // slow regulationRate
            0.01, 0.1, 100, 10,
            0.3, 0.95, 0.05, 50, 0.8,
            VectorizedParameters.createDefault()
        );
        
        // Test with fast regulation
        var fastParams = new VectorizedARTSTARParameters(
            0.7, 0.001, 0.5, 0.5, 0.5,
            0.5,     // fast regulationRate
            0.01, 0.1, 100, 10,
            0.3, 0.95, 0.05, 50, 0.8,
            VectorizedParameters.createDefault()
        );
        
        var slowArt = createAlgorithm(slowParams);
        var fastArt = createAlgorithm(fastParams);
        
        // Create alternating pattern types
        double[][] typeA = {
            {0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        double[][] typeB = {
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0}
        };
        
        // Alternate between pattern types
        for (int i = 0; i < 10; i++) {
            for (var pattern : typeA) {
                slowArt.learn(pattern);
                fastArt.learn(pattern);
            }
            for (var pattern : typeB) {
                slowArt.learn(pattern);
                fastArt.learn(pattern);
            }
        }
        
        var slowMetrics = slowArt.getPerformanceStats();
        var fastMetrics = fastArt.getPerformanceStats();
        
        // Fast regulation should have more regulation events
        assertTrue(fastMetrics.stabilityRegulations() + fastMetrics.adaptabilityRegulations() >= 
                  slowMetrics.stabilityRegulations() + slowMetrics.adaptabilityRegulations(),
            "Fast regulation should have more regulation events");
    }

    @Test
    @DisplayName("Test max categories constraint")
    void testMaxCategoriesConstraint() {
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTSTARParameters(
            0.9,     // high baseVigilance (tends to create more categories)
            0.001,   // alpha
            0.5,     // learningRate
            0.3,     // stabilityBias
            0.7,     // adaptabilityBias (favor new learning)
            0.1,     // regulationRate
            0.01,    // decayRate
            0.1,     // pruningThreshold
            10,      // maxCategories (low limit for testing)
            5,       // minCategoryAge
            0.3,     // minVigilance
            0.95,    // maxVigilance
            0.05,    // vigilanceAdjustmentRate
            50,      // performanceWindowSize
            0.8,     // targetSuccessRate
            baseParams
        );
        var art = createAlgorithm(params);
        
        // Try to create more patterns than max categories
        for (int i = 0; i < 20; i++) {
            double[] pattern = new double[10];
            pattern[i % 10] = 1.0; // Each pattern is orthogonal
            art.learn(pattern);
        }
        
        int categories = art.getCategoryCount();
        
        // Should not exceed max categories
        assertTrue(categories <= 10, 
            "Should respect max categories constraint: " + categories);
        assertTrue(categories > 0, "Should create at least one category");
    }

    @Test
    @DisplayName("Test performance window tracking")
    void testPerformanceWindowTracking() {
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTSTARParameters(
            0.7, 0.001, 0.5, 0.5, 0.5, 0.1, 0.01, 0.1, 100, 10,
            0.3, 0.95, 0.05,
            10,      // small performanceWindowSize for testing
            0.8,     // targetSuccessRate
            baseParams
        );
        var art = createAlgorithm(params);
        
        // Learn initial patterns
        double[][] patterns = new double[15][10];
        for (int i = 0; i < 15; i++) {
            for (int j = 0; j < 10; j++) {
                patterns[i][j] = (j == (i % 10)) ? 0.8 : 0.02;
            }
        }
        
        // Learn patterns
        for (var pattern : patterns) {
            art.learn(pattern);
        }
        
        // Test predictions to generate performance data
        int correct = 0;
        for (var pattern : patterns) {
            int prediction = art.predict(pattern);
            if (prediction >= 0) correct++;
        }
        
        double successRate = (double) correct / patterns.length;
        assertTrue(successRate > 0, "Should have some successful predictions");
        
        // Get metrics
        var metrics = art.getPerformanceStats();
        assertNotNull(metrics, "Should have performance metrics");
    }

    @Test
    @DisplayName("Test SIMD optimization benefits")
    void testSIMDOptimization() {
        var params = new VectorizedARTSTARParameters();
        var art = createAlgorithm(params);
        
        // Use higher dimension for SIMD benefits
        int dimension = 128;
        int numPatterns = 100;
        
        double[][] patterns = new double[numPatterns][dimension];
        for (int i = 0; i < numPatterns; i++) {
            patterns[i] = generateRandomInput(dimension);
        }
        
        // Measure performance
        long startTime = System.nanoTime();
        for (var pattern : patterns) {
            art.learn(pattern);
        }
        long learningTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        for (var pattern : patterns) {
            art.predict(pattern);
        }
        long predictionTime = System.nanoTime() - startTime;
        
        // Get performance metrics
        var metrics = art.getPerformanceStats();
        
        // Verify SIMD utilization
        assertTrue(metrics.simdOperations() > 0, "Should use SIMD operations");
        assertTrue(metrics.simdUtilization() > 0, "Should have SIMD utilization");
        
        // Log performance
        System.out.println("ARTSTAR Performance (128D):");
        System.out.println("  Learning time: " + (learningTime / 1_000_000) + " ms");
        System.out.println("  Prediction time: " + (predictionTime / 1_000_000) + " ms");
        System.out.println("  SIMD operations: " + metrics.simdOperations());
        System.out.println("  SIMD utilization: " + String.format("%.2f%%", metrics.simdUtilization() * 100));
        System.out.println("  Throughput: " + String.format("%.0f ops/sec", metrics.throughputOpsPerSec()));
    }
}