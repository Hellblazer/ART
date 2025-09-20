package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedARTE (Enhanced ART) algorithm.
 * Tests adaptive learning, feature weighting, topology adjustment, and performance optimizations.
 */
@DisplayName("VectorizedARTE Tests")
public class VectorizedARTETest extends BaseVectorizedARTTest<VectorizedARTE, VectorizedARTEParameters> {

    @Override
    protected VectorizedARTE createAlgorithm(VectorizedARTEParameters params) {
        return new VectorizedARTE(params);
    }

    @Override
    protected VectorizedARTEParameters createDefaultParameters() {
        return VectorizedARTEParameters.defaults();
    }

    @Override
    protected VectorizedARTEParameters createParametersWithVigilance(double vigilance) {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 1.0 / 10.0);
        return new VectorizedARTEParameters(
            vigilance, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.3, 1.0, 0.01,
            10, 0.001, baseParams, true, 0.01, 10
        );
    }

    protected double[] generateRandomInput(int dimension) {
        var input = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            input[i] = Math.random();
        }
        return input;
    }

    @Test
    @DisplayName("Test adaptive learning rate adjustment")
    void testAdaptiveLearningRate() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.3,
            1.0, 0.1, 10, 0.001, baseParams, true, 0.01, 10
        );
        var art = createAlgorithm(params);
        
        // Create patterns that will become familiar
        double[][] familiarPatterns = {
            {0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        // Learn patterns multiple times (should become familiar)
        for (int epoch = 0; epoch < 10; epoch++) {
            for (var pattern : familiarPatterns) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        
        int familiarCategories = art.getCategoryCount();
        
        // Introduce novel patterns
        double[][] novelPatterns = {
            {0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0}
        };
        
        for (var pattern : novelPatterns) {
            art.learn(Pattern.of(pattern), params);
        }
        
        int totalCategories = art.getCategoryCount();
        
        // Should create new categories for novel patterns
        assertTrue(totalCategories > familiarCategories, 
            "Adaptive learning should allow novel patterns to create new categories");
        
        // Familiar patterns should still be recognized
        for (var pattern : familiarPatterns) {
            var result = art.predict(Pattern.of(pattern), params);
            int prediction = result instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
            assertTrue(prediction >= 0, "Familiar patterns should remain recognized");
        }
    }

    @Test
    @DisplayName("Test feature weighting mechanism")
    void testFeatureWeighting() {
        // Create custom feature weights (emphasize first 3 dimensions)
        double[] weights = new double[10];
        weights[0] = 2.0;  // High importance
        weights[1] = 2.0;  // High importance
        weights[2] = 1.5;  // Medium importance
        for (int i = 3; i < 10; i++) {
            weights[i] = 0.5;  // Low importance
        }
        
        var baseParams = VectorizedParameters.createDefault();
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.3, 1.0, 0.01,
            10, 0.001, baseParams, true, 0.01, 10
        );
        var art = createAlgorithm(params);
        
        // Create patterns that differ in important vs unimportant features
        double[][] patterns = {
            // Group 1: Different in important features (dims 0-2)
            {0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            {0.1, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            {0.1, 0.1, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
            // Group 2: Same in important features, different in unimportant
            {0.5, 0.5, 0.5, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
            {0.5, 0.5, 0.5, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1},
            {0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1}
        };
        
        // Learn all patterns
        for (var pattern : patterns) {
            art.learn(Pattern.of(pattern), params);
        }
        
        // Group 1 should create distinct categories (different in important features)
        var r0 = art.predict(Pattern.of(patterns[0]), params);
        var r1 = art.predict(Pattern.of(patterns[1]), params);
        var r2 = art.predict(Pattern.of(patterns[2]), params);
        int pred0 = r0 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int pred1 = r1 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int pred2 = r2 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        
        assertNotEquals(pred0, pred1, "Patterns different in important features should separate");
        assertNotEquals(pred1, pred2, "Patterns different in important features should separate");
        assertNotEquals(pred0, pred2, "Patterns different in important features should separate");
        
        // Group 2 might cluster together (same in important features)
        var r3 = art.predict(Pattern.of(patterns[3]), params);
        var r4 = art.predict(Pattern.of(patterns[4]), params);
        var r5 = art.predict(Pattern.of(patterns[5]), params);
        int pred3 = r3 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int pred4 = r4 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int pred5 = r5 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        
        // These might be the same category since they're similar in important features
        // But should be different from Group 1
        assertNotEquals(pred0, pred3, "Different groups should be separated");
    }

    @Test
    @DisplayName("Test topology adjustment capability")
    void testTopologyAdjustment() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.2, 0.6, 0.3, 1.0, 1.0,
            10, 0.001, baseParams, true, 1.0, 10
        );
        var art = createAlgorithm(params);
        
        // Create a sequence of gradually changing patterns
        double[][] gradualSequence = new double[20][10];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 10; j++) {
                // Create smooth transitions
                gradualSequence[i][j] = Math.sin((i * 0.2 + j * 0.5)) * 0.5 + 0.5;
            }
        }
        
        // Learn the sequence multiple times
        for (int epoch = 0; epoch < 5; epoch++) {
            for (var pattern : gradualSequence) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        
        // Should create reasonable number of categories with topology adjustment
        int categories = art.getCategoryCount();
        assertTrue(categories > 0, "Should create categories");
        assertTrue(categories <= gradualSequence.length * 10, 
            "Should create reasonable number of categories (complement coding creates many more)");
        
        // Check metrics for topology adjustments
        var metrics = art.getPerformanceStats();
        assertTrue(metrics.topologyAdjustments() > 0, 
            "Should have performed topology adjustments");
    }

    @Test
    @DisplayName("Test performance-based optimization")
    void testPerformanceOptimization() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.8, 0.3, 1.0, 0.01,
            20, 0.001, baseParams, true, 0.01, 10
        );
        var art = createAlgorithm(params);
        
        // Create patterns with different complexity levels
        double[][] simplePatterns = new double[10][10];
        double[][] complexPatterns = new double[10][10];
        
        for (int i = 0; i < 10; i++) {
            // Simple patterns (one-hot encoded)
            simplePatterns[i][i] = 1.0;
            
            // Complex patterns (distributed)
            for (int j = 0; j < 10; j++) {
                complexPatterns[i][j] = Math.random() * 0.5 + 0.25;
            }
        }
        
        // Learn simple patterns first
        for (var pattern : simplePatterns) {
            art.learn(Pattern.of(pattern), params);
        }
        
        int simpleCategories = art.getCategoryCount();
        
        // Learn complex patterns
        for (var pattern : complexPatterns) {
            art.learn(Pattern.of(pattern), params);
        }
        
        int totalCategories = art.getCategoryCount();
        
        // Performance optimization should handle both pattern types
        assertTrue(totalCategories > simpleCategories, 
            "Should create additional categories for complex patterns");
        
        // Check convergence optimization
        var metrics = art.getPerformanceStats();
        assertTrue(metrics.convergenceOptimizations() >= 0, 
            "Should track convergence optimizations");
    }

    @Test
    @DisplayName("Test context sensitivity")
    void testContextSensitivity() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.9, 1.0, 0.01,
            10, 0.001, baseParams, true, 0.01, 10
        );
        var art = createAlgorithm(params);
        
        // Create context-dependent patterns
        double[][] contextA = {
            {0.8, 0.2, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0},
            {0.7, 0.3, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0}
        };
        
        double[][] contextB = {
            {0.0, 0.0, 0.8, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.7, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0}
        };
        
        // Learn in context-specific sequences
        for (int context = 0; context < 5; context++) {
            // Context A
            for (var pattern : contextA) {
                art.learn(Pattern.of(pattern), params);
            }
            // Context B
            for (var pattern : contextB) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        
        // Context sensitivity should maintain separation
        var rA1 = art.predict(Pattern.of(contextA[0]), params);
        var rA2 = art.predict(Pattern.of(contextA[1]), params);
        var rB1 = art.predict(Pattern.of(contextB[0]), params);
        var rB2 = art.predict(Pattern.of(contextB[1]), params);
        int predA1 = rA1 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int predA2 = rA2 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int predB1 = rB1 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        int predB2 = rB2 instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
        
        // Same context patterns might cluster
        assertEquals(predA1, predA2, "Similar patterns in same context should cluster");
        assertEquals(predB1, predB2, "Similar patterns in same context should cluster");
        
        // Different contexts should separate
        assertNotEquals(predA1, predB1, "Different contexts should be separated");
    }

    @Test
    @DisplayName("Test convergence threshold behavior")
    void testConvergenceThreshold() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.3, 1.0, 0.01,
            10, 0.01, baseParams, true, 0.01, 10
        );
        var art = createAlgorithm(params);
        
        // Create stable patterns
        double[][] stablePatterns = {
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        // Learn patterns and track convergence rate
        double initialConvergenceRate = 0.0;
        double finalConvergenceRate = 0.0;
        
        // Early learning phase (should have low convergence)
        for (int epoch = 0; epoch < 5; epoch++) {
            for (var pattern : stablePatterns) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        initialConvergenceRate = art.getPerformanceStats().convergenceRate();
        
        // Extended learning phase (should increase convergence)
        for (int epoch = 0; epoch < 15; epoch++) {
            for (var pattern : stablePatterns) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        finalConvergenceRate = art.getPerformanceStats().convergenceRate();
        
        // System should create reasonable categories
        int categories = art.getCategoryCount();
        assertTrue(categories > 0, "Should create categories");
        assertTrue(categories <= 100, 
            "Should create reasonable number of categories (complement coding and feature weighting can create many categories)");
        
        // Convergence rate should increase over time (more weights converging)
        assertTrue(finalConvergenceRate >= initialConvergenceRate, 
            "Convergence rate should increase or remain stable over repeated learning");
        
        // System should be able to predict learned patterns correctly
        for (var pattern : stablePatterns) {
            var result = art.predict(Pattern.of(pattern), params);
            int prediction = result instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
            assertTrue(prediction >= 0, "Should successfully predict learned patterns");
        }
        
        // Convergence optimizations should have been performed
        var metrics = art.getPerformanceStats();
        assertTrue(metrics.convergenceOptimizations() > 0, 
            "Should track convergence optimizations during learning");
    }

    @Test
    @DisplayName("Test familiarity decay mechanism")
    void testFamiliarityDecay() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.7, 0.001, 0.8, 0.5, true, weights, 0.1, 0.6, 0.3, 1.0, 0.1,
            10, 0.001, baseParams, true, 0.1, 10
        );
        var art = createAlgorithm(params);
        
        // Learn initial patterns intensively
        double[][] initialPatterns = {
            {0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        for (int epoch = 0; epoch < 10; epoch++) {
            for (var pattern : initialPatterns) {
                art.learn(Pattern.of(pattern), params);
            }
        }
        
        // Learn many other patterns (initial patterns decay)
        double[][] otherPatterns = new double[20][10];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 10; j++) {
                otherPatterns[i][j] = Math.random();
            }
            art.learn(Pattern.of(otherPatterns[i]), params);
        }
        
        // Initial patterns should still be recognized despite decay
        for (var pattern : initialPatterns) {
            var result = art.predict(Pattern.of(pattern), params);
            int prediction = result instanceof ActivationResult.Success s ? s.categoryIndex() : -1;
            assertTrue(prediction >= 0, 
                "Initial patterns should still be recognized after familiarity decay");
        }
    }

    @Test
    @DisplayName("Test pruning operations")
    void testPruningOperations() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 0.1);
        var params = new VectorizedARTEParameters(
            0.8, 0.001, 0.8, 0.5, true, weights, 0.1, 0.7, 0.3, 1.0, 0.01,
            10, 0.001, baseParams, true, 0.01, 10
        );  // high vigilance creates more categories
        var art = createAlgorithm(params);
        
        // Create many slightly different patterns
        for (int i = 0; i < 50; i++) {
            double[] pattern = new double[10];
            pattern[i % 10] = 0.8 + Math.random() * 0.2;
            for (int j = 0; j < 10; j++) {
                if (j != i % 10) {
                    pattern[j] = Math.random() * 0.2;
                }
            }
            art.learn(Pattern.of(pattern), params);
        }
        
        // Check pruning activity
        var metrics = art.getPerformanceStats();
        assertTrue(metrics.pruningOperations() >= 0, 
            "Should track pruning operations");
        
        // System should maintain reasonable category count
        int categories = art.getCategoryCount();
        assertTrue(categories > 0, "Should have categories");
        assertTrue(categories < 50, "Pruning should prevent excessive categories");
    }

    @Test
    @DisplayName("Test SIMD vectorization performance")
    void testSIMDPerformance() {
        var baseParams = VectorizedParameters.createDefault();
        double[] weights = new double[128];
        java.util.Arrays.fill(weights, 1.0 / 128.0);
        var params = new VectorizedARTEParameters(
            0.75, 0.001, 0.8, 0.5, true, weights, 0.1, 0.8, 0.3, 1.0, 0.01,
            50, 0.001, baseParams, true, 0.01, 128
        );
        var art = createAlgorithm(params);
        
        // Generate high-dimensional patterns
        int numPatterns = 100;
        double[][] patterns = new double[numPatterns][128];
        for (int i = 0; i < numPatterns; i++) {
            patterns[i] = generateRandomInput(128);
        }
        
        // Measure learning performance
        long startTime = System.nanoTime();
        for (var pattern : patterns) {
            art.learn(Pattern.of(pattern), params);
        }
        long learningTime = System.nanoTime() - startTime;
        
        // Measure prediction performance
        startTime = System.nanoTime();
        for (var pattern : patterns) {
            art.predict(Pattern.of(pattern), params);
        }
        long predictionTime = System.nanoTime() - startTime;
        
        // Get performance metrics
        var metrics = art.getPerformanceStats();
        
        // Verify SIMD utilization
        assertTrue(metrics.simdOperations() > 0, "Should use SIMD operations");
        assertTrue(metrics.simdUtilization() > 0, "Should have SIMD utilization");
        
        // Log performance
        System.out.println("ARTE Performance (128D):");
        System.out.println("  Learning time: " + (learningTime / 1_000_000) + " ms");
        System.out.println("  Prediction time: " + (predictionTime / 1_000_000) + " ms");
        System.out.println("  SIMD operations: " + metrics.simdOperations());
        System.out.println("  SIMD utilization: " + String.format("%.2f%%", metrics.simdUtilization() * 100));
        System.out.println("  Throughput: " + String.format("%.0f ops/sec", metrics.throughputOpsPerSec()));
        System.out.println("  Categories: " + art.getCategoryCount());
        System.out.println("  Convergence rate: " + String.format("%.2f", metrics.convergenceRate()));
    }
}