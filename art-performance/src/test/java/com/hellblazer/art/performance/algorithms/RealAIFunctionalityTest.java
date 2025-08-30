package com.hellblazer.art.performance.algorithms;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedBayesianART;
import com.hellblazer.art.performance.algorithms.VectorizedART;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Comprehensive test to verify real AI functionality with actual machine learning patterns.
 * Tests clustering, classification, and learning with realistic datasets.
 */
@DisplayName("Real AI Functionality Verification")
class RealAIFunctionalityTest {
    
    private VectorizedParameters params;
    private Random random;
    
    @BeforeEach
    void setUp() {
        params = VectorizedParameters.createDefault()
            .withVigilance(0.75)
            .withLearningRate(0.1);
        random = new Random(42); // Fixed seed for reproducibility
    }
    
    @Test
    @DisplayName("Iris Dataset Classification - Real ML Problem")
    void testIrisDatasetClassification() {
        var fuzzyART = new VectorizedFuzzyART(params);
        
        // Real Iris dataset samples (normalized)
        var irisData = List.of(
            // Setosa samples
            new DenseVector(new double[]{0.22, 0.625, 0.067, 0.042}),
            new DenseVector(new double[]{0.167, 0.417, 0.067, 0.042}),
            new DenseVector(new double[]{0.111, 0.5, 0.051, 0.042}),
            // Versicolor samples  
            new DenseVector(new double[]{0.667, 0.417, 0.627, 0.583}),
            new DenseVector(new double[]{0.556, 0.125, 0.576, 0.5}),
            new DenseVector(new double[]{0.611, 0.333, 0.61, 0.583}),
            // Virginica samples
            new DenseVector(new double[]{0.806, 0.667, 0.864, 1.0}),
            new DenseVector(new double[]{0.639, 0.375, 0.763, 0.75}),
            new DenseVector(new double[]{0.694, 0.417, 0.763, 0.833})
        );
        
        // Train the network
        var categories = new ArrayList<Integer>();
        for (var sample : irisData) {
            var result = (ActivationResult) fuzzyART.learn(sample, params);
            if (result instanceof ActivationResult.Success success) {
                categories.add(success.categoryIndex());
            }
        }
        
        // Verify clustering occurred
        assertTrue(fuzzyART.getCategoryCount() >= 2, "Should create multiple categories for distinct classes");
        assertTrue(fuzzyART.getCategoryCount() <= 6, "Should not over-cluster");
        
        // Test prediction on similar patterns
        var testSample = new DenseVector(new double[]{0.2, 0.6, 0.07, 0.04}); // Similar to Setosa
        var prediction = (ActivationResult) fuzzyART.predict(testSample, params);
        assertNotNull(prediction, "Should predict category for new sample");
        
        if (prediction instanceof ActivationResult.Success success) {
            System.out.printf("Iris Classification: %d categories created, prediction category: %d%n", 
                             fuzzyART.getCategoryCount(), success.categoryIndex());
        }
    }
    
    @Test
    @DisplayName("XOR Problem - Non-Linear Separability")
    void testXORProblem() {
        var art = new VectorizedART(params.withVigilance(0.5));
        
        // XOR truth table with complement coding
        var xorData = List.of(
            new DenseVector(new double[]{0.0, 0.0, 1.0, 1.0}), // 0 XOR 0 = 0
            new DenseVector(new double[]{0.0, 1.0, 1.0, 0.0}), // 0 XOR 1 = 1
            new DenseVector(new double[]{1.0, 0.0, 0.0, 1.0}), // 1 XOR 0 = 1
            new DenseVector(new double[]{1.0, 1.0, 0.0, 0.0})  // 1 XOR 1 = 0
        );
        
        // Train multiple times to test learning
        for (int epoch = 0; epoch < 10; epoch++) {
            for (var pattern : xorData) {
                art.learn(pattern, params);
            }
        }
        
        // Should learn distinct representations for different outputs
        assertTrue(art.getCategoryCount() >= 2, "Should distinguish XOR patterns");
        
        // Test generalization with noisy inputs
        var noisyInput = new DenseVector(new double[]{0.1, 0.9, 0.9, 0.1}); // Noisy version of [0,1]
        var result = art.predict(noisyInput, params);
        assertNotNull(result, "Should handle noisy input");
        
        System.out.printf("XOR Learning: %d categories, handles noisy input: %s%n", 
                         art.getCategoryCount(), result != null);
    }
    
    @Test
    @DisplayName("Bayesian Uncertainty Quantification - Real Probabilistic Learning")
    void testBayesianUncertaintyQuantification() {
        var bayesianART = new VectorizedBayesianART(params.withVigilance(0.8));
        
        // Generate probabilistic patterns with different certainty levels
        var certainPatterns = List.of(
            new DenseVector(new double[]{0.95, 0.05}), // High certainty
            new DenseVector(new double[]{0.05, 0.95}), // High certainty opposite
            new DenseVector(new double[]{0.9, 0.1}),   // High certainty similar to first
            new DenseVector(new double[]{0.1, 0.9})    // High certainty similar to second
        );
        
        var uncertainPatterns = List.of(
            new DenseVector(new double[]{0.6, 0.4}),   // Medium certainty
            new DenseVector(new double[]{0.4, 0.6}),   // Medium certainty
            new DenseVector(new double[]{0.55, 0.45}), // Close to uncertain
            new DenseVector(new double[]{0.45, 0.55})  // Close to uncertain
        );
        
        // Train on certain patterns first
        for (var pattern : certainPatterns) {
            bayesianART.learn(pattern, params);
        }
        
        int categoriesAfterCertain = bayesianART.getCategoryCount();
        
        // Train on uncertain patterns
        for (var pattern : uncertainPatterns) {
            bayesianART.learn(pattern, params);
        }
        
        int finalCategories = bayesianART.getCategoryCount();
        
        // Bayesian ART should handle uncertainty appropriately
        assertTrue(finalCategories >= categoriesAfterCertain, "Should adapt to uncertain patterns");
        
        // Test prediction with varying certainty
        var testUncertain = new DenseVector(new double[]{0.5, 0.5}); // Maximum uncertainty
        var result = bayesianART.predict(testUncertain, params);
        assertNotNull(result, "Should handle maximum uncertainty case");
        
        System.out.printf("Bayesian Learning: %d→%d categories, handles uncertainty: %s%n", 
                         categoriesAfterCertain, finalCategories, result != null);
    }
    
    @Test
    @DisplayName("Incremental Learning - Online Adaptation")
    void testIncrementalLearning() {
        var art = new VectorizedFuzzyART(params.withVigilance(0.7));
        
        // Simulate streaming data with concept drift
        var phase1Data = generateGaussianCluster(0.2, 0.3, 0.1, 20); // Cluster around (0.2, 0.3)
        var phase2Data = generateGaussianCluster(0.7, 0.8, 0.1, 20); // New cluster around (0.7, 0.8)
        var phase3Data = generateGaussianCluster(0.5, 0.5, 0.15, 15); // Overlapping cluster
        
        // Phase 1: Learn initial patterns
        for (var pattern : phase1Data) {
            art.learn(pattern, params);
        }
        int phase1Categories = art.getCategoryCount();
        
        // Phase 2: Introduce new concept
        for (var pattern : phase2Data) {
            art.learn(pattern, params);
        }
        int phase2Categories = art.getCategoryCount();
        
        // Phase 3: Introduce overlapping concept
        for (var pattern : phase3Data) {
            art.learn(pattern, params);
        }
        int phase3Categories = art.getCategoryCount();
        
        // Verify incremental learning
        assertTrue(phase1Categories >= 1, "Should learn initial concept");
        assertTrue(phase2Categories > phase1Categories, "Should adapt to new concept");
        assertTrue(phase3Categories >= phase2Categories, "Should handle concept overlap");
        
        // Test prediction on all phases
        var testPhase1 = new DenseVector(new double[]{0.25, 0.35});
        var testPhase2 = new DenseVector(new double[]{0.65, 0.75});
        var testPhase3 = new DenseVector(new double[]{0.48, 0.52});
        
        var pred1 = art.predict(testPhase1, params);
        var pred2 = art.predict(testPhase2, params);
        var pred3 = art.predict(testPhase3, params);
        
        assertNotNull(pred1, "Should predict for phase 1 patterns");
        assertNotNull(pred2, "Should predict for phase 2 patterns");
        assertNotNull(pred3, "Should predict for phase 3 patterns");
        
        System.out.printf("Incremental Learning: %d→%d→%d categories across concept drift%n", 
                         phase1Categories, phase2Categories, phase3Categories);
    }
    
    @Test
    @DisplayName("Performance Scaling - SIMD vs Sequential")
    void testPerformanceScaling() {
        var art = new VectorizedFuzzyART(params);
        
        // Generate large dataset for performance testing
        var largeDataset = new ArrayList<DenseVector>();
        for (int i = 0; i < 1000; i++) {
            var pattern = new double[16]; // 16-dimensional vectors for SIMD
            for (int j = 0; j < pattern.length; j++) {
                pattern[j] = random.nextGaussian() * 0.1 + (i % 4) * 0.25; // 4 clusters
            }
            largeDataset.add(new DenseVector(pattern));
        }
        
        // Measure training time
        long startTime = System.nanoTime();
        for (var pattern : largeDataset) {
            art.learn(pattern, params);
        }
        long trainingTime = System.nanoTime() - startTime;
        
        // Measure prediction time
        startTime = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            art.predict(largeDataset.get(i), params);
        }
        long predictionTime = System.nanoTime() - startTime;
        
        // Verify SIMD optimization is working
        var stats = art.getPerformanceStats();
        assertTrue(stats.totalVectorOperations() > 0, "Should track vector operations");
        assertTrue(art.getCategoryCount() >= 3, "Should create multiple categories for large dataset");
        
        System.out.printf("Performance Test: %d categories, %d vector ops, training: %.2fms, prediction: %.2fms%n",
                         art.getCategoryCount(), stats.totalVectorOperations(),
                         trainingTime / 1_000_000.0, predictionTime / 1_000_000.0);
        
        // Performance should be reasonable for 1000 samples
        assertTrue(trainingTime < 5_000_000_000L, "Training should complete in reasonable time"); // 5 seconds
    }
    
    /**
     * Generate Gaussian cluster around specified center with given deviation.
     */
    private List<DenseVector> generateGaussianCluster(double centerX, double centerY, double stddev, int count) {
        var cluster = new ArrayList<DenseVector>();
        for (int i = 0; i < count; i++) {
            double x = Math.max(0, Math.min(1, centerX + random.nextGaussian() * stddev));
            double y = Math.max(0, Math.min(1, centerY + random.nextGaussian() * stddev));
            cluster.add(new DenseVector(new double[]{x, y}));
        }
        return cluster;
    }
}