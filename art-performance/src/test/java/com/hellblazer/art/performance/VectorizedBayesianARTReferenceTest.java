package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.results.BayesianActivationResult;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedBayesianART;

/**
 * Test suite for VectorizedBayesianART implementation with vectorized SIMD optimizations.
 * 
 * Tests probabilistic learning, uncertainty quantification, and Bayesian parameter updates
 * using high-performance vectorized operations.
 */
@DisplayName("Vectorized Bayesian ART Reference Tests")
class VectorizedBayesianARTReferenceTest {
    
    private VectorizedParameters referenceParams;
    private VectorizedBayesianART vectorizedBayesianART;
    
    @BeforeEach
    void setUp() {
        referenceParams = VectorizedParameters.createDefault()
            .withVigilance(0.8)
            .withLearningRate(0.1);
        
        vectorizedBayesianART = new VectorizedBayesianART(referenceParams);
    }
    
    @Test
    @DisplayName("Basic VectorizedBayesianART Instantiation")
    void testBayesianArtInstantiation() {
        assertNotNull(vectorizedBayesianART, "VectorizedBayesianART should be instantiated");
        assertNotNull(referenceParams, "Parameters should be configured");
        assertEquals(referenceParams, vectorizedBayesianART.getParameters());
    }
    
    @Test
    @DisplayName("Parameter Configuration Validation")
    void testParameterConfiguration() {
        assertTrue(referenceParams.vigilanceThreshold() > 0, "Vigilance threshold should be positive");
        assertTrue(referenceParams.learningRate() > 0, "Learning rate should be positive");
        assertTrue(referenceParams.alpha() > 0, "Alpha parameter should be positive");
        
        // These parameter ranges should be suitable for Bayesian operations
        assertTrue(referenceParams.vigilanceThreshold() <= 1.0, "Vigilance should not exceed 1.0");
        assertTrue(referenceParams.learningRate() <= 1.0, "Learning rate should not exceed 1.0");
    }
    
    @Test
    @DisplayName("Bayesian Learning with Probabilistic Patterns")
    void testBayesianLearning() {
        // Test patterns with probability-like values for Bayesian learning
        var probabilisticPatterns = new DenseVector[] {
            new DenseVector(new double[]{0.1, 0.9}),  // High certainty pattern
            new DenseVector(new double[]{0.5, 0.5}),  // Uncertain pattern  
            new DenseVector(new double[]{0.9, 0.1})   // High certainty opposite pattern
        };
        
        // Learn each pattern
        var results = new BayesianActivationResult[probabilisticPatterns.length];
        for (int i = 0; i < probabilisticPatterns.length; i++) {
            results[i] = (BayesianActivationResult) vectorizedBayesianART.learn(probabilisticPatterns[i], referenceParams);
            assertNotNull(results[i], "Learning result should not be null");
            assertTrue(results[i].categoryIndex() >= 0, "Category index should be valid");
        }
        
        // Verify categories were created
        assertTrue(vectorizedBayesianART.getCategoryCount() > 0, "Should have created categories");
        assertTrue(vectorizedBayesianART.getCategoryCount() <= probabilisticPatterns.length, 
                  "Should not exceed maximum possible categories");
    }
    
    @Test
    @DisplayName("Bayesian Prediction and Classification")
    void testBayesianPrediction() {
        // First train with some patterns
        var trainingPatterns = new DenseVector[] {
            new DenseVector(new double[]{0.2, 0.8}),
            new DenseVector(new double[]{0.8, 0.2})
        };
        
        for (var pattern : trainingPatterns) {
            vectorizedBayesianART.learn(pattern, referenceParams);
        }
        
        // Test prediction on similar patterns
        var testPatterns = new DenseVector[] {
            new DenseVector(new double[]{0.25, 0.75}),  // Similar to first training pattern
            new DenseVector(new double[]{0.75, 0.25})   // Similar to second training pattern
        };
        
        for (var testPattern : testPatterns) {
            var result = vectorizedBayesianART.predict(testPattern, referenceParams);
            assertNotNull(result, "Prediction result should not be null");
        }
    }
    
    @Test
    @DisplayName("Performance Metrics and Statistics")
    void testPerformanceMetrics() {
        // Train on several patterns to generate performance data
        var patterns = new DenseVector[] {
            new DenseVector(new double[]{0.1, 0.9, 0.5}),
            new DenseVector(new double[]{0.9, 0.1, 0.5}),
            new DenseVector(new double[]{0.5, 0.5, 0.1}),
            new DenseVector(new double[]{0.5, 0.5, 0.9})
        };
        
        for (var pattern : patterns) {
            vectorizedBayesianART.learn(pattern, referenceParams);
        }
        
        var stats = vectorizedBayesianART.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
        
        // Verify performance tracking
        assertTrue(stats.totalVectorOperations() >= 0, "Vector operations should be tracked");
        assertTrue(stats.totalParallelTasks() >= 0, "Parallel tasks should be tracked");
        assertTrue(stats.categoryCount() > 0, "Category count should be positive");
    }
    
    @Test
    @DisplayName("SIMD Vector Species Length")
    void testVectorSpeciesLength() {
        var speciesLength = vectorizedBayesianART.getVectorSpeciesLength();
        assertTrue(speciesLength > 0, "Vector species length should be positive");
        assertTrue(speciesLength <= 64, "Vector species length should be reasonable"); // Typical SIMD widths
    }
    
    @Test
    @DisplayName("Resource Management and Cleanup")
    void testResourceManagement() {
        // Test performance tracking reset
        vectorizedBayesianART.resetPerformanceTracking();
        var stats = vectorizedBayesianART.getPerformanceStats();
        assertEquals(0, stats.totalVectorOperations(), "Vector operations should be reset");
        
        // Test resource cleanup
        assertDoesNotThrow(() -> vectorizedBayesianART.close(), "Close should not throw exceptions");
    }
}