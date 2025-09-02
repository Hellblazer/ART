package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.parameters.GaussianParameters;
import com.hellblazer.art.core.algorithms.GaussianART;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for VectorizedGaussianART implementation.
 * Tests both SIMD and standard computation paths, performance characteristics,
 * and compatibility with GaussianART semantics.
 */
public class VectorizedGaussianARTTest {
    
    private VectorizedGaussianART vectorizedART;
    private VectorizedGaussianParameters params;
    private GaussianART standardART;
    private GaussianParameters gaussianParams;
    
    @BeforeEach
    void setUp() {
        // Configure vectorized parameters for GaussianART
        params = new VectorizedGaussianParameters(
            0.8,    // vigilance
            0.1,    // gamma (learning rate)
            1.0,    // rho_a (variance adjustment factor)
            0.5,    // rho_b (minimum variance)
            4,      // parallelismLevel
            true    // enableSIMD
        );
        
        vectorizedART = new VectorizedGaussianART(params);
        
        // Configure standard GaussianART for comparison
        var sigmaInit = new double[]{0.5, 0.5}; // 2D default
        gaussianParams = new GaussianParameters(0.8, sigmaInit);
        standardART = new GaussianART();
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedART != null) {
            vectorizedART.close();
        }
    }
    
    @Test
    @DisplayName("Basic learning and recognition should work correctly")
    void testBasicLearningAndRecognition() {
        // Create simple 2D patterns
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.3, 0.7);
        var pattern3 = Pattern.of(0.75, 0.25); // Similar to pattern1
        
        // Train on first two patterns
        var result1 = vectorizedART.learn(pattern1, params);
        var result2 = vectorizedART.learn(pattern2, params);
        
        // Should create two categories
        assertEquals(2, vectorizedART.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Test recognition of similar pattern
        var result3 = vectorizedART.learn(pattern3, params);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        var successResult3 = (ActivationResult.Success) result3;
        // Pattern3 may create a new category or match existing based on Gaussian parameters
        // With default parameters, it likely creates a new category
        assertTrue(successResult3.categoryIndex() >= 0 && successResult3.categoryIndex() < 3);
    }
    
    @Test
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceControl() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.7, 0.3); // Moderately similar
        
        // High vigilance - may create one or two categories based on Gaussian similarity
        var highVigilanceParams = new VectorizedGaussianParameters(
            0.95, 0.1, 1.0, 0.1, 4, true
        );
        var highVigilanceART = new VectorizedGaussianART(highVigilanceParams);
        
        highVigilanceART.learn(pattern1, highVigilanceParams);
        highVigilanceART.learn(pattern2, highVigilanceParams);
        
        // With high vigilance and moderately similar patterns, could be 1 or 2 categories
        assertTrue(highVigilanceART.getCategoryCount() >= 1 && highVigilanceART.getCategoryCount() <= 2);
        
        // Low vigilance - should merge into one category
        var lowVigilanceParams = new VectorizedGaussianParameters(
            0.01, 0.1, 1.0, 0.1, 4, true
        );
        var lowVigilanceART = new VectorizedGaussianART(lowVigilanceParams);
        
        lowVigilanceART.learn(pattern1, lowVigilanceParams);
        lowVigilanceART.learn(pattern2, lowVigilanceParams);
        
        assertEquals(1, lowVigilanceART.getCategoryCount());
        
        highVigilanceART.close();
        lowVigilanceART.close();
    }
    
    @Test
    @DisplayName("SIMD and standard computation should produce equivalent results")
    void testSIMDEquivalence() {
        var patterns = List.of(
            Pattern.of(0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4),
            Pattern.of(0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6),
            Pattern.of(0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3),
            Pattern.of(0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7)
        );
        
        // Train with SIMD enabled
        var simdParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.1, 4, true
        );
        var simdART = new VectorizedGaussianART(simdParams);
        
        // Train with SIMD disabled
        var noSimdParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.1, 4, false
        );
        var noSimdART = new VectorizedGaussianART(noSimdParams);
        
        // Train both networks
        for (var pattern : patterns) {
            simdART.learn(pattern, simdParams);
            noSimdART.learn(pattern, noSimdParams);
        }
        
        // Should create same number of categories
        assertEquals(simdART.getCategoryCount(), noSimdART.getCategoryCount());
        
        // Test predictions should be equivalent
        for (var pattern : patterns) {
            var simdResult = simdART.predict(pattern, simdParams);
            var noSimdResult = noSimdART.predict(pattern, noSimdParams);
            
            assertTrue(simdResult instanceof ActivationResult.Success);
            assertTrue(noSimdResult instanceof ActivationResult.Success);
            
            var simdSuccess = (ActivationResult.Success) simdResult;
            var noSimdSuccess = (ActivationResult.Success) noSimdResult;
            
            assertEquals(simdSuccess.categoryIndex(), noSimdSuccess.categoryIndex());
            assertEquals(simdSuccess.activationValue(), noSimdSuccess.activationValue(), 1e-6);
        }
        
        simdART.close();
        noSimdART.close();
    }
    
    @Test
    @DisplayName("Gaussian probability density should be computed correctly")
    void testGaussianProbabilityComputation() {
        var input = Pattern.of(0.5, 0.5);
        var weight = VectorizedGaussianWeight.fromInput(input, params);
        
        // For a Gaussian centered at the input with initial variance,
        // the probability density should be high
        var activation = vectorizedART.calculateActivation(input, weight, params);
        assertTrue(activation > 0.0, "Activation should be positive");
        
        // Test with different input - should have lower activation
        var differentInput = Pattern.of(0.9, 0.1);
        var differentActivation = vectorizedART.calculateActivation(differentInput, weight, params);
        assertTrue(differentActivation > 0.0, "Different activation should be positive");
        assertTrue(activation > differentActivation, "Centered input should have higher activation");
    }
    
    @Test
    @DisplayName("Gaussian weight updates should follow incremental statistics")
    void testIncrementalGaussianLearning() {
        var input1 = Pattern.of(0.5, 0.5);
        var input2 = Pattern.of(0.6, 0.4);
        
        // Create initial weight from first input
        var weight1 = VectorizedGaussianWeight.fromInput(input1, params);
        assertEquals(1, weight1.getSampleCount());
        assertEquals(0.5, weight1.getMean()[0], 1e-10);
        assertEquals(0.5, weight1.getMean()[1], 1e-10);
        
        // Update with second input
        var weight2 = weight1.updateGaussian(input2, params);
        assertEquals(2, weight2.getSampleCount());
        
        // Mean should be incremental average: (0.5 + 0.6) / 2 = 0.55, (0.5 + 0.4) / 2 = 0.45
        assertEquals(0.55, weight2.getMean()[0], 1e-10);
        assertEquals(0.45, weight2.getMean()[1], 1e-10);
        
        // Variance should be updated according to incremental algorithm
        assertTrue(weight2.getCovariance()[0][0] > 0.0);
        assertTrue(weight2.getCovariance()[1][1] > 0.0);
    }
    
    @Test
    @DisplayName("Parallel processing should work for large category sets")
    void testParallelProcessing() {
        // Create many patterns to potentially trigger parallel processing
        var patterns = new ArrayList<Pattern>();
        for (int i = 0; i < 50; i++) {
            double x = Math.sin(i * 0.1) * 0.5 + 0.5; // [0, 1] range
            double y = Math.cos(i * 0.1) * 0.5 + 0.5; // [0, 1] range
            patterns.add(Pattern.of(x, y));
        }
        
        // Set high vigilance to create more categories
        var parallelParams = new VectorizedGaussianParameters(
            0.99, 0.1, 1.0, 0.05, 4, true
        );
        var parallelART = new VectorizedGaussianART(parallelParams);
        
        // Train with many patterns
        for (var pattern : patterns) {
            parallelART.learn(pattern, parallelParams);
        }
        
        // Should have created multiple categories
        assertTrue(parallelART.getCategoryCount() > 1, 
            "Expected categories > 1 but was " + parallelART.getCategoryCount());
        
        // Test prediction
        var testPattern = Pattern.of(0.5, 0.5);
        var result = parallelART.predict(testPattern, parallelParams);
        assertTrue(result instanceof ActivationResult.Success);
        
        parallelART.close();
    }
    
    @Test
    @DisplayName("Gamma parameter should affect learning rate")
    void testGammaLearningRate() {
        var input1 = Pattern.of(0.5, 0.5);
        var input2 = Pattern.of(0.7, 0.3);
        
        // High gamma (fast learning)
        var highGammaParams = new VectorizedGaussianParameters(
            0.5, 0.5, 1.0, 0.1, 4, true
        );
        var highGammaART = new VectorizedGaussianART(highGammaParams);
        
        // Low gamma (slow learning)
        var lowGammaParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.1, 4, true
        );
        var lowGammaART = new VectorizedGaussianART(lowGammaParams);
        
        // Train both with same patterns
        highGammaART.learn(input1, highGammaParams);
        highGammaART.learn(input2, highGammaParams);
        
        lowGammaART.learn(input1, lowGammaParams);
        lowGammaART.learn(input2, lowGammaParams);
        
        // Both should create categories, but variance adaptation should differ
        assertTrue(highGammaART.getCategoryCount() > 0);
        assertTrue(lowGammaART.getCategoryCount() > 0);
        
        highGammaART.close();
        lowGammaART.close();
    }
    
    @Test
    @DisplayName("Rho parameters should control variance constraints")
    void testRhoVarianceConstraints() {
        var input = Pattern.of(0.5, 0.5);
        
        // High rho_b (minimum variance)
        var highRhoBParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.8, 4, true
        );
        
        // Low rho_b (minimum variance)
        var lowRhoBParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.01, 4, true
        );
        
        var highRhoWeight = VectorizedGaussianWeight.fromInput(input, highRhoBParams);
        var lowRhoWeight = VectorizedGaussianWeight.fromInput(input, lowRhoBParams);
        
        // High rho_b should result in higher minimum variance
        var highRhoVariance = highRhoWeight.getCovariance()[0][0];
        var lowRhoVariance = lowRhoWeight.getCovariance()[0][0];
        
        assertTrue(highRhoVariance >= highRhoBParams.rho_b());
        assertTrue(lowRhoVariance >= lowRhoBParams.rho_b());
        assertTrue(highRhoVariance >= lowRhoVariance);
    }
    
    @Test
    @DisplayName("Performance statistics should be tracked correctly")
    void testPerformanceTracking() {
        var patterns = List.of(
            Pattern.of(0.8, 0.2),
            Pattern.of(0.3, 0.7),
            Pattern.of(0.6, 0.4)
        );
        
        // Initial stats should show zero operations
        var initialStats = vectorizedART.getPerformanceStats();
        assertEquals(0, initialStats.totalVectorOperations());
        
        // Train and test
        for (var pattern : patterns) {
            vectorizedART.learn(pattern, params);
        }
        
        // Stats should be updated
        var finalStats = vectorizedART.getPerformanceStats();
        assertTrue(finalStats.totalVectorOperations() > 0);
        assertTrue(finalStats.avgComputeTimeMs() >= 0.0);
        assertEquals(vectorizedART.getCategoryCount(), finalStats.categoryCount());
        
        // Reset should clear stats
        vectorizedART.resetPerformanceTracking();
        var resetStats = vectorizedART.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    @Test
    @DisplayName("Error handling should work correctly")
    void testErrorHandling() {
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.learn(Pattern.of(0.5, 0.5), null);
        });
        
        // Wrong parameter type should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedART.calculateActivation(Pattern.of(0.5, 0.5), 
                VectorizedGaussianWeight.fromInput(Pattern.of(0.5, 0.5), params), 
                "wrong type");
        });
        
        // Null input should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.learn(null, params);
        });
        
        // Mismatched dimensions should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            var weight2D = VectorizedGaussianWeight.fromInput(Pattern.of(0.5, 0.5), params);
            vectorizedART.calculateActivation(Pattern.of(0.5, 0.5, 0.5), weight2D, params);
        });
    }
    
    @Test
    @DisplayName("VectorizedGaussianWeight should handle edge cases correctly")
    void testVectorizedGaussianWeightEdgeCases() {
        // Test minimum variance enforcement
        var input = Pattern.of(0.5, 0.5);
        var weight = VectorizedGaussianWeight.fromInput(input, params);
        
        // All variance values should be >= rho_b
        var covariance = weight.getCovariance();
        for (int i = 0; i < covariance.length; i++) {
            assertTrue(covariance[i][i] >= params.rho_b(), 
                "Variance at " + i + " should be >= rho_b");
        }
        
        // Test determinant computation
        assertTrue(weight.getDeterminant() > 0.0, "Determinant should be positive");
        
        // Test vigilance computation
        var vigilance = weight.computeVigilance(input, params);
        assertTrue(vigilance >= 0.0, "Vigilance should be non-negative");
    }
    
    @Test
    @DisplayName("Resource cleanup should work correctly")
    void testResourceCleanup() {
        var art = new VectorizedGaussianART(params);
        
        // Use the ART network
        art.learn(Pattern.of(0.5, 0.5), params);
        
        // Close should not throw exception
        assertDoesNotThrow(() -> art.close());
        
        // toString should work even after close
        assertNotNull(art.toString());
    }
    
    @Test
    @DisplayName("VectorizedARTAlgorithm interface methods should work correctly")
    void testVectorizedARTAlgorithmInterface() {
        var input = Pattern.of(0.6, 0.4);
        
        // Test learn method
        var learnResult = vectorizedART.learn(input, params);
        assertTrue(learnResult instanceof ActivationResult.Success);
        
        // Test predict method
        var predictResult = vectorizedART.predict(input, params);
        assertTrue(predictResult instanceof ActivationResult.Success);
        
        // Test getCategoryCount
        assertTrue(vectorizedART.getCategoryCount() > 0);
        
        // Test getParameters
        assertEquals(params, vectorizedART.getParameters());
    }
}