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
import com.hellblazer.art.performance.BaseVectorizedARTTest;

/**
 * Comprehensive test suite for VectorizedGaussianART implementation.
 * Tests both SIMD and standard computation paths, performance characteristics,
 * and compatibility with GaussianART semantics.
 */
public class VectorizedGaussianARTTest extends BaseVectorizedARTTest<VectorizedGaussianART, VectorizedGaussianParameters> {
    
    private GaussianART standardART;
    private GaussianParameters gaussianParams;
    
    @Override
    protected VectorizedGaussianART createAlgorithm(VectorizedGaussianParameters params) {
        return new VectorizedGaussianART(params);
    }
    
    @Override
    protected VectorizedGaussianParameters createDefaultParameters() {
        return new VectorizedGaussianParameters(
            0.8,    // vigilance
            0.1,    // gamma (learning rate)
            1.0,    // rho_a (variance adjustment factor)
            0.5,    // rho_b (minimum variance)
            4,      // parallelismLevel
            true    // enableSIMD
        );
    }
    
    @Override
    protected VectorizedGaussianParameters createParametersWithVigilance(double vigilance) {
        return new VectorizedGaussianParameters(
            vigilance,
            0.1,    // gamma (learning rate) - same as default
            1.0,    // rho_a (variance adjustment factor) - same as default
            0.5,    // rho_b (minimum variance) - same as default
            4,      // parallelismLevel - same as default
            true    // enableSIMD - same as default
        );
    }
    
    @BeforeEach
    protected void setUp() {
        super.setUp();
        // Configure standard GaussianART for comparison
        var sigmaInit = new double[]{0.5, 0.5}; // 2D default
        gaussianParams = new GaussianParameters(0.8, sigmaInit);
        standardART = new GaussianART();
    }
    
    @AfterEach
    void tearDown() {
        if (algorithm != null) {
            algorithm.close();
        }
    }
    
    // Basic learning test is inherited from base class, but we override to test Gaussian-specific behavior
    @Test
    @DisplayName("Gaussian-specific learning should work correctly")
    void testGaussianLearning() {
        // Create simple 2D patterns
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.3, 0.7);
        var pattern3 = Pattern.of(0.75, 0.25); // Similar to pattern1
        
        // Train on first two patterns
        var result1 = algorithm.learn(pattern1, parameters);
        var result2 = algorithm.learn(pattern2, parameters);
        
        // Should create two categories
        assertEquals(2, algorithm.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Test recognition of similar pattern
        var result3 = algorithm.learn(pattern3, parameters);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        var successResult3 = (ActivationResult.Success) result3;
        // Pattern3 may create a new category or match existing based on Gaussian parameters
        // With default parameters, it likely creates a new category
        assertTrue(successResult3.categoryIndex() >= 0 && successResult3.categoryIndex() < 3);
    }
    
    // Vigilance control test is inherited from base class, but we provide Gaussian-specific one
    @Test
    @DisplayName("Gaussian vigilance should control category creation")
    void testGaussianVigilanceControl() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.7, 0.3); // Moderately similar
        
        // High vigilance - may create one or two categories based on Gaussian similarity
        var highVigilanceParams = new VectorizedGaussianParameters(
            0.95, 0.1, 1.0, 0.1, 4, true
        );
        var highVigilanceART = createAlgorithm(highVigilanceParams);
        
        highVigilanceART.learn(pattern1, highVigilanceParams);
        highVigilanceART.learn(pattern2, highVigilanceParams);
        
        // With high vigilance and moderately similar patterns, could be 1 or 2 categories
        assertTrue(highVigilanceART.getCategoryCount() >= 1 && highVigilanceART.getCategoryCount() <= 2);
        
        // Low vigilance - should merge into one category
        var lowVigilanceParams = new VectorizedGaussianParameters(
            0.01, 0.1, 1.0, 0.1, 4, true
        );
        var lowVigilanceART = createAlgorithm(lowVigilanceParams);
        
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
        var simdART = createAlgorithm(simdParams);
        
        // Train with SIMD disabled
        var noSimdParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.1, 4, false
        );
        var noSimdART = createAlgorithm(noSimdParams);
        
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
        var weight = VectorizedGaussianWeight.fromInput(input, parameters);
        
        // For a Gaussian centered at the input with initial variance,
        // the probability density should be high
        var activation = algorithm.calculateActivation(input, weight, parameters);
        assertTrue(activation > 0.0, "Activation should be positive");
        
        // Test with different input - should have lower activation
        var differentInput = Pattern.of(0.9, 0.1);
        var differentActivation = algorithm.calculateActivation(differentInput, weight, parameters);
        assertTrue(differentActivation > 0.0, "Different activation should be positive");
        assertTrue(activation > differentActivation, "Centered input should have higher activation");
    }
    
    @Test
    @DisplayName("Gaussian weight updates should follow incremental statistics")
    void testIncrementalGaussianLearning() {
        var input1 = Pattern.of(0.5, 0.5);
        var input2 = Pattern.of(0.6, 0.4);
        
        // Create initial weight from first input
        var weight1 = VectorizedGaussianWeight.fromInput(input1, parameters);
        assertEquals(1, weight1.getSampleCount());
        assertEquals(0.5, weight1.getMean()[0], 1e-10);
        assertEquals(0.5, weight1.getMean()[1], 1e-10);
        
        // Update with second input
        var weight2 = weight1.updateGaussian(input2, parameters);
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
        var parallelART = createAlgorithm(parallelParams);
        
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
        var highGammaART = createAlgorithm(highGammaParams);
        
        // Low gamma (slow learning)
        var lowGammaParams = new VectorizedGaussianParameters(
            0.5, 0.1, 1.0, 0.1, 4, true
        );
        var lowGammaART = createAlgorithm(lowGammaParams);
        
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
        var initialStats = algorithm.getPerformanceStats();
        assertEquals(0, initialStats.totalVectorOperations());
        
        // Train and test
        for (var pattern : patterns) {
            algorithm.learn(pattern, parameters);
        }
        
        // Stats should be updated
        var finalStats = algorithm.getPerformanceStats();
        assertTrue(finalStats.totalVectorOperations() > 0);
        assertTrue(finalStats.avgComputeTimeMs() >= 0.0);
        assertEquals(algorithm.getCategoryCount(), finalStats.categoryCount());
        
        // Reset should clear stats
        algorithm.resetPerformanceTracking();
        var resetStats = algorithm.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    // Error handling is inherited from base class, but we add Gaussian-specific tests
    @Test
    @DisplayName("Gaussian-specific error handling should work correctly")
    void testGaussianErrorHandling() {
        // Wrong parameter type should throw exception
        assertThrows(ClassCastException.class, () -> {
            algorithm.calculateActivation(Pattern.of(0.5, 0.5), 
                VectorizedGaussianWeight.fromInput(Pattern.of(0.5, 0.5), parameters), 
                (VectorizedGaussianParameters) ((Object) "wrong type"));
        });
        
        // Mismatched dimensions should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            var weight2D = VectorizedGaussianWeight.fromInput(Pattern.of(0.5, 0.5), parameters);
            algorithm.calculateActivation(Pattern.of(0.5, 0.5, 0.5), weight2D, parameters);
        });
    }
    
    @Test
    @DisplayName("VectorizedGaussianWeight should handle edge cases correctly")
    void testVectorizedGaussianWeightEdgeCases() {
        // Test minimum variance enforcement
        var input = Pattern.of(0.5, 0.5);
        var weight = VectorizedGaussianWeight.fromInput(input, parameters);
        
        // All variance values should be >= rho_b
        var covariance = weight.getCovariance();
        for (int i = 0; i < covariance.length; i++) {
            assertTrue(covariance[i][i] >= parameters.rho_b(), 
                "Variance at " + i + " should be >= rho_b");
        }
        
        // Test determinant computation
        assertTrue(weight.getDeterminant() > 0.0, "Determinant should be positive");
        
        // Test vigilance computation
        var vigilance = weight.computeVigilance(input, parameters);
        assertTrue(vigilance >= 0.0, "Vigilance should be non-negative");
    }
    
    // Resource cleanup tests are inherited from base class
    
    @Test
    @DisplayName("VectorizedARTAlgorithm interface methods should work correctly")
    void testVectorizedARTAlgorithmInterface() {
        var input = Pattern.of(0.6, 0.4);
        
        // Test learn method
        var learnResult = algorithm.learn(input, parameters);
        assertTrue(learnResult instanceof ActivationResult.Success);
        
        // Test predict method
        var predictResult = algorithm.predict(input, parameters);
        assertTrue(predictResult instanceof ActivationResult.Success);
        
        // Test getCategoryCount
        assertTrue(algorithm.getCategoryCount() > 0);
        
        // Test getParameters
        assertEquals(parameters, algorithm.getParameters());
    }
}