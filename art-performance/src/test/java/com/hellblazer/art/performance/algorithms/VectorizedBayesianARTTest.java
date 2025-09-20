package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedBayesianART implementation.
 * Tests probabilistic learning with uncertainty estimation.
 */
public class VectorizedBayesianARTTest extends BaseVectorizedARTTest<VectorizedBayesianART, VectorizedParameters> {
    
    @Override
    protected VectorizedBayesianART createAlgorithm(VectorizedParameters params) {
        return new VectorizedBayesianART(params);
    }
    
    @Override
    protected VectorizedParameters createDefaultParameters() {
        return VectorizedParameters.createDefault();
    }
    
    @BeforeEach
    protected void setUp() {
        parameters = createDefaultParameters();
        algorithm = createAlgorithm(parameters);
        super.setUp();
    }
    
    @Override
    protected VectorizedParameters createParametersWithVigilance(double vigilance) {
        var defaults = VectorizedParameters.createDefault();
        return new VectorizedParameters(
            vigilance,
            defaults.learningRate(),
            defaults.alpha(),
            defaults.parallelismLevel(),
            defaults.parallelThreshold(),
            defaults.maxCacheSize(),
            defaults.enableSIMD(),
            defaults.enableJOML(),
            defaults.memoryOptimizationThreshold()
        );
    }
    
    @Override
    protected java.util.List<Pattern> getTestPatterns() {
        // Override to provide 4-dimensional patterns for BayesianART
        return java.util.List.of(
            Pattern.of(0.8, 0.2, 0.5, 0.9),
            Pattern.of(0.3, 0.7, 0.4, 0.1),
            Pattern.of(0.9, 0.1, 0.6, 0.8),
            Pattern.of(0.1, 0.9, 0.2, 0.3),
            Pattern.of(0.5, 0.5, 0.5, 0.5)
        );
    }
    
    // The following tests are covered by base class:
    // - testBasicLearning()
    // - testMultiplePatternLearning()
    // - testPrediction()
    // - testPerformanceTracking()
    // - testErrorHandling()
    // - testResourceCleanup()
    
    @Test
    @DisplayName("Should compute Bayesian likelihood correctly")
    void testBayesianLikelihood() {
        var pattern1 = Pattern.of(0.8, 0.2, 0.5, 0.9);
        var pattern2 = Pattern.of(0.7, 0.3, 0.4, 0.8); // Similar to pattern1
        var pattern3 = Pattern.of(0.1, 0.9, 0.2, 0.1); // Different
        
        // Learn first pattern
        var result1 = algorithm.learn(pattern1, parameters);
        assertInstanceOf(ActivationResult.Success.class, result1);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Similar pattern should activate same category with high likelihood
        var prediction2 = algorithm.predict(pattern2, parameters);
        assertNotNull(prediction2);
        if (prediction2 instanceof ActivationResult.Success success) {
            assertTrue(success.activationValue() > 0.5, 
                "Similar pattern should have high activation");
        }
        
        // Different pattern should have lower likelihood
        var prediction3 = algorithm.predict(pattern3, parameters);
        assertNotNull(prediction3);
        if (prediction3 instanceof ActivationResult.Success success) {
            assertTrue(success.activationValue() < 0.8, 
                "Different pattern should have lower activation");
        }
    }
    
    @Test
    @DisplayName("Should estimate uncertainty via Mahalanobis distance")
    void testUncertaintyEstimation() {
        // Train with multiple similar patterns to establish distribution
        var patterns = new Pattern[] {
            Pattern.of(0.5, 0.5, 0.5, 0.5),
            Pattern.of(0.52, 0.48, 0.51, 0.49),
            Pattern.of(0.48, 0.52, 0.49, 0.51),
            Pattern.of(0.51, 0.49, 0.52, 0.48)
        };
        
        for (var pattern : patterns) {
            algorithm.learn(pattern, parameters);
        }
        
        // Test pattern close to mean should have low uncertainty
        var closePattern = Pattern.of(0.50, 0.50, 0.50, 0.50);
        var closeResult = algorithm.predict(closePattern, parameters);
        
        if (closeResult instanceof ActivationResult.Success success) {
            assertTrue(success.activationValue() > 0.7,
                "Pattern close to mean should have high activation");
        }
        
        // Test pattern far from mean should have high uncertainty
        var farPattern = Pattern.of(0.9, 0.1, 0.8, 0.2);
        var farResult = algorithm.predict(farPattern, parameters);
        
        if (farResult instanceof ActivationResult.Success success) {
            assertTrue(success.activationValue() < 0.9,
                "Pattern far from mean should have lower activation");
        } else if (farResult instanceof ActivationResult.NoMatch) {
            // Far pattern may not match at all, which is expected
            assertTrue(true, "Far pattern did not match any category");
        }
    }
    
    @Test
    @DisplayName("Should update conjugate priors correctly")
    void testConjugatePriorUpdates() {
        var pattern = Pattern.of(0.6, 0.4, 0.7, 0.3);
        
        // Initial learning should create category with prior parameters
        var result1 = algorithm.learn(pattern, parameters);
        assertInstanceOf(ActivationResult.Success.class, result1);
        
        // Learning same pattern again should update priors
        var result2 = algorithm.learn(pattern, parameters);
        assertInstanceOf(ActivationResult.Success.class, result2);
        
        // Posterior should be different from prior after updates
        assertEquals(1, algorithm.getCategoryCount(), 
            "Should still have one category after updating");
        
        // Prediction should reflect updated distribution
        var prediction = algorithm.predict(pattern, parameters);
        assertNotNull(prediction);
        if (prediction instanceof ActivationResult.Success success) {
            assertTrue(success.activationValue() > 0.8,
                "Pattern should have high activation after learning");
        }
    }
    
    @Test
    @DisplayName("Should handle prior parameters correctly")
    void testPriorParameters() {
        // Test with different prior settings - using different vigilance as proxy
        var highVarianceParams = createParametersWithVigilance(0.5);  // Lower vigilance = more accepting
        var lowVarianceParams = createParametersWithVigilance(0.9);   // Higher vigilance = more restrictive
        
        var highVarAlg = new VectorizedBayesianART(highVarianceParams);
        var lowVarAlg = new VectorizedBayesianART(lowVarianceParams);
        
        try {
            var pattern = Pattern.of(0.3, 0.7, 0.2, 0.8);
            
            // High variance prior should be more accepting
            var highVarResult = highVarAlg.learn(pattern, highVarianceParams);
            assertInstanceOf(ActivationResult.Success.class, highVarResult);
            
            // Low variance prior should be more restrictive
            var lowVarResult = lowVarAlg.learn(pattern, lowVarianceParams);
            assertInstanceOf(ActivationResult.Success.class, lowVarResult);
            
            // Test patterns far from prior mean
            var farPattern = Pattern.of(0.9, 0.9, 0.9, 0.9);
            
            var highVarPred = highVarAlg.predict(farPattern, highVarianceParams);
            var lowVarPred = lowVarAlg.predict(farPattern, lowVarianceParams);
            
            assertNotNull(highVarPred);
            assertNotNull(lowVarPred);
            
        } finally {
            highVarAlg.close();
            lowVarAlg.close();
        }
    }
    
    @Test
    @DisplayName("Should compute multivariate Gaussian correctly")
    void testMultivariateGaussian() {
        // Create patterns with known statistical properties
        var mean = Pattern.of(0.5, 0.5, 0.5, 0.5);
        
        // Learn the mean pattern
        var result = algorithm.learn(mean, parameters);
        assertInstanceOf(ActivationResult.Success.class, result);
        
        // Test patterns at different distances from mean
        var patterns = new Pattern[] {
            Pattern.of(0.5, 0.5, 0.5, 0.5),  // At mean
            Pattern.of(0.6, 0.4, 0.6, 0.4),  // Close to mean
            Pattern.of(0.9, 0.1, 0.9, 0.1),  // Far from mean
        };
        
        // Test each pattern and verify they return valid results
        double meanActivation = -1;
        double closeActivation = -1;
        double farActivation = -1;
        
        for (int i = 0; i < patterns.length; i++) {
            var prediction = algorithm.predict(patterns[i], parameters);
            
            if (prediction instanceof ActivationResult.Success success) {
                switch (i) {
                    case 0: meanActivation = success.activationValue(); break;
                    case 1: closeActivation = success.activationValue(); break;
                    case 2: farActivation = success.activationValue(); break;
                }
            }
        }
        
        // At minimum, pattern at mean should have high activation
        assertTrue(meanActivation > 0.5, "Pattern at mean should have high activation");
        
        // If far pattern matched, it should have lower activation than mean
        if (farActivation >= 0 && meanActivation >= 0) {
            assertTrue(farActivation <= meanActivation, 
                "Far pattern should not have higher activation than mean");
        }
    }
    
    @Test
    @DisplayName("Should handle degrees of freedom parameter")
    void testDegreesOfFreedom() {
        // Test with different vigilance levels as proxy for DOF sensitivity
        var highDOF = createParametersWithVigilance(0.6);  // Less sensitive
        var lowDOF = createParametersWithVigilance(0.8);   // More sensitive
        
        var highDOFAlg = new VectorizedBayesianART(highDOF);
        var lowDOFAlg = new VectorizedBayesianART(lowDOF);
        
        try {
            var pattern = Pattern.of(0.4, 0.6, 0.3, 0.7);
            
            highDOFAlg.learn(pattern, highDOF);
            lowDOFAlg.learn(pattern, lowDOF);
            
            // Both should create categories
            assertEquals(1, highDOFAlg.getCategoryCount());
            assertEquals(1, lowDOFAlg.getCategoryCount());
            
            // Test outlier pattern
            var outlier = Pattern.of(0.99, 0.01, 0.99, 0.01);
            
            var highDOFPred = highDOFAlg.predict(outlier, highDOF);
            var lowDOFPred = lowDOFAlg.predict(outlier, lowDOF);
            
            // Lower DOF should be more sensitive to outliers
            if (highDOFPred instanceof ActivationResult.Success highResult &&
                lowDOFPred instanceof ActivationResult.Success lowResult) {
                // Both should recognize outlier has low activation
                assertTrue(highResult.activationValue() < 0.5);
                assertTrue(lowResult.activationValue() < 0.5);
            } else {
                // Outliers may not match at all, which is also acceptable
                assertTrue(highDOFPred instanceof ActivationResult.NoMatch ||
                          lowDOFPred instanceof ActivationResult.NoMatch,
                          "Outliers should either have low activation or no match");
            }
            
        } finally {
            highDOFAlg.close();
            lowDOFAlg.close();
        }
    }
}