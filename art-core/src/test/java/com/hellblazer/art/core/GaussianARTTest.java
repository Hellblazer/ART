package com.hellblazer.art.core;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

/**
 * Comprehensive test suite for GaussianART implementation.
 * Tests probabilistic learning, Gaussian statistics, and incremental updates.
 */
class GaussianARTTest {
    
    private GaussianART gaussianART;
    private GaussianParameters defaultParams;
    private GaussianParameters lowVigilanceParams;
    private GaussianParameters highVigilanceParams;
    private GaussianParameters tightParams;
    
    @BeforeEach
    void setUp() {
        gaussianART = new GaussianART();
        
        // Default parameters (vigilance 0.1, sigma [0.5, 0.5])
        defaultParams = GaussianParameters.of(0.1, new double[]{0.5, 0.5});
        
        // Low vigilance for easy clustering
        lowVigilanceParams = GaussianParameters.of(0.001, new double[]{0.3, 0.3});
        
        // High vigilance for strict clustering  
        highVigilanceParams = GaussianParameters.of(0.5, new double[]{0.2, 0.2});
        
        // Tight initial sigma for precise clusters
        tightParams = GaussianParameters.of(0.01, new double[]{0.1, 0.1});
    }
    
    @Test
    @DisplayName("GaussianART should start with zero categories")
    void testInitialState() {
        assertEquals(0, gaussianART.getCategoryCount());
        assertTrue(gaussianART.getCategories().isEmpty());
    }
    
    @Test
    @DisplayName("First input should create single category")
    void testFirstInput() {
        var input = Pattern.of(0.3, 0.7);
        var result = gaussianART.stepFit(input, defaultParams);
        
        // Should create first category
        assertEquals(1, gaussianART.getCategoryCount());
        
        // Should be success with correct index
        assertInstanceOf(ActivationResult.Success.class, result);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), 1e-10); // Perfect match for first category
        
        // Check that category weight matches input
        var weight = gaussianART.getCategory(0);
        assertInstanceOf(GaussianWeight.class, weight);
        var gaussianWeight = (GaussianWeight) weight;
        
        assertArrayEquals(new double[]{0.3, 0.7}, gaussianWeight.mean(), 1e-10);
        assertEquals(1L, gaussianWeight.sampleCount());
    }
    
    @Test
    @DisplayName("Similar inputs should be classified to same category")
    void testSimilarInputsClustering() {
        // First input
        var input1 = Pattern.of(0.5, 0.5);
        var result1 = gaussianART.stepFit(input1, lowVigilanceParams);
        
        // Similar input should go to same category
        var input2 = Pattern.of(0.52, 0.48);
        var result2 = gaussianART.stepFit(input2, lowVigilanceParams);
        
        // Should still have only 1 category
        assertEquals(1, gaussianART.getCategoryCount());
        
        // Both should be classified to category 0
        var success1 = (ActivationResult.Success) result1;
        var success2 = (ActivationResult.Success) result2;
        assertEquals(0, success1.categoryIndex());
        assertEquals(0, success2.categoryIndex());
        
        // Second input should have lower activation (not perfect match)
        assertTrue(success2.activationValue() < success1.activationValue());
        
        // Weight should be updated with both samples
        var weight = (GaussianWeight) gaussianART.getCategory(0);
        assertEquals(2L, weight.sampleCount());
        
        // Mean should be average of two inputs
        var expectedMean = new double[]{(0.5 + 0.52) / 2, (0.5 + 0.48) / 2};
        assertArrayEquals(expectedMean, weight.mean(), 1e-10);
    }
    
    @Test
    @DisplayName("Dissimilar inputs should create separate categories")
    void testDissimilarInputsSeparation() {
        // First input in one region
        var input1 = Pattern.of(0.2, 0.2);
        gaussianART.stepFit(input1, highVigilanceParams);
        
        // Very different input should create new category
        var input2 = Pattern.of(0.8, 0.8);
        var result2 = gaussianART.stepFit(input2, highVigilanceParams);
        
        // Should have 2 categories now
        assertEquals(2, gaussianART.getCategoryCount());
        
        var success2 = (ActivationResult.Success) result2;
        assertEquals(1, success2.categoryIndex()); // Second category
        
        // Verify both categories maintain their original means
        var weight1 = (GaussianWeight) gaussianART.getCategory(0);
        var weight2 = (GaussianWeight) gaussianART.getCategory(1);
        
        assertArrayEquals(new double[]{0.2, 0.2}, weight1.mean(), 1e-10);
        assertArrayEquals(new double[]{0.8, 0.8}, weight2.mean(), 1e-10);
        assertEquals(1L, weight1.sampleCount());
        assertEquals(1L, weight2.sampleCount());
    }
    
    @Test
    @DisplayName("Winner-take-all competition should select highest probability category")
    void testActivationBasedCompetition() {
        // Create two well-separated categories
        var input1 = Pattern.of(0.1, 0.1);
        var input2 = Pattern.of(0.9, 0.9);
        gaussianART.stepFit(input1, defaultParams);
        gaussianART.stepFit(input2, defaultParams);
        
        // Input closer to first category should activate it
        var testInput = Pattern.of(0.15, 0.12);
        var result = gaussianART.stepFit(testInput, defaultParams);
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Should select category 0
        
        // Category 0 should now have 2 samples
        var weight0 = (GaussianWeight) gaussianART.getCategory(0);
        assertEquals(2L, weight0.sampleCount());
    }
    
    @Test
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceControl() {
        var input1 = Pattern.of(0.5, 0.5);
        
        // Low vigilance - should accept similar inputs
        gaussianART.stepFit(input1, lowVigilanceParams);
        var input2 = Pattern.of(0.6, 0.4);
        gaussianART.stepFit(input2, lowVigilanceParams);
        assertEquals(1, gaussianART.getCategoryCount());
        
        // Reset and try with high vigilance (very high threshold)
        gaussianART.clear();
        var veryHighVigilanceParams = GaussianParameters.of(0.9, new double[]{0.1, 0.1});
        gaussianART.stepFit(input1, veryHighVigilanceParams);
        var input3 = Pattern.of(0.8, 0.2); // More different from input1
        gaussianART.stepFit(input3, veryHighVigilanceParams);
        assertEquals(2, gaussianART.getCategoryCount()); // Should create separate categories
    }
    
    @Test
    @DisplayName("Incremental learning should update Gaussian statistics")
    void testIncrementalLearning() {
        // Use tight parameters with small initial sigma
        var smallSigmaParams = GaussianParameters.of(0.1, new double[]{0.1, 0.1});
        
        // Start with one input
        var input1 = Pattern.of(0.4, 0.6);
        gaussianART.stepFit(input1, smallSigmaParams);
        
        var weight1 = (GaussianWeight) gaussianART.getCategory(0);
        var initialMean = weight1.mean().clone();
        var initialSigma = weight1.sigma().clone();
        assertEquals(1L, weight1.sampleCount());
        
        // Add significantly different input - should update statistics
        var input2 = Pattern.of(0.8, 0.2);  // Very different from first input
        gaussianART.stepFit(input2, smallSigmaParams);
        
        var weight2 = (GaussianWeight) gaussianART.getCategory(0);
        assertEquals(2L, weight2.sampleCount());
        
        // Mean should change (moving average)
        assertFalse(java.util.Arrays.equals(initialMean, weight2.mean()));
        
        // Sigma should increase due to variance from two different samples
        assertTrue(weight2.sigma()[0] > initialSigma[0] && weight2.sigma()[1] > initialSigma[1]);
    }
    
    @Test
    @DisplayName("Mathematical properties should be preserved")
    void testMathematicalProperties() {
        var input = Pattern.of(0.3, 0.7);
        gaussianART.stepFit(input, defaultParams);
        
        var weight = (GaussianWeight) gaussianART.getCategory(0);
        
        // Sigma values should be positive
        for (double sigma : weight.sigma()) {
            assertTrue(sigma > 0, "Sigma values must be positive");
        }
        
        // Inverse sigma should be reciprocal
        for (int i = 0; i < weight.sigma().length; i++) {
            assertEquals(1.0 / weight.sigma()[i], weight.invSigma()[i], 1e-10);
        }
        
        // Square root determinant should be valid
        double expectedSqrtDet = 1.0;
        for (double sigma : weight.sigma()) {
            expectedSqrtDet *= sigma;
        }
        assertEquals(Math.sqrt(expectedSqrtDet), weight.sqrtDetSigma(), 1e-10);
    }
    
    @Test
    @DisplayName("Edge cases should be handled correctly")
    void testEdgeCases() {
        // Very small inputs
        var smallInput = Pattern.of(0.001, 0.001);
        var result1 = gaussianART.stepFit(smallInput, defaultParams);
        assertInstanceOf(ActivationResult.Success.class, result1);
        
        // Very large inputs (within [0,1] range)
        var largeInput = Pattern.of(0.999, 0.999);
        var result2 = gaussianART.stepFit(largeInput, defaultParams);
        assertInstanceOf(ActivationResult.Success.class, result2);
        
        // Should have created separate categories due to distance
        assertEquals(2, gaussianART.getCategoryCount());
    }
    
    @Test
    @DisplayName("Parameter validation should work correctly")
    void testParameterValidation() {
        var input = Pattern.of(0.5, 0.5);
        
        // Add a category first to enable parameter validation
        gaussianART.stepFit(input, defaultParams);
        
        // Wrong parameter type should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            gaussianART.stepFit(Pattern.of(0.6, 0.4), "wrong_params");
        });
        
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> {
            gaussianART.stepFit(input, null);
        });
        
        // Null input should throw exception
        assertThrows(NullPointerException.class, () -> {
            gaussianART.stepFit(null, defaultParams);
        });
    }
    
    @Test
    @DisplayName("Dimension mismatch should throw exception")
    void testDimensionValidation() {
        // Create 2D parameters
        var params2D = GaussianParameters.of(0.1, new double[]{0.5, 0.5});
        
        // Try with 3D input - should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            var input3D = Pattern.of(0.3, 0.4, 0.5);
            gaussianART.stepFit(input3D, params2D);
        });
    }
    
    @Test
    @DisplayName("Multiple learning cycles should maintain consistency")
    void testMultipleLearnignCycles() {
        var inputs = new Pattern[]{
            Pattern.of(0.2, 0.3),
            Pattern.of(0.25, 0.35),
            Pattern.of(0.8, 0.7),
            Pattern.of(0.75, 0.72),
            Pattern.of(0.22, 0.32)
        };
        
        // Process all inputs
        for (var input : inputs) {
            gaussianART.stepFit(input, defaultParams);
        }
        
        // Should have created appropriate number of categories
        assertTrue(gaussianART.getCategoryCount() >= 1);
        assertTrue(gaussianART.getCategoryCount() <= inputs.length);
        
        // All categories should have valid statistics
        for (int i = 0; i < gaussianART.getCategoryCount(); i++) {
            var weight = (GaussianWeight) gaussianART.getCategory(i);
            assertTrue(weight.sampleCount() > 0);
            for (double sigma : weight.sigma()) {
                assertTrue(sigma > 0);
            }
        }
    }
    
    @Test
    @DisplayName("toString should provide meaningful representation")
    void testToString() {
        var toString = gaussianART.toString();
        assertTrue(toString.contains("GaussianART"));
        assertTrue(toString.contains("categories=0"));
        
        // Add a category and check again
        gaussianART.stepFit(Pattern.of(0.5, 0.5), defaultParams);
        toString = gaussianART.toString();
        assertTrue(toString.contains("categories=1"));
    }
    
    @Test
    @DisplayName("Probability density calculation should be mathematically correct")
    void testProbabilityDensityCalculation() {
        var input = Pattern.of(0.5, 0.5);
        var result = gaussianART.stepFit(input, tightParams);
        
        var success = (ActivationResult.Success) result;
        // First input should have activation 1.0 (perfect probability at center)
        assertEquals(1.0, success.activationValue(), 1e-10);
        
        // Second identical input should also have high probability
        var result2 = gaussianART.stepFit(input, tightParams);
        var success2 = (ActivationResult.Success) result2;
        assertTrue(success2.activationValue() > 0.5); // Should still be high probability
    }
}