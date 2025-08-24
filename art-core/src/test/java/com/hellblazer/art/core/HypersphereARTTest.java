package com.hellblazer.art.core;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

/**
 * Comprehensive test suite for HypersphereART implementation.
 * Tests geometric learning, distance-based activation, and radius expansion.
 */
class HypersphereARTTest {
    
    private HypersphereART hypersphereART;
    private HypersphereParameters defaultParams;
    private HypersphereParameters smallRadiusParams;
    private HypersphereParameters largeRadiusParams;
    private HypersphereParameters adaptiveParams;
    
    @BeforeEach
    void setUp() {
        hypersphereART = new HypersphereART();
        
        // Default parameters (low vigilance for easy clustering, defaultRadius 0.5)
        defaultParams = HypersphereParameters.of(0.1, 0.5, false);
        
        // Small radius for tight clustering
        smallRadiusParams = HypersphereParameters.of(0.1, 0.1, false);
        
        // Large radius for loose clustering  
        largeRadiusParams = HypersphereParameters.of(0.1, 1.0, false);
        
        // Adaptive radius parameters
        adaptiveParams = HypersphereParameters.of(0.1, 0.3, true);
    }
    
    @Test
    @DisplayName("HypersphereART should start with zero categories")
    void testInitialState() {
        assertEquals(0, hypersphereART.getCategoryCount());
        assertTrue(hypersphereART.getCategories().isEmpty());
    }
    
    @Test
    @DisplayName("First input should create single hypersphere category")
    void testFirstInput() {
        var input = Pattern.of(0.3, 0.7);
        var result = hypersphereART.stepFit(input, defaultParams);
        
        // Should create first category
        assertEquals(1, hypersphereART.getCategoryCount());
        
        // Should be success with correct index
        assertInstanceOf(ActivationResult.Success.class, result);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), 1e-10); // Perfect match for first category
        
        // Check that hypersphere is centered at input with default radius
        var weight = hypersphereART.getCategory(0);
        assertInstanceOf(HypersphereWeight.class, weight);
        var hypersphereWeight = (HypersphereWeight) weight;
        
        assertArrayEquals(new double[]{0.3, 0.7}, hypersphereWeight.center(), 1e-10);
        assertEquals(defaultParams.defaultRadius(), hypersphereWeight.radius(), 1e-10);
    }
    
    @Test
    @DisplayName("Points within hypersphere radius should be classified to same category")
    void testPointsWithinRadius() {
        // First input creates hypersphere at (0.5, 0.5) with radius 0.5
        var input1 = Pattern.of(0.5, 0.5);
        var result1 = hypersphereART.stepFit(input1, defaultParams);
        
        // Point within radius should go to same category
        var input2 = Pattern.of(0.7, 0.6); // Distance ≈ 0.22, within radius 0.5
        var result2 = hypersphereART.stepFit(input2, defaultParams);
        
        // Should still have only 1 category
        assertEquals(1, hypersphereART.getCategoryCount());
        
        // Both should be classified to category 0
        var success1 = (ActivationResult.Success) result1;
        var success2 = (ActivationResult.Success) result2;
        assertEquals(0, success1.categoryIndex());
        assertEquals(0, success2.categoryIndex());
        
        // Second input should have lower activation (greater distance)
        assertTrue(success2.activationValue() < success1.activationValue());
        
        // Hypersphere center should remain unchanged
        var weight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertArrayEquals(new double[]{0.5, 0.5}, weight.center(), 1e-10);
    }
    
    @Test
    @DisplayName("Points outside hypersphere radius should create new categories")
    void testPointsOutsideRadius() {
        // First input with small radius
        var input1 = Pattern.of(0.2, 0.2);
        hypersphereART.stepFit(input1, smallRadiusParams);
        
        // Distant point should create new category
        var input2 = Pattern.of(0.8, 0.8); // Distance ≈ 0.85, outside radius 0.1
        var result2 = hypersphereART.stepFit(input2, smallRadiusParams);
        
        // Should have 2 categories now
        assertEquals(2, hypersphereART.getCategoryCount());
        
        var success2 = (ActivationResult.Success) result2;
        assertEquals(1, success2.categoryIndex()); // Second category
        
        // Verify both categories maintain their centers
        var weight1 = (HypersphereWeight) hypersphereART.getCategory(0);
        var weight2 = (HypersphereWeight) hypersphereART.getCategory(1);
        
        assertArrayEquals(new double[]{0.2, 0.2}, weight1.center(), 1e-10);
        assertArrayEquals(new double[]{0.8, 0.8}, weight2.center(), 1e-10);
        assertEquals(smallRadiusParams.defaultRadius(), weight1.radius(), 1e-10);
        assertEquals(smallRadiusParams.defaultRadius(), weight2.radius(), 1e-10);
    }
    
    @Test
    @DisplayName("Distance-based activation should work correctly")
    void testDistanceBasedActivation() {
        // Create hypersphere at origin
        var center = Pattern.of(0.0, 0.0);
        hypersphereART.stepFit(center, defaultParams);
        
        // Test points at various distances (all within vigilance range)
        var closePoint = Pattern.of(0.1, 0.0); // Distance = 0.1
        var mediumPoint = Pattern.of(0.2, 0.0); // Distance = 0.2  
        var farPoint = Pattern.of(0.3, 0.0); // Distance = 0.3
        
        var result1 = hypersphereART.stepFit(closePoint, defaultParams);
        var result2 = hypersphereART.stepFit(mediumPoint, defaultParams);
        var result3 = hypersphereART.stepFit(farPoint, defaultParams);
        
        var success1 = (ActivationResult.Success) result1;
        var success2 = (ActivationResult.Success) result2;
        var success3 = (ActivationResult.Success) result3;
        
        // Closer points should have higher activation
        assertTrue(success1.activationValue() > success2.activationValue());
        assertTrue(success2.activationValue() > success3.activationValue());
        
        // All should classify to the same category (within vigilance range)
        assertEquals(0, success1.categoryIndex());
        assertEquals(0, success2.categoryIndex()); 
        assertEquals(0, success3.categoryIndex());
    }
    
    @Test
    @DisplayName("Hypersphere radius should expand when point is outside")
    void testRadiusExpansion() {
        // Create hypersphere with zero initial radius for guaranteed expansion
        var zeroRadiusParams = HypersphereParameters.of(0.1, 0.0, false);
        var input1 = Pattern.of(0.5, 0.5);
        hypersphereART.stepFit(input1, zeroRadiusParams);
        
        var initialWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        var initialRadius = initialWeight.radius();
        assertEquals(0.0, initialRadius, 1e-10);
        
        // Add point that requires radius expansion
        var input2 = Pattern.of(0.55, 0.52); // Distance ≈ 0.054
        
        // Should expand the radius to include this point
        hypersphereART.stepFit(input2, zeroRadiusParams);
        
        // Still should have 1 category (expanded)
        assertEquals(1, hypersphereART.getCategoryCount());
        
        var expandedWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertTrue(expandedWeight.radius() > initialRadius);
        
        // Center should remain unchanged
        assertArrayEquals(new double[]{0.5, 0.5}, expandedWeight.center(), 1e-10);
    }
    
    @Test
    @DisplayName("Winner-take-all competition should select closest center")
    void testWinnerTakeAllCompetition() {
        // Create two well-separated hyperspheres
        var input1 = Pattern.of(0.1, 0.1);
        var input2 = Pattern.of(0.9, 0.9);
        hypersphereART.stepFit(input1, defaultParams);
        hypersphereART.stepFit(input2, defaultParams);
        
        // Input closer to first hypersphere should activate it
        var testInput = Pattern.of(0.15, 0.12);
        var result = hypersphereART.stepFit(testInput, defaultParams);
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Should select category 0 (closer center)
        
        // The radius of category 0 might have expanded to include the test input
        var weight0 = (HypersphereWeight) hypersphereART.getCategory(0);
        assertArrayEquals(new double[]{0.1, 0.1}, weight0.center(), 1e-10);
    }
    
    @Test
    @DisplayName("Default radius parameter should control initial hypersphere size")
    void testDefaultRadiusControl() {
        var input = Pattern.of(0.5, 0.5);
        
        // Test with small default radius
        hypersphereART.stepFit(input, smallRadiusParams);
        var smallWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertEquals(smallRadiusParams.defaultRadius(), smallWeight.radius(), 1e-10);
        
        // Reset and test with large default radius
        hypersphereART.clear();
        hypersphereART.stepFit(input, largeRadiusParams);
        var largeWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertEquals(largeRadiusParams.defaultRadius(), largeWeight.radius(), 1e-10);
        
        assertTrue(largeWeight.radius() > smallWeight.radius());
    }
    
    @Test
    @DisplayName("Mathematical properties should be preserved")
    void testMathematicalProperties() {
        var input = Pattern.of(0.3, 0.7);
        hypersphereART.stepFit(input, defaultParams);
        
        var weight = (HypersphereWeight) hypersphereART.getCategory(0);
        
        // Radius should be non-negative
        assertTrue(weight.radius() >= 0, "Radius must be non-negative");
        
        // Center should match input for first category
        assertArrayEquals(new double[]{0.3, 0.7}, weight.center(), 1e-10);
        
        // Distance from center to itself should be 0
        double centerDistance = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            var diff = input.get(i) - weight.center()[i];
            centerDistance += diff * diff;
        }
        assertEquals(0.0, Math.sqrt(centerDistance), 1e-10);
    }
    
    @Test
    @DisplayName("Edge cases should be handled correctly")
    void testEdgeCases() {
        // Very small coordinates
        var smallInput = Pattern.of(0.001, 0.001);
        var result1 = hypersphereART.stepFit(smallInput, defaultParams);
        assertInstanceOf(ActivationResult.Success.class, result1);
        
        // Very large coordinates (within reasonable range)
        var largeInput = Pattern.of(0.999, 0.999);
        var result2 = hypersphereART.stepFit(largeInput, defaultParams);
        assertInstanceOf(ActivationResult.Success.class, result2);
        
        // Points should create separate hyperspheres due to distance
        assertEquals(2, hypersphereART.getCategoryCount());
        
        // Zero radius should work
        var zeroRadiusParams = HypersphereParameters.of(0.5, 0.0, false);
        hypersphereART.clear();
        var result3 = hypersphereART.stepFit(Pattern.of(0.5, 0.5), zeroRadiusParams);
        assertInstanceOf(ActivationResult.Success.class, result3);
        
        var weight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertEquals(0.0, weight.radius(), 1e-10);
    }
    
    @Test
    @DisplayName("Parameter validation should work correctly")
    void testParameterValidation() {
        var input = Pattern.of(0.5, 0.5);
        
        // Add a category first to enable parameter validation
        hypersphereART.stepFit(input, defaultParams);
        
        // Wrong parameter type should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            hypersphereART.stepFit(Pattern.of(0.6, 0.4), "wrong_params");
        });
        
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> {
            hypersphereART.stepFit(input, null);
        });
        
        // Null input should throw exception
        assertThrows(NullPointerException.class, () -> {
            hypersphereART.stepFit(null, defaultParams);
        });
    }
    
    @Test
    @DisplayName("Dimension validation should work correctly")
    void testDimensionValidation() {
        // Create 2D hypersphere
        var input2D = Pattern.of(0.3, 0.4);
        hypersphereART.stepFit(input2D, defaultParams);
        
        // Try with 3D input - should throw exception when trying to calculate activation
        var input3D = Pattern.of(0.3, 0.4, 0.5);
        assertThrows(IllegalArgumentException.class, () -> {
            hypersphereART.stepFit(input3D, defaultParams);
        });
        
        // Should still have only the original category
        assertEquals(1, hypersphereART.getCategoryCount());
    }
    
    @Test
    @DisplayName("Multiple learning cycles should maintain consistency")
    void testMultipleLearningCycles() {
        var inputs = new Pattern[]{
            Pattern.of(0.2, 0.3),
            Pattern.of(0.25, 0.35),
            Pattern.of(0.8, 0.7),
            Pattern.of(0.75, 0.72),
            Pattern.of(0.22, 0.32)
        };
        
        // Process all inputs
        for (var input : inputs) {
            hypersphereART.stepFit(input, defaultParams);
        }
        
        // Should have created appropriate number of categories
        assertTrue(hypersphereART.getCategoryCount() >= 1);
        assertTrue(hypersphereART.getCategoryCount() <= inputs.length);
        
        // All hyperspheres should have valid properties
        for (int i = 0; i < hypersphereART.getCategoryCount(); i++) {
            var weight = (HypersphereWeight) hypersphereART.getCategory(i);
            assertTrue(weight.radius() >= 0);
            assertEquals(inputs[0].dimension(), weight.dimension());
        }
    }
    
    @Test
    @DisplayName("Activation function should be mathematically correct")
    void testActivationFunction() {
        var center = Pattern.of(0.5, 0.5);
        hypersphereART.stepFit(center, defaultParams);
        
        // Test activation at center (distance = 0)
        var result1 = hypersphereART.stepFit(center, defaultParams);
        var success1 = (ActivationResult.Success) result1;
        assertEquals(1.0, success1.activationValue(), 1e-10); // 1/(1+0) = 1
        
        // Test activation at known distance
        var testPoint = Pattern.of(0.5, 0.6); // Distance = 0.1
        hypersphereART.clear();
        hypersphereART.stepFit(center, defaultParams);
        var result2 = hypersphereART.stepFit(testPoint, defaultParams);
        var success2 = (ActivationResult.Success) result2;
        
        // Should be approximately 1/(1+0.1) = 0.909
        var expectedActivation = 1.0 / (1.0 + 0.1);
        assertEquals(expectedActivation, success2.activationValue(), 1e-3);
    }
    
    @Test
    @DisplayName("Hypersphere expansion should work correctly")
    void testHypersphereExpansion() {
        // Start with zero radius
        var params = HypersphereParameters.of(0.5, 0.0, false);
        var center = Pattern.of(0.5, 0.5);
        hypersphereART.stepFit(center, params);
        
        var initialWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertEquals(0.0, initialWeight.radius(), 1e-10);
        
        // Add point at distance 0.3 - should expand radius
        var distantPoint = Pattern.of(0.8, 0.5); // Distance = 0.3
        hypersphereART.stepFit(distantPoint, params);
        
        assertEquals(1, hypersphereART.getCategoryCount()); // Should still be one category
        
        var expandedWeight = (HypersphereWeight) hypersphereART.getCategory(0);
        assertEquals(0.3, expandedWeight.radius(), 1e-10);
        
        // Center should remain unchanged
        assertArrayEquals(new double[]{0.5, 0.5}, expandedWeight.center(), 1e-10);
    }
    
    @Test
    @DisplayName("toString should provide meaningful representation")
    void testToString() {
        var toString = hypersphereART.toString();
        assertTrue(toString.contains("HypersphereART"));
        assertTrue(toString.contains("categories=0"));
        
        // Add a category and check again
        hypersphereART.stepFit(Pattern.of(0.5, 0.5), defaultParams);
        toString = hypersphereART.toString();
        assertTrue(toString.contains("categories=1"));
    }
}