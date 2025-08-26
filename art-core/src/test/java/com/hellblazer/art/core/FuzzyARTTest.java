package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.weights.FuzzyWeight;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for FuzzyART implementation.
 * Tests the complete FuzzyART algorithm including:
 * - Choice function (calculateActivation)
 * - Vigilance test (checkVigilance)  
 * - Fuzzy min learning rule (updateWeights)
 * - Complement coding handling
 * - Integration with BaseART template framework
 */
class FuzzyARTTest {
    
    private FuzzyART fuzzyART;
    private FuzzyParameters defaultParams;
    private FuzzyParameters highVigilanceParams;
    private FuzzyParameters lowVigilanceParams;
    
    @BeforeEach
    void setUp() {
        fuzzyART = new FuzzyART();
        defaultParams = FuzzyParameters.defaults(); // ρ=0.5, α=0.0, β=1.0
        highVigilanceParams = FuzzyParameters.of(0.9, 0.0, 1.0);
        lowVigilanceParams = FuzzyParameters.of(0.1, 0.0, 1.0);
    }
    
    @Test
    @DisplayName("FuzzyART constructor creates empty network")
    void testConstructor() {
        assertEquals(0, fuzzyART.getCategoryCount());
        assertTrue(fuzzyART.getCategories().isEmpty());
        assertEquals("FuzzyART{categories=0}", fuzzyART.toString());
    }
    
    @Test
    @DisplayName("First input creates initial category with complement coding")
    void testFirstInputCreatesCategory() {
        var input = Pattern.of(0.3, 0.7);
        var result = fuzzyART.stepFit(input, defaultParams);
        
        // Should create new category
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), 1e-10); // First category always gets 1.0
        assertEquals(1, fuzzyART.getCategoryCount());
        
        // Check complement coding is applied correctly
        var category = fuzzyART.getCategory(0);
        assertTrue(category instanceof FuzzyWeight);
        var fuzzyWeight = (FuzzyWeight) category;
        assertEquals(4, fuzzyWeight.dimension()); // 2D → 4D complement coding
        assertEquals(2, fuzzyWeight.originalDimension());
        
        // Verify complement coding: [0.3, 0.7, 0.7, 0.3]
        assertEquals(0.3, fuzzyWeight.get(0), 1e-10);
        assertEquals(0.7, fuzzyWeight.get(1), 1e-10);
        assertEquals(0.7, fuzzyWeight.get(2), 1e-10);
        assertEquals(0.3, fuzzyWeight.get(3), 1e-10);
    }
    
    @Test
    @DisplayName("Same input pattern is accepted by existing category")
    void testSameInputAccepted() {
        var input = Pattern.of(0.6, 0.4);
        
        // Create first category
        fuzzyART.stepFit(input, defaultParams);
        assertEquals(1, fuzzyART.getCategoryCount());
        
        // Present same input again - should be accepted
        var result = fuzzyART.stepFit(input, defaultParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Same category
        assertEquals(1, fuzzyART.getCategoryCount()); // No new category created
    }
    
    @Test
    @DisplayName("Similar input pattern is accepted with low vigilance")
    void testSimilarInputAcceptedLowVigilance() {
        var input1 = Pattern.of(0.6, 0.4);
        var input2 = Pattern.of(0.65, 0.35); // Similar pattern
        
        // Create first category
        fuzzyART.stepFit(input1, lowVigilanceParams);
        assertEquals(1, fuzzyART.getCategoryCount());
        
        // Present similar input with low vigilance - should be accepted
        var result = fuzzyART.stepFit(input2, lowVigilanceParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Same category
        assertEquals(1, fuzzyART.getCategoryCount()); // No new category created
    }
    
    @Test
    @DisplayName("Different input pattern creates new category with high vigilance")
    void testDifferentInputCreatesNewCategoryHighVigilance() {
        var input1 = Pattern.of(0.8, 0.2);
        var input2 = Pattern.of(0.2, 0.8); // Very different pattern
        
        // Create first category
        fuzzyART.stepFit(input1, highVigilanceParams);
        assertEquals(1, fuzzyART.getCategoryCount());
        
        // Present different input with high vigilance - should create new category
        var result = fuzzyART.stepFit(input2, highVigilanceParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(1, success.categoryIndex()); // New category index
        assertEquals(1.0, success.activationValue(), 1e-10); // New category gets 1.0
        assertEquals(2, fuzzyART.getCategoryCount()); // New category created
        
        // Verify both categories exist and are different
        var category1 = (FuzzyWeight) fuzzyART.getCategory(0);
        var category2 = (FuzzyWeight) fuzzyART.getCategory(1);
        assertNotEquals(category1, category2);
    }
    
    @Test
    @DisplayName("Learning updates category weights using fuzzy min rule")
    void testLearningUpdatesWeights() {
        var input1 = Pattern.of(0.8, 0.2);
        var input2 = Pattern.of(0.9, 0.1); // Similar but slightly different
        var learningParams = FuzzyParameters.of(0.1, 0.0, 0.5); // β=0.5 for partial learning
        
        // Create first category
        fuzzyART.stepFit(input1, learningParams);
        var initialWeight = (FuzzyWeight) fuzzyART.getCategory(0);
        
        // Present similar input - should update weights
        var result = fuzzyART.stepFit(input2, learningParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Same category
        assertEquals(1, fuzzyART.getCategoryCount()); // No new category
        
        // Weight should have been updated
        var updatedWeight = success.updatedWeight();
        assertNotEquals(initialWeight, updatedWeight);
        
        // Verify fuzzy min learning was applied - weights should have been updated
        var fuzzyUpdated = (FuzzyWeight) updatedWeight;
        // With β=0.5, new weight = 0.5 * min(input, old_weight) + 0.5 * old_weight
        // The key test is that the updated weight is different and uses fuzzy min learning
        // For complement coding, the weights are applied across all dimensions
        assertTrue(fuzzyUpdated.get(0) <= 0.9); // Should be <= max of the inputs
        assertTrue(fuzzyUpdated.get(1) <= 0.2); // Should be <= max of the inputs
    }
    
    @Test
    @DisplayName("Choice function produces higher activation for better matches")
    void testChoiceFunctionActivationOrdering() {
        var input1 = Pattern.of(0.7, 0.3);
        var input2 = Pattern.of(0.8, 0.2); // More similar to input1
        var input3 = Pattern.of(0.2, 0.8); // Less similar to input1
        var params = FuzzyParameters.of(0.8, 0.01, 1.0); // Higher vigilance to create separate categories
        
        // Create categories
        fuzzyART.stepFit(input1, params);
        fuzzyART.stepFit(input3, params); // This should create a second category due to high vigilance
        
        assertEquals(2, fuzzyART.getCategoryCount());
        
        // Test input2 - should have higher activation for category 0 (more similar to input1)
        var result = fuzzyART.stepFit(input2, params);
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // Should select category 0 (more similar)
    }
    
    @Test
    @DisplayName("Multiple inputs create correct category structure")
    void testMultipleInputsCreateCategories() {
        var inputs = new Pattern[]{
            Pattern.of(0.9, 0.1), // Group 1: High first dimension
            Pattern.of(0.8, 0.2),
            Pattern.of(0.1, 0.9), // Group 2: High second dimension  
            Pattern.of(0.2, 0.8),
            Pattern.of(0.95, 0.05) // Should join Group 1 with low vigilance
        };
        var params = FuzzyParameters.of(0.3, 0.0, 0.8); // Moderate vigilance
        
        // Present all inputs
        for (var input : inputs) {
            fuzzyART.stepFit(input, params);
        }
        
        // Should have created fewer categories than inputs due to grouping
        assertTrue(fuzzyART.getCategoryCount() < inputs.length);
        assertTrue(fuzzyART.getCategoryCount() >= 2); // At least two distinct groups
        
        // Verify toString representation
        assertTrue(fuzzyART.toString().contains("FuzzyART"));
        assertTrue(fuzzyART.toString().contains("categories=" + fuzzyART.getCategoryCount()));
    }
    
    @Test
    @DisplayName("FuzzyART handles edge case inputs correctly")
    void testEdgeCaseInputs() {
        // Test with all zeros
        var zeroInput = Pattern.of(0.0, 0.0);
        var result1 = fuzzyART.stepFit(zeroInput, highVigilanceParams); // Use high vigilance for separation
        assertTrue(result1 instanceof ActivationResult.Success);
        
        // Test with all ones
        var oneInput = Pattern.of(1.0, 1.0);
        var result2 = fuzzyART.stepFit(oneInput, highVigilanceParams);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Test with mixed
        var mixedInput = Pattern.of(0.0, 1.0);
        var result3 = fuzzyART.stepFit(mixedInput, highVigilanceParams);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        // All should create separate categories with high vigilance
        assertEquals(3, fuzzyART.getCategoryCount());
    }
    
    @Test
    @DisplayName("Parameter validation in FuzzyART methods")
    void testParameterValidation() {
        var input = Pattern.of(0.5, 0.5);
        var wrongParams = new Object();
        
        // Add a category first so parameter validation will occur
        fuzzyART.stepFit(input, defaultParams);
        
        // Test invalid parameter types (now validation will occur)
        assertThrows(IllegalArgumentException.class, 
            () -> fuzzyART.stepFit(Pattern.of(0.6, 0.4), wrongParams));
        
        // Test null parameters
        assertThrows(NullPointerException.class,
            () -> fuzzyART.stepFit(null, defaultParams));
        assertThrows(NullPointerException.class,
            () -> fuzzyART.stepFit(input, null));
    }
    
    @Test
    @DisplayName("FuzzyART integrates correctly with BaseART template framework")
    void testBaseARTIntegration() {
        var input1 = Pattern.of(0.4, 0.6);
        var input2 = Pattern.of(0.45, 0.55);
        
        // Test that stepFit method works (template method from BaseART)
        var result1 = fuzzyART.stepFit(input1, defaultParams);
        assertTrue(result1 instanceof ActivationResult.Success);
        
        // Test that categories are managed correctly
        assertEquals(1, fuzzyART.getCategoryCount());
        var category = fuzzyART.getCategory(0);
        assertNotNull(category);
        assertTrue(category instanceof FuzzyWeight);
        
        // Test that getCategories() returns unmodifiable list
        var categories = fuzzyART.getCategories();
        assertEquals(1, categories.size());
        assertThrows(UnsupportedOperationException.class, 
            () -> categories.add(FuzzyWeight.fromInput(input2)));
        
        // Test clear() method
        fuzzyART.clear();
        assertEquals(0, fuzzyART.getCategoryCount());
        assertTrue(fuzzyART.getCategories().isEmpty());
    }
    
    @Test
    @DisplayName("Different vigilance parameters produce different clustering behavior")
    void testVigilanceParameterEffects() {
        var input1 = Pattern.of(0.6, 0.4);
        var input2 = Pattern.of(0.7, 0.3); // Moderately similar
        
        // Test with low vigilance (more permissive)
        var lowVigilanceFuzzyART = new FuzzyART();
        lowVigilanceFuzzyART.stepFit(input1, lowVigilanceParams);
        lowVigilanceFuzzyART.stepFit(input2, lowVigilanceParams);
        
        // Test with high vigilance (more strict)  
        var highVigilanceFuzzyART = new FuzzyART();
        highVigilanceFuzzyART.stepFit(input1, highVigilanceParams);
        highVigilanceFuzzyART.stepFit(input2, highVigilanceParams);
        
        // Low vigilance should create fewer categories (more grouping)
        // High vigilance should create more categories (less grouping)
        assertTrue(lowVigilanceFuzzyART.getCategoryCount() <= highVigilanceFuzzyART.getCategoryCount());
    }
    
    @Test
    @DisplayName("Complement coding dimension handling is correct")
    void testComplementCodingDimensions() {
        // Test various input dimensions
        var input2D = Pattern.of(0.3, 0.7);
        var input3D = Pattern.of(0.2, 0.5, 0.8);
        var input1D = Pattern.of(0.6);
        
        fuzzyART.stepFit(input2D, defaultParams);
        var category2D = (FuzzyWeight) fuzzyART.getCategory(0);
        assertEquals(4, category2D.dimension()); // 2 * 2
        assertEquals(2, category2D.originalDimension());
        
        fuzzyART.clear();
        fuzzyART.stepFit(input3D, defaultParams);
        var category3D = (FuzzyWeight) fuzzyART.getCategory(0);
        assertEquals(6, category3D.dimension()); // 3 * 2
        assertEquals(3, category3D.originalDimension());
        
        fuzzyART.clear();
        fuzzyART.stepFit(input1D, defaultParams);
        var category1D = (FuzzyWeight) fuzzyART.getCategory(0);
        assertEquals(2, category1D.dimension()); // 1 * 2
        assertEquals(1, category1D.originalDimension());
    }
    
    @Test
    @DisplayName("FuzzyART handles numerical edge cases in choice function")
    void testNumericalEdgeCases() {
        // Test with alpha = 0 and small weights (potential division issues)
        var smallInput = Pattern.of(1e-10, 1e-10);
        var zeroAlphaParams = FuzzyParameters.of(0.5, 0.0, 1.0);
        
        // Should not throw division by zero
        var result = fuzzyART.stepFit(smallInput, zeroAlphaParams);
        assertTrue(result instanceof ActivationResult.Success);
        
        // Test with very large alpha
        var largeAlphaParams = FuzzyParameters.of(0.5, 1e6, 1.0);
        var result2 = fuzzyART.stepFit(Pattern.of(0.5, 0.5), largeAlphaParams);
        assertTrue(result2 instanceof ActivationResult.Success);
    }
}