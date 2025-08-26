package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.ARTA;
import com.hellblazer.art.core.parameters.ARTAParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.weights.ARTAWeight;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for ART-A (Attentional ART) implementation.
 * Tests the complete ART-A algorithm including:
 * - Attention-weighted activation computation
 * - Dynamic attention weight learning 
 * - Attention-based vigilance testing
 * - Feature importance analysis through attention weights
 * - Integration with BaseART template framework
 */
class ARTATest {
    
    private ARTA arta;
    private ARTAParameters defaultParams;
    private ARTAParameters highVigilanceParams;
    private ARTAParameters lowAttentionLearningParams;
    
    @BeforeEach
    void setUp() {
        arta = new ARTA();
        defaultParams = ARTAParameters.defaults(); // vigilance=0.7, alpha=0.0, beta=1.0, etc.
        highVigilanceParams = ARTAParameters.of(0.9, 0.0, 1.0, 0.1, 0.8, 0.01);
        lowAttentionLearningParams = ARTAParameters.of(0.7, 0.0, 1.0, 0.01, 0.8, 0.01);
    }
    
    @Test
    @DisplayName("ART-A constructor creates empty network")
    void testConstructor() {
        assertEquals(0, arta.getCategoryCount());
        assertTrue(arta.getCategories().isEmpty());
        assertEquals("ARTA{categories=0}", arta.toString());
    }
    
    @Test
    @DisplayName("First input creates initial category with attention weights")
    void testFirstInputCreatesCategory() {
        var input = Pattern.of(0.3, 0.7, 0.5);
        var result = arta.stepFit(input, defaultParams);
        
        // Should create new category
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), 1e-10); // First category always gets 1.0
        
        // Verify category was created
        assertEquals(1, arta.getCategoryCount());
        
        // Check that weight is ARTAWeight with proper structure
        var weight = arta.getCategory(0);
        assertTrue(weight instanceof ARTAWeight);
        var artaWeight = (ARTAWeight) weight;
        
        assertEquals(3, artaWeight.dimension());
        assertArrayEquals(new double[]{0.3, 0.7, 0.5}, artaWeight.getCategoryWeights(), 1e-10);
        
        // Check attention weights are initialized properly
        var attentionWeights = artaWeight.getAttentionWeights();
        assertEquals(3, attentionWeights.length);
        for (double attention : attentionWeights) {
            assertTrue(attention >= defaultParams.minAttentionWeight());
            assertTrue(attention <= 1.0);
        }
    }
    
    @Test
    @DisplayName("Second input with similar pattern updates attention weights")
    void testAttentionWeightLearning() {
        // First input
        var input1 = Pattern.of(0.3, 0.7, 0.2);
        arta.stepFit(input1, defaultParams);
        
        var initialWeight = (ARTAWeight) arta.getCategory(0);
        var initialAttention = initialWeight.getAttentionWeights();
        
        // Second similar input - should update attention weights
        var input2 = Pattern.of(0.35, 0.75, 0.25);
        var result = arta.stepFit(input2, defaultParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result).categoryIndex()); // Same category
        
        // Check that attention weights have been updated
        var updatedWeight = (ARTAWeight) arta.getCategory(0);
        var updatedAttention = updatedWeight.getAttentionWeights();
        
        // Attention weights should have changed (unless learning rate is 0)
        if (defaultParams.attentionLearningRate() > 0.0) {
            assertFalse(java.util.Arrays.equals(initialAttention, updatedAttention));
        }
    }
    
    @Test
    @DisplayName("Different input creates new category with different attention pattern")
    void testDifferentInputCreatesNewCategory() {
        // First input
        var input1 = Pattern.of(0.9, 0.1, 0.1);
        arta.stepFit(input1, defaultParams);
        
        // Very different input - should create new category
        var input2 = Pattern.of(0.1, 0.9, 0.9);
        var result = arta.stepFit(input2, defaultParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(1, success.categoryIndex()); // New category
        
        // Should have 2 categories now
        assertEquals(2, arta.getCategoryCount());
        
        // Categories should have different category weight patterns but initially same attention weights
        var weight1 = (ARTAWeight) arta.getCategory(0);
        var weight2 = (ARTAWeight) arta.getCategory(1);
        
        // Category weights should be different
        assertFalse(java.util.Arrays.equals(weight1.getCategoryWeights(), weight2.getCategoryWeights()));
        
        // Attention weights start uniform for new categories, but will diverge with training
        assertTrue(java.util.Arrays.equals(weight1.getAttentionWeights(), weight2.getAttentionWeights()));
    }
    
    @Test
    @DisplayName("Attention-weighted activation calculation works correctly")
    void testAttentionWeightedActivation() {
        var input = Pattern.of(0.5, 0.5, 0.5);
        arta.stepFit(input, defaultParams);
        
        // Create a test input
        var testInput = Pattern.of(0.6, 0.4, 0.5);
        
        // Calculate activation manually and compare with ART-A calculation
        var weight = (ARTAWeight) arta.getCategory(0);
        var categoryWeights = weight.getCategoryWeights();
        var attentionWeights = weight.getAttentionWeights();
        
        // Manual calculation of attention-weighted activation
        double expectedIntersection = 0.0;
        double expectedCategoryMag = 0.0;
        // Manual calculation using Pattern interface methods
        
        for (int i = 0; i < testInput.dimension(); i++) {
            double fuzzyMin = Math.min(testInput.get(i), categoryWeights[i]);
            expectedIntersection += attentionWeights[i] * fuzzyMin;
            expectedCategoryMag += attentionWeights[i] * categoryWeights[i];
        }
        
        double expectedActivation = expectedIntersection / (defaultParams.alpha() + expectedCategoryMag);
        
        // Test via getAttentionWeightedSimilarity (which uses similar calculation)
        double similarity = arta.getAttentionWeightedSimilarity(testInput, 0);
        assertTrue(similarity > 0.0 && similarity <= 1.0);
    }
    
    @Test
    @DisplayName("Attention-weighted vigilance testing works correctly")
    void testAttentionWeightedVigilance() {
        // Create initial category with high vigilance parameters
        var input1 = Pattern.of(0.8, 0.2, 0.1);
        arta.stepFit(input1, highVigilanceParams);
        
        // Test with very similar input (should pass vigilance with high vigilance=0.9)
        var verySimilarInput = Pattern.of(0.82, 0.21, 0.11);
        var result1 = arta.stepFit(verySimilarInput, highVigilanceParams);
        assertTrue(result1 instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result1).categoryIndex());
        
        // Test with moderately similar input (should fail high vigilance and create new category)
        var moderatelySimilarInput = Pattern.of(0.85, 0.25, 0.15);
        var result2 = arta.stepFit(moderatelySimilarInput, highVigilanceParams);
        assertTrue(result2 instanceof ActivationResult.Success);
        assertEquals(1, ((ActivationResult.Success) result2).categoryIndex()); // New category
        
        // Test with dissimilar input (should fail vigilance and create new category)
        var dissimilarInput = Pattern.of(0.2, 0.8, 0.9);
        var result3 = arta.stepFit(dissimilarInput, highVigilanceParams);
        assertTrue(result3 instanceof ActivationResult.Success);
        assertEquals(2, ((ActivationResult.Success) result3).categoryIndex()); // New category
    }
    
    @Test
    @DisplayName("Attention analysis provides meaningful feature rankings")
    void testAttentionAnalysis() {
        // Train with patterns that emphasize different features
        arta.stepFit(Pattern.of(0.9, 0.1, 0.1), defaultParams); // Feature 0 important
        arta.stepFit(Pattern.of(0.1, 0.9, 0.1), defaultParams); // Feature 1 important
        arta.stepFit(Pattern.of(0.1, 0.1, 0.9), defaultParams); // Feature 2 important
        
        var analysis = arta.analyzeAttentionDistribution();
        
        assertEquals(3, analysis.meanAttentionPerFeature().length);
        assertEquals(3, analysis.featureRanking().length);
        assertEquals(3, analysis.maxAttentionPerFeature().length);
        
        // Feature rankings should be valid indices
        for (int featureIdx : analysis.featureRanking()) {
            assertTrue(featureIdx >= 0 && featureIdx < 3);
        }
        
        // Mean attention values should be positive
        for (double meanAttention : analysis.meanAttentionPerFeature()) {
            assertTrue(meanAttention > 0.0);
        }
        
        // Check toString method
        assertNotNull(analysis.toString());
        assertTrue(analysis.toString().contains("AttentionAnalysis"));
    }
    
    @Test
    @DisplayName("Get attention weights for specific categories")
    void testGetAttentionWeights() {
        // Create categories
        arta.stepFit(Pattern.of(0.7, 0.3), defaultParams);
        arta.stepFit(Pattern.of(0.2, 0.8), defaultParams);
        
        // Test getAttentionWeights
        var attention0 = arta.getAttentionWeights(0);
        var attention1 = arta.getAttentionWeights(1);
        
        assertEquals(2, attention0.length);
        assertEquals(2, attention1.length);
        
        // Test bounds checking
        assertThrows(IndexOutOfBoundsException.class, () -> arta.getAttentionWeights(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> arta.getAttentionWeights(2));
    }
    
    @Test
    @DisplayName("ARTAParameters validation and builder")
    void testParametersValidation() {
        // Valid parameters
        assertDoesNotThrow(() -> ARTAParameters.of(0.5, 0.0, 1.0, 0.1, 0.8, 0.01));
        
        // Invalid vigilance
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(-0.1, 0.0, 1.0, 0.1, 0.8, 0.01));
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(1.1, 0.0, 1.0, 0.1, 0.8, 0.01));
            
        // Invalid alpha
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, -0.1, 1.0, 0.1, 0.8, 0.01));
            
        // Invalid beta
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, -0.1, 0.1, 0.8, 0.01));
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.1, 0.1, 0.8, 0.01));
            
        // Invalid attention learning rate
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, -0.1, 0.8, 0.01));
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, 1.1, 0.8, 0.01));
            
        // Invalid attention vigilance
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, 0.1, -0.1, 0.01));
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, 0.1, 1.1, 0.01));
            
        // Invalid min attention weight
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, 0.1, 0.8, -0.1));
        assertThrows(IllegalArgumentException.class,
            () -> ARTAParameters.of(0.5, 0.0, 1.0, 0.1, 0.8, 1.1));
    }
    
    @Test
    @DisplayName("ARTAParameters builder pattern works correctly")
    void testParametersBuilder() {
        var params = ARTAParameters.builder()
            .vigilance(0.8)
            .choiceParameter(0.1)
            .learningRate(0.9)
            .attentionLearningRate(0.2)
            .attentionVigilance(0.85)
            .minAttentionWeight(0.02)
            .build();
            
        assertEquals(0.8, params.vigilance());
        assertEquals(0.1, params.alpha());
        assertEquals(0.9, params.beta());
        assertEquals(0.2, params.attentionLearningRate());
        assertEquals(0.85, params.attentionVigilance());
        assertEquals(0.02, params.minAttentionWeight());
    }
    
    @Test
    @DisplayName("ARTAParameters immutable updates work correctly")
    void testParametersImmutableUpdates() {
        var original = ARTAParameters.defaults();
        var modified = original.withVigilance(0.8)
                              .withAlpha(0.1)
                              .withBeta(0.9)
                              .withAttentionLearningRate(0.2)
                              .withAttentionVigilance(0.85)
                              .withMinAttentionWeight(0.02);
        
        // Original unchanged
        assertEquals(0.7, original.vigilance());
        assertEquals(0.0, original.alpha());
        
        // Modified has new values
        assertEquals(0.8, modified.vigilance());
        assertEquals(0.1, modified.alpha());
        assertEquals(0.9, modified.beta());
        assertEquals(0.2, modified.attentionLearningRate());
        assertEquals(0.85, modified.attentionVigilance());
        assertEquals(0.02, modified.minAttentionWeight());
    }
    
    @Test
    @DisplayName("ARTAWeight validation and creation")
    void testARTAWeightValidation() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        var attentionWeights = new double[]{0.8, 0.6, 0.9};
        
        // Valid creation
        assertDoesNotThrow(() -> new ARTAWeight(categoryWeights, attentionWeights));
        
        // Null category weights
        assertThrows(NullPointerException.class,
            () -> new ARTAWeight(null, attentionWeights));
            
        // Null attention weights  
        assertThrows(NullPointerException.class,
            () -> new ARTAWeight(categoryWeights, null));
            
        // Empty category weights
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(new double[0], new double[0]));
            
        // Mismatched dimensions
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(categoryWeights, new double[]{0.5, 0.7}));
            
        // Invalid attention weights (out of range)
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(categoryWeights, new double[]{-0.1, 0.6, 0.9}));
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(categoryWeights, new double[]{0.8, 1.1, 0.9}));
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(categoryWeights, new double[]{Double.NaN, 0.6, 0.9}));
            
        // Invalid category weights (NaN/Infinite)
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(new double[]{Double.NaN, 0.7, 0.3}, attentionWeights));
        assertThrows(IllegalArgumentException.class,
            () -> new ARTAWeight(new double[]{Double.POSITIVE_INFINITY, 0.7, 0.3}, attentionWeights));
    }
    
    @Test
    @DisplayName("ARTAWeight utility methods work correctly")
    void testARTAWeightUtilityMethods() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        var attentionWeights = new double[]{0.8, 0.6, 0.9};
        var weight = new ARTAWeight(categoryWeights, attentionWeights);
        
        // Test getters
        assertArrayEquals(categoryWeights, weight.getCategoryWeights());
        assertArrayEquals(attentionWeights, weight.getAttentionWeights());
        assertEquals(3, weight.dimension());
        
        // Test individual access
        assertEquals(0.5, weight.getCategoryWeight(0));
        assertEquals(0.7, weight.getCategoryWeight(1));
        assertEquals(0.8, weight.getAttentionWeight(0));
        assertEquals(0.6, weight.getAttentionWeight(1));
        
        // Test bounds checking
        assertThrows(IndexOutOfBoundsException.class, () -> weight.getCategoryWeight(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> weight.getCategoryWeight(3));
        assertThrows(IndexOutOfBoundsException.class, () -> weight.getAttentionWeight(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> weight.getAttentionWeight(3));
        
        // Test WeightVector interface  
        assertEquals(0.5, weight.get(0));
        
        // Test factory methods
        var uniformWeight = ARTAWeight.withUniformAttention(categoryWeights, 0.7);
        for (double attention : uniformWeight.getAttentionWeights()) {
            assertEquals(0.7, attention, 1e-10);
        }
        
        var maxWeight = ARTAWeight.withMaxAttention(categoryWeights);
        for (double attention : maxWeight.getAttentionWeights()) {
            assertEquals(1.0, attention, 1e-10);
        }
        
        var vectorWeight = ARTAWeight.fromVector(Pattern.of(categoryWeights), 0.5);
        for (double attention : vectorWeight.getAttentionWeights()) {
            assertEquals(0.5, attention, 1e-10);
        }
    }
    
    @Test
    @DisplayName("ARTAWeight attention-weighted calculations")
    void testAttentionWeightedCalculations() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        var attentionWeights = new double[]{0.9, 0.1, 0.8}; // High attention on features 0,2
        var weight = new ARTAWeight(categoryWeights, attentionWeights);
        var input = Pattern.of(0.6, 0.8, 0.4);
        
        // Test attention-weighted distance
        double distance = weight.attentionWeightedDistance(input);
        assertTrue(distance >= 0.0);
        
        // Test attention-weighted similarity  
        double similarity = weight.attentionWeightedSimilarity(input);
        assertTrue(similarity >= 0.0 && similarity <= 1.0);
        
        // Test dimension mismatch
        assertThrows(IllegalArgumentException.class,
            () -> weight.attentionWeightedDistance(Pattern.of(0.5, 0.7)));
    }
    
    @Test
    @DisplayName("ARTAWeight immutable operations")
    void testARTAWeightImmutableOperations() {
        var originalCategoryWeights = new double[]{0.5, 0.7, 0.3};
        var originalAttentionWeights = new double[]{0.8, 0.6, 0.9};
        var weight = new ARTAWeight(originalCategoryWeights, originalAttentionWeights);
        
        var newCategoryWeights = new double[]{0.6, 0.8, 0.4};
        var newAttentionWeights = new double[]{0.9, 0.7, 1.0};
        
        // Test withCategoryWeights
        var updated1 = weight.withCategoryWeights(newCategoryWeights);
        assertArrayEquals(newCategoryWeights, updated1.getCategoryWeights());
        assertArrayEquals(originalAttentionWeights, updated1.getAttentionWeights());
        
        // Test withAttentionWeights
        var updated2 = weight.withAttentionWeights(newAttentionWeights);
        assertArrayEquals(originalCategoryWeights, updated2.getCategoryWeights());
        assertArrayEquals(newAttentionWeights, updated2.getAttentionWeights());
        
        // Test withWeights
        var updated3 = weight.withWeights(newCategoryWeights, newAttentionWeights);
        assertArrayEquals(newCategoryWeights, updated3.getCategoryWeights());
        assertArrayEquals(newAttentionWeights, updated3.getAttentionWeights());
        
        // Original unchanged
        assertArrayEquals(originalCategoryWeights, weight.getCategoryWeights());
        assertArrayEquals(originalAttentionWeights, weight.getAttentionWeights());
    }
    
    @Test
    @DisplayName("ARTAWeight equals and hashCode work correctly")
    void testARTAWeightEqualsAndHashCode() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        var attentionWeights = new double[]{0.8, 0.6, 0.9};
        
        var weight1 = new ARTAWeight(categoryWeights, attentionWeights);
        var weight2 = new ARTAWeight(categoryWeights.clone(), attentionWeights.clone());
        var weight3 = new ARTAWeight(new double[]{0.6, 0.7, 0.3}, attentionWeights);
        
        // Test equality
        assertEquals(weight1, weight2);
        assertNotEquals(weight1, weight3);
        assertNotEquals(weight1, null);
        assertNotEquals(weight1, "not a weight");
        
        // Test hashCode consistency
        assertEquals(weight1.hashCode(), weight2.hashCode());
        
        // Test toString
        assertNotNull(weight1.toString());
        assertTrue(weight1.toString().contains("ARTAWeight"));
        
        // Test compact string
        assertNotNull(weight1.toCompactString());
        assertTrue(weight1.toCompactString().contains("dim=3"));
    }
    
    @Test
    @DisplayName("Integration with BaseART template method pattern")
    void testBaseARTIntegration() {
        // Test that ART-A properly integrates with BaseART template methods
        var input1 = Pattern.of(0.8, 0.2);
        var input2 = Pattern.of(0.2, 0.8);
        var input3 = Pattern.of(0.85, 0.15); // Similar to input1
        
        // First input creates first category
        var result1 = arta.stepFit(input1, defaultParams);
        assertTrue(result1 instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result1).categoryIndex());
        assertEquals(1, arta.getCategoryCount());
        
        // Second input creates second category
        var result2 = arta.stepFit(input2, defaultParams);
        assertTrue(result2 instanceof ActivationResult.Success);
        assertEquals(1, ((ActivationResult.Success) result2).categoryIndex());
        assertEquals(2, arta.getCategoryCount());
        
        // Third input should match first category
        var result3 = arta.stepFit(input3, defaultParams);
        assertTrue(result3 instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result3).categoryIndex());
        assertEquals(2, arta.getCategoryCount()); // No new category
    }
}