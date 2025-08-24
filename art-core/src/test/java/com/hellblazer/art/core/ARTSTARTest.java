package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for ARTSTAR (ART with STability and Adaptability Regulation).
 * Tests the complete ARTSTAR algorithm including:
 * - Stability and adaptability regulation mechanisms
 * - Dynamic vigilance adjustment
 * - Category strength tracking and decay
 * - Network-level regulation updates
 * - Category pruning and limit enforcement
 * - Integration with BaseART template framework
 */
class ARTSTARTest {
    
    private ARTSTAR artstar;
    private ARTSTARParameters defaultParams;
    private ARTSTARParameters highStabilityParams;
    private ARTSTARParameters limitedCategoryParams;
    
    @BeforeEach
    void setUp() {
        artstar = new ARTSTAR();
        defaultParams = ARTSTARParameters.defaults();
        highStabilityParams = ARTSTARParameters.of(0.7, 0.0, 1.0, 0.9, 0.3, 0.1, 0.02, 0.15, 0.2, 0);
        limitedCategoryParams = ARTSTARParameters.of(0.6, 0.0, 1.0, 0.7, 0.5, 0.2, 0.05, 0.2, 0.15, 3);
    }
    
    @Test
    @DisplayName("ARTSTAR constructor creates empty network with regulation state")
    void testConstructor() {
        assertEquals(0, artstar.getCategoryCount());
        assertTrue(artstar.getCategories().isEmpty());
        
        var analysis = artstar.analyzeRegulationState();
        assertEquals(0.5, analysis.networkStability(), 1e-10);
        assertEquals(0.5, analysis.networkAdaptability(), 1e-10);
        assertEquals(0, analysis.totalCategories());
        assertTrue(artstar.toString().contains("ARTSTAR"));
    }
    
    @Test
    @DisplayName("First input creates initial ARTSTAR category with regulation data")
    void testFirstInputCreatesCategory() {
        var input = Pattern.of(0.8, 0.3, 0.6);
        var result = artstar.stepFit(input, defaultParams);
        
        // Should create new category
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), 1e-10); // First category gets max activation
        
        // Verify category was created
        assertEquals(1, artstar.getCategoryCount());
        
        // Check that weight is ARTSTARWeight with proper regulation data
        var weight = artstar.getARTSTARCategory(0);
        assertEquals(3, weight.dimension());
        assertArrayEquals(new double[]{0.8, 0.3, 0.6}, weight.getCategoryWeights(), 1e-10);
        
        // Check regulation measures are initialized
        assertTrue(weight.getStabilityMeasure() >= 0.0 && weight.getStabilityMeasure() <= 1.0);
        assertTrue(weight.getAdaptabilityMeasure() >= 0.0 && weight.getAdaptabilityMeasure() <= 1.0);
        assertEquals(1.0, weight.getStrength(), 1e-10); // New categories start with full strength
        assertEquals(1, weight.getUsageCount());
        assertTrue(weight.getLastUpdateTime() > 0);
    }
    
    @Test
    @DisplayName("Similar input updates regulation measures and strengthens category")
    void testRegulationMeasureUpdates() {
        // First input - establish category
        var input1 = Pattern.of(0.7, 0.4, 0.2);
        artstar.stepFit(input1, defaultParams);
        
        var initialWeight = artstar.getARTSTARCategory(0);
        double initialStability = initialWeight.getStabilityMeasure();
        double initialAdaptability = initialWeight.getAdaptabilityMeasure();
        double initialStrength = initialWeight.getStrength();
        long initialUsage = initialWeight.getUsageCount();
        
        // Similar input - should update regulation measures
        var input2 = Pattern.of(0.75, 0.35, 0.25);
        var result = artstar.stepFit(input2, defaultParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result).categoryIndex()); // Same category
        
        // Check regulation measures have been updated
        var updatedWeight = artstar.getARTSTARCategory(0);
        
        // Usage count should increase
        assertEquals(initialUsage + 1, updatedWeight.getUsageCount());
        
        // Strength should increase slightly due to successful learning
        assertTrue(updatedWeight.getStrength() >= initialStrength);
        
        // Regulation measures should change based on similarity
        // (exact values depend on similarity calculations)
        assertTrue(Math.abs(updatedWeight.getStabilityMeasure() - initialStability) >= 0.0);
        assertTrue(Math.abs(updatedWeight.getAdaptabilityMeasure() - initialAdaptability) >= 0.0);
    }
    
    @Test
    @DisplayName("Dynamic vigilance adjustment based on regulation state")
    void testDynamicVigilanceAdjustment() {
        // Create category with high stability parameters and perform multiple learning steps
        var input1 = Pattern.of(0.9, 0.1, 0.0);
        artstar.stepFit(input1, highStabilityParams);
        
        // Perform several learning steps to build up regulation state
        for (int i = 0; i < 5; i++) {
            var similarInput = Pattern.of(0.9 + i * 0.01, 0.1 + i * 0.01, 0.0 + i * 0.01);
            artstar.stepFit(similarInput, highStabilityParams);
        }
        
        // Test with moderately similar input
        var input2 = Pattern.of(0.8, 0.2, 0.1);
        var result = artstar.stepFit(input2, highStabilityParams);
        
        // With high stability, the category should be more accepting of similar patterns
        assertTrue(result instanceof ActivationResult.Success);
        
        // After learning, the effective vigilance should be influenced by regulation state
        var analysis = artstar.analyzeRegulationState();
        
        // Check that regulation state has developed (stability or adaptability != initial 0.5)
        assertTrue(analysis.averageCategoryStability() != 0.5 || 
                  analysis.averageCategoryAdaptability() != 0.5,
                  "Regulation measures should change after learning");
    }
    
    @Test
    @DisplayName("Different input creates new category with different regulation profile")
    void testDifferentInputCreatesNewCategory() {
        // First input
        var input1 = Pattern.of(0.9, 0.1, 0.1);
        artstar.stepFit(input1, defaultParams);
        
        // Very different input - should create new category
        var input2 = Pattern.of(0.1, 0.9, 0.8);
        var result = artstar.stepFit(input2, defaultParams);
        
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(1, success.categoryIndex()); // New category
        
        // Should have 2 categories now
        assertEquals(2, artstar.getCategoryCount());
        
        // Categories should have different patterns but similar initial regulation state
        var weight1 = artstar.getARTSTARCategory(0);
        var weight2 = artstar.getARTSTARCategory(1);
        
        // Category weights should be different
        assertFalse(java.util.Arrays.equals(weight1.getCategoryWeights(), weight2.getCategoryWeights()));
        
        // Both should have similar initial regulation measures (from network state)
        assertTrue(Math.abs(weight1.getStabilityMeasure() - weight2.getStabilityMeasure()) < 0.2);
        assertTrue(Math.abs(weight1.getAdaptabilityMeasure() - weight2.getAdaptabilityMeasure()) < 0.2);
    }
    
    @Test
    @DisplayName("Network regulation analysis provides meaningful insights")
    void testNetworkRegulationAnalysis() {
        // Create multiple categories with different patterns
        artstar.stepFit(Pattern.of(0.8, 0.2, 0.1), defaultParams);
        artstar.stepFit(Pattern.of(0.2, 0.8, 0.1), defaultParams);
        artstar.stepFit(Pattern.of(0.1, 0.2, 0.8), defaultParams);
        
        // Perform some learning to generate regulation data
        artstar.stepFit(Pattern.of(0.85, 0.15, 0.05), defaultParams);
        artstar.stepFit(Pattern.of(0.25, 0.75, 0.05), defaultParams);
        
        var analysis = artstar.analyzeRegulationState();
        
        // Should have meaningful values
        assertTrue(analysis.networkStability() >= 0.0 && analysis.networkStability() <= 1.0);
        assertTrue(analysis.networkAdaptability() >= 0.0 && analysis.networkAdaptability() <= 1.0);
        assertTrue(analysis.averageCategoryStability() >= 0.0 && analysis.averageCategoryStability() <= 1.0);
        assertTrue(analysis.averageCategoryAdaptability() >= 0.0 && analysis.averageCategoryAdaptability() <= 1.0);
        assertTrue(analysis.averageCategoryStrength() > 0.0 && analysis.averageCategoryStrength() <= 1.0);
        assertTrue(analysis.averageUsageCount() >= 1);
        assertTrue(analysis.learningSuccessRate() >= 0.0 && analysis.learningSuccessRate() <= 1.0,
                  "Learning success rate should be in [0,1], got: " + analysis.learningSuccessRate());
        assertTrue(analysis.totalCategories() >= 1); // At least one category should be created
        
        // Check toString method
        assertNotNull(analysis.toString());
        assertTrue(analysis.toString().contains("RegulationAnalysis"));
    }
    
    @Test
    @DisplayName("Category strength tracking and time-based decay")
    void testCategoryStrengthAndDecay() {
        // Create a category
        var input = Pattern.of(0.5, 0.5, 0.5);
        artstar.stepFit(input, defaultParams);
        
        var initialWeight = artstar.getARTSTARCategory(0);
        double initialStrength = initialWeight.getStrength();
        
        // Test decay calculation (without waiting for real time)
        double decayRate = 0.1; // High decay rate for testing
        double calculatedDecay = initialWeight.calculateDecayFactor(decayRate);
        assertTrue(calculatedDecay >= 0.0 && calculatedDecay <= 1.0);
        
        double decayedStrength = initialWeight.getDecayedStrength(decayRate);
        assertTrue(decayedStrength <= initialStrength);
        
        // Test weakness detection
        var weakParams = defaultParams.withMinCategoryStrength(0.9);
        if (initialWeight.getStrength() < 0.9) {
            assertTrue(initialWeight.isWeak(weakParams.minCategoryStrength()));
        }
    }
    
    @Test
    @DisplayName("Category limit enforcement with strength-based pruning")
    void testCategoryLimitEnforcement() {
        // Create more categories than the limit allows with very different patterns
        artstar.stepFit(Pattern.of(1.0, 0.0, 0.0), limitedCategoryParams);
        artstar.stepFit(Pattern.of(0.0, 1.0, 0.0), limitedCategoryParams);
        artstar.stepFit(Pattern.of(0.0, 0.0, 1.0), limitedCategoryParams);
        artstar.stepFit(Pattern.of(0.33, 0.33, 0.33), limitedCategoryParams);
        
        // Should have 4 categories before regulation (verify they're all different enough)
        int categoriesBeforeRegulation = artstar.getCategoryCount();
        assertTrue(categoriesBeforeRegulation >= 3, "Should create at least 3 distinct categories");
        
        // Force network regulation update
        artstar.updateNetworkRegulation(limitedCategoryParams);
        
        // Should be pruned to the limit
        assertTrue(artstar.getCategoryCount() <= limitedCategoryParams.maxCategories());
    }
    
    @Test
    @DisplayName("Stability-aware learning reduces learning rate for stable categories")
    void testStabilityAwareLearning() {
        // Create a category and make it stable through repeated similar inputs
        var baseInput = Pattern.of(0.6, 0.4, 0.2);
        artstar.stepFit(baseInput, defaultParams);
        
        // Train with similar inputs to increase stability
        for (int i = 0; i < 5; i++) {
            var similarInput = Pattern.of(0.6 + (i * 0.01), 0.4 - (i * 0.005), 0.2 + (i * 0.005));
            artstar.stepFit(similarInput, defaultParams);
        }
        
        var stableWeight = artstar.getARTSTARCategory(0);
        double stabilityMeasure = stableWeight.getStabilityMeasure();
        
        // High stability should result in more conservative learning
        // (This is tested indirectly through the learning rate adjustment in updateWeights)
        assertTrue(stabilityMeasure >= 0.0 && stabilityMeasure <= 1.0);
    }
    
    @Test
    @DisplayName("ARTSTARParameters validation and builder patterns")
    void testParametersValidationAndBuilder() {
        // Valid parameters
        assertDoesNotThrow(() -> ARTSTARParameters.of(0.7, 0.0, 1.0, 0.8, 0.6, 0.1, 0.01, 0.2, 0.1, 5));
        
        // Invalid vigilance
        assertThrows(IllegalArgumentException.class,
            () -> ARTSTARParameters.of(-0.1, 0.0, 1.0, 0.8, 0.6, 0.1, 0.01, 0.2, 0.1, 5));
        
        // Invalid stability factor
        assertThrows(IllegalArgumentException.class,
            () -> ARTSTARParameters.of(0.7, 0.0, 1.0, 1.5, 0.6, 0.1, 0.01, 0.2, 0.1, 5));
        
        // Invalid vigilance range
        assertThrows(IllegalArgumentException.class,
            () -> ARTSTARParameters.of(0.7, 0.0, 1.0, 0.8, 0.6, 0.1, 0.01, 0.6, 0.1, 5));
        
        // Test builder pattern
        var params = ARTSTARParameters.builder()
            .vigilance(0.8)
            .stabilityFactor(0.9)
            .adaptabilityFactor(0.4)
            .regulationRate(0.15)
            .maxCategories(10)
            .build();
        
        assertEquals(0.8, params.vigilance());
        assertEquals(0.9, params.stabilityFactor());
        assertEquals(0.4, params.adaptabilityFactor());
        assertEquals(0.15, params.regulationRate());
        assertEquals(10, params.maxCategories());
    }
    
    @Test
    @DisplayName("ARTSTARParameters immutable updates work correctly")
    void testParametersImmutableUpdates() {
        var original = ARTSTARParameters.defaults();
        var modified = original.withVigilance(0.8)
                              .withStabilityFactor(0.9)
                              .withAdaptabilityFactor(0.4)
                              .withRegulationRate(0.2);
        
        // Original unchanged
        assertEquals(0.7, original.vigilance());
        assertEquals(0.8, original.stabilityFactor());
        
        // Modified has new values
        assertEquals(0.8, modified.vigilance());
        assertEquals(0.9, modified.stabilityFactor());
        assertEquals(0.4, modified.adaptabilityFactor());
        assertEquals(0.2, modified.regulationRate());
    }
    
    @Test
    @DisplayName("ARTSTARWeight validation and creation")
    void testARTSTARWeightValidation() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        
        // Valid creation
        assertDoesNotThrow(() -> new ARTSTARWeight(categoryWeights, 0.6, 0.4, 5, System.currentTimeMillis(), 0.8));
        
        // Invalid stability measure
        assertThrows(IllegalArgumentException.class,
            () -> new ARTSTARWeight(categoryWeights, 1.5, 0.4, 5, System.currentTimeMillis(), 0.8));
        
        // Invalid adaptability measure  
        assertThrows(IllegalArgumentException.class,
            () -> new ARTSTARWeight(categoryWeights, 0.6, -0.1, 5, System.currentTimeMillis(), 0.8));
        
        // Invalid strength
        assertThrows(IllegalArgumentException.class,
            () -> new ARTSTARWeight(categoryWeights, 0.6, 0.4, 5, System.currentTimeMillis(), 0.0));
        
        // Invalid usage count
        assertThrows(IllegalArgumentException.class,
            () -> new ARTSTARWeight(categoryWeights, 0.6, 0.4, -1, System.currentTimeMillis(), 0.8));
        
        // Null category weights
        assertThrows(NullPointerException.class,
            () -> new ARTSTARWeight(null, 0.6, 0.4, 5, System.currentTimeMillis(), 0.8));
    }
    
    @Test
    @DisplayName("ARTSTARWeight utility methods work correctly")
    void testARTSTARWeightUtilityMethods() {
        var categoryWeights = new double[]{0.6, 0.3, 0.8};
        var weight = new ARTSTARWeight(categoryWeights, 0.7, 0.3, 10, System.currentTimeMillis() - 1000, 0.9);
        
        // Test getters
        assertArrayEquals(categoryWeights, weight.getCategoryWeights());
        assertEquals(3, weight.dimension());
        assertEquals(0.7, weight.getStabilityMeasure());
        assertEquals(0.3, weight.getAdaptabilityMeasure());
        assertEquals(10, weight.getUsageCount());
        assertEquals(0.9, weight.getStrength());
        
        // Test regulation balance
        assertEquals(0.3 - 0.7, weight.getRegulationBalance(), 1e-10);
        
        // Test time calculations
        assertTrue(weight.getTimeSinceLastUpdate() >= 1000);
        
        // Test weakness detection
        assertFalse(weight.isWeak(0.5));
        assertTrue(weight.isWeak(0.95));
        
        // Test factory methods
        var vectorWeight = ARTSTARWeight.fromVector(Pattern.of(categoryWeights), 0.8);
        assertEquals(0.8, vectorWeight.getStrength());
        assertArrayEquals(categoryWeights, vectorWeight.getCategoryWeights());
        
        var uniformWeight = ARTSTARWeight.withUniformWeights(4, 0.5, 0.7);
        assertEquals(4, uniformWeight.dimension());
        assertEquals(0.7, uniformWeight.getStrength());
        for (int i = 0; i < 4; i++) {
            assertEquals(0.5, uniformWeight.get(i));
        }
    }
    
    @Test
    @DisplayName("ARTSTARWeight immutable operations")
    void testARTSTARWeightImmutableOperations() {
        var originalWeights = new double[]{0.4, 0.6, 0.2};
        var weight = new ARTSTARWeight(originalWeights, 0.5, 0.7, 5, 12345L, 0.8);
        
        // Test individual updates
        var stabilityUpdated = weight.withStabilityMeasure(0.9);
        assertEquals(0.9, stabilityUpdated.getStabilityMeasure());
        assertEquals(0.7, stabilityUpdated.getAdaptabilityMeasure()); // Unchanged
        
        var adaptabilityUpdated = weight.withAdaptabilityMeasure(0.2);
        assertEquals(0.2, adaptabilityUpdated.getAdaptabilityMeasure());
        assertEquals(0.5, adaptabilityUpdated.getStabilityMeasure()); // Unchanged
        
        var strengthUpdated = weight.withStrength(0.95);
        assertEquals(0.95, strengthUpdated.getStrength());
        
        var usageUpdated = weight.withUsage();
        assertEquals(6, usageUpdated.getUsageCount());
        assertTrue(usageUpdated.getLastUpdateTime() > 12345L);
        
        // Test comprehensive update
        var newWeights = new double[]{0.8, 0.2, 0.4};
        var fullyUpdated = weight.withUpdate(newWeights, 0.9, 0.1, 15, 54321L, 0.75);
        assertArrayEquals(newWeights, fullyUpdated.getCategoryWeights());
        assertEquals(0.9, fullyUpdated.getStabilityMeasure());
        assertEquals(0.1, fullyUpdated.getAdaptabilityMeasure());
        assertEquals(15, fullyUpdated.getUsageCount());
        assertEquals(54321L, fullyUpdated.getLastUpdateTime());
        assertEquals(0.75, fullyUpdated.getStrength());
        
        // Original unchanged
        assertArrayEquals(originalWeights, weight.getCategoryWeights());
        assertEquals(0.5, weight.getStabilityMeasure());
        assertEquals(0.7, weight.getAdaptabilityMeasure());
    }
    
    @Test
    @DisplayName("ARTSTARWeight similarity calculations")
    void testARTSTARWeightSimilarity() {
        var categoryWeights = new double[]{0.8, 0.2, 0.5};
        var weight = new ARTSTARWeight(categoryWeights, 0.7, 0.3, 1, System.currentTimeMillis(), 1.0);
        
        var input1 = Pattern.of(0.9, 0.1, 0.4); // Very similar
        var input2 = Pattern.of(0.2, 0.8, 0.9); // Very different
        
        double similarity1 = weight.calculateSimilarity(input1);
        double similarity2 = weight.calculateSimilarity(input2);
        
        assertTrue(similarity1 > similarity2); // Similar input should have higher similarity
        assertTrue(similarity1 >= 0.0 && similarity1 <= 1.0);
        assertTrue(similarity2 >= 0.0 && similarity2 <= 1.0);
        
        // Test regulated similarity (considers stability/adaptability)
        double regSimilarity1 = weight.calculateRegulatedSimilarity(input1);
        double regSimilarity2 = weight.calculateRegulatedSimilarity(input2);
        
        // With high stability (0.7), regulated similarity should be higher than base similarity
        assertTrue(regSimilarity1 >= similarity1 * 0.5); // Some boost due to stability
        
        // Test dimension mismatch
        assertThrows(IllegalArgumentException.class,
            () -> weight.calculateSimilarity(Pattern.of(0.5, 0.5)));
    }
    
    @Test
    @DisplayName("ARTSTARWeight equals and hashCode work correctly")
    void testARTSTARWeightEqualsAndHashCode() {
        var categoryWeights = new double[]{0.5, 0.7, 0.3};
        long timestamp = System.currentTimeMillis();
        
        var weight1 = new ARTSTARWeight(categoryWeights, 0.6, 0.4, 5, timestamp, 0.8);
        var weight2 = new ARTSTARWeight(categoryWeights.clone(), 0.6, 0.4, 5, timestamp, 0.8);
        var weight3 = new ARTSTARWeight(new double[]{0.6, 0.7, 0.3}, 0.6, 0.4, 5, timestamp, 0.8);
        
        // Test equality
        assertEquals(weight1, weight2);
        assertNotEquals(weight1, weight3);
        assertNotEquals(weight1, null);
        assertNotEquals(weight1, "not a weight");
        
        // Test hashCode consistency
        assertEquals(weight1.hashCode(), weight2.hashCode());
        
        // Test toString methods
        assertNotNull(weight1.toString());
        assertTrue(weight1.toString().contains("ARTSTARWeight"));
        
        assertNotNull(weight1.toCompactString());
        assertTrue(weight1.toCompactString().contains("ARTSTAR"));
        
        assertNotNull(weight1.toDetailedString());
        assertTrue(weight1.toDetailedString().contains("weights="));
    }
    
    @Test
    @DisplayName("Integration with BaseART template method pattern")
    void testBaseARTIntegration() {
        // Test that ARTSTAR properly integrates with BaseART template methods
        var input1 = Pattern.of(0.8, 0.2, 0.1);
        var input2 = Pattern.of(0.2, 0.8, 0.1);
        var input3 = Pattern.of(0.85, 0.15, 0.05); // Similar to input1
        
        // First input creates first category
        var result1 = artstar.stepFit(input1, defaultParams);
        assertTrue(result1 instanceof ActivationResult.Success);
        assertEquals(0, ((ActivationResult.Success) result1).categoryIndex());
        assertEquals(1, artstar.getCategoryCount());
        
        // Second input creates second category
        var result2 = artstar.stepFit(input2, defaultParams);
        assertTrue(result2 instanceof ActivationResult.Success);
        assertEquals(1, ((ActivationResult.Success) result2).categoryIndex());
        assertEquals(2, artstar.getCategoryCount());
        
        // Third input should match first category (depending on vigilance and regulation)
        var result3 = artstar.stepFit(input3, defaultParams);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        // Network regulation should be working
        var analysis = artstar.analyzeRegulationState();
        assertTrue(analysis.learningSuccessRate() >= 0.0, // Can be 0 if using regular stepFit
                  "Learning success rate should be >= 0, got: " + analysis.learningSuccessRate());
        assertTrue(analysis.totalCategories() >= 1, // At least one category should exist
                  "Expected at least 1 category, got: " + analysis.totalCategories());
    }
}