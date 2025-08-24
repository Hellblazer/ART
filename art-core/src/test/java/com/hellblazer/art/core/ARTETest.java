package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * Comprehensive test suite for ART-E (Enhanced ART) implementation.
 * 
 * Tests cover all key ART-E features:
 * - Adaptive learning rates based on familiarity
 * - Feature importance weighting
 * - Performance monitoring and optimization
 * - Context-sensitive vigilance adjustment
 * - Convergence detection
 * - Network topology adjustment
 */
class ARTETest {
    
    private ARTE arte;
    private ARTEParameters params;
    private static final double TOLERANCE = 1e-6;
    
    @BeforeEach
    void setUp() {
        arte = new ARTE(42); // Fixed seed for reproducible tests
        params = ARTEParameters.createDefault(3); // 3-dimensional input
    }
    
    // ARTEParameters Tests
    
    @Test
    void testParametersCreation() {
        assertNotNull(params);
        assertEquals(0.75, params.vigilance(), TOLERANCE);
        assertEquals(0.001, params.alpha(), TOLERANCE);
        assertEquals(0.1, params.baseLearningRate(), TOLERANCE);
        assertTrue(params.featureWeightingEnabled());
        assertEquals(3, params.featureWeights().length);
    }
    
    @Test
    void testParametersValidation() {
        assertThrows(IllegalArgumentException.class, () -> 
            ARTEParameters.createDefault(-1));
        assertThrows(IllegalArgumentException.class, () ->
            new ARTEParameters(1.5, 0.001, 0.1, 0.2, true, new double[]{0.5}, 0.05, 0.3, 0.1, 0.8, 0.01, 10, 0.001));
        assertThrows(IllegalArgumentException.class, () ->
            new ARTEParameters(0.75, -0.1, 0.1, 0.2, true, new double[]{0.5}, 0.05, 0.3, 0.1, 0.8, 0.01, 10, 0.001));
    }
    
    @Test
    void testParametersBuilder() {
        var customParams = ARTEParameters.builder()
            .vigilance(0.8)
            .alpha(0.002)
            .baseLearningRate(0.15)
            .featureWeightingEnabled(false)
            .uniformFeatureWeights(4)
            .build();
        
        assertEquals(0.8, customParams.vigilance(), TOLERANCE);
        assertEquals(0.002, customParams.alpha(), TOLERANCE);
        assertEquals(0.15, customParams.baseLearningRate(), TOLERANCE);
        assertFalse(customParams.featureWeightingEnabled());
        assertEquals(4, customParams.featureWeights().length);
    }
    
    @Test
    void testAdaptiveLearningRate() {
        // High familiarity should result in lower learning rate
        double highFamiliarityRate = params.getAdaptiveLearningRate(0.9);
        double lowFamiliarityRate = params.getAdaptiveLearningRate(0.1);
        
        assertTrue(lowFamiliarityRate > highFamiliarityRate);
        assertTrue(highFamiliarityRate >= params.minLearningRate());
        assertTrue(lowFamiliarityRate <= params.maxLearningRate());
    }
    
    @Test
    void testEffectiveVigilance() {
        double baseVigilance = params.vigilance();
        double lowContextVigilance = params.getEffectiveVigilance(0.2);
        double highContextVigilance = params.getEffectiveVigilance(0.8);
        
        assertTrue(lowContextVigilance < baseVigilance);
        assertTrue(highContextVigilance > baseVigilance);
    }
    
    // ARTEWeight Tests
    
    @Test
    void testARTEWeightCreation() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight = ARTEWeight.fromInput(input, params);
        
        assertNotNull(weight);
        assertEquals(3, weight.dimension());
        assertEquals(0.8, weight.get(0), TOLERANCE);
        assertEquals(0.6, weight.get(1), TOLERANCE);
        assertEquals(0.4, weight.get(2), TOLERANCE);
        assertEquals(0.0, weight.getFamiliarityScore(), TOLERANCE);
        assertEquals(0.5, weight.getContextAdaptation(), TOLERANCE);
    }
    
    @Test
    void testARTEWeightValidation() {
        var categoryWeights = new double[]{0.5, 0.5, 0.5};
        var featureImportances = new double[]{0.3, 0.4, 0.3};
        var performanceHistory = List.of(0.8, 0.9);
        long currentTime = System.currentTimeMillis();
        
        assertThrows(IllegalArgumentException.class, () ->
            new ARTEWeight(categoryWeights, new double[]{0.5}, performanceHistory, 0.5, 0.5, currentTime, currentTime, 1, 0.1));
        assertThrows(IllegalArgumentException.class, () ->
            new ARTEWeight(categoryWeights, featureImportances, performanceHistory, 1.5, 0.5, currentTime, currentTime, 1, 0.1));
        assertThrows(IllegalArgumentException.class, () ->
            new ARTEWeight(categoryWeights, featureImportances, performanceHistory, 0.5, 0.5, currentTime - 1000, currentTime, 1, 0.1));
    }
    
    @Test
    void testARTEWeightUpdate() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight = ARTEWeight.fromInput(input, params);
        
        var newInput = Vector.of(0.7, 0.5, 0.3);
        var updatedWeight = (ARTEWeight) weight.update(newInput, params);
        
        assertNotNull(updatedWeight);
        assertTrue(updatedWeight.getUpdateCount() > weight.getUpdateCount());
        assertTrue(updatedWeight.getFamiliarityScore() > 0.0);
        assertFalse(updatedWeight.getPerformanceHistory().isEmpty());
    }
    
    @Test
    void testARTEWeightSimilarity() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight = ARTEWeight.fromInput(input, params);
        
        // Same input should have high familiarity
        double sameSimilarity = weight.calculateFamiliarity(input);
        assertEquals(1.0, sameSimilarity, TOLERANCE);
        
        // Different input should have lower familiarity
        var differentInput = Vector.of(0.2, 0.3, 0.1);
        double differentSimilarity = weight.calculateFamiliarity(differentInput);
        assertTrue(differentSimilarity < sameSimilarity);
    }
    
    @Test
    void testARTEWeightPerformanceTracking() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight = ARTEWeight.fromInput(input, params);
        
        assertEquals(1.0, weight.getAveragePerformance(), TOLERANCE);
        assertTrue(weight.isPerforming(0.5));
        assertFalse(weight.hasConverged(0.001));
    }
    
    // ARTE Algorithm Tests
    
    @Test
    void testARTEConstruction() {
        assertNotNull(arte);
        assertEquals(0, arte.getCategoryCount());
        assertEquals(0.0, arte.getNetworkPerformance(), TOLERANCE);
        assertEquals(0, arte.getTotalLearningSteps());
    }
    
    @Test
    void testFirstInputCreatesCategory() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var result = arte.stepFit(input, params);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        assertEquals(1, arte.getCategoryCount());
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1.0, success.activationValue(), TOLERANCE);
        assertInstanceOf(ARTEWeight.class, success.updatedWeight());
    }
    
    @Test
    void testSimilarInputMatchesCategory() {
        var input1 = Vector.of(0.8, 0.6, 0.4);
        var input2 = Vector.of(0.75, 0.65, 0.45);
        
        arte.stepFit(input1, params);
        var result = arte.stepFit(input2, params);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        assertEquals(1, arte.getCategoryCount()); // Should still be 1 category
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
    }
    
    @Test
    void testDifferentInputCreatesNewCategory() {
        var input1 = Vector.of(0.8, 0.6, 0.4);
        var input2 = Vector.of(0.1, 0.9, 0.8); // Pattern that won't match well with first
        
        arte.stepFit(input1, params);
        var result = arte.stepFit(input2, params);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        assertEquals(2, arte.getCategoryCount()); // Should create new category
        
        var success = (ActivationResult.Success) result;
        assertEquals(1, success.categoryIndex()); // Index of new category
    }
    
    @Test
    void testEnhancedStepFit() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var result = arte.stepFitEnhanced(input, params);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        assertEquals(1, arte.getTotalLearningSteps());
        assertTrue(arte.getNetworkPerformance() > 0.0);
    }
    
    @Test
    void testFeatureWeightingEnabled() {
        var customParams = ARTEParameters.builder()
            .vigilance(0.7)
            .featureWeightingEnabled(true)
            .uniformFeatureWeights(3)
            .build();
        
        var input = Vector.of(0.8, 0.6, 0.4);
        var result = arte.stepFit(input, customParams);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        var weight = (ARTEWeight) ((ActivationResult.Success) result).updatedWeight();
        assertEquals(3, weight.getFeatureImportances().length);
    }
    
    @Test
    void testFeatureWeightingDisabled() {
        var customParams = ARTEParameters.builder()
            .vigilance(0.7)
            .featureWeightingEnabled(false)
            .uniformFeatureWeights(3)
            .build();
        
        var input = Vector.of(0.8, 0.6, 0.4);
        var result = arte.stepFit(input, customParams);
        
        assertInstanceOf(ActivationResult.Success.class, result);
        // Should still work even with feature weighting disabled
    }
    
    @Test
    void testNetworkOptimization() {
        // Train several categories
        for (int i = 0; i < 5; i++) {
            var input = Vector.of(0.1 * i, 0.2 * i, 0.3 * i);
            arte.stepFitEnhanced(input, params);
        }
        
        int categoriesBeforeOptimization = arte.getCategoryCount();
        arte.optimizeNetwork(params);
        
        // Network should still be valid after optimization
        assertTrue(arte.getCategoryCount() > 0);
        assertTrue(arte.getCategoryCount() <= categoriesBeforeOptimization);
    }
    
    @Test
    void testPerformancePruning() {
        var lowPerformanceParams = ARTEParameters.builder()
            .vigilance(0.9) // High vigilance to force new categories
            .performanceThreshold(0.8) // High performance threshold
            .performanceWindowSize(3)
            .baseLearningRate(0.01) // Low learning rate
            .uniformFeatureWeights(3) // Need feature weights for 3D input
            .build();
        
        // Create some categories that might underperform
        for (int i = 0; i < 10; i++) {
            var input = Vector.of(Math.random(), Math.random(), Math.random());
            arte.stepFitEnhanced(input, lowPerformanceParams);
        }
        
        int categoriesBeforePruning = arte.getCategoryCount();
        arte.optimizeNetwork(lowPerformanceParams);
        
        // Should have pruned some categories or maintained them if they're performing well
        assertTrue(arte.getCategoryCount() <= categoriesBeforePruning);
    }
    
    @Test
    void testNetworkAnalysis() {
        // Train network with several inputs
        for (int i = 0; i < 10; i++) {
            var input = Vector.of(0.1 * i, 0.2 * i, 0.3 * i);
            arte.stepFitEnhanced(input, params);
        }
        
        var analysis = arte.analyzeNetwork();
        assertNotNull(analysis);
        assertTrue(analysis.totalCategories() > 0);
        assertTrue(analysis.networkPerformance() >= 0.0);
        assertTrue(analysis.averageFamiliarity() >= 0.0);
        assertTrue(analysis.averagePerformance() >= 0.0);
        assertTrue(analysis.totalUpdates() > 0);
    }
    
    @Test
    void testARTECategoryAccess() {
        var input = Vector.of(0.8, 0.6, 0.4);
        arte.stepFit(input, params);
        
        var category = arte.getARTECategory(0);
        assertNotNull(category);
        assertInstanceOf(ARTEWeight.class, category);
        assertEquals(3, category.dimension());
    }
    
    @Test
    void testInvalidCategoryAccess() {
        assertThrows(IndexOutOfBoundsException.class, () ->
            arte.getARTECategory(0));
        
        var input = Vector.of(0.8, 0.6, 0.4);
        arte.stepFit(input, params);
        
        assertThrows(IndexOutOfBoundsException.class, () ->
            arte.getARTECategory(1));
    }
    
    @Test
    void testConvergenceDetection() {
        var input = Vector.of(0.8, 0.6, 0.4);
        
        // Train with same input multiple times
        for (int i = 0; i < 20; i++) {
            arte.stepFitEnhanced(input, params);
        }
        
        // Should show some convergence pattern
        assertTrue(arte.getConvergenceRate() >= 0.0);
        assertTrue(arte.getTotalLearningSteps() > 0);
    }
    
    @Test
    void testParameterImmutability() {
        var originalParams = ARTEParameters.createDefault(3);
        var newParams = originalParams.withVigilance(0.9);
        
        assertNotEquals(originalParams.vigilance(), newParams.vigilance());
        assertEquals(0.75, originalParams.vigilance(), TOLERANCE);
        assertEquals(0.9, newParams.vigilance(), TOLERANCE);
    }
    
    @Test
    void testWeightVectorEqualsAndHashCode() throws InterruptedException {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight1 = ARTEWeight.fromInput(input, params);
        Thread.sleep(1); // Ensure different timestamp
        var weight2 = ARTEWeight.fromInput(input, params);
        
        // Should not be equal (different timestamps)
        assertNotEquals(weight1, weight2);
        assertNotEquals(weight1.hashCode(), weight2.hashCode());
        
        // Self equality
        assertEquals(weight1, weight1);
        assertEquals(weight1.hashCode(), weight1.hashCode());
    }
    
    @Test
    void testToStringMethods() {
        var input = Vector.of(0.8, 0.6, 0.4);
        arte.stepFit(input, params);
        
        assertNotNull(params.toString());
        assertNotNull(arte.toString());
        assertNotNull(arte.getARTECategory(0).toString());
        assertNotNull(arte.analyzeNetwork().toString());
        
        assertTrue(params.toString().contains("ARTEParameters"));
        assertTrue(arte.toString().contains("ARTE"));
    }
    
    @Test
    void testLearningRateAdaptation() {
        var input = Vector.of(0.8, 0.6, 0.4);
        arte.stepFit(input, params);
        
        // Get initial weight
        var initialWeight = arte.getARTECategory(0);
        var initialFamiliarity = initialWeight.getFamiliarityScore();
        
        // Train with similar input multiple times
        for (int i = 0; i < 5; i++) {
            var similarInput = Vector.of(0.79, 0.59, 0.39);
            arte.stepFit(similarInput, params);
        }
        
        // Familiarity should increase with similar inputs
        var finalWeight = arte.getARTECategory(0);
        assertTrue(finalWeight.getFamiliarityScore() >= initialFamiliarity);
    }
    
    @Test
    void testMostImportantFeature() {
        var input = Vector.of(0.8, 0.6, 0.4);
        arte.stepFit(input, params);
        
        var weight = arte.getARTECategory(0);
        int mostImportant = weight.getMostImportantFeature();
        
        assertTrue(mostImportant >= 0);
        assertTrue(mostImportant < weight.dimension());
        
        // The most important feature should have the highest importance
        var importances = weight.getFeatureImportances();
        for (int i = 0; i < importances.length; i++) {
            if (i != mostImportant) {
                assertTrue(importances[mostImportant] >= importances[i]);
            }
        }
    }
    
    @Test
    void testWeightedSimilarity() {
        var input = Vector.of(0.8, 0.6, 0.4);
        var weight = ARTEWeight.fromInput(input, params);
        
        // Same input should have similarity close to 1.0
        double sameSimilarity = weight.calculateWeightedSimilarity(input);
        assertTrue(sameSimilarity > 0.8);
        
        // Different input should have lower similarity
        var differentInput = Vector.of(0.2, 0.3, 0.1);
        double differentSimilarity = weight.calculateWeightedSimilarity(differentInput);
        assertTrue(differentSimilarity < sameSimilarity);
    }
}