package com.hellblazer.art.core;

import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.CategoryResult;
import com.hellblazer.art.core.weights.FuzzyWeight;
import com.hellblazer.art.core.utils.DataBounds;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for BaseART template framework and result interfaces.
 * Tests the template method pattern and integration with existing Vector/WeightVector system.
 */
class BaseARTTest {
    
    private TestableBaseART art;
    private FuzzyParameters params;
    
    @BeforeEach
    void setUp() {
        art = new TestableBaseART();
        params = FuzzyParameters.defaults(); // vigilance=0.5, alpha=0.0, beta=1.0
    }
    
    @Test
    @DisplayName("ActivationResult Success creation and validation")
    void testActivationResultSuccess() {
        var weight = FuzzyWeight.fromInput(Pattern.of(0.5, 0.5));
        var result = new ActivationResult.Success(0, 1.5, weight);
        
        assertEquals(0, result.categoryIndex());
        assertEquals(1.5, result.activationValue(), 1e-10);
        assertEquals(weight, result.updatedWeight());
        
        // Test validation
        assertThrows(IllegalArgumentException.class, 
            () -> new ActivationResult.Success(-1, 1.0, weight)); // negative index
        assertThrows(IllegalArgumentException.class,
            () -> new ActivationResult.Success(0, Double.NaN, weight)); // NaN activation
        assertThrows(NullPointerException.class,
            () -> new ActivationResult.Success(0, 1.0, null)); // null weight
    }
    
    @Test
    @DisplayName("ActivationResult NoMatch singleton behavior")
    void testActivationResultNoMatch() {
        var noMatch1 = ActivationResult.NoMatch.instance();
        var noMatch2 = ActivationResult.NoMatch.instance();
        var noMatch3 = new ActivationResult.NoMatch();
        
        assertSame(noMatch1, noMatch2); // Singleton
        assertEquals(noMatch1, noMatch3); // Equal but not same
    }
    
    @Test
    @DisplayName("ActivationResult pattern matching")
    void testActivationResultPatternMatching() {
        var weight = FuzzyWeight.fromInput(Pattern.of(0.5, 0.5));
        var successInstance = new ActivationResult.Success(1, 2.5, weight);
        var noMatchInstance = ActivationResult.NoMatch.instance();
        
        // Pattern matching with records using match method
        var successResult = successInstance.match(
            s -> "success:" + s.categoryIndex(),
            () -> "no-match"
        );
        assertEquals("success:1", successResult);
        
        var noMatchResult = noMatchInstance.match(
            s -> "success:" + s.categoryIndex(),
            () -> "no-match"
        );
        assertEquals("no-match", noMatchResult);
        
        // Using match method
        var matchResult = successInstance.match(
            s -> "Found category " + s.categoryIndex(),
            () -> "No match found"
        );
        assertEquals("Found category 1", matchResult);
    }
    
    @Test
    @DisplayName("MatchResult Accepted creation and validation")
    void testMatchResultAccepted() {
        var accepted = new MatchResult.Accepted(0.7, 0.5);
        
        assertEquals(0.7, accepted.matchValue(), 1e-10);
        assertEquals(0.5, accepted.vigilanceThreshold(), 1e-10);
        assertTrue(accepted.isAccepted());
        assertFalse(accepted.isRejected());
        
        // Test validation
        assertThrows(IllegalArgumentException.class,
            () -> new MatchResult.Accepted(-0.1, 0.5)); // negative match value
        assertThrows(IllegalArgumentException.class,
            () -> new MatchResult.Accepted(0.3, 0.5)); // match < vigilance
    }
    
    @Test
    @DisplayName("MatchResult Rejected creation and validation")
    void testMatchResultRejected() {
        var rejected = new MatchResult.Rejected(0.3, 0.5);
        
        assertEquals(0.3, rejected.matchValue(), 1e-10);
        assertEquals(0.5, rejected.vigilanceThreshold(), 1e-10);
        assertFalse(rejected.isAccepted());
        assertTrue(rejected.isRejected());
        
        // Test validation
        assertThrows(IllegalArgumentException.class,
            () -> new MatchResult.Rejected(0.7, 0.5)); // match >= vigilance
    }
    
    @Test
    @DisplayName("MatchResult utility methods")
    void testMatchResultUtilityMethods() {
        var accepted = new MatchResult.Accepted(0.8, 0.6);
        var rejected = new MatchResult.Rejected(0.4, 0.6);
        
        assertEquals(0.8, accepted.matchValue(), 1e-10);
        assertEquals(0.6, accepted.getVigilanceThreshold(), 1e-10);
        assertEquals(0.4, rejected.matchValue(), 1e-10);
        assertEquals(0.6, rejected.getVigilanceThreshold(), 1e-10);
        
        // Pattern matching
        var acceptedResult = accepted.match(a -> "accepted", r -> "rejected");
        var rejectedResult = rejected.match(a -> "accepted", r -> "rejected");
        
        assertEquals("accepted", acceptedResult);
        assertEquals("rejected", rejectedResult);
    }
    
    @Test
    @DisplayName("CategoryResult creation and validation")
    void testCategoryResultCreation() {
        var weight = FuzzyWeight.fromInput(Pattern.of(0.8, 0.2));
        var activations = new double[]{0.5, 0.9, 0.3};
        var result = new CategoryResult(1, 0.9, weight, activations);
        
        assertEquals(1, result.winnerIndex());
        assertEquals(0.9, result.winnerActivation(), 1e-10);
        assertEquals(weight, result.winnerWeight());
        assertEquals(3, result.categoryCount());
        assertTrue(result.isWinnerUnique());
        
        // Test validation
        assertThrows(IllegalArgumentException.class,
            () -> new CategoryResult(-1, 0.9, weight, activations)); // negative index
        assertThrows(IllegalArgumentException.class,
            () -> new CategoryResult(1, 0.8, weight, activations)); // activation mismatch
    }
    
    @Test
    @DisplayName("CategoryResult winner margin calculation")
    void testCategoryResultWinnerMargin() {
        var weight = FuzzyWeight.fromInput(Pattern.of(0.5, 0.5));
        var activations = new double[]{0.7, 0.9, 0.3, 0.6};
        var result = new CategoryResult(1, 0.9, weight, activations);
        
        assertEquals(0.2, result.getWinnerMargin(), 1e-10); // 0.9 - 0.7
        
        // Test tied winners
        var tiedActivations = new double[]{0.9, 0.9, 0.3};
        var tiedResult = new CategoryResult(0, 0.9, weight, tiedActivations);
        assertFalse(tiedResult.isWinnerUnique());
        assertEquals(0.6, tiedResult.getWinnerMargin(), 1e-10); // 0.9 - 0.3
    }
    
    @Test
    @DisplayName("BaseART empty categories initialization")
    void testBaseARTEmptyInitialization() {
        var input = Pattern.of(0.3, 0.7);
        var result = art.stepFit(input, params);
        
        // First input should create first category
        assertTrue(result instanceof ActivationResult.Success);
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1, art.getCategoryCount());
        
        // Verify the category was created with complement coding
        var category = art.getCategory(0);
        assertTrue(category instanceof FuzzyWeight);
        var fuzzyWeight = (FuzzyWeight) category;
        assertEquals(4, fuzzyWeight.dimension()); // complement coded
        assertEquals(2, fuzzyWeight.originalDimension());
    }
    
    @Test
    @DisplayName("BaseART category management")
    void testBaseARTCategoryManagement() {
        assertEquals(0, art.getCategoryCount());
        assertTrue(art.getCategories().isEmpty());
        
        // Add first category
        art.stepFit(Pattern.of(0.8, 0.2), params);
        assertEquals(1, art.getCategoryCount());
        
        // Categories should be unmodifiable
        assertThrows(UnsupportedOperationException.class,
            () -> art.getCategories().add(FuzzyWeight.fromInput(Pattern.of(0.5, 0.5))));
        
        // Test bounds checking
        assertThrows(IndexOutOfBoundsException.class,
            () -> art.getCategory(1));
        assertThrows(IndexOutOfBoundsException.class,
            () -> art.getCategory(-1));
        
        // Clear categories
        art.clear();
        assertEquals(0, art.getCategoryCount());
    }
    
    @Test
    @DisplayName("BaseART template method workflow")
    void testBaseARTTemplateMethodWorkflow() {
        // Test that template method calls abstract methods in correct order
        art.clearCallLog();
        
        var input1 = Pattern.of(0.9, 0.1);
        var result1 = art.stepFit(input1, params);
        
        // First input: createInitialWeight should be called
        var log1 = art.getCallLog();
        assertTrue(log1.contains("createInitialWeight"));
        assertFalse(log1.contains("calculateActivation")); // No existing categories
        
        art.clearCallLog();
        
        // Second input: all methods should be called
        var input2 = Pattern.of(0.7, 0.3);
        var result2 = art.stepFit(input2, params);
        
        var log2 = art.getCallLog();
        assertTrue(log2.contains("calculateActivation"));
        assertTrue(log2.contains("checkVigilance"));
        // Either updateWeights OR createInitialWeight depending on vigilance test
        assertTrue(log2.contains("updateWeights") || log2.contains("createInitialWeight"));
    }
    
    @Test
    @DisplayName("BaseART winner selection")
    void testBaseARTWinnerSelection() {
        // Create multiple categories by presenting different inputs
        art.stepFit(Pattern.of(0.9, 0.1), params); // Category 0
        art.stepFit(Pattern.of(0.1, 0.9), params); // Category 1 (different enough to create new)
        
        assertEquals(2, art.getCategoryCount());
        
        // Test winner finding
        var activations = new double[]{0.3, 0.7, 0.1};
        var winner = art.findWinner(activations);
        assertEquals(1, winner.winnerIndex());
        assertEquals(0.7, winner.winnerActivation(), 1e-10);
    }
    
    @Test
    @DisplayName("BaseART integration with existing foundation")
    void testBaseARTFoundationIntegration() {
        // Test integration with Pattern system
        var input = Pattern.of(0.6, 0.4);
        var normalized = input.normalize(DataBounds.of(
            new double[]{0.0, 0.0}, new double[]{1.0, 1.0}
        ));
        
        var result = art.stepFit(normalized, params);
        assertTrue(result instanceof ActivationResult.Success);
        
        // Test integration with FuzzyWeight
        var category = art.getCategory(0);
        assertTrue(category instanceof FuzzyWeight);
        
        // Test integration with FuzzyParameters
        var customParams = FuzzyParameters.of(0.8, 0.1, 0.5);
        var result2 = art.stepFit(Pattern.of(0.5, 0.5), customParams);
        assertTrue(result2 instanceof ActivationResult.Success);
    }
    
    @Test
    @DisplayName("BaseART null parameter handling")
    void testBaseARTNullHandling() {
        assertThrows(NullPointerException.class,
            () -> art.stepFit(null, params));
        assertThrows(NullPointerException.class,
            () -> art.stepFit(Pattern.of(0.5, 0.5), null));
    }
    
    @Test
    @DisplayName("BaseART toString representation")
    void testBaseARTToString() {
        assertTrue(art.toString().contains("TestableBaseART"));
        assertTrue(art.toString().contains("categories=0"));
        
        art.stepFit(Pattern.of(0.5, 0.5), params);
        assertTrue(art.toString().contains("categories=1"));
    }
    
    /**
     * Testable concrete implementation of BaseART for testing the template method pattern.
     */
    private static class TestableBaseART extends BaseART {
        private final StringBuilder callLog = new StringBuilder();
        
        @Override
        protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
            callLog.append("calculateActivation;");
            // Simple activation: L1 norm of fuzzy intersection
            if (weight instanceof FuzzyWeight fuzzyWeight) {
                // Apply complement coding to input to match weight dimension
                var complementCoded = FuzzyWeight.fromInput(input);
                var intersection = Pattern.of(complementCoded.data()).min(Pattern.of(fuzzyWeight.data()));
                return intersection.l1Norm();
            }
            return 1.0; // Default activation
        }
        
        @Override
        protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
            callLog.append("checkVigilance;");
            var fuzzyParams = (FuzzyParameters) parameters;
            
            if (weight instanceof FuzzyWeight fuzzyWeight) {
                // Compute match ratio: |input ∧ weight| / |input|
                // Apply complement coding to input to match weight dimension
                var complementCoded = FuzzyWeight.fromInput(input);
                var intersection = Pattern.of(complementCoded.data()).min(Pattern.of(fuzzyWeight.data()));
                var matchValue = intersection.l1Norm() / input.l1Norm();
                
                if (matchValue >= fuzzyParams.vigilance()) {
                    return new MatchResult.Accepted(matchValue, fuzzyParams.vigilance());
                } else {
                    return new MatchResult.Rejected(matchValue, fuzzyParams.vigilance());
                }
            }
            
            return new MatchResult.Accepted(1.0, fuzzyParams.vigilance());
        }
        
        @Override
        protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
            callLog.append("updateWeights;");
            var fuzzyParams = (FuzzyParameters) parameters;
            
            if (currentWeight instanceof FuzzyWeight fuzzyWeight) {
                // Use existing FuzzyWeight update method
                var complementCoded = FuzzyWeight.fromInput(input);
                return fuzzyWeight.update(Pattern.of(complementCoded.data()), fuzzyParams);
            }
            
            return currentWeight; // No update for other types
        }
        
        @Override
        protected WeightVector createInitialWeight(Pattern input, Object parameters) {
            callLog.append("createInitialWeight;");
            // Create FuzzyWeight with complement coding
            return FuzzyWeight.fromInput(input);
        }
        
        public String getCallLog() {
            return callLog.toString();
        }
        
        public void clearCallLog() {
            callLog.setLength(0);
        }
        
        // Make findWinner accessible for testing
        public CategoryResult findWinner(double[] activations) {
            return super.findWinner(activations);
        }

        @Override
        public void close() {
            // No resources to close in test implementation
        }
    }
}