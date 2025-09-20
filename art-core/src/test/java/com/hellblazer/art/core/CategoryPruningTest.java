package com.hellblazer.art.core;

import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for category pruning functionality in BaseART.
 */
class CategoryPruningTest {
    
    private TestableARTWithPruning art;
    private FuzzyParameters params;
    
    @BeforeEach
    void setUp() {
        art = new TestableARTWithPruning();
        params = FuzzyParameters.of(0.9, 0.0, 1.0); // High vigilance to force new categories
    }
    
    @Test
    @DisplayName("Pruning by usage frequency removes low-usage categories")
    void testPruneByUsageFrequency() {
        // Create multiple categories with different patterns
        art.stepFit(Pattern.of(0.1, 0.9), params); // Category 0
        art.stepFit(Pattern.of(0.9, 0.1), params); // Category 1
        art.stepFit(Pattern.of(0.5, 0.5), params); // Category 2
        art.stepFit(Pattern.of(0.3, 0.7), params); // Category 3
        
        assertEquals(4, art.getCategoryCount());
        
        // Use some categories more than others
        // Category 0: used heavily
        for (int i = 0; i < 10; i++) {
            art.stepFit(Pattern.of(0.1, 0.9), params);
        }
        
        // Category 1: used moderately
        for (int i = 0; i < 5; i++) {
            art.stepFit(Pattern.of(0.9, 0.1), params);
        }
        
        // Categories 2 and 3: used only once (initial creation)
        
        // Verify usage counts
        assertEquals(11, art.getCategoryUsageCount(0)); // 1 initial + 10 additional
        assertEquals(6, art.getCategoryUsageCount(1));  // 1 initial + 5 additional
        assertEquals(1, art.getCategoryUsageCount(2));  // Only initial
        assertEquals(1, art.getCategoryUsageCount(3));  // Only initial
        
        // Prune categories with usage less than 50% of mean
        // Mean usage = (11 + 6 + 1 + 1) / 4 = 4.75
        // Threshold = 4.75 * 0.5 = 2.375
        // Categories 2 and 3 should be pruned
        int pruned = art.pruneByUsageFrequency(0.5);
        
        assertEquals(2, pruned);
        assertEquals(2, art.getCategoryCount());
        
        // Verify the right categories remain
        assertEquals(11, art.getCategoryUsageCount(0));
        assertEquals(6, art.getCategoryUsageCount(1));
    }
    
    @Test
    @DisplayName("Pruning by age removes old unused categories")
    void testPruneByAge() throws InterruptedException {
        // Create categories at different times
        art.stepFit(Pattern.of(0.1, 0.9), params); // Category 0
        Thread.sleep(50);
        art.stepFit(Pattern.of(0.9, 0.1), params); // Category 1
        Thread.sleep(50);
        art.stepFit(Pattern.of(0.5, 0.5), params); // Category 2
        Thread.sleep(50);
        art.stepFit(Pattern.of(0.3, 0.7), params); // Category 3
        
        assertEquals(4, art.getCategoryCount());
        
        // Update categories 2 and 3 to be "recently used"
        art.stepFit(Pattern.of(0.5, 0.5), params); // Updates category 2
        art.stepFit(Pattern.of(0.3, 0.7), params); // Updates category 3
        
        // Prune categories older than 100ms
        // Categories 0 and 1 should be pruned as they haven't been used recently
        int pruned = art.pruneByAge(100);
        
        assertTrue(pruned >= 1); // At least one should be pruned
        assertTrue(art.getCategoryCount() <= 3);
    }
    
    @Test
    @DisplayName("Pruning to max size keeps most frequently used categories")
    void testPruneToMaxSize() {
        // Create 6 categories
        art.stepFit(Pattern.of(0.1, 0.9), params); // Category 0
        art.stepFit(Pattern.of(0.9, 0.1), params); // Category 1
        art.stepFit(Pattern.of(0.5, 0.5), params); // Category 2
        art.stepFit(Pattern.of(0.3, 0.7), params); // Category 3
        art.stepFit(Pattern.of(0.7, 0.3), params); // Category 4
        art.stepFit(Pattern.of(0.2, 0.8), params); // Category 5
        
        assertEquals(6, art.getCategoryCount());
        
        // Use categories with different frequencies
        for (int i = 0; i < 10; i++) {
            art.stepFit(Pattern.of(0.1, 0.9), params); // Category 0: 11 total
        }
        for (int i = 0; i < 8; i++) {
            art.stepFit(Pattern.of(0.9, 0.1), params); // Category 1: 9 total
        }
        for (int i = 0; i < 5; i++) {
            art.stepFit(Pattern.of(0.5, 0.5), params); // Category 2: 6 total
        }
        for (int i = 0; i < 3; i++) {
            art.stepFit(Pattern.of(0.3, 0.7), params); // Category 3: 4 total
        }
        // Categories 4 and 5: only 1 use each
        
        // Prune to max 3 categories
        int pruned = art.pruneToMaxSize(3);
        
        assertEquals(3, pruned);
        assertEquals(3, art.getCategoryCount());
        
        // Verify categories are sorted by usage (highest first)
        assertTrue(art.getCategoryUsageCount(0) >= art.getCategoryUsageCount(1));
        assertTrue(art.getCategoryUsageCount(1) >= art.getCategoryUsageCount(2));
    }
    
    @Test
    @DisplayName("Pruning empty network returns 0")
    void testPruneEmptyNetwork() {
        assertEquals(0, art.getCategoryCount());
        
        assertEquals(0, art.pruneByUsageFrequency(0.5));
        assertEquals(0, art.pruneByAge(1000));
        assertEquals(0, art.pruneToMaxSize(10));
    }
    
    @Test
    @DisplayName("Pruning validates parameters")
    void testPruningParameterValidation() {
        art.stepFit(Pattern.of(0.5, 0.5), params);
        
        // Invalid usage ratio
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneByUsageFrequency(-0.1));
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneByUsageFrequency(1.1));
        
        // Invalid age
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneByAge(0));
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneByAge(-100));
        
        // Invalid max size
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneToMaxSize(0));
        assertThrows(IllegalArgumentException.class, 
            () -> art.pruneToMaxSize(-5));
    }
    
    @Test
    @DisplayName("No pruning when threshold not met")
    void testNoPruningWhenThresholdNotMet() {
        // Create categories with similar usage
        art.stepFit(Pattern.of(0.1, 0.9), params);
        art.stepFit(Pattern.of(0.9, 0.1), params);
        art.stepFit(Pattern.of(0.5, 0.5), params);
        
        // Use all categories equally
        for (int i = 0; i < 5; i++) {
            art.stepFit(Pattern.of(0.1, 0.9), params);
            art.stepFit(Pattern.of(0.9, 0.1), params);
            art.stepFit(Pattern.of(0.5, 0.5), params);
        }
        
        assertEquals(3, art.getCategoryCount());
        
        // With equal usage, pruning at 0.9 ratio should remove nothing
        // All categories have usage of 6, mean is 6, threshold is 5.4
        int pruned = art.pruneByUsageFrequency(0.9);
        assertEquals(0, pruned);
        assertEquals(3, art.getCategoryCount());
        
        // Pruning to size >= current size should do nothing
        pruned = art.pruneToMaxSize(3);
        assertEquals(0, pruned);
        assertEquals(3, art.getCategoryCount());
        
        pruned = art.pruneToMaxSize(5);
        assertEquals(0, pruned);
        assertEquals(3, art.getCategoryCount());
    }
    
    @Test
    @DisplayName("Usage statistics tracking")
    void testUsageStatisticsTracking() {
        // Test that statistics are properly tracked
        // Use lower vigilance to allow category reuse  
        var lowVigilanceParams = FuzzyParameters.of(0.5, 0.0, 1.0);
        
        art.stepFit(Pattern.of(0.1, 0.9), lowVigilanceParams);
        art.stepFit(Pattern.of(0.9, 0.1), lowVigilanceParams);
        
        assertEquals(1, art.getCategoryUsageCount(0));
        assertEquals(1, art.getCategoryUsageCount(1));
        assertEquals(2, art.getTotalActivations());
        
        // Reuse first category with same pattern
        art.stepFit(Pattern.of(0.11, 0.89), lowVigilanceParams); // Similar to first pattern
        assertEquals(2, art.getCategoryUsageCount(0));
        assertEquals(1, art.getCategoryUsageCount(1));
        assertEquals(3, art.getTotalActivations());
        
        // Verify timestamps are updated
        var time0 = art.getCategoryLastUsedTimestamp(0);
        var time1 = art.getCategoryLastUsedTimestamp(1);
        assertTrue(time0 >= time1); // Category 0 was used more recently or at same time
    }
    
    @Test
    @DisplayName("Statistics reset on clear")
    void testStatisticsResetOnClear() {
        // Create some categories
        art.stepFit(Pattern.of(0.1, 0.9), params);
        art.stepFit(Pattern.of(0.9, 0.1), params);
        art.stepFit(Pattern.of(0.5, 0.5), params);
        
        assertEquals(3, art.getCategoryCount());
        assertEquals(3, art.getTotalActivations());
        
        // Clear the network
        art.clear();
        
        assertEquals(0, art.getCategoryCount());
        assertEquals(0, art.getTotalActivations());
        
        // Add new category
        art.stepFit(Pattern.of(0.3, 0.7), params);
        assertEquals(1, art.getCategoryCount());
        assertEquals(1, art.getTotalActivations());
        assertEquals(1, art.getCategoryUsageCount(0));
    }
    
    @Test
    @DisplayName("Statistics bounds checking")
    void testStatisticsBoundsChecking() {
        art.stepFit(Pattern.of(0.5, 0.5), params);
        
        // Valid access
        assertDoesNotThrow(() -> art.getCategoryUsageCount(0));
        assertDoesNotThrow(() -> art.getCategoryLastUsedTimestamp(0));
        
        // Invalid access
        assertThrows(IndexOutOfBoundsException.class, 
            () -> art.getCategoryUsageCount(-1));
        assertThrows(IndexOutOfBoundsException.class, 
            () -> art.getCategoryUsageCount(1));
        assertThrows(IndexOutOfBoundsException.class, 
            () -> art.getCategoryLastUsedTimestamp(-1));
        assertThrows(IndexOutOfBoundsException.class, 
            () -> art.getCategoryLastUsedTimestamp(1));
    }
    
    /**
     * Testable implementation that exposes BaseART pruning capabilities.
     */
    private static class TestableARTWithPruning extends BaseART {
        
        @Override
        protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
            // Simple activation based on cosine similarity
            if (weight instanceof SimpleWeight sw) {
                // Compute cosine similarity
                var dotProduct = computeDotProduct(input, sw.pattern);
                return dotProduct / (input.l2Norm() * sw.pattern.l2Norm() + 1e-10);
            }
            return 1.0;
        }
        
        @Override
        protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
            var params = (FuzzyParameters) parameters;
            if (weight instanceof SimpleWeight sw) {
                // Compute similarity using min operation (Fuzzy ART style)
                var minPattern = input.min(sw.pattern);
                var similarity = minPattern.l1Norm() / (input.l1Norm() + 1e-10);
                if (similarity >= params.vigilance()) {
                    return new MatchResult.Accepted(similarity, params.vigilance());
                } else {
                    return new MatchResult.Rejected(similarity, params.vigilance());
                }
            }
            return new MatchResult.Accepted(1.0, params.vigilance());
        }
        
        @Override
        protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
            var params = (FuzzyParameters) parameters;
            if (currentWeight instanceof SimpleWeight sw) {
                // Simple averaging update using scale
                var beta = params.beta();
                var scaledCurrent = sw.pattern.scale(1 - beta);
                var scaledInput = input.scale(beta);
                // Add manually since Pattern doesn't have add method
                var result = new double[input.dimension()];
                for (int i = 0; i < input.dimension(); i++) {
                    result[i] = scaledCurrent.get(i) + scaledInput.get(i);
                }
                return new SimpleWeight(Pattern.of(result));
            }
            return currentWeight;
        }
        
        @Override
        protected WeightVector createInitialWeight(Pattern input, Object parameters) {
            return new SimpleWeight(input);
        }
        
        // Helper method to compute dot product
        private double computeDotProduct(Pattern a, Pattern b) {
            if (a.dimension() != b.dimension()) {
                throw new IllegalArgumentException("Dimensions must match");
            }
            double sum = 0.0;
            for (int i = 0; i < a.dimension(); i++) {
                sum += a.get(i) * b.get(i);
            }
            return sum;
        }
        
        // Simple weight implementation for testing
        private record SimpleWeight(Pattern pattern) implements WeightVector {
            @Override
            public double get(int index) {
                return pattern.get(index);
            }
            
            @Override
            public int dimension() {
                return pattern.dimension();
            }
            
            @Override
            public double l1Norm() {
                return pattern.l1Norm();
            }
            
            @Override
            public WeightVector update(Pattern input, Object parameters) {
                // Delegate to parent's updateWeights method
                return new SimpleWeight(input);
            }
        }

        @Override
        public void close() {
            // No resources to close in test implementation
        }
    }
}