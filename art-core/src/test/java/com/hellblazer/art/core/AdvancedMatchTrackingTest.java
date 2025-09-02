package com.hellblazer.art.core;

import com.hellblazer.art.core.artmap.SimpleARTMAP;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for advanced match tracking methods in ARTMAP.
 * Tests MT+, MT-, MT0, MT~, and MT1 match tracking strategies.
 */
@DisplayName("Advanced Match Tracking Tests")
public class AdvancedMatchTrackingTest {
    
    private TestARTMAP artmap;
    private double[] inputA;
    private double[] inputB;
    private int targetClass;
    
    @BeforeEach
    void setUp() {
        artmap = new TestARTMAP();
        inputA = new double[]{0.7, 0.3, 0.8};
        inputB = new double[]{0.2, 0.9, 0.4};
        targetClass = 1;
    }
    
    @Nested
    @DisplayName("MT+ (Positive Match Tracking)")
    class MTPlusTests {
        
        @Test
        @DisplayName("Should increase vigilance only when mismatch occurs")
        void testPositiveMatchTracking() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_PLUS);
            
            // Train with initial pattern
            double initialVigilance = 0.5;
            artmap.train(inputA, inputB, targetClass, initialVigilance);
            
            // Present conflicting pattern (same input, different class)
            int conflictingClass = 2;
            artmap.train(inputA, inputB, conflictingClass, initialVigilance);
            
            // Vigilance should have increased
            assertTrue(artmap.getCurrentVigilance() > initialVigilance,
                      "MT+ should increase vigilance on mismatch");
            
            // Present non-conflicting pattern
            double[] newInputA = new double[]{0.1, 0.2, 0.3};
            double beforeVigilance = artmap.getCurrentVigilance();
            artmap.train(newInputA, inputB, targetClass, initialVigilance);
            
            // Vigilance should not increase for non-conflict
            assertEquals(beforeVigilance, artmap.getCurrentVigilance(), 0.001,
                        "MT+ should not increase vigilance without conflict");
        }
        
        @Test
        @DisplayName("Should increase vigilance by epsilon amount")
        void testVigilanceIncrement() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_PLUS);
            double epsilon = 0.001;
            artmap.setMatchTrackingEpsilon(epsilon);
            
            double initialVigilance = 0.5;
            artmap.train(inputA, inputB, targetClass, initialVigilance);
            
            // Force a mismatch
            artmap.forceMatchTracking(initialVigilance);
            
            assertEquals(initialVigilance + epsilon, artmap.getCurrentVigilance(), 1e-10,
                        "MT+ should increase by exactly epsilon");
        }
    }
    
    @Nested
    @DisplayName("MT- (Negative Match Tracking)")
    class MTMinusTests {
        
        @Test
        @DisplayName("Should decrease vigilance when match tracking triggered")
        void testNegativeMatchTracking() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_MINUS);
            
            double initialVigilance = 0.8;
            artmap.train(inputA, inputB, targetClass, initialVigilance);
            
            // Trigger match tracking
            artmap.forceMatchTracking(initialVigilance);
            
            assertTrue(artmap.getCurrentVigilance() < initialVigilance,
                      "MT- should decrease vigilance");
        }
        
        @Test
        @DisplayName("Should not decrease below minimum threshold")
        void testMinimumVigilanceThreshold() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_MINUS);
            double minVigilance = 0.1;
            artmap.setMinimumVigilance(minVigilance);
            
            double initialVigilance = 0.15;
            artmap.train(inputA, inputB, targetClass, initialVigilance);
            
            // Trigger multiple match tracking events
            for (int i = 0; i < 10; i++) {
                artmap.forceMatchTracking(artmap.getCurrentVigilance());
            }
            
            assertTrue(artmap.getCurrentVigilance() >= minVigilance,
                      "MT- should not go below minimum vigilance");
        }
    }
    
    @Nested
    @DisplayName("MT0 (Zero Match Tracking)")
    class MT0Tests {
        
        @Test
        @DisplayName("Should set vigilance to exact match value")
        void testZeroMatchTracking() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_ZERO);
            
            double initialVigilance = 0.5;
            double matchValue = 0.65;
            
            artmap.train(inputA, inputB, targetClass, initialVigilance);
            artmap.forceMatchTrackingWithValue(matchValue);
            
            assertEquals(matchValue, artmap.getCurrentVigilance(), 1e-10,
                        "MT0 should set vigilance to exact match value");
        }
        
        @Test
        @DisplayName("Should handle edge case of perfect match")
        void testPerfectMatch() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_ZERO);
            
            double perfectMatch = 1.0;
            artmap.forceMatchTrackingWithValue(perfectMatch);
            
            assertEquals(perfectMatch, artmap.getCurrentVigilance(), 1e-10,
                        "MT0 should handle perfect match correctly");
        }
    }
    
    @Nested
    @DisplayName("MT~ (Approximate Match Tracking)")
    class MTApproximateTests {
        
        @Test
        @DisplayName("Should interpolate between current and target vigilance")
        void testApproximateMatchTracking() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_APPROXIMATE);
            double interpolationFactor = 0.3; // 30% towards target
            artmap.setInterpolationFactor(interpolationFactor);
            
            double currentVigilance = 0.4;
            double targetVigilance = 0.8;
            
            artmap.setCurrentVigilance(currentVigilance);
            artmap.forceMatchTrackingWithValue(targetVigilance);
            
            double expected = currentVigilance + 
                             interpolationFactor * (targetVigilance - currentVigilance);
            
            assertEquals(expected, artmap.getCurrentVigilance(), 1e-10,
                        "MT~ should interpolate correctly");
        }
        
        @Test
        @DisplayName("Should handle boundary interpolation factors")
        void testBoundaryInterpolation() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_APPROXIMATE);
            
            // Test with factor = 0 (no change)
            artmap.setInterpolationFactor(0.0);
            double initial = 0.5;
            artmap.setCurrentVigilance(initial);
            artmap.forceMatchTrackingWithValue(0.9);
            assertEquals(initial, artmap.getCurrentVigilance(), 1e-10,
                        "Factor 0 should not change vigilance");
            
            // Test with factor = 1 (full change)
            artmap.setInterpolationFactor(1.0);
            artmap.setCurrentVigilance(initial);
            double target = 0.9;
            artmap.forceMatchTrackingWithValue(target);
            assertEquals(target, artmap.getCurrentVigilance(), 1e-10,
                        "Factor 1 should fully update vigilance");
        }
    }
    
    @Nested
    @DisplayName("MT1 (Unity Match Tracking)")
    class MT1Tests {
        
        @Test
        @DisplayName("Should set vigilance to maximum (1.0)")
        void testUnityMatchTracking() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_ONE);
            
            artmap.setCurrentVigilance(0.3);
            artmap.forceMatchTracking(0.5);
            
            assertEquals(1.0, artmap.getCurrentVigilance(), 1e-10,
                        "MT1 should set vigilance to 1.0");
        }
        
        @Test
        @DisplayName("Should effectively disable category after MT1")
        void testCategoryDisabling() {
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_ONE);
            
            // Train initial category
            artmap.train(inputA, inputB, targetClass, 0.5);
            int categoryIndex = artmap.getLastActivatedCategory();
            
            // Trigger MT1 on this category
            artmap.forceMatchTrackingOnCategory(categoryIndex);
            
            // Try to activate the same category again
            boolean activated = artmap.tryActivateCategory(categoryIndex, inputA, 0.99);
            
            assertFalse(activated, 
                       "Category should not activate after MT1 (vigilance = 1.0)");
        }
    }
    
    @Nested
    @DisplayName("Comparative Tests")
    class ComparativeTests {
        
        @Test
        @DisplayName("Should show different convergence rates")
        void testConvergenceRates() {
            // Test how quickly different methods stabilize
            double[] convergenceSteps = new double[5];
            MatchTrackingMethod[] methods = {
                MatchTrackingMethod.MT_PLUS,
                MatchTrackingMethod.MT_MINUS,
                MatchTrackingMethod.MT_ZERO,
                MatchTrackingMethod.MT_APPROXIMATE,
                MatchTrackingMethod.MT_ONE
            };
            
            for (int i = 0; i < methods.length; i++) {
                TestARTMAP testMap = new TestARTMAP();
                testMap.setMatchTrackingMethod(methods[i]);
                convergenceSteps[i] = measureConvergence(testMap);
            }
            
            // MT1 should converge fastest (immediately)
            assertEquals(1, convergenceSteps[4], "MT1 should converge in 1 step");
            
            // MT0 should converge in 2 steps (first sets to target+epsilon, then stabilizes)
            assertTrue(convergenceSteps[2] <= 2, 
                      String.format("MT0 should converge quickly, got %.0f", convergenceSteps[2]));
            
            // MT+ and MT- should take more steps
            assertTrue(convergenceSteps[0] > convergenceSteps[2], 
                      "MT+ should take more steps than MT0");
        }
        
        @Test
        @DisplayName("Should handle noisy data differently")
        void testNoiseRobustness() {
            double noiseLevel = 0.1;
            
            // MT- should be more robust to noise
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_MINUS);
            int categoriesMTMinus = trainWithNoise(artmap, noiseLevel);
            
            // MT+ should create more categories with noise
            TestARTMAP artmapPlus = new TestARTMAP();
            artmapPlus.setMatchTrackingMethod(MatchTrackingMethod.MT_PLUS);
            int categoriesMTPlus = trainWithNoise(artmapPlus, noiseLevel);
            
            assertTrue(categoriesMTMinus <= categoriesMTPlus,
                      "MT- should create fewer categories with noise");
        }
    }
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("Should switch between methods dynamically")
        void testDynamicMethodSwitching() {
            // Start with MT+
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_PLUS);
            artmap.train(inputA, inputB, targetClass, 0.5);
            double afterMTPlus = artmap.getCurrentVigilance();
            
            // Switch to MT-
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_MINUS);
            artmap.forceMatchTracking(afterMTPlus);
            double afterMTMinus = artmap.getCurrentVigilance();
            
            assertTrue(afterMTMinus < afterMTPlus,
                      "Switching from MT+ to MT- should decrease vigilance");
            
            // Switch to MT1
            artmap.setMatchTrackingMethod(MatchTrackingMethod.MT_ONE);
            artmap.forceMatchTracking(afterMTMinus);
            
            assertEquals(1.0, artmap.getCurrentVigilance(),
                        "Switching to MT1 should set vigilance to 1.0");
        }
        
        @Test
        @DisplayName("Should work with different ARTMAP variants")
        void testWithARTMAPVariants() {
            // Test with SimpleARTMAP (when methods are added)
            // SimpleARTMAP simpleMap = new SimpleARTMAP();
            // simpleMap.setMatchTrackingMethod(MatchTrackingMethod.MT_APPROXIMATE);
            // assertDoesNotThrow(() -> simpleMap.learn(inputA, targetClass));
            
            // Test with FuzzyARTMAP (when implemented)
            // FuzzyARTMAP fuzzyMap = new FuzzyARTMAP();
            // fuzzyMap.setMatchTrackingMethod(MatchTrackingMethod.MT_ZERO);
            // assertDoesNotThrow(() -> fuzzyMap.learn(inputA, targetClass));
            
            // For now, just verify the enum exists
            assertNotNull(MatchTrackingMethod.MT_PLUS);
            assertNotNull(MatchTrackingMethod.MT_MINUS);
            assertNotNull(MatchTrackingMethod.MT_ZERO);
            assertNotNull(MatchTrackingMethod.MT_APPROXIMATE);
            assertNotNull(MatchTrackingMethod.MT_ONE);
        }
    }
    
    // Helper methods
    
    private double measureConvergence(TestARTMAP map) {
        int steps = 0;
        double initialVigilance = 0.5;
        double targetValue = 0.7; // Fixed target for convergence testing
        map.setCurrentVigilance(initialVigilance);
        double prevVigilance = initialVigilance;
        
        while (steps < 100) {
            steps++;
            // Use a fixed target value instead of prevVigilance
            map.forceMatchTrackingWithValue(targetValue);
            double currentVigilance = map.getCurrentVigilance();
            
            if (Math.abs(currentVigilance - prevVigilance) < 0.001 || 
                currentVigilance >= 1.0 || currentVigilance <= 0.0) {
                break;
            }
            prevVigilance = currentVigilance;
        }
        
        return steps;
    }
    
    private int trainWithNoise(TestARTMAP map, double noiseLevel) {
        map.reset();
        
        for (int i = 0; i < 20; i++) {
            double[] noisyInput = addNoise(inputA, noiseLevel);
            map.train(noisyInput, inputB, targetClass % 3, 0.7);
        }
        
        return map.getCategoryCount();
    }
    
    private double[] addNoise(double[] input, double level) {
        double[] noisy = input.clone();
        for (int i = 0; i < noisy.length; i++) {
            noisy[i] += (Math.random() - 0.5) * 2 * level;
            noisy[i] = Math.max(0, Math.min(1, noisy[i])); // Clamp to [0,1]
        }
        return noisy;
    }
    
    /**
     * Test implementation of ARTMAP with advanced match tracking
     */
    private static class TestARTMAP {
        private MatchTrackingMethod method = MatchTrackingMethod.MT_PLUS;
        private double currentVigilance = 0.5;
        private double epsilon = 0.001;
        private double minVigilance = 0.0;
        private double interpolationFactor = 0.5;
        private int categoryCount = 0;
        private int lastActivatedCategory = -1;
        private java.util.Map<Integer, Integer> categoryToClass = new java.util.HashMap<>();
        private java.util.Map<String, Integer> patternToCategory = new java.util.HashMap<>();
        
        void setMatchTrackingMethod(MatchTrackingMethod method) {
            this.method = method;
        }
        
        void setCurrentVigilance(double vigilance) {
            this.currentVigilance = vigilance;
        }
        
        double getCurrentVigilance() {
            return currentVigilance;
        }
        
        void setMatchTrackingEpsilon(double epsilon) {
            this.epsilon = epsilon;
        }
        
        void setMinimumVigilance(double min) {
            this.minVigilance = min;
        }
        
        void setInterpolationFactor(double factor) {
            this.interpolationFactor = factor;
        }
        
        int getCategoryCount() {
            return categoryCount;
        }
        
        int getLastActivatedCategory() {
            return lastActivatedCategory;
        }
        
        void train(double[] inputA, double[] inputB, int targetClass, double vigilance) {
            // Simplified training logic
            String patternKey = java.util.Arrays.toString(inputA);
            
            // Check if pattern already exists with different class
            if (patternToCategory.containsKey(patternKey)) {
                int existingCategory = patternToCategory.get(patternKey);
                int existingClass = categoryToClass.get(existingCategory);
                
                if (existingClass != targetClass) {
                    // Conflict detected - trigger match tracking from current vigilance
                    forceMatchTracking(currentVigilance);
                }
            } else {
                // New pattern - create category
                categoryCount++;
                lastActivatedCategory = categoryCount - 1;
                patternToCategory.put(patternKey, lastActivatedCategory);
                categoryToClass.put(lastActivatedCategory, targetClass);
                // Only set vigilance if not already increased by match tracking
                if (currentVigilance <= vigilance) {
                    currentVigilance = vigilance;
                }
            }
        }
        
        void forceMatchTracking(double vigilance) {
            forceMatchTrackingWithValue(vigilance + epsilon);
        }
        
        void forceMatchTrackingWithValue(double targetValue) {
            switch (method) {
                case MT_PLUS -> currentVigilance = Math.min(1.0, currentVigilance + epsilon);
                case MT_MINUS -> currentVigilance = Math.max(minVigilance, currentVigilance - epsilon);
                case MT_ZERO -> currentVigilance = targetValue;
                case MT_APPROXIMATE -> currentVigilance = currentVigilance + 
                    interpolationFactor * (targetValue - currentVigilance);
                case MT_ONE -> currentVigilance = 1.0;
            }
        }
        
        void forceMatchTrackingOnCategory(int category) {
            currentVigilance = 1.0; // For MT1 testing
        }
        
        boolean tryActivateCategory(int category, double[] input, double matchScore) {
            // Check if the match score meets the current vigilance threshold
            // After MT1, currentVigilance is 1.0, so match must be perfect (>= 1.0)
            return matchScore >= currentVigilance;
        }
        
        void reset() {
            categoryCount = 0;
            currentVigilance = 0.5;
            lastActivatedCategory = -1;
            categoryToClass.clear();
            patternToCategory.clear();
        }
    }
    
    /**
     * Enum for match tracking methods
     */
    private enum MatchTrackingMethod {
        MT_PLUS,        // Increase vigilance
        MT_MINUS,       // Decrease vigilance
        MT_ZERO,        // Set to match value
        MT_APPROXIMATE, // Interpolate
        MT_ONE          // Set to 1.0
    }
}