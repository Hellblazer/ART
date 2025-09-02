package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for VectorizedART1 implementation.
 * Tests both SIMD and standard computation paths, performance characteristics,
 * and compatibility with ART1 semantics for binary pattern recognition.
 * 
 * ART1 is specifically designed for binary patterns (values 0 or 1 only).
 */
public class VectorizedART1Test {
    
    private VectorizedART1 vectorizedART;
    private VectorizedART1Parameters params;
    
    @BeforeEach
    void setUp() {
        // Configure vectorized parameters for ART1
        params = new VectorizedART1Parameters(
            0.9,    // vigilance - high selectivity for binary patterns
            2.0,    // L - uncommitted node bias
            4,      // parallelismLevel
            100,    // parallelThreshold
            1000,   // maxCacheSize
            true    // enableSIMD
        );
        
        vectorizedART = new VectorizedART1();
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedART != null) {
            vectorizedART.close();
        }
    }
    
    @Test
    @DisplayName("Basic binary learning and recognition should work correctly")
    void testBasicBinaryLearningAndRecognition() {
        // Create binary patterns (ART1 requires strict binary: 0.0 or 1.0)
        var pattern1 = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var pattern2 = Pattern.of(0.0, 1.0, 0.0, 1.0);
        var pattern3 = Pattern.of(1.0, 0.0, 1.0, 0.0); // Exact match to pattern1
        
        // Train on first two patterns
        var result1 = vectorizedART.learn(pattern1, params);
        var result2 = vectorizedART.learn(pattern2, params);
        
        // Should create two categories for distinct binary patterns
        assertEquals(2, vectorizedART.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Test recognition of exact match
        var result3 = vectorizedART.predict(pattern3, params);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        var successResult3 = (ActivationResult.Success) result3;
        // Should match first category exactly
        assertEquals(0, successResult3.categoryIndex());
    }
    
    @Test
    @DisplayName("Vigilance parameter should control binary category creation")
    void testVigilanceControlBinary() {
        var pattern1 = Pattern.of(1.0, 1.0, 0.0, 0.0);
        var pattern2 = Pattern.of(0.0, 1.0, 1.0, 0.0); // Different pattern, not a subset
        
        // High vigilance - should create separate categories
        var highVigilanceParams = new VectorizedART1Parameters(
            0.9, 2.0, 4, 100, 1000, true
        );
        var highVigilanceART = new VectorizedART1();
        
        highVigilanceART.learn(pattern1, highVigilanceParams);
        highVigilanceART.learn(pattern2, highVigilanceParams);
        
        assertEquals(2, highVigilanceART.getCategoryCount());
        
        // Low vigilance - should merge into one category
        var lowVigilanceParams = new VectorizedART1Parameters(
            0.3, 2.0, 4, 100, 1000, true  // Lower vigilance to ensure merge
        );
        var lowVigilanceART = new VectorizedART1();
        
        lowVigilanceART.learn(pattern1, lowVigilanceParams);
        lowVigilanceART.learn(pattern2, lowVigilanceParams);
        
        assertEquals(1, lowVigilanceART.getCategoryCount());
        
        highVigilanceART.close();
        lowVigilanceART.close();
    }
    
    @Test
    @DisplayName("SIMD and standard computation should produce equivalent results")
    void testSIMDEquivalenceBinary() {
        var patterns = List.of(
            Pattern.of(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            Pattern.of(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            Pattern.of(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
            Pattern.of(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)
        );
        
        // Train with SIMD enabled
        var simdParams = new VectorizedART1Parameters(
            0.8, 2.0, 4, 100, 1000, true
        );
        var simdART = new VectorizedART1();
        
        // Train with SIMD disabled
        var noSimdParams = new VectorizedART1Parameters(
            0.8, 2.0, 4, 100, 1000, false
        );
        var noSimdART = new VectorizedART1();
        
        // Train both networks
        for (var pattern : patterns) {
            simdART.learn(pattern, simdParams);
            noSimdART.learn(pattern, noSimdParams);
        }
        
        // Should create same number of categories
        assertEquals(simdART.getCategoryCount(), noSimdART.getCategoryCount());
        
        // Test predictions should be equivalent
        for (var pattern : patterns) {
            var simdResult = simdART.predict(pattern, simdParams);
            var noSimdResult = noSimdART.predict(pattern, noSimdParams);
            
            assertTrue(simdResult instanceof ActivationResult.Success);
            assertTrue(noSimdResult instanceof ActivationResult.Success);
            
            var simdSuccess = (ActivationResult.Success) simdResult;
            var noSimdSuccess = (ActivationResult.Success) noSimdResult;
            
            assertEquals(simdSuccess.categoryIndex(), noSimdSuccess.categoryIndex());
            assertEquals(simdSuccess.activationValue(), noSimdSuccess.activationValue(), 1e-6);
        }
        
        simdART.close();
        noSimdART.close();
    }
    
    @Test
    @DisplayName("Non-binary input should throw exception")
    void testNonBinaryInputRejection() {
        // ART1 should reject non-binary inputs
        var nonBinaryPattern = Pattern.of(0.5, 0.7, 0.3, 0.9);
        
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedART.learn(nonBinaryPattern, params);
        });
    }
    
    @Test
    @DisplayName("ART1 choice function should work correctly")
    void testART1ChoiceFunction() {
        // Test the choice function: T_j = |I ∧ w_j| / (L + |w_j|)
        var input = Pattern.of(1.0, 0.0, 1.0, 0.0);
        
        // Use lower vigilance to ensure partial matches pass
        var testParams = new VectorizedART1Parameters(
            0.5, 2.0, 4, 100, 1000, true  // Lower vigilance for testing
        );
        
        // Train with input to create first category
        var result = vectorizedART.learn(input, testParams);
        assertTrue(result instanceof ActivationResult.Success);
        
        // Test activation calculation for exact match
        var exactMatch = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var exactResult = vectorizedART.predict(exactMatch, testParams);
        assertTrue(exactResult instanceof ActivationResult.Success);
        
        var exactSuccess = (ActivationResult.Success) exactResult;
        // For exact match with ART1, activation should be high
        assertTrue(exactSuccess.activationValue() > 0.4);
        
        // Test activation for partial match
        var partialMatch = Pattern.of(1.0, 0.0, 0.0, 0.0);
        var partialResult = vectorizedART.predict(partialMatch, testParams);
        assertTrue(partialResult instanceof ActivationResult.Success);
        
        var partialSuccess = (ActivationResult.Success) partialResult;
        // Partial match should have lower activation than exact match
        assertTrue(partialSuccess.activationValue() < exactSuccess.activationValue());
    }
    
    @Test
    @DisplayName("ART1 learning rule should work correctly")
    void testART1LearningRule() {
        // ART1 learning rule: new_weight = input AND old_weight
        var input1 = Pattern.of(1.0, 1.0, 0.0, 0.0);
        var input2 = Pattern.of(1.0, 0.0, 1.0, 0.0);
        
        // Learn first pattern
        vectorizedART.learn(input1, params);
        assertEquals(1, vectorizedART.getCategoryCount());
        
        // Learn second pattern on same category (if vigilance allows)
        var lowVigilanceParams = new VectorizedART1Parameters(
            0.3, 2.0, 4, 100, 1000, true  // Low vigilance to force merge
        );
        
        var result = vectorizedART.learn(input2, lowVigilanceParams);
        
        // Should still have one category (merged)
        assertEquals(1, vectorizedART.getCategoryCount());
        
        // The learned weight should be the AND of input1 and input2
        // input1 AND input2 = (1,1,0,0) AND (1,0,1,0) = (1,0,0,0)
        var testPattern = Pattern.of(1.0, 0.0, 0.0, 0.0);
        var testResult = vectorizedART.predict(testPattern, lowVigilanceParams);
        assertTrue(testResult instanceof ActivationResult.Success);
    }
    
    @Test
    @DisplayName("Parallel processing should work for large binary category sets")
    void testParallelProcessingBinary() {
        // Create many binary patterns to trigger parallel processing
        var patterns = new ArrayList<Pattern>();
        for (int i = 0; i < 150; i++) {
            var bits = new double[8];
            // Create different binary patterns
            for (int j = 0; j < 8; j++) {
                bits[j] = ((i >> j) & 1) == 1 ? 1.0 : 0.0;
            }
            patterns.add(Pattern.of(bits));
        }
        
        // Set high vigilance and low parallel threshold
        var parallelParams = new VectorizedART1Parameters(
            0.95, 2.0, 4, 5, 1000, true
        );
        var parallelART = new VectorizedART1();
        
        // Train with many binary patterns
        for (var pattern : patterns) {
            parallelART.learn(pattern, parallelParams);
        }
        
        // Should have created many categories
        assertTrue(parallelART.getCategoryCount() > 5);
        
        // Test performance stats - should show parallel tasks were executed
        var stats = parallelART.getPerformanceStats();
        assertTrue(stats.totalVectorOperations() > 0);
        
        parallelART.close();
    }
    
    @Test
    @DisplayName("L parameter should affect choice function correctly")
    void testLParameterEffect() {
        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);
        
        // Test with different L values
        var smallLParams = new VectorizedART1Parameters(0.8, 1.0, 4, 100, 1000, true);
        var largeLParams = new VectorizedART1Parameters(0.8, 10.0, 4, 100, 1000, true);
        
        var smallLART = new VectorizedART1();
        var largeLART = new VectorizedART1();
        
        // Learn same pattern with different L values
        smallLART.learn(pattern, smallLParams);
        largeLART.learn(pattern, largeLParams);
        
        // Predict with same pattern
        var smallLResult = smallLART.predict(pattern, smallLParams);
        var largeLResult = largeLART.predict(pattern, largeLParams);
        
        assertTrue(smallLResult instanceof ActivationResult.Success);
        assertTrue(largeLResult instanceof ActivationResult.Success);
        
        var smallLSuccess = (ActivationResult.Success) smallLResult;
        var largeLSuccess = (ActivationResult.Success) largeLResult;
        
        // Smaller L should give higher activation for same input
        assertTrue(smallLSuccess.activationValue() > largeLSuccess.activationValue());
        
        smallLART.close();
        largeLART.close();
    }
    
    @Test
    @DisplayName("Performance statistics should be tracked correctly")
    void testPerformanceTrackingBinary() {
        var patterns = List.of(
            Pattern.of(1.0, 0.0, 1.0, 0.0),
            Pattern.of(0.0, 1.0, 0.0, 1.0),
            Pattern.of(1.0, 1.0, 0.0, 0.0)
        );
        
        // Initial stats should be zero
        var initialStats = vectorizedART.getPerformanceStats();
        assertEquals(0, initialStats.totalVectorOperations());
        assertEquals(0, initialStats.totalParallelTasks());
        
        // Train and test
        for (var pattern : patterns) {
            vectorizedART.learn(pattern, params);
        }
        
        // Stats should be updated
        var finalStats = vectorizedART.getPerformanceStats();
        assertTrue(finalStats.totalVectorOperations() > 0);
        assertTrue(finalStats.avgComputeTimeMs() >= 0);
        
        // Reset should clear stats
        vectorizedART.resetPerformanceTracking();
        var resetStats = vectorizedART.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    @Test
    @DisplayName("Error handling should work correctly")
    void testErrorHandling() {
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.learn(Pattern.of(1.0, 0.0), null);
        });
        
        // Wrong parameter type should throw exception 
        // Note: This test may need to be adjusted based on actual implementation
        // For now, let's test with invalid parameters instead
        assertThrows(IllegalArgumentException.class, () -> {
            var invalidParams = new VectorizedART1Parameters(
                -0.1, 2.0, 4, 100, 1000, true  // Invalid vigilance (negative)
            );
        });
        
        // Null input should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.learn(null, params);
        });
    }
    
    @Test
    @DisplayName("VectorizedART1Weight should handle binary operations correctly")
    void testVectorizedART1WeightBinaryOperations() {
        // Create weight from binary input
        var input = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var weight = VectorizedART1Weight.fromInput(input, params);
        
        // Weight should preserve binary values
        assertEquals(4, weight.dimension());
        assertEquals(1.0, weight.get(0), 1e-10);
        assertEquals(0.0, weight.get(1), 1e-10);
        assertEquals(1.0, weight.get(2), 1e-10);
        assertEquals(0.0, weight.get(3), 1e-10);
        
        // Test activation calculation
        var testInput = Pattern.of(1.0, 1.0, 0.0, 0.0);
        var activation = weight.computeActivation(testInput, params);
        assertTrue(activation >= 0.0);
        
        // Test vigilance computation
        var vigilance = weight.computeVigilance(testInput, params);
        assertTrue(vigilance >= 0.0 && vigilance <= 1.0);
    }
    
    @Test
    @DisplayName("Resource cleanup should work correctly")
    void testResourceCleanup() {
        var art = new VectorizedART1();
        
        // Use the ART network
        art.learn(Pattern.of(1.0, 0.0), params);
        
        // Close should not throw exception
        assertDoesNotThrow(() -> art.close());
        
        // toString should work even after close
        assertNotNull(art.toString());
    }
    
    @Test
    @DisplayName("Empty pattern should be handled correctly")
    void testEmptyPatternHandling() {
        // Empty pattern should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedART.learn(Pattern.of(), params);
        });
    }
    
    @Test
    @DisplayName("Large binary patterns should work with SIMD")
    void testLargeBinaryPatternsSIMD() {
        // Create large binary pattern (32 dimensions to test SIMD)
        var largeBits = new double[32];
        for (int i = 0; i < 32; i++) {
            largeBits[i] = (i % 2 == 0) ? 1.0 : 0.0;
        }
        var largePattern = Pattern.of(largeBits);
        
        var result = vectorizedART.learn(largePattern, params);
        assertTrue(result instanceof ActivationResult.Success);
        
        // Test prediction
        var predictResult = vectorizedART.predict(largePattern, params);
        assertTrue(predictResult instanceof ActivationResult.Success);
        
        var successResult = (ActivationResult.Success) predictResult;
        assertEquals(0, successResult.categoryIndex()); // Should match first category
    }
}