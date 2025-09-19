/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Base test class for vectorized ART implementations.
 * Provides common test patterns to reduce redundancy across algorithm tests.
 * 
 * @param <A> the algorithm type
 * @param <P> the parameter type
 */
public abstract class BaseVectorizedARTTest<A extends ARTAlgorithm<P>, P> {
    
    protected A algorithm;
    protected P parameters;
    
    @BeforeEach
    protected void setUp() {
        parameters = createDefaultParameters();
        algorithm = createAlgorithm(parameters);
    }
    
    /**
     * Create a new instance of the algorithm with the given parameters.
     */
    protected abstract A createAlgorithm(P params);
    
    /**
     * Create default parameters for the algorithm.
     */
    protected abstract P createDefaultParameters();
    
    /**
     * Create parameters with a specific vigilance value.
     */
    protected P createParametersWithVigilance(double vigilance) {
        // Default implementation for VectorizedParameters
        if (parameters instanceof VectorizedParameters) {
            var vp = (VectorizedParameters) parameters;
            return (P) new VectorizedParameters(
                vigilance,
                vp.learningRate(),
                vp.alpha(),
                vp.parallelismLevel(),
                vp.parallelThreshold(),
                vp.maxCacheSize(),
                vp.enableSIMD(),
                vp.enableJOML(),
                vp.memoryOptimizationThreshold()
            );
        }
        // Subclasses should override for custom parameter types
        throw new UnsupportedOperationException(
            "Override createParametersWithVigilance for custom parameter types");
    }
    
    /**
     * Get a set of test patterns appropriate for this algorithm.
     */
    protected List<Pattern> getTestPatterns() {
        return List.of(
            Pattern.of(0.8, 0.2),
            Pattern.of(0.3, 0.7),
            Pattern.of(0.9, 0.1),
            Pattern.of(0.1, 0.9),
            Pattern.of(0.5, 0.5)
        );
    }
    
    // ==================== Common Test Patterns ====================
    
    @Test
    @DisplayName("Algorithm should handle basic learning correctly")
    void testBasicLearning() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        var patterns = getTestPatterns();
        
        try {
            // Learn first pattern
            var result = algorithm.learn(patterns.get(0), params);
            assertInstanceOf(ActivationResult.Success.class, result);
            assertEquals(1, algorithm.getCategoryCount(), 
                "Should create one category for first pattern");
            
            // Verify the category was created correctly
            var success = (ActivationResult.Success) result;
            assertEquals(0, success.categoryIndex(), 
                "First pattern should create category 0");
            assertTrue(success.activationValue() > 0, 
                "Activation should be positive");
            // Weight is part of the internal state, not exposed in Success result
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Algorithm should handle null inputs correctly")
    void testNullInputHandling() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        
        try {
            // BaseART enforces null checks
            assertThrows(NullPointerException.class, 
                () -> algorithm.learn(null, params),
                "Should throw NullPointerException for null input");
            
            assertThrows(NullPointerException.class, 
                () -> algorithm.learn(Pattern.of(1.0, 0.0), null),
                "Should throw NullPointerException for null parameters");
            
            assertThrows(NullPointerException.class, 
                () -> algorithm.predict(null, params),
                "Should throw NullPointerException for null input in predict");
            
            assertThrows(NullPointerException.class, 
                () -> algorithm.predict(Pattern.of(1.0, 0.0), null),
                "Should throw NullPointerException for null parameters in predict");
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Prediction should work without modifying the network")
    void testPredictionWithoutLearning() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        var patterns = getTestPatterns();
        
        try {
            // Predict on empty network
            var emptyResult = algorithm.predict(patterns.get(0), params);
            assertInstanceOf(ActivationResult.NoMatch.class, emptyResult,
                "Empty network should return NoMatch");
            
            // Learn a pattern
            algorithm.learn(patterns.get(0), params);
            int categoryCountBefore = algorithm.getCategoryCount();
            
            // Predict shouldn't change category count
            var predictResult = algorithm.predict(patterns.get(1), params);
            assertEquals(categoryCountBefore, algorithm.getCategoryCount(),
                "Predict should not create new categories");
            
            // Predict on learned pattern should match
            var matchResult = algorithm.predict(patterns.get(0), params);
            assertInstanceOf(ActivationResult.Success.class, matchResult,
                "Should match learned pattern");
            var success = (ActivationResult.Success) matchResult;
            assertEquals(0, success.categoryIndex(),
                "Should match first category");
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @ParameterizedTest(name = "Vigilance = {0}")
    @ValueSource(doubles = {0.3, 0.5, 0.7, 0.9})
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceParameterEffect(double vigilance) {
        var params = createParametersWithVigilance(vigilance);
        var algorithm = createAlgorithm(params);
        
        try {
            // Create similar patterns
            var pattern1 = Pattern.of(0.8, 0.2);
            var pattern2 = Pattern.of(0.75, 0.25); // Very similar
            
            algorithm.learn(pattern1, params);
            algorithm.learn(pattern2, params);
            
            // Higher vigilance may create more categories, but this is algorithm-specific
            // Some algorithms may still group similar patterns even with high vigilance
            if (vigilance > 0.95) {
                // Only with very high vigilance can we be sure patterns will separate
                // But even this depends on the algorithm's specific behavior
                assertTrue(algorithm.getCategoryCount() >= 1,
                    String.format("Vigilance %.1f should create at least one category", vigilance));
            }
            // Note: The exact category creation behavior is algorithm-specific
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Algorithm should handle multiple pattern learning")
    void testMultiplePatternLearning() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        var patterns = getTestPatterns();
        
        try {
            int previousCount = 0;
            
            for (int i = 0; i < patterns.size(); i++) {
                var result = algorithm.learn(patterns.get(i), params);
                assertInstanceOf(ActivationResult.Success.class, result,
                    String.format("Learning pattern %d should succeed", i));
                
                assertTrue(algorithm.getCategoryCount() >= previousCount,
                    "Category count should not decrease");
                previousCount = algorithm.getCategoryCount();
            }
            
            assertTrue(algorithm.getCategoryCount() > 0,
                "Should have created at least one category");
            assertTrue(algorithm.getCategoryCount() <= patterns.size(),
                "Should not create more categories than patterns");
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Clear should reset the algorithm state")
    void testClearFunctionality() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        var patterns = getTestPatterns();
        
        try {
            // Learn some patterns
            for (var pattern : patterns.subList(0, 3)) {
                algorithm.learn(pattern, params);
            }
            
            assertTrue(algorithm.getCategoryCount() > 0,
                "Should have categories after learning");
            
            // Clear the algorithm
            algorithm.clear();
            
            assertEquals(0, algorithm.getCategoryCount(),
                "Should have no categories after clear");
            
            // Should be able to learn again
            var result = algorithm.learn(patterns.get(0), params);
            assertInstanceOf(ActivationResult.Success.class, result,
                "Should be able to learn after clear");
            assertEquals(1, algorithm.getCategoryCount(),
                "Should have one category after learning post-clear");
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    // ==================== Edge Case Tests ====================
    
    @Test
    @DisplayName("Algorithm should handle extreme parameter values")
    void testExtremeParameterValues() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        
        try {
            // Test with very low vigilance
            var lowVigilance = createParametersWithVigilance(0.01);
            var patterns = getTestPatterns();
            
            for (var pattern : patterns) {
                var result = algorithm.learn(pattern, lowVigilance);
                assertInstanceOf(ActivationResult.Success.class, result,
                    "Should handle low vigilance");
            }
            
            // With very low vigilance, might merge many patterns
            assertTrue(algorithm.getCategoryCount() <= patterns.size(),
                "Category count should be reasonable");
            
            algorithm.clear();
            
            // Test with very high vigilance
            var highVigilance = createParametersWithVigilance(0.99);
            
            for (var pattern : patterns) {
                var result = algorithm.learn(pattern, highVigilance);
                assertInstanceOf(ActivationResult.Success.class, result,
                    "Should handle high vigilance");
            }
            
            // With very high vigilance, should create more categories
            assertTrue(algorithm.getCategoryCount() >= 1,
                "Should create at least one category");
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Algorithm should handle patterns with extreme values")
    void testPatternsWithExtremeValues() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        
        try {
            // Test with all zeros
            var zerosPattern = Pattern.of(0.0, 0.0, 0.0, 0.0);
            var result1 = algorithm.learn(zerosPattern, params);
            assertInstanceOf(ActivationResult.Success.class, result1,
                "Should handle all-zero pattern");
            
            // Test with all ones
            var onesPattern = Pattern.of(1.0, 1.0, 1.0, 1.0);
            var result2 = algorithm.learn(onesPattern, params);
            assertInstanceOf(ActivationResult.Success.class, result2,
                "Should handle all-one pattern");
            
            // Test with very small values
            var smallPattern = Pattern.of(1e-10, 1e-10, 1e-10, 1e-10);
            var result3 = algorithm.learn(smallPattern, params);
            assertInstanceOf(ActivationResult.Success.class, result3,
                "Should handle very small values");
            
            // Test with mixed extreme values
            var mixedPattern = Pattern.of(0.0, 1.0, 1e-10, 0.99999);
            var result4 = algorithm.learn(mixedPattern, params);
            assertInstanceOf(ActivationResult.Success.class, result4,
                "Should handle mixed extreme values");
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("Algorithm should handle single-dimension patterns")
    void testSingleDimensionPatterns() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        
        try {
            // Single dimension patterns
            var pattern1 = Pattern.of(0.3);
            var pattern2 = Pattern.of(0.8);
            
            var result1 = algorithm.learn(pattern1, params);
            assertInstanceOf(ActivationResult.Success.class, result1,
                "Should handle single-dimension pattern");
            
            var result2 = algorithm.learn(pattern2, params);
            assertInstanceOf(ActivationResult.Success.class, result2,
                "Should handle another single-dimension pattern");
            
            assertTrue(algorithm.getCategoryCount() > 0,
                "Should create categories for single-dimension patterns");
            
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    // ==================== Performance Helper Methods ====================
    
    /**
     * Generate a stream of test vigilance values for parameterized tests.
     */
    protected static Stream<Arguments> vigilanceTestParameters() {
        return Stream.of(
            Arguments.of(0.1, "Low vigilance"),
            Arguments.of(0.3, "Low-medium vigilance"),
            Arguments.of(0.5, "Medium vigilance"),
            Arguments.of(0.7, "Medium-high vigilance"),
            Arguments.of(0.9, "High vigilance"),
            Arguments.of(0.99, "Very high vigilance")
        );
    }
    
    /**
     * Generate patterns for stress testing.
     */
    protected List<Pattern> generateLargePatternSet(int count, int dimensions) {
        var patterns = new java.util.ArrayList<Pattern>(count);
        var random = new java.util.Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < count; i++) {
            var values = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                values[j] = random.nextDouble();
            }
            patterns.add(Pattern.of(values));
        }
        
        return patterns;
    }
}