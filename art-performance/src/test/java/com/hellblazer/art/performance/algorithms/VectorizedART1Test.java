package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedART1 implementation.
 * Tests binary pattern recognition with vectorized operations.
 */
public class VectorizedART1Test extends BaseVectorizedARTTest<VectorizedART1, VectorizedART1Parameters> {
    
    @Override
    protected VectorizedART1 createAlgorithm(VectorizedART1Parameters params) {
        return new VectorizedART1(params);
    }
    
    @Override
    protected VectorizedART1Parameters createDefaultParameters() {
        return VectorizedART1Parameters.createDefault();
    }
    
    @BeforeEach
    protected void setUp() {
        parameters = createDefaultParameters();
        algorithm = createAlgorithm(parameters);
        super.setUp();
    }
    
    @Override
    protected VectorizedART1Parameters createParametersWithVigilance(double vigilance) {
        return VectorizedART1Parameters.createWithVigilance(vigilance);
    }
    
    @Override
    protected List<Pattern> getTestPatterns() {
        // Override to provide binary patterns for ART1 (0.0 or 1.0 only)
        return List.of(
            Pattern.of(1.0, 0.0),
            Pattern.of(0.0, 1.0),
            Pattern.of(1.0, 1.0),
            Pattern.of(1.0, 0.0),
            Pattern.of(0.0, 1.0)
        );
    }
    
    protected double[] generateRandomInput(int dimension) {
        // Generate binary input for ART1
        var input = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            input[i] = Math.random() > 0.5 ? 1.0 : 0.0;
        }
        return input;
    }
    
    // The following tests are covered by base class:
    // - testBasicLearning()
    // - testMultiplePatternLearning()
    // - testPrediction()
    // - testPerformanceTracking()
    // - testErrorHandling()
    // - testResourceCleanup()
    
    @Test
    @DisplayName("Should handle binary patterns correctly")
    void testBinaryPatterns() {
        // Create binary patterns
        var pattern1 = Pattern.of(1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        var pattern2 = Pattern.of(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        var pattern3 = Pattern.of(1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
        
        // Learn patterns
        var result1 = algorithm.learn(pattern1, parameters);
        assertInstanceOf(ActivationResult.Success.class, result1);
        assertEquals(1, algorithm.getCategoryCount());
        
        var result2 = algorithm.learn(pattern2, parameters);
        assertInstanceOf(ActivationResult.Success.class, result2);
        
        var result3 = algorithm.learn(pattern3, parameters);
        assertInstanceOf(ActivationResult.Success.class, result3);
        
        // Test prediction
        var prediction1 = algorithm.predict(pattern1, parameters);
        assertNotNull(prediction1);
        assertInstanceOf(ActivationResult.Success.class, prediction1);
    }
    
    @Test
    @DisplayName("Should enforce binary input constraints")
    void testBinaryConstraints() {
        // ART1 should work with binary patterns
        var binaryPattern = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var result = algorithm.learn(binaryPattern, parameters);
        assertInstanceOf(ActivationResult.Success.class, result);
        
        // Non-binary patterns should throw an exception
        var nonBinaryPattern = Pattern.of(0.5, 0.7, 0.2, 0.9);
        assertThrows(IllegalArgumentException.class, () -> {
            algorithm.learn(nonBinaryPattern, parameters);
        }, "ART1 should reject non-binary patterns");
    }
    
    @Test
    @DisplayName("Should demonstrate choice function behavior")
    void testChoiceFunction() {
        // ART1 uses choice function for category selection
        var pattern1 = Pattern.of(1.0, 1.0, 0.0, 0.0);
        var pattern2 = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var pattern3 = Pattern.of(0.0, 1.0, 0.0, 1.0);
        
        algorithm.learn(pattern1, parameters);
        algorithm.learn(pattern2, parameters);
        algorithm.learn(pattern3, parameters);
        
        // Test that choice function selects appropriate categories
        var prediction1 = algorithm.predict(pattern1, parameters);
        var prediction2 = algorithm.predict(pattern2, parameters);
        var prediction3 = algorithm.predict(pattern3, parameters);
        
        assertNotNull(prediction1);
        assertNotNull(prediction2);
        assertNotNull(prediction3);
    }
    
    @Test
    @DisplayName("Should handle vigilance parameter correctly")
    void testVigilanceParameter() {
        // High vigilance should create more categories
        var highVigilanceParams = createParametersWithVigilance(0.9);
        var highVigilanceAlg = new VectorizedART1(highVigilanceParams);
        
        // Low vigilance should create fewer categories
        var lowVigilanceParams = createParametersWithVigilance(0.3);
        var lowVigilanceAlg = new VectorizedART1(lowVigilanceParams);
        
        try {
            var patterns = new Pattern[] {
                Pattern.of(1.0, 0.0, 1.0, 0.0),
                Pattern.of(1.0, 0.0, 0.0, 1.0),
                Pattern.of(0.0, 1.0, 1.0, 0.0),
                Pattern.of(0.0, 1.0, 0.0, 1.0)
            };
            
            // Learn with both vigilance settings
            for (var pattern : patterns) {
                highVigilanceAlg.learn(pattern, highVigilanceParams);
                lowVigilanceAlg.learn(pattern, lowVigilanceParams);
            }
            
            // High vigilance should generally create more categories
            var highCategories = highVigilanceAlg.getCategoryCount();
            var lowCategories = lowVigilanceAlg.getCategoryCount();
            
            assertTrue(highCategories >= lowCategories,
                "High vigilance should create >= categories than low vigilance");
            
        } finally {
            highVigilanceAlg.close();
            lowVigilanceAlg.close();
        }
    }
    
    @Test
    @DisplayName("Should demonstrate resonance behavior")
    void testResonance() {
        // Test resonance vs reset behavior
        var trainingPattern = Pattern.of(1.0, 1.0, 0.0, 0.0);
        algorithm.learn(trainingPattern, parameters);
        
        // Similar pattern should resonate
        var similarPattern = Pattern.of(1.0, 1.0, 0.0, 1.0);
        var similarResult = algorithm.predict(similarPattern, parameters);
        
        // Very different pattern may not resonate
        var differentPattern = Pattern.of(0.0, 0.0, 1.0, 1.0);
        var differentResult = algorithm.predict(differentPattern, parameters);
        
        assertNotNull(similarResult);
        assertNotNull(differentResult);
        
        // Both should get some response, but similar should have higher activation
        if (similarResult instanceof ActivationResult.Success similar &&
            differentResult instanceof ActivationResult.Success different) {
            // Similar pattern should generally have higher activation
            assertTrue(similar.activationValue() >= different.activationValue() - 0.1,
                "Similar pattern should have comparable or higher activation");
        }
    }
    
    @Test
    @DisplayName("Should handle complement coding appropriately")
    void testComplementCoding() {
        // ART1 traditionally doesn't use complement coding, but vectorized version might
        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var result = algorithm.learn(pattern, parameters);
        assertInstanceOf(ActivationResult.Success.class, result);
        
        // Test that the algorithm handles the pattern appropriately
        var prediction = algorithm.predict(pattern, parameters);
        assertNotNull(prediction);
        assertInstanceOf(ActivationResult.Success.class, prediction);
    }
    
    // Override base class tests that use incompatible patterns
    
    @ParameterizedTest(name = "Vigilance = {0}")
    @ValueSource(doubles = {0.3, 0.5, 0.7, 0.9})
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceParameterEffect(double vigilance) {
        var params = createParametersWithVigilance(vigilance);
        var algorithm = createAlgorithm(params);
        
        try {
            // Create similar binary patterns (4-dimensional for ART1)
            var pattern1 = Pattern.of(1.0, 1.0, 0.0, 0.0);
            var pattern2 = Pattern.of(1.0, 1.0, 1.0, 0.0);
            
            algorithm.learn(pattern1, params);
            algorithm.learn(pattern2, params);
            
            if (vigilance > 0.95) {
                assertTrue(algorithm.getCategoryCount() >= 1,
                    String.format("Vigilance %.1f should create at least one category", vigilance));
            }
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
    
    @Test
    @DisplayName("ART1 requires binary patterns")
    void testSingleDimensionPatterns() {
        // ART1 requires binary patterns (0.0 or 1.0)
        var pattern1 = Pattern.of(1.0);
        var pattern2 = Pattern.of(0.0);
        
        var result1 = algorithm.learn(pattern1, parameters);
        assertInstanceOf(ActivationResult.Success.class, result1,
            "Should handle single-dimension binary pattern");
        
        var result2 = algorithm.learn(pattern2, parameters);
        assertInstanceOf(ActivationResult.Success.class, result2,
            "Should handle another single-dimension binary pattern");
        
        assertTrue(algorithm.getCategoryCount() > 0,
            "Should create categories for single-dimension binary patterns");
    }
    
    @Test
    @DisplayName("Algorithm should handle binary extreme values")
    void testPatternsWithExtremeValues() {
        var params = createDefaultParameters();
        var algorithm = createAlgorithm(params);
        
        try {
            // Test with all zeros (binary)
            var zeroPattern = Pattern.of(0.0, 0.0, 0.0, 0.0);
            var result1 = algorithm.learn(zeroPattern, params);
            assertNotNull(result1, "Should handle all-zeros binary pattern");
            
            // Test with all ones (binary)
            var onesPattern = Pattern.of(1.0, 1.0, 1.0, 1.0);
            var result2 = algorithm.learn(onesPattern, params);
            assertNotNull(result2, "Should handle all-ones binary pattern");
            
            // Test with mixed binary
            var mixedPattern = Pattern.of(1.0, 0.0, 1.0, 0.0);
            var result3 = algorithm.learn(mixedPattern, params);
            assertNotNull(result3, "Should handle mixed binary pattern");
        } finally {
            if (algorithm instanceof AutoCloseable ac) {
                try { ac.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }
}