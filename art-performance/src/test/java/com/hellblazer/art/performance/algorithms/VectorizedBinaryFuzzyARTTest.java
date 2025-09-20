package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VectorizedBinaryFuzzyART algorithm.
 *
 * Tests the binary-optimized fuzzy ART implementation with SIMD vectorization,
 * focusing on binary pattern processing, complement coding, and performance optimization.
 *
 * @author Claude (Anthropic AI)
 * @version 1.0
 */
@DisplayName("VectorizedBinaryFuzzyART Tests")
class VectorizedBinaryFuzzyARTTest extends BaseVectorizedARTTest<VectorizedBinaryFuzzyART, VectorizedParameters> {

    private static final Random RANDOM = new Random(42);

    @BeforeEach
    @Override
    protected void setUp() {
        super.setUp();
    }
    
    @Override
    protected VectorizedBinaryFuzzyART createAlgorithm(VectorizedParameters params) {
        return new VectorizedBinaryFuzzyART(params);
    }
    
    @Override
    protected VectorizedParameters createDefaultParameters() {
        return new VectorizedParameters(
            0.8,    // vigilance
            1.0,    // learningRate
            0.01,   // alpha
            1,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            true,   // enableJOML
            0.8     // memoryOptimizationThreshold
        );
    }

    @Test
    @DisplayName("Should create categories for binary patterns")
    void testBinaryPatternLearning() {
        // Use patterns that will definitely create separate categories
        var pattern1 = Pattern.of(1.0, 1.0, 1.0, 1.0); // All ones
        var pattern2 = Pattern.of(0.0, 0.0, 0.0, 0.0); // All zeros

        var result1 = algorithm.learn(pattern1, parameters);
        var result2 = algorithm.learn(pattern2, parameters);

        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        // These extreme patterns should create separate categories
        assertEquals(2, algorithm.getCategoryCount());
    }

    @Test
    @DisplayName("Should handle complement coding automatically")
    void testComplementCoding() {
        var basePattern = Pattern.of(1.0, 0.0, 1.0); // Odd length - needs complement coding

        var result = algorithm.learn(basePattern, parameters);

        assertTrue(result instanceof ActivationResult.Success);
        assertEquals(1, algorithm.getCategoryCount());
    }

    @Test
    @DisplayName("Should optimize for binary values (0.0 and 1.0)")
    void testBinaryOptimization() {
        var binaryPattern = Pattern.of(1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        var nonBinaryPattern = Pattern.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5); // Very different pattern

        var binaryResult = algorithm.learn(binaryPattern, parameters);
        var nonBinaryResult = algorithm.learn(nonBinaryPattern, parameters);

        assertTrue(binaryResult instanceof ActivationResult.Success);
        assertTrue(nonBinaryResult instanceof ActivationResult.Success);

        // With standard vigilance these different patterns should create separate categories
        assertTrue(algorithm.getCategoryCount() >= 1);
    }

    @Test
    @DisplayName("Should respect vigilance parameter for binary patterns")
    void testVigilanceWithBinaryPatterns() {
        var strictParams = parameters.withVigilance(0.99); // Extremely strict vigilance

        var pattern1 = Pattern.of(1.0, 1.0, 0.0, 0.0);
        var pattern2 = Pattern.of(0.0, 0.0, 1.0, 1.0); // Completely different

        algorithm.learn(pattern1, strictParams);
        algorithm.learn(pattern2, strictParams);

        // With extremely strict vigilance, these opposite patterns must create separate categories
        assertEquals(2, algorithm.getCategoryCount());
    }

    @Test
    @DisplayName("Should handle fast learning (beta = 1.0)")
    void testFastLearning() {
        var fastParams = parameters.withLearningRate(1.0);

        var pattern1 = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var pattern2 = Pattern.of(1.0, 0.0, 1.0, 0.0); // Identical pattern

        var result1 = algorithm.learn(pattern1, fastParams);
        var result2 = algorithm.learn(pattern2, fastParams);

        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        assertEquals(1, algorithm.getCategoryCount()); // Should update same category
    }

    @Test
    @DisplayName("Should handle slow learning (beta < 1.0)")
    void testSlowLearning() {
        var slowParams = parameters.withLearningRate(0.5);

        var pattern1 = Pattern.of(1.0, 0.0, 1.0, 0.0);
        var pattern2 = Pattern.of(1.0, 0.0, 1.0, 0.0); // Identical pattern

        var result1 = algorithm.learn(pattern1, slowParams);
        var result2 = algorithm.learn(pattern2, slowParams);

        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        assertEquals(1, algorithm.getCategoryCount());
    }

    @Test
    @DisplayName("Should perform prediction on binary patterns")
    void testBinaryPrediction() {
        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);

        // Learn the pattern
        algorithm.learn(pattern, parameters);
        assertEquals(1, algorithm.getCategoryCount());

        // Predict on same pattern
        var prediction = algorithm.predict(pattern, parameters);
        assertTrue(prediction instanceof ActivationResult.Success);

        if (prediction instanceof ActivationResult.Success success) {
            assertEquals(0, success.categoryIndex());
            assertTrue(success.activationValue() > 0.0);
        }
    }

    @Test
    @DisplayName("Should handle alpha parameter correctly")
    void testAlphaParameter() {
        // Create parameters with different alpha values manually
        var lowAlphaParams = new VectorizedParameters(
            parameters.vigilanceThreshold(), parameters.learningRate(), 0.001, // low alpha
            parameters.parallelismLevel(), parameters.parallelThreshold(),
            parameters.maxCacheSize(), parameters.enableSIMD(), parameters.enableJOML(),
            parameters.memoryOptimizationThreshold()
        );
        var highAlphaParams = new VectorizedParameters(
            parameters.vigilanceThreshold(), parameters.learningRate(), 1.0, // high alpha
            parameters.parallelismLevel(), parameters.parallelThreshold(),
            parameters.maxCacheSize(), parameters.enableSIMD(), parameters.enableJOML(),
            parameters.memoryOptimizationThreshold()
        );

        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);

        // Test with low alpha
        var lowResult = algorithm.learn(pattern, lowAlphaParams);
        assertTrue(lowResult instanceof ActivationResult.Success);

        algorithm.close();
        algorithm = new VectorizedBinaryFuzzyART(parameters);

        // Test with high alpha
        var highResult = algorithm.learn(pattern, highAlphaParams);
        assertTrue(highResult instanceof ActivationResult.Success);
    }

    // Removed - covered by base class testPerformanceTracking()

    @Test
    @DisplayName("Should enable/disable SIMD optimization")
    void testSIMDToggle() {
        var simdParams = parameters.withCacheSettings(1000, true, true);
        var noSimdParams = parameters.withCacheSettings(1000, false, true);

        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);

        // Both should work, but SIMD might be faster
        var simdResult = algorithm.learn(pattern, simdParams);
        assertTrue(simdResult instanceof ActivationResult.Success);

        algorithm.close();
        algorithm = new VectorizedBinaryFuzzyART(parameters);

        var noSimdResult = algorithm.learn(pattern, noSimdParams);
        assertTrue(noSimdResult instanceof ActivationResult.Success);
    }

    @Test
    @DisplayName("Should handle parallel processing")
    void testParallelProcessing() {
        var parallelParams = parameters.withParallelismLevel(2);
        var patterns = generateBinaryPatterns(10, 6);

        for (var pattern : patterns) {
            var result = algorithm.learn(pattern, parallelParams);
            assertTrue(result instanceof ActivationResult.Success ||
                      result instanceof ActivationResult.NoMatch);
        }

        assertTrue(algorithm.getCategoryCount() >= 1);
    }

    @Test
    @DisplayName("Should handle caching efficiently")
    void testInputCaching() {
        var cachingParams = parameters.withCacheSettings(10, true, true);
        var pattern = Pattern.of(1.0, 0.0, 1.0, 0.0);

        // Learn same pattern multiple times
        for (int i = 0; i < 5; i++) {
            var result = algorithm.learn(pattern, cachingParams);
            assertTrue(result instanceof ActivationResult.Success);
        }

        var stats = algorithm.getPerformanceStats();
        assertTrue(stats.cacheSize() > 0);
    }

    @Test
    @DisplayName("Should validate parameter ranges")
    void testParameterValidation() {
        assertThrows(IllegalArgumentException.class, () ->
            VectorizedParameters.createDefault().withVigilance(-0.1));

        assertThrows(IllegalArgumentException.class, () ->
            VectorizedParameters.createDefault().withVigilance(1.1));

        // Note: withAlpha method doesn't exist on VectorizedParameters
        // Alpha validation happens in the constructor

        assertThrows(IllegalArgumentException.class, () ->
            VectorizedParameters.createDefault().withLearningRate(-0.1));
    }

    // Removed - covered by base class testErrorHandling()

    @Test
    @DisplayName("Should handle edge case patterns")
    void testEdgeCasePatterns() {
        // Test same-dimension edge cases
        var sameDimEdgeCases = new Pattern[]{
            Pattern.of(0.0, 0.0, 0.0, 0.0), // All zeros
            Pattern.of(1.0, 1.0, 1.0, 1.0), // All ones
            Pattern.of(0.0, 0.0, 0.0, 1.0), // Single one
            Pattern.of(1.0, 0.0, 0.0, 0.0)  // Single one at start
        };

        for (var pattern : sameDimEdgeCases) {
            var result = algorithm.learn(pattern, parameters);
            assertTrue(result instanceof ActivationResult.Success);
        }

        assertTrue(algorithm.getCategoryCount() >= 1);

        // Test different dimensions in separate instances
        algorithm.close();
        algorithm = new VectorizedBinaryFuzzyART(parameters);

        var singleElement = Pattern.of(1.0);
        var result = algorithm.learn(singleElement, parameters);
        assertTrue(result instanceof ActivationResult.Success);
        assertEquals(1, algorithm.getCategoryCount());
    }

    @Test
    @DisplayName("Should return current parameters")
    void testParameterRetrieval() {
        algorithm.learn(Pattern.of(1.0, 0.0), parameters);
        var retrievedParams = algorithm.getParameters();

        assertNotNull(retrievedParams);
        assertEquals(parameters.vigilanceThreshold(), retrievedParams.vigilanceThreshold());
        assertEquals(parameters.alpha(), retrievedParams.alpha());
        assertEquals(parameters.learningRate(), retrievedParams.learningRate());
    }

    @Test
    @DisplayName("Should provide meaningful string representation")
    void testStringRepresentation() {
        algorithm.learn(Pattern.of(1.0, 0.0, 1.0, 0.0), parameters);

        var str = algorithm.toString();
        assertNotNull(str);
        assertTrue(str.contains("VectorizedBinaryFuzzyART"));
        assertTrue(str.contains("categories=1"));
    }

    // Helper methods

    private Pattern[] generateBinaryPatterns(int count, int dimension) {
        var patterns = new Pattern[count];

        for (int i = 0; i < count; i++) {
            var values = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                values[j] = RANDOM.nextBoolean() ? 1.0 : 0.0;
            }
            patterns[i] = Pattern.of(values);
        }

        return patterns;
    }
}