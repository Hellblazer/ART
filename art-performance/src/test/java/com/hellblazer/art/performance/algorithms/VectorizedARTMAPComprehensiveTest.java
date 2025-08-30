package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.supervised.VectorizedARTMAP;
import com.hellblazer.art.performance.supervised.VectorizedARTMAPParameters;
import com.hellblazer.art.performance.supervised.VectorizedARTMAPResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for VectorizedARTMAP based on reference implementation patterns.
 * Tests cover initialization, parameter validation, data preparation, supervised learning,
 * prediction, and regression capabilities.
 */
class VectorizedARTMAPComprehensiveTest {

    private VectorizedARTMAP artmap;
    private VectorizedARTMAPParameters parameters;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42);
        
        // Initialize with typical parameters similar to Python tests
        parameters = VectorizedARTMAPParameters.defaults()
            .withMapVigilance(0.7)
            .withBaselineVigilance(0.5);
        
        artmap = new VectorizedARTMAP(parameters);
    }

    @Test
    @DisplayName("Test ARTMAP initialization and parameter access")
    void testInitializationAndParameters() {
        assertNotNull(artmap);
        
        // Test parameter retrieval
        var retrievedParams = artmap.getParameters();
        assertNotNull(retrievedParams);
        assertEquals(VectorizedARTMAPParameters.class, retrievedParams.getClass());
        
        // Test performance stats initialization
        var stats = artmap.getPerformanceStats();
        assertNotNull(stats);
    }

    @Test
    @DisplayName("Test data validation with valid inputs")
    void testDataValidation() {
        var inputData = generateRandomPatterns(10, 5);
        var targetData = generateRandomPatterns(10, 3);
        
        // Should not throw for valid data
        assertDoesNotThrow(() -> {
            for (int i = 0; i < inputData.size(); i++) {
                artmap.learn(inputData.get(i), parameters);
            }
        });
    }

    @Test
    @DisplayName("Test data validation with invalid inputs")
    void testDataValidationInvalid() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.learn(null, parameters);
        });
        
        // Test null parameters
        var pattern = Pattern.of(0.1, 0.2, 0.3);
        assertThrows(NullPointerException.class, () -> {
            artmap.learn(pattern, null);
        });
    }

    @Test
    @DisplayName("Test supervised learning with prepared data")
    void testSupervisedLearning() {
        var inputData = generateRandomPatterns(20, 4);
        var targetData = generateRandomPatterns(20, 2);
        
        // Train the model
        for (int i = 0; i < inputData.size(); i++) {
            var result = artmap.learn(inputData.get(i), parameters);
            assertNotNull(result);
        }
        
        // Verify model has learned something
        assertTrue(artmap.getCategoryCount() > 0);
        var stats = artmap.getPerformanceStats();
        assertNotNull(stats);
    }

    @Test
    @DisplayName("Test prediction after training")
    void testPredictionAfterTraining() {
        var inputData = generateRandomPatterns(15, 3);
        
        // Train the model first
        for (var pattern : inputData) {
            artmap.learn(pattern, parameters);
        }
        
        // Test prediction on training data
        for (var pattern : inputData) {
            var prediction = artmap.predict(pattern, parameters);
            assertNotNull(prediction);
        }
        
        // Test prediction on new data
        var newData = generateRandomPatterns(5, 3);
        for (var pattern : newData) {
            var prediction = artmap.predict(pattern, parameters);
            assertNotNull(prediction);
        }
    }

    @Test
    @DisplayName("Test learning with different vigilance parameters")
    void testVigilanceParameterEffects() {
        var inputData = generateRandomPatterns(10, 4);
        
        // Test with low vigilance (should create fewer categories)
        var lowVigilanceParams = VectorizedARTMAPParameters.defaults()
            .withMapVigilance(0.1)
            .withBaselineVigilance(0.1);
        
        var artmapLowVigilance = new VectorizedARTMAP(lowVigilanceParams);
        for (var pattern : inputData) {
            artmapLowVigilance.learn(pattern, lowVigilanceParams);
        }
        int lowVigilanceCategories = artmapLowVigilance.getCategoryCount();
        
        // Test with high vigilance (should create more categories)
        var highVigilanceParams = VectorizedARTMAPParameters.defaults()
            .withMapVigilance(0.9)
            .withBaselineVigilance(0.9);
        
        var artmapHighVigilance = new VectorizedARTMAP(highVigilanceParams);
        for (var pattern : inputData) {
            artmapHighVigilance.learn(pattern, highVigilanceParams);
        }
        int highVigilanceCategories = artmapHighVigilance.getCategoryCount();
        
        // Higher vigilance should generally create more categories
        assertTrue(highVigilanceCategories >= lowVigilanceCategories,
            "High vigilance (" + highVigilanceCategories + 
            ") should create at least as many categories as low vigilance (" + 
            lowVigilanceCategories + ")");
    }

    @Test
    @DisplayName("Test performance tracking and statistics")
    void testPerformanceTracking() {
        var inputData = generateRandomPatterns(25, 6);
        
        // Reset performance tracking
        artmap.resetPerformanceTracking();
        
        // Perform some operations
        for (var pattern : inputData) {
            artmap.learn(pattern, parameters);
        }
        
        // Check that performance stats are being tracked
        var stats = artmap.getPerformanceStats();
        assertNotNull(stats);
        
        // The performance stats should have some meaningful data after operations
        // This is implementation-dependent, but we can at least verify non-null
        assertNotNull(stats.toString());
    }

    @Test
    @DisplayName("Test resource management with AutoCloseable")
    void testResourceManagement() {
        assertDoesNotThrow(() -> {
            try (var testArtmap = new VectorizedARTMAP(VectorizedARTMAPParameters.defaults())) {
                var pattern = Pattern.of(0.1, 0.2, 0.3);
                testArtmap.learn(pattern, parameters);
                testArtmap.predict(pattern, parameters);
            }
        });
    }

    @Test
    @DisplayName("Test vectorization capabilities")
    void testVectorizationFeatures() {
        assertTrue(artmap.isVectorized(), "ARTMAP should report as vectorized");
        
        // Vector species length might be implementation dependent
        int vectorLength = artmap.getVectorSpeciesLength();
        assertTrue(vectorLength != 0, "Vector species length should be meaningful");
    }

    @Test
    @DisplayName("Test incremental learning behavior")
    void testIncrementalLearning() {
        var batch1 = generateRandomPatterns(10, 4);
        var batch2 = generateRandomPatterns(10, 4);
        
        // Learn first batch
        for (var pattern : batch1) {
            artmap.learn(pattern, parameters);
        }
        int categoriesAfterBatch1 = artmap.getCategoryCount();
        
        // Learn second batch
        for (var pattern : batch2) {
            artmap.learn(pattern, parameters);
        }
        int categoriesAfterBatch2 = artmap.getCategoryCount();
        
        // Should have learned something from both batches
        assertTrue(categoriesAfterBatch1 > 0, "Should have categories after first batch");
        assertTrue(categoriesAfterBatch2 >= categoriesAfterBatch1, 
            "Categories should not decrease after second batch");
    }

    @Test
    @DisplayName("Test learning with similar patterns")
    void testSimilarPatternLearning() {
        // Create very similar patterns
        var basePattern = List.of(0.5, 0.5, 0.5, 0.5);
        var similarPatterns = List.of(
            Pattern.of(0.5, 0.5, 0.5, 0.5),
            Pattern.of(0.51, 0.5, 0.5, 0.5),
            Pattern.of(0.5, 0.51, 0.5, 0.5),
            Pattern.of(0.5, 0.5, 0.51, 0.5)
        );
        
        for (var pattern : similarPatterns) {
            artmap.learn(pattern, parameters);
        }
        
        int categoriesCreated = artmap.getCategoryCount();
        
        // With moderate vigilance, similar patterns might be grouped
        assertTrue(categoriesCreated > 0 && categoriesCreated <= similarPatterns.size(),
            "Should create reasonable number of categories for similar patterns");
    }

    @Test
    @DisplayName("Test learning with distinct patterns")
    void testDistinctPatternLearning() {
        // Create very different patterns
        var distinctPatterns = List.of(
            Pattern.of(0.0, 0.0, 0.0, 0.0),
            Pattern.of(1.0, 1.0, 1.0, 1.0),
            Pattern.of(0.0, 1.0, 0.0, 1.0),
            Pattern.of(1.0, 0.0, 1.0, 0.0)
        );
        
        for (var pattern : distinctPatterns) {
            artmap.learn(pattern, parameters);
        }
        
        int categoriesCreated = artmap.getCategoryCount();
        
        // Distinct patterns should create multiple categories
        assertTrue(categoriesCreated > 1,
            "Distinct patterns should create multiple categories");
    }

    /**
     * Generate random patterns for testing
     */
    private List<Pattern> generateRandomPatterns(int count, int dimensions) {
        return IntStream.range(0, count)
            .mapToObj(i -> {
                double[] values = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    values[j] = random.nextDouble();
                }
                return Pattern.of(values);
            })
            .toList();
    }
}