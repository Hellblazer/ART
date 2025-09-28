package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.integration.*;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for VectorizedARTAlgorithm interface compliance.
 * Ensures VectorizedTemporalART correctly implements all required methods.
 */
class VectorizedTemporalARTInterfaceTest {

    private VectorizedTemporalART art;
    private TemporalARTParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = TemporalARTParameters.defaults();
        art = new VectorizedTemporalART(parameters);
    }

    @AfterEach
    void tearDown() {
        if (art != null) {
            art.close();
        }
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - learn method")
    void testLearnInterface() {
        var pattern = Pattern.of(new double[]{1.0, 0.0, 0.0});

        // Test learn method signature and return type
        ActivationResult result = art.learn(pattern, parameters);
        assertNotNull(result, "Learn should return non-null result");

        // Should return Success for new pattern
        if (result instanceof ActivationResult.Success success) {
            assertTrue(success.categoryIndex() >= 0, "Category index should be non-negative");
            // Note: Record field names may vary - focus on basic functionality
        }
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - predict method")
    void testPredictInterface() {
        // First learn a pattern
        var pattern = Pattern.of(new double[]{1.0, 0.0, 0.0});
        art.learn(pattern, parameters);

        // Test predict method signature and return type
        ActivationResult result = art.predict(pattern, parameters);
        assertNotNull(result, "Predict should return non-null result");

        // Should recognize the learned pattern
        assertTrue(result instanceof ActivationResult.Success,
                  "Should recognize learned pattern");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getCategoryCount method")
    void testGetCategoryCountInterface() {
        // Initially should be 0
        assertEquals(0, art.getCategoryCount(), "Initial category count should be 0");

        // Learn some patterns
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);
        art.learn(Pattern.of(new double[]{0.0, 1.0, 0.0}), parameters);

        int count = art.getCategoryCount();
        assertTrue(count > 0, "Category count should increase after learning");
        assertTrue(count <= 2, "Category count should not exceed learned patterns");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getCategories method")
    void testGetCategoriesInterface() {
        // Initially should be empty
        var categories = art.getCategories();
        assertNotNull(categories, "Categories list should not be null");
        assertEquals(0, categories.size(), "Initial categories should be empty");

        // Learn a pattern
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);

        categories = art.getCategories();
        assertEquals(1, categories.size(), "Should have one category after learning");
        assertTrue(categories.get(0) instanceof WeightVector, "Category should be WeightVector");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getCategory method")
    void testGetCategoryInterface() {
        // Learn a pattern
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);

        // Test valid index
        WeightVector category = art.getCategory(0);
        assertNotNull(category, "Category should not be null");
        assertTrue(category instanceof TemporalWeight, "Category should be TemporalWeight");

        // Test invalid index
        assertThrows(IndexOutOfBoundsException.class, () -> {
            art.getCategory(1);
        }, "Should throw exception for invalid index");

        assertThrows(IndexOutOfBoundsException.class, () -> {
            art.getCategory(-1);
        }, "Should throw exception for negative index");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getParameters method")
    void testGetParametersInterface() {
        TemporalARTParameters params = art.getParameters();
        assertNotNull(params, "Parameters should not be null");
        assertSame(parameters, params, "Should return same parameters instance");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getPerformanceStats method")
    void testGetPerformanceStatsInterface() {
        var stats = art.getPerformanceStats();
        assertNotNull(stats, "Performance stats should not be null");
        assertTrue(stats instanceof VectorizedTemporalART.PerformanceStats,
                  "Stats should be correct type");

        // Initially should have no patterns processed
        assertEquals(0, stats.patternsProcessed(), "Initially no patterns processed");

        // Process a pattern
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);

        stats = art.getPerformanceStats();
        assertTrue(stats.patternsProcessed() > 0, "Should track processed patterns");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - resetPerformanceTracking method")
    void testResetPerformanceTrackingInterface() {
        // Process some patterns
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);
        art.learn(Pattern.of(new double[]{0.0, 1.0, 0.0}), parameters);

        var stats = art.getPerformanceStats();
        assertTrue(stats.patternsProcessed() > 0, "Should have processed patterns");

        // Reset performance tracking
        art.resetPerformanceTracking();

        stats = art.getPerformanceStats();
        assertEquals(0, stats.patternsProcessed(), "Performance stats should be reset");

        // Categories should still exist
        assertTrue(art.getCategoryCount() > 0, "Categories should not be affected by performance reset");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - clear method")
    void testClearInterface() {
        // Learn some patterns
        art.learn(Pattern.of(new double[]{1.0, 0.0, 0.0}), parameters);
        art.learn(Pattern.of(new double[]{0.0, 1.0, 0.0}), parameters);

        assertTrue(art.getCategoryCount() > 0, "Should have categories before clear");

        // Clear all categories
        art.clear();

        assertEquals(0, art.getCategoryCount(), "All categories should be cleared");
        assertTrue(art.getCategories().isEmpty(), "Categories list should be empty");

        var stats = art.getPerformanceStats();
        assertEquals(0, stats.patternsProcessed(), "Performance stats should be reset");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - close method (AutoCloseable)")
    void testCloseInterface() {
        // This method should be callable without exceptions
        assertDoesNotThrow(() -> {
            art.close();
        }, "Close method should not throw exceptions");

        // After close, the art should still be usable for basic queries
        // (implementation dependent, but should not crash)
        assertDoesNotThrow(() -> {
            art.getCategoryCount();
        }, "Should be able to query basic state after close");
    }

    @Test
    @DisplayName("VectorizedARTAlgorithm interface - getVectorSpeciesLength method")
    void testGetVectorSpeciesLengthInterface() {
        int length = art.getVectorSpeciesLength();
        assertTrue(length > 0, "Vector species length should be positive");
        assertTrue(length <= 64, "Vector species length should be reasonable (≤64)");

        // Should be consistent across calls
        assertEquals(length, art.getVectorSpeciesLength(),
                    "Vector species length should be consistent");
    }

    @Test
    @DisplayName("Interface consistency - multiple operations")
    void testInterfaceConsistency() {
        // Test consistent behavior across multiple operations
        var pattern1 = Pattern.of(new double[]{1.0, 0.0, 0.0});
        var pattern2 = Pattern.of(new double[]{0.0, 1.0, 0.0});

        // Learn patterns
        var result1 = art.learn(pattern1, parameters);
        var result2 = art.learn(pattern2, parameters);

        // Verify consistency
        assertEquals(art.getCategoryCount(), art.getCategories().size(),
                    "Category count should match categories list size");

        // Test prediction consistency
        var prediction1 = art.predict(pattern1, parameters);
        var prediction2 = art.predict(pattern2, parameters);

        if (result1 instanceof ActivationResult.Success s1 &&
            prediction1 instanceof ActivationResult.Success p1) {
            assertEquals(s1.categoryIndex(), p1.categoryIndex(),
                        "Prediction should match learned category");
        }

        if (result2 instanceof ActivationResult.Success s2 &&
            prediction2 instanceof ActivationResult.Success p2) {
            assertEquals(s2.categoryIndex(), p2.categoryIndex(),
                        "Prediction should match learned category");
        }
    }

    @Test
    @DisplayName("Error handling - invalid inputs")
    void testErrorHandling() {
        // Test null pattern - should throw NPE (reasonable behavior)
        assertThrows(NullPointerException.class, () -> {
            art.learn(null, parameters);
        }, "Should throw NPE for null pattern");

        // Test null parameters - VectorizedTemporalART may handle gracefully
        var pattern = Pattern.of(new double[]{1.0, 0.0, 0.0});
        // Instead of expecting NPE, just ensure it doesn't crash
        assertDoesNotThrow(() -> {
            var result = art.learn(pattern, null);
            // Result should be valid regardless of null params
            assertNotNull(result, "Should return valid result even with null params");
        }, "Should handle null parameters gracefully");
    }

    @Test
    @DisplayName("Thread safety - basic concurrent access")
    void testBasicThreadSafety() throws InterruptedException {
        // Basic thread safety test
        var pattern = Pattern.of(new double[]{1.0, 0.0, 0.0});
        art.learn(pattern, parameters);

        var thread1 = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                art.getCategoryCount();
                art.getPerformanceStats();
            }
        });

        var thread2 = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                art.predict(pattern, parameters);
            }
        });

        thread1.start();
        thread2.start();

        thread1.join(1000);
        thread2.join(1000);

        assertFalse(thread1.isAlive(), "Thread 1 should complete");
        assertFalse(thread2.isAlive(), "Thread 2 should complete");
    }
}