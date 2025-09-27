package com.hellblazer.art.hybrid.pan;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PAN implementation.
 */
class PANTest {

    private PAN pan;
    private PANParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = PANParameters.defaultParameters();
        pan = new PAN(parameters);
    }

    @AfterEach
    void tearDown() {
        if (pan != null) {
            pan.close();
        }
    }

    @Test
    void testBasicLearning() {
        // Create a simple pattern (simulated 28x28 image)
        double[] imageData = new double[784];
        for (int i = 0; i < 784; i++) {
            imageData[i] = Math.random();
        }
        Pattern input = new DenseVector(imageData);

        // Learn the pattern
        ActivationResult result = pan.learn(input, parameters);

        assertNotNull(result);
        assertInstanceOf(ActivationResult.Success.class, result);

        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertTrue(success.activationValue() >= 0.0);
        assertNotNull(success.updatedWeight());

        // Verify category was created
        assertEquals(1, pan.getCategoryCount());
    }

    @Test
    void testSupervisedLearning() {
        // Create input pattern
        double[] imageData = new double[784];
        Random rand = new Random(42);
        for (int i = 0; i < 784; i++) {
            imageData[i] = rand.nextDouble();
        }
        Pattern input = new DenseVector(imageData);

        // Create target pattern (one-hot encoded for digit 3)
        double[] targetData = new double[10];
        targetData[3] = 1.0;
        Pattern target = new DenseVector(targetData);

        // Supervised learning
        ActivationResult result = pan.learnSupervised(input, target, parameters);

        assertNotNull(result);
        assertInstanceOf(ActivationResult.Success.class, result);

        assertEquals(1, pan.getCategoryCount());
    }

    @Test
    void testPrediction() {
        // Learn multiple patterns
        Random rand = new Random(42);
        List<Pattern> patterns = new ArrayList<>();

        for (int i = 0; i < 3; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                data[j] = rand.nextDouble() * (i + 1) / 3.0;  // Different scales
            }
            patterns.add(new DenseVector(data));
        }

        // Learn all patterns
        for (Pattern p : patterns) {
            pan.learn(p, parameters);
        }

        assertEquals(3, pan.getCategoryCount());

        // Predict on first pattern (should match category 0)
        ActivationResult prediction = pan.predict(patterns.get(0), parameters);
        assertNotNull(prediction);
        assertInstanceOf(ActivationResult.Success.class, prediction);

        var success = (ActivationResult.Success) prediction;
        assertEquals(0, success.categoryIndex());
    }

    @Test
    void testBatchOperations() {
        // Create batch of patterns
        List<Pattern> batch = new ArrayList<>();
        Random rand = new Random(42);

        for (int i = 0; i < 5; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                data[j] = rand.nextDouble();
            }
            batch.add(new DenseVector(data));
        }

        // Batch learning
        List<ActivationResult> results = pan.learnBatch(batch, parameters);

        assertNotNull(results);
        assertEquals(5, results.size());
        for (var result : results) {
            assertInstanceOf(ActivationResult.Success.class, result);
        }

        // Batch prediction
        List<ActivationResult> predictions = pan.predictBatch(batch, parameters);
        assertNotNull(predictions);
        assertEquals(5, predictions.size());
    }

    @Test
    void testMaxCategories() {
        // Set low max categories
        var limitedParams = new PANParameters(
            0.7, 2,  // Only 2 max categories
            parameters.cnnConfig(),
            false,
            0.01, 0.9, 0.0001, true, 64,
            0.95, 0.8,
            100, 10, 0.1,
            0.1
        );

        try (var limitedPAN = new PAN(limitedParams)) {
            Random rand = new Random(42);

            // Try to learn 5 different patterns
            for (int i = 0; i < 5; i++) {
                double[] data = new double[784];
                for (int j = 0; j < 784; j++) {
                    data[j] = rand.nextDouble() * (i + 1);
                }
                Pattern pattern = new DenseVector(data);
                limitedPAN.learn(pattern, limitedParams);
            }

            // Should be capped at 2 categories
            assertEquals(2, limitedPAN.getCategoryCount());
        }
    }

    @Test
    void testClearAndReset() {
        // Learn some patterns
        Random rand = new Random(42);
        for (int i = 0; i < 3; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                data[j] = rand.nextDouble();
            }
            pan.learn(new DenseVector(data), parameters);
        }

        assertEquals(3, pan.getCategoryCount());

        // Clear
        pan.clear();
        assertEquals(0, pan.getCategoryCount());

        // Can learn again after clear
        double[] newData = new double[784];
        for (int i = 0; i < 784; i++) {
            newData[i] = rand.nextDouble();
        }
        pan.learn(new DenseVector(newData), parameters);
        assertEquals(1, pan.getCategoryCount());
    }

    @Test
    void testPerformanceTracking() {
        // Learn some patterns
        Random rand = new Random(42);
        for (int i = 0; i < 5; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                data[j] = rand.nextDouble();
            }
            pan.learn(new DenseVector(data), parameters);
        }

        // Get performance stats
        var stats = pan.getPerformanceStats();
        assertNotNull(stats);
        assertInstanceOf(java.util.Map.class, stats);

        @SuppressWarnings("unchecked")
        var statsMap = (java.util.Map<String, Object>) stats;

        assertEquals(5L, statsMap.get("totalSamples"));
        assertEquals(5, statsMap.get("categoryCount"));
        assertTrue((Long) statsMap.get("trainingTimeMs") >= 0);

        // Reset tracking
        pan.resetPerformanceTracking();

        stats = pan.getPerformanceStats();
        @SuppressWarnings("unchecked")
        var resetStats = (java.util.Map<String, Object>) stats;
        assertEquals(0L, resetStats.get("totalSamples"));
    }

    @Test
    void testGetCategories() {
        // Learn patterns
        Random rand = new Random(42);
        for (int i = 0; i < 3; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                data[j] = rand.nextDouble();
            }
            pan.learn(new DenseVector(data), parameters);
        }

        // Get all categories
        var categories = pan.getCategories();
        assertNotNull(categories);
        assertEquals(3, categories.size());

        // Get specific category
        var category0 = pan.getCategory(0);
        assertNotNull(category0);
        assertEquals(categories.get(0), category0);

        // Test out of bounds
        assertThrows(IndexOutOfBoundsException.class, () -> pan.getCategory(5));
        assertThrows(IndexOutOfBoundsException.class, () -> pan.getCategory(-1));
    }

    @Test
    void testNoMatchResult() {
        // Create pattern but don't learn
        double[] data = new double[784];
        Pattern pattern = new DenseVector(data);

        // With no categories, predict should return NoMatch
        ActivationResult result = pan.predict(pattern, parameters);
        assertNotNull(result);
        assertInstanceOf(ActivationResult.NoMatch.class, result);
    }
}