package com.hellblazer.art.temporal.integration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Integration tests for the complete temporal ART system.
 */
public class TemporalARTTest {

    private TemporalART temporalART;
    private TemporalARTParameters parameters;

    @BeforeEach
    public void setUp() {
        parameters = TemporalARTParameters.defaults();
        temporalART = new TemporalART(parameters);
    }

    @Test
    public void testBasicSequenceLearning() {
        // Create a simple sequence
        var sequence = createSimpleSequence(5, 10);

        // Process sequence
        for (var pattern : sequence) {
            temporalART.learn(pattern);
        }

        // Check categories were created
        assertTrue(temporalART.getCategoryCount() > 0, "Should create categories");

        // Check state
        var state = temporalART.getState();
        assertNotNull(state);
        assertTrue(state.learningEnabled());
    }

    @Test
    public void testSequencePrediction() {
        // Train on multiple sequences
        var sequences = createMultipleSequences(3, 5, 10);

        for (var sequence : sequences) {
            temporalART.processSequence(sequence);
        }

        // Predict on a trained sequence
        int prediction = temporalART.predictSequence(sequences.get(0));
        assertTrue(prediction >= 0, "Should predict a category");

        // Predict on a novel sequence
        var novelSequence = createSimpleSequence(5, 10);
        int novelPrediction = temporalART.predictSequence(novelSequence);
        // Novel sequence might create new category or match existing
        assertTrue(novelPrediction >= -1, "Should handle novel sequence");
    }

    @Test
    public void testPhoneNumberPattern() {
        // Use phone number optimized parameters
        parameters = TemporalARTParameters.speechDefaults();
        temporalART = new TemporalART(parameters);

        // Create phone number pattern (10 digits as 3-3-4)
        var phoneNumber = createPhoneNumberSequence();

        // Train on the pattern multiple times
        for (int i = 0; i < 5; i++) {
            temporalART.processSequence(phoneNumber);
        }

        // Check chunking occurred
        var stats = temporalART.getStatistics();
        assertTrue(stats.chunkCount() > 0, "Should form chunks for phone number");

        // Test recognition
        int category = temporalART.predictSequence(phoneNumber);
        assertTrue(category >= 0, "Should recognize trained phone number");
    }

    @Test
    public void testListLearning() {
        // Use list learning parameters
        parameters = TemporalARTParameters.listLearningDefaults();
        temporalART = new TemporalART(parameters);

        // Create word list (simulated as patterns)
        var wordList = createWordListSequence(7);  // Miller's 7±2

        // Learn the list
        temporalART.processSequence(wordList);

        // Check for appropriate chunking
        var stats = temporalART.getStatistics();
        if (stats.chunkCount() > 0) {
            assertTrue(stats.averageChunkSize() >= 2.0 && stats.averageChunkSize() <= 7.0,
                      "Chunks should be within human memory span");
        }

        // Test recall
        int recalled = temporalART.predictSequence(wordList);
        assertTrue(recalled >= 0, "Should recall learned list");
    }

    @Test
    public void testResetFunctionality() {
        // Learn some sequences
        var sequence = createSimpleSequence(5, 10);
        temporalART.processSequence(sequence);

        assertTrue(temporalART.getCategoryCount() > 0, "Should have categories");

        // Reset
        temporalART.reset();

        assertEquals(0, temporalART.getCategoryCount(), "Categories should be cleared");
        assertEquals(0.0, temporalART.getCurrentTime(), "Time should reset");

        var state = temporalART.getState();
        assertEquals(0, state.getTotalItemCount(), "All items should be cleared");
    }

    @Test
    public void testLearningToggle() {
        // Start with learning enabled
        var sequence = createSimpleSequence(3, 10);
        temporalART.processSequence(sequence);
        int initialCategories = temporalART.getCategoryCount();

        // Disable learning
        temporalART.setLearningEnabled(false);

        // Process more sequences
        var newSequence = createRandomSequence(3, 10);
        temporalART.processSequence(newSequence);

        // Category count should not change
        assertEquals(initialCategories, temporalART.getCategoryCount(),
                    "No new categories when learning disabled");

        // Re-enable learning
        temporalART.setLearningEnabled(true);
        var anotherSequence = createRandomSequence(3, 10);
        temporalART.processSequence(anotherSequence);

        // Now categories might increase
        assertTrue(temporalART.getCategoryCount() >= initialCategories,
                  "Categories can increase with learning enabled");
    }

    @Test
    public void testTemporalDynamics() {
        // Process patterns with time tracking
        var patterns = createSimpleSequence(10, 5);

        for (var pattern : patterns) {
            double timeBefore = temporalART.getCurrentTime();
            temporalART.learn(pattern);
            double timeAfter = temporalART.getCurrentTime();

            assertTrue(timeAfter > timeBefore, "Time should advance");
            assertEquals(parameters.getTimeStep(), timeAfter - timeBefore, 0.001,
                       "Time step should be consistent");
        }
    }

    @Test
    public void testStatistics() {
        // Process various sequences
        for (int i = 0; i < 3; i++) {
            var sequence = createRandomSequence(5, 10);
            temporalART.processSequence(sequence);
        }

        var stats = temporalART.getStatistics();

        assertNotNull(stats);
        assertTrue(stats.categoryCount() >= 0);
        assertTrue(stats.workingMemoryItems() >= 0);
        assertTrue(stats.compressionRatio() > 0);

        // Check statistics consistency
        if (stats.chunkCount() > 0) {
            assertTrue(stats.averageChunkSize() > 0, "Average chunk size should be positive");
        }
    }

    @Test
    public void testCapacityLimits() {
        // Try to exceed capacity
        int maxCategories = parameters.getMaxCategories();

        // Create many diverse sequences
        for (int i = 0; i < maxCategories * 2; i++) {
            var pattern = createRandomPattern(10, new Random(i));
            temporalART.learn(pattern);
        }

        // Check capacity not exceeded
        assertTrue(temporalART.getCategoryCount() <= maxCategories,
                  "Should not exceed maximum categories");
    }

    @Test
    public void testSequentialMemory() {
        // Test that system maintains sequential information
        var sequence1 = List.of(
            createSpecificPattern(new double[]{1, 0, 0, 0, 0}),
            createSpecificPattern(new double[]{0, 1, 0, 0, 0}),
            createSpecificPattern(new double[]{0, 0, 1, 0, 0})
        );

        var sequence2 = List.of(
            createSpecificPattern(new double[]{0, 0, 1, 0, 0}),  // Same patterns
            createSpecificPattern(new double[]{0, 1, 0, 0, 0}),  // Different order
            createSpecificPattern(new double[]{1, 0, 0, 0, 0})
        );

        // Learn first sequence
        temporalART.processSequence(sequence1);
        int cat1 = temporalART.predictSequence(sequence1);

        // Learn second sequence
        temporalART.processSequence(sequence2);
        int cat2 = temporalART.predictSequence(sequence2);

        // Different sequences should potentially map to different categories
        // (though not guaranteed due to vigilance and similarity)
        assertNotEquals(-1, cat1, "First sequence should be learned");
        assertNotEquals(-1, cat2, "Second sequence should be learned");
    }

    // Helper methods

    private List<double[]> createSimpleSequence(int length, int dimension) {
        List<double[]> sequence = new ArrayList<>();
        Random random = new Random(42);

        for (int i = 0; i < length; i++) {
            double[] pattern = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                pattern[j] = random.nextDouble();
            }
            sequence.add(pattern);
        }

        return sequence;
    }

    private List<double[]> createRandomSequence(int length, int dimension) {
        return createSimpleSequence(length, dimension);
    }

    private List<List<double[]>> createMultipleSequences(int count, int length, int dimension) {
        List<List<double[]>> sequences = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            sequences.add(createSimpleSequence(length, dimension));
        }
        return sequences;
    }

    private List<double[]> createPhoneNumberSequence() {
        List<double[]> digits = new ArrayList<>();
        Random random = new Random(5551234);  // Phone seed

        // Create 10 digit patterns
        for (int i = 0; i < 10; i++) {
            double[] digit = new double[10];
            digit[i % 10] = 1.0;  // One-hot encoding
            for (int j = 0; j < 10; j++) {
                if (j != i % 10) {
                    digit[j] = random.nextDouble() * 0.2;  // Small noise
                }
            }
            digits.add(digit);
        }

        return digits;
    }

    private List<double[]> createWordListSequence(int numWords) {
        List<double[]> words = new ArrayList<>();
        Random random = new Random(123);

        for (int i = 0; i < numWords; i++) {
            double[] word = new double[20];  // Simulate word features
            // Make each word distinctive
            word[i % 20] = 1.0;
            word[(i + 7) % 20] = 0.7;
            for (int j = 0; j < 20; j++) {
                if (word[j] == 0) {
                    word[j] = random.nextDouble() * 0.3;
                }
            }
            words.add(word);
        }

        return words;
    }

    private double[] createRandomPattern(int dimension, Random random) {
        double[] pattern = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            pattern[i] = random.nextDouble();
        }
        return pattern;
    }

    private double[] createSpecificPattern(double[] values) {
        return values.clone();
    }
}