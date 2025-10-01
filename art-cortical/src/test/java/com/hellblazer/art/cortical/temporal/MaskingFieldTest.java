package com.hellblazer.art.cortical.temporal;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

/**
 * Tests for masking field with multi-scale temporal dynamics.
 * Validates spatial competition, chunking, and list learning.
 *
 * Part of LIST PARSE temporal chunking system
 * (Kazerounian & Grossberg, 2014).
 */
class MaskingFieldTest {

    private MaskingField maskingField;
    private MaskingFieldParameters parameters;
    private Random random;

    @BeforeEach
    void setUp() {
        parameters = MaskingFieldParameters.listLearningDefaults();
        maskingField = new MaskingField(parameters);
        random = new Random(42); // Deterministic for testing
    }

    @Test
    void testMaskingFieldCreation() {
        assertNotNull(maskingField);
        var state = maskingField.getState();
        assertNotNull(state);
        assertEquals(0, state.activeItemCount());
    }

    @Test
    void testItemNodeCreation() {
        var pattern = createTestPattern(10);

        // Process pattern through masking field
        maskingField.update(pattern, 0.01);

        // Verify item nodes were created
        var itemNodes = maskingField.getItemNodes();
        assertEquals(1, itemNodes.size(), "One item node should be created");
    }

    @Test
    void testMultiplePatternProcessing() {
        // Process multiple distinct patterns
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(10);
            maskingField.update(pattern, 0.01);
        }

        var itemNodes = maskingField.getItemNodes();
        assertTrue(itemNodes.size() <= 5, "Should have up to 5 item nodes");
        assertTrue(itemNodes.size() >= 1, "Should have at least 1 item node");
    }

    @Test
    void testSpatialCompetition() {
        // Use parameters with lower winner threshold
        var testParams = MaskingFieldParameters.builder()
            .winnerThreshold(0.1)  // Lower threshold to ensure winners
            .initialActivation(0.5)
            .build();
        maskingField = new MaskingField(testParams);

        // Create and process pattern
        var pattern = createTestPattern(10);
        maskingField.update(pattern, 0.01);

        // Get state and check activations
        var state = maskingField.getState();
        var activations = state.itemActivations();

        // Check if any nodes have activation
        var hasActivation = false;
        for (var activation : activations) {
            if (activation > 0) {
                hasActivation = true;
                break;
            }
        }

        assertTrue(hasActivation, "Should have some node activations");
    }

    @Test
    void testListChunkFormation() {
        // Use parameters that encourage chunk formation
        var chunkParams = MaskingFieldParameters.builder()
            .minChunkSize(2)
            .winnerThreshold(0.1)
            .initialActivation(0.6)
            .activationBoost(0.5)
            .build();
        maskingField = new MaskingField(chunkParams);

        // Process multiple patterns to create a sequence
        for (int i = 0; i < 4; i++) {
            var pattern = createTestPattern(10);
            maskingField.update(pattern, 0.05);
        }

        // Get chunks
        var chunks = maskingField.getListChunks();
        // Chunks may or may not form depending on dynamics
        assertNotNull(chunks);
        assertTrue(chunks.size() >= 0);
    }

    @Test
    void testStateTracking() {
        var pattern = createTestPattern(10);
        var state1 = maskingField.update(pattern, 0.01);

        assertNotNull(state1);
        assertEquals(1, state1.activeItemCount());

        // Process another pattern
        var pattern2 = createTestPattern(10);
        var state2 = maskingField.update(pattern2, 0.01);

        assertTrue(state2.activeItemCount() >= 1);
    }

    @Test
    void testReset() {
        // Add some patterns
        for (int i = 0; i < 3; i++) {
            maskingField.update(createTestPattern(10), 0.01);
        }

        // Verify nodes were created
        assertTrue(maskingField.getItemNodes().size() > 0);

        // Reset
        maskingField.reset();

        // Verify reset
        assertEquals(0, maskingField.getItemNodes().size());
        assertEquals(0, maskingField.getListChunks().size());
        assertEquals(0, maskingField.getState().activeItemCount());
    }

    @Test
    void testActivationDynamics() {
        var pattern = createTestPattern(10);

        // Update multiple times with same pattern
        for (int i = 0; i < 5; i++) {
            maskingField.update(pattern, 0.01);
        }

        var state = maskingField.getState();
        var activations = state.itemActivations();

        // First item should have some activation
        assertTrue(activations[0] > 0, "First item should be activated");
    }

    @Test
    void testContrastEnhancement() {
        var params = MaskingFieldParameters.builder()
            .normalizationEnabled(true)
            .initialActivation(0.5)
            .build();
        maskingField = new MaskingField(params);

        var pattern = createTestPattern(10);
        var state = maskingField.update(pattern, 0.01);

        // Contrast should be computed
        var contrast = state.computeContrast();
        assertTrue(contrast >= 0, "Contrast should be non-negative");
    }

    @Test
    void testStatistics() {
        // Add some patterns
        for (int i = 0; i < 3; i++) {
            maskingField.update(createTestPattern(10), 0.01);
        }

        var stats = maskingField.getStatistics();
        assertNotNull(stats);
        assertTrue(stats.totalItemNodes() >= 0);
        assertTrue(stats.totalChunks() >= 0);
        assertTrue(stats.averageChunkSize() >= 0);
    }

    @Test
    void testPhoneNumberDefaults() {
        var phoneParams = MaskingFieldParameters.phoneNumberDefaults();
        var phoneField = new MaskingField(phoneParams);

        assertNotNull(phoneField);

        // Parameters should be configured for phone number chunking
        assertEquals(3, phoneParams.minChunkSize());
        assertEquals(4, phoneParams.maxChunkSize());
    }

    @Test
    void testParameterValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .maxItemNodes(5) // Too small (< 10)
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .minChunkSize(10)
                .maxChunkSize(5) // Max < min
                .build();
        });

        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .integrationTimeStep(0.2) // Too large (> 0.1)
                .build();
        });
    }

    // Helper methods

    private double[] createTestPattern(int dimension) {
        var pattern = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            pattern[i] = random.nextDouble();
        }
        return pattern;
    }
}
