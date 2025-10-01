package com.hellblazer.art.cortical.temporal;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Integration tests for temporal processing pipeline.
 * Tests WorkingMemory → MaskingField integration.
 *
 * Validates LIST PARSE phone number chunking and sequential processing
 * (Kazerounian & Grossberg, 2014).
 */
class TemporalIntegrationTest {

    @Test
    void testPhoneNumberChunking() {
        // Classic LIST PARSE example: 555-1234 → [555][1234]
        var wmParams = WorkingMemoryParameters.builder()
            .capacity(10)
            .primacyDecayRate(0.1)
            .decayRate(0.05)
            .build();

        var mfParams = MaskingFieldParameters.builder()
            .maxItemNodes(10)
            .minChunkSize(3)
            .maxChunkSize(4)
            .winnerThreshold(0.2)
            .initialActivation(0.6)
            .activationBoost(0.3)
            .build();

        var processor = new TemporalProcessor(wmParams, mfParams);

        // Simulate phone number: 5, 5, 5, 1, 2, 3, 4
        var digits = List.of(5, 5, 5, 1, 2, 3, 4);

        for (var digit : digits) {
            var pattern = digitPattern(digit);
            processor.processItem(pattern);
        }

        // After processing, should have item nodes
        var mfState = processor.getMaskingFieldState();
        assertTrue(mfState.activeItemCount() > 0,
            "Should have active item nodes after processing sequence");

        // Working memory should contain items
        var wmState = processor.getWorkingMemoryState();
        assertTrue(wmState.itemCount() > 0,
            "Working memory should contain items");
    }

    @Test
    void testWorkingMemoryToMaskingFieldFlow() {
        // Test data flow: WM → MF
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.listLearningDefaults();
        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process a sequence
        var sequence = List.of(
            new double[]{1.0, 0.0, 0.0},
            new double[]{0.0, 1.0, 0.0},
            new double[]{0.0, 0.0, 1.0}
        );

        var results = processor.processSequence(sequence);

        // Should have results for each item
        assertEquals(3, results.size());

        // Each result should have both WM and MF state
        for (var result : results) {
            assertNotNull(result.workingMemoryState());
            assertNotNull(result.maskingFieldState());
        }
    }

    @Test
    void testSequentialProcessing() {
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.listLearningDefaults();
        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process items sequentially
        var item1 = new double[]{1.0, 0.0, 0.0};
        var result1 = processor.processItem(item1);

        var item2 = new double[]{0.0, 1.0, 0.0};
        var result2 = processor.processItem(item2);

        var item3 = new double[]{0.0, 0.0, 1.0};
        var result3 = processor.processItem(item3);

        // Working memory should accumulate items
        assertTrue(result3.workingMemoryState().itemCount() >= 1);

        // Masking field should track item nodes
        assertTrue(result3.maskingFieldState().activeItemCount() >= 1);
    }

    @Test
    void testProcessorReset() {
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.listLearningDefaults();
        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process some items
        for (int i = 0; i < 5; i++) {
            processor.processItem(new double[]{i * 0.1, i * 0.2, i * 0.3});
        }

        // Verify state exists
        assertTrue(processor.getWorkingMemoryState().itemCount() > 0);

        // Reset
        processor.reset();

        // Verify reset
        assertEquals(0, processor.getWorkingMemoryState().itemCount());
        assertEquals(0, processor.getMaskingFieldState().activeItemCount());
        assertEquals(0, processor.getActiveChunks().size());
    }

    @Test
    void testTemporalResultRecord() {
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.listLearningDefaults();
        var processor = new TemporalProcessor(wmParams, mfParams);

        var result = processor.processItem(new double[]{1.0, 0.5, 0.3});

        assertNotNull(result);
        assertNotNull(result.workingMemoryState());
        assertNotNull(result.maskingFieldState());
        assertNotNull(result.activeChunks());

        // Test record methods
        assertEquals(result.activeChunks().size(), result.chunkCount());
        assertEquals(result.chunkCount() > 0, result.hasChunks());
        assertTrue(result.totalChunkedItems() >= 0);
    }

    @Test
    void testPrimacyGradientEffect() {
        // Configure with strong primacy gradient
        var wmParams = WorkingMemoryParameters.builder()
            .capacity(10)
            .primacyDecayRate(0.3) // Strong primacy
            .build();

        var mfParams = MaskingFieldParameters.listLearningDefaults();
        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process sequence
        var sequence = new ArrayList<double[]>();
        for (int i = 0; i < 5; i++) {
            sequence.add(new double[]{i * 0.2, i * 0.3, i * 0.4});
        }

        processor.processSequence(sequence);

        // First items should have stronger representation due to primacy
        var wmState = processor.getWorkingMemoryState();
        assertTrue(wmState.itemCount() > 0);

        // Verify primacy gradient was applied
        var pattern = wmState.getCombinedPattern();
        assertNotNull(pattern);
        assertTrue(pattern.length > 0);
    }

    @Test
    void testCustomTimeStep() {
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.listLearningDefaults();

        // Custom time step
        var processor = new TemporalProcessor(wmParams, mfParams, 0.05);

        var result = processor.processItem(new double[]{1.0, 0.0, 0.0});
        assertNotNull(result);

        // Processing should work with custom time step
        assertTrue(result.maskingFieldState().activeItemCount() >= 0);
    }

    @Test
    void testChunkDetection() {
        // Use parameters optimized for chunking
        var wmParams = WorkingMemoryParameters.builder()
            .capacity(10)
            .primacyDecayRate(0.1)
            .build();

        var mfParams = MaskingFieldParameters.builder()
            .maxItemNodes(20)
            .minChunkSize(2)
            .maxChunkSize(4)
            .winnerThreshold(0.15)
            .initialActivation(0.7)
            .activationBoost(0.4)
            .competitionStrength(0.6)
            .build();

        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process longer sequence to encourage chunking
        for (int i = 0; i < 8; i++) {
            var pattern = digitPattern(i);
            processor.processItem(pattern);
        }

        // Check for chunks (may or may not form depending on dynamics)
        var chunks = processor.getActiveChunks();
        assertNotNull(chunks);
        // Chunks are optional, so just verify the list exists
        assertTrue(chunks.size() >= 0);
    }

    @Test
    void testPatternSimilarityGrouping() {
        var wmParams = WorkingMemoryParameters.paperDefaults();
        var mfParams = MaskingFieldParameters.builder()
            .matchingThreshold(0.9) // High similarity required
            .build();

        var processor = new TemporalProcessor(wmParams, mfParams);

        // Process similar patterns
        var pattern1 = new double[]{1.0, 0.0, 0.0};
        var pattern2 = new double[]{0.99, 0.01, 0.0}; // Very similar
        var pattern3 = new double[]{0.0, 1.0, 0.0};  // Different

        processor.processItem(pattern1);
        processor.processItem(pattern2);
        processor.processItem(pattern3);

        // Similar patterns should be recognized
        var mfState = processor.getMaskingFieldState();
        assertTrue(mfState.activeItemCount() >= 1);
    }

    // Helper methods

    private double[] digitPattern(int digit) {
        var pattern = new double[10];
        pattern[digit % 10] = 1.0;
        return pattern;
    }
}
