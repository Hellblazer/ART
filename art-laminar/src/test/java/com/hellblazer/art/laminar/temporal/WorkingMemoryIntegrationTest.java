package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.CanonicalCircuitTestBase;
import com.hellblazer.art.laminar.core.Layer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.temporal.memory.WorkingMemory;
import com.hellblazer.art.temporal.memory.WorkingMemoryParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for WorkingMemory integration with layer chunking.
 * Validates STORE 2 + LIST PARSE integration.
 *
 * @author Hal Hildebrand
 */
@DisplayName("WorkingMemory Integration Tests")
class WorkingMemoryIntegrationTest extends CanonicalCircuitTestBase {

    private WorkingMemoryLayerBridge bridge;
    private WorkingMemory workingMemory;
    private TemporalChunkingLayer chunkingLayer;

    @BeforeEach
    void setUp() {
        // Create working memory with capacity 7 (Miller's number)
        var wmParams = WorkingMemoryParameters.builder()
            .capacity(7)
            .primacyDecayRate(0.3)
            .overflowResetEnabled(false)
            .build();

        workingMemory = new WorkingMemory(wmParams);

        // Create chunking layer
        var baseLayer = new MockLayer(10);
        var chunkParams = ChunkingParameters.paperDefaults();
        chunkingLayer = new TemporalChunkingLayerDecorator(baseLayer, chunkParams);

        // Create bridge
        bridge = new WorkingMemoryLayerBridge(workingMemory, chunkingLayer, 0.1);
    }

    @Test
    @DisplayName("Bridge processes patterns through both systems")
    void testBasicProcessing() {
        var pattern = createTestPattern(0.8, 10);
        var result = bridge.processItem(pattern);

        assertNotNull(result, "Should return processed pattern");
        assertEquals(1, bridge.getSequencePosition(), "Should track position");
    }

    @Test
    @DisplayName("Primacy gradient forms in working memory")
    void testPrimacyGradient() {
        // Store sequence
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(0.7, 10);
            bridge.processItem(pattern);
        }

        var primacyWeights = bridge.getWorkingMemoryActivations();

        // First item should have highest primacy weight
        assertTrue(primacyWeights[0] > primacyWeights[4],
                  "Early items should have higher primacy");

        // Should decrease monotonically (or near-monotonically)
        for (int i = 0; i < 4; i++) {
            assertTrue(primacyWeights[i] >= primacyWeights[i + 1] * 0.9,
                      String.format("Primacy should decrease: pos %d (%.3f) >= pos %d (%.3f)",
                                   i, primacyWeights[i], i + 1, primacyWeights[i + 1]));
        }
    }

    @Test
    @DisplayName("Temporal chunks form from coherent sequences")
    void testChunkFormation() {
        // Process coherent sequence
        for (int i = 0; i < 7; i++) {
            var pattern = createTestPattern(0.8, 10);
            bridge.processItem(pattern);
        }

        var chunks = bridge.getTemporalChunks();
        assertTrue(chunks.size() > 0,
                  "Should form chunks from coherent sequence");
    }

    @Test
    @DisplayName("Integrated statistics combine both systems")
    void testIntegratedStatistics() {
        // Process sequence
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(0.7 + i * 0.05, 10);
            bridge.processItem(pattern);
        }

        var stats = bridge.getStatistics();

        assertNotNull(stats.primacyWeights(), "Should have primacy weights");
        assertNotNull(stats.recencyWeights(), "Should have recency weights");
        assertEquals(5, stats.sequenceLength(), "Should track sequence length");
        assertTrue(stats.primacyGradientStrength() > 0,
                  "Should have primacy gradient");
    }

    @Test
    @DisplayName("Combined representation blends primacy and chunks")
    void testCombinedRepresentation() {
        // Process sequence to form chunks
        for (int i = 0; i < 7; i++) {
            var pattern = createTestPattern(0.8, 10);
            bridge.processItem(pattern);
        }

        var combined = bridge.getCombinedRepresentation();

        assertNotNull(combined, "Should create combined representation");
        assertTrue(combined.dimension() > 0, "Should have non-zero dimension");

        // Combined should be non-zero if chunks formed
        double magnitude = 0.0;
        for (int i = 0; i < combined.dimension(); i++) {
            magnitude += combined.get(i) * combined.get(i);
        }
        assertTrue(magnitude > 0, "Combined representation should be non-zero");
    }

    @Test
    @DisplayName("Sequence processing maintains order")
    void testSequenceProcessing() {
        List<Pattern> sequence = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            sequence.add(createTestPattern(0.6 + i * 0.05, 10));
        }

        var finalState = bridge.processSequence(sequence);

        assertNotNull(finalState, "Should return final state");
        assertEquals(5, bridge.getSequencePosition(), "Should process all patterns");

        var processed = bridge.getProcessedSequence();
        assertEquals(5, processed.size(), "Should track all processed patterns");
    }

    @Test
    @DisplayName("Reset clears both systems")
    void testReset() {
        // Process some patterns
        for (int i = 0; i < 3; i++) {
            bridge.processItem(createTestPattern(0.7, 10));
        }

        assertTrue(bridge.getSequencePosition() > 0, "Should have position");

        // Reset
        bridge.reset();

        assertEquals(0, bridge.getSequencePosition(), "Position should be reset");
        assertTrue(bridge.getProcessedSequence().isEmpty(),
                  "Sequence should be cleared");
        assertTrue(bridge.getTemporalChunks().isEmpty(),
                  "Chunks should be cleared");
    }

    @Test
    @DisplayName("Layer state reflects both primacy and chunking")
    void testLayerState() {
        // Process sequence to build state
        for (int i = 0; i < 6; i++) {
            bridge.processItem(createTestPattern(0.75, 10));
        }

        var layerState = bridge.getLayerState();

        assertNotNull(layerState, "Should have layer state");
        assertNotNull(layerState.currentActivation(), "Should have activation");
        assertTrue(layerState.timestamp() >= 0, "Should have timestamp");
    }

    @Test
    @DisplayName("Capacity limits respected")
    void testCapacityLimit() {
        // Try to exceed capacity (7 items)
        for (int i = 0; i < 10; i++) {
            bridge.processItem(createTestPattern(0.7, 10));
        }

        // Should stop at capacity or handle overflow
        var wmState = workingMemory.getState();
        assertTrue(wmState.getCurrentPosition() <= wmState.getCapacity(),
                  "Should not exceed capacity");
    }

    @Test
    @DisplayName("Different patterns create different primacy profiles")
    void testPatternVariation() {
        bridge.reset();

        // Process varying patterns
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(0.5 + i * 0.1, 10);
            bridge.processItem(pattern);
        }

        var primacyWeights = bridge.getWorkingMemoryActivations();

        // Primacy gradient should still exist despite pattern variation
        assertTrue(primacyWeights[0] > primacyWeights[4],
                  "Primacy should exist despite pattern variation");
    }

    // Helper methods

    private Pattern createTestPattern(double value, int size) {
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            values[i] = value;
        }
        return new DenseVector(values);
    }

    // Mock Layer implementation
    private static class MockLayer implements Layer {
        private final int size;
        private Pattern activation;

        MockLayer(int size) {
            this.size = size;
            this.activation = new DenseVector(new double[size]);
        }

        @Override public String getId() { return "mock"; }
        @Override public int size() { return size; }
        @Override public LayerType getType() { return LayerType.INPUT; }
        @Override public Pattern getActivation() { return activation; }
        @Override public void setActivation(Pattern activation) { this.activation = activation; }

        @Override
        public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
            setActivation(input);
            return input;
        }

        @Override
        public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
            return expectation;
        }

        @Override
        public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
            return lateral;
        }

        @Override public com.hellblazer.art.laminar.core.WeightMatrix getWeights() { return null; }
        @Override public void setWeights(com.hellblazer.art.laminar.core.WeightMatrix weights) {}
        @Override public void updateWeights(Pattern input, double learningRate) {}
        @Override public void reset() { activation = new DenseVector(new double[size]); }
        @Override public void addActivationListener(LayerActivationListener listener) {}
    }
}