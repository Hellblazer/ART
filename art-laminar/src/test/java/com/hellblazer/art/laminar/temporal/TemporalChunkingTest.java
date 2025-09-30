package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.CanonicalCircuitTestBase;
import com.hellblazer.art.laminar.core.Layer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for temporal chunking implementation.
 * Validates LIST PARSE model integration into laminar layers.
 *
 * @author Hal Hildebrand
 */
@DisplayName("Temporal Chunking Tests")
class TemporalChunkingTest extends CanonicalCircuitTestBase {

    private MockLayer baseLayer;
    private TemporalChunkingLayerDecorator chunkingLayer;
    private ChunkingParameters parameters;
    private Random random;  // Deterministic random for reproducibility

    @BeforeEach
    void setUp() {
        baseLayer = new MockLayer(10);
        parameters = ChunkingParameters.paperDefaults();
        chunkingLayer = new TemporalChunkingLayerDecorator(baseLayer, parameters);
        random = new Random(42);  // Fixed seed for reproducibility
    }

    @Test
    @DisplayName("Chunk formation requires minimum history")
    void testMinimumHistoryRequired() {
        // Single pattern - cannot form chunk
        var pattern1 = createTestPattern(0.8);
        chunkingLayer.processWithChunking(pattern1, 0.01);

        assertFalse(chunkingLayer.shouldFormChunk(),
                   "Should not form chunk with single pattern");

        // Add second pattern - now can form chunk
        var pattern2 = createTestPattern(0.75);
        chunkingLayer.processWithChunking(pattern2, 0.01);

        var state = chunkingLayer.getChunkingState();
        assertEquals(2, state.getHistorySize(),
                    "Should have 2 patterns in history");
    }

    @Test
    @DisplayName("Chunk formation requires sufficient activation")
    void testActivationThreshold() {
        // Add weak patterns below formation threshold
        // Pattern with all 0.1 has L2 norm = sqrt(10 * 0.01) = 0.316 < 0.5
        for (int i = 0; i < 5; i++) {
            var weakPattern = createTestPattern(0.1);  // L2 norm ~0.316 < 0.5 threshold
            chunkingLayer.processWithChunking(weakPattern, 0.01);
        }

        assertFalse(chunkingLayer.shouldFormChunk(),
                   "Weak patterns should not trigger chunk formation");

        // Add strong patterns above threshold
        chunkingLayer.resetChunking();
        baseLayer.setActivation(createTestPattern(0.8));  // Set base layer activation
        for (int i = 0; i < 5; i++) {
            var strongPattern = createTestPattern(0.8);
            chunkingLayer.processWithChunking(strongPattern, 0.01);
        }

        // Debug output
        var state = chunkingLayer.getChunkingState();
        System.out.println("History size: " + state.getHistorySize());

        assertTrue(chunkingLayer.shouldFormChunk(),
                  "Strong coherent patterns should trigger chunk formation");
    }

    @Test
    @DisplayName("Chunk formation requires temporal coherence")
    void testCoherenceRequirement() {
        var params = ChunkingParameters.builder()
            .chunkFormationThreshold(0.3)  // Low activation threshold
            .chunkCoherenceThreshold(0.8)  // High coherence requirement
            .minChunkSize(2)
            .maxChunkSize(7)
            .maxHistorySize(12)
            .chunkDecayRate(0.01)
            .activityThreshold(0.1)
            .temporalWindowSize(0.5)
            .build();

        chunkingLayer = new TemporalChunkingLayerDecorator(baseLayer, params);

        // Add truly incoherent patterns (orthogonal patterns for guaranteed low coherence)
        // Each pattern activates different dimensions with no overlap
        for (int i = 0; i < 5; i++) {
            var orthogonalPattern = createOrthogonalPattern(i);
            chunkingLayer.processWithChunking(orthogonalPattern, 0.01);
        }

        assertFalse(chunkingLayer.shouldFormChunk(),
                   "Incoherent patterns should not form chunks");

        // Add coherent patterns (similar)
        chunkingLayer.resetChunking();
        var basePattern = createTestPattern(0.7);
        for (int i = 0; i < 5; i++) {
            var similarPattern = addNoise(basePattern, 0.05);
            chunkingLayer.processWithChunking(similarPattern, 0.01);
        }

        assertTrue(chunkingLayer.shouldFormChunk(),
                  "Coherent patterns should form chunks");
    }

    @Test
    @DisplayName("Chunks form within size constraints")
    void testChunkSizeConstraints() {
        // Form chunk and verify size
        for (int i = 0; i < parameters.getMinChunkSize() + 2; i++) {
            var pattern = createTestPattern(0.8);
            chunkingLayer.processWithChunking(pattern, 0.01);
        }

        if (chunkingLayer.shouldFormChunk()) {
            var chunk = chunkingLayer.formChunk();
            assertNotNull(chunk, "Should form valid chunk");

            int size = chunk.size();
            assertTrue(size >= parameters.getMinChunkSize(),
                      "Chunk size should be >= minChunkSize");
            assertTrue(size <= parameters.getMaxChunkSize(),
                      "Chunk size should be <= maxChunkSize");
        }
    }

    @Test
    @DisplayName("Representative pattern computed correctly")
    void testRepresentativePattern() {
        // Create chunk with known patterns
        List<TemporalChunk.ChunkItem> items = new ArrayList<>();
        items.add(new TemporalChunk.ChunkItem(
            new DenseVector(new double[]{1.0, 0.0}),
            0.8, 0.0, 0
        ));
        items.add(new TemporalChunk.ChunkItem(
            new DenseVector(new double[]{0.9, 0.1}),
            0.9, 0.01, 1
        ));
        items.add(new TemporalChunk.ChunkItem(
            new DenseVector(new double[]{0.8, 0.2}),
            0.7, 0.02, 2
        ));

        var chunk = new TemporalChunk(items, 0.0, 0);
        var repr = chunk.getRepresentativePattern();

        assertNotNull(repr, "Representative pattern should not be null");
        assertEquals(2, repr.dimension(), "Pattern dimension should match");

        // Should be weighted average favoring high-activation patterns
        assertTrue(repr.get(0) > 0.8, "First dimension should be high");
        assertTrue(repr.get(1) < 0.2, "Second dimension should be low");
    }

    @Test
    @DisplayName("Chunk strength decays over time")
    void testChunkDecay() {
        var items = createChunkItems(5);
        var chunk = new TemporalChunk(items, 0.0, 0);

        double initialStrength = chunk.getStrength();
        assertTrue(initialStrength > 0, "Initial strength should be positive");

        // Apply decay
        double decayRate = 0.05;
        double timeElapsed = 1.0;
        chunk.decay(decayRate, timeElapsed);

        double decayedStrength = chunk.getStrength();
        assertTrue(decayedStrength < initialStrength,
                  "Strength should decrease after decay");

        // Verify exponential decay formula
        double expectedStrength = initialStrength * Math.exp(-decayRate * timeElapsed);
        assertEquals(expectedStrength, decayedStrength, 1e-10,
                    "Decay should follow exponential formula");
    }

    @Test
    @DisplayName("Inactive chunks are pruned")
    void testChunkPruning() {
        var state = new ChunkingState(12);

        // Add several chunks
        for (int i = 0; i < 3; i++) {
            var items = createChunkItems(3);
            var chunk = new TemporalChunk(items, 0.0, i);
            state.addChunk(chunk);
        }

        assertEquals(3, state.getActiveChunks().size(),
                    "Should have 3 active chunks");

        // Decay chunks heavily over time
        double decayRate = 0.5;
        for (int i = 0; i < 10; i++) {
            // Advance time and add dummy activation to update currentTime
            state.addActivation(createTestPattern(0.1), 0.1, (i + 1) * 1.0);
            state.decayChunks(decayRate);
        }

        // Prune with threshold
        state.pruneInactiveChunks(0.1);

        assertTrue(state.getActiveChunks().size() < 3,
                  "Heavily decayed chunks should be pruned");
    }

    @Test
    @DisplayName("Activation history maintains max size")
    void testHistoryManagement() {
        int maxSize = 5;
        var params = ChunkingParameters.builder()
            .maxHistorySize(maxSize)
            .chunkFormationThreshold(0.5)
            .chunkCoherenceThreshold(0.6)
            .chunkDecayRate(0.01)
            .activityThreshold(0.1)
            .minChunkSize(2)
            .maxChunkSize(7)
            .temporalWindowSize(0.5)
            .build();

        chunkingLayer = new TemporalChunkingLayerDecorator(baseLayer, params);

        // Add more patterns than max history
        for (int i = 0; i < maxSize + 5; i++) {
            var pattern = createTestPattern(0.7);
            chunkingLayer.processWithChunking(pattern, 0.01);
        }

        var state = chunkingLayer.getChunkingState();
        assertEquals(maxSize, state.getHistorySize(),
                    "History should not exceed max size");
    }

    @Test
    @DisplayName("Temporal context aggregates chunks")
    void testTemporalContext() {
        // Form several chunks
        for (int iteration = 0; iteration < 3; iteration++) {
            for (int i = 0; i < 5; i++) {
                var pattern = createTestPattern(0.8);
                chunkingLayer.processWithChunking(pattern, 0.01);
            }

            if (chunkingLayer.shouldFormChunk()) {
                var chunk = chunkingLayer.formChunk();
                if (chunk != null) {
                    chunkingLayer.getChunkingState().addChunk(chunk);
                    chunkingLayer.resetChunking();
                }
            }
        }

        var context = chunkingLayer.getTemporalContext();
        assertNotNull(context, "Temporal context should not be null");
        assertEquals(baseLayer.size(), context.dimension(),
                    "Context dimension should match layer size");

        // Context should be non-zero if chunks exist
        var chunks = chunkingLayer.getTemporalChunks();
        if (!chunks.isEmpty()) {
            double magnitude = computeMagnitude(context);
            assertTrue(magnitude > 0, "Context should be non-zero with active chunks");
        }
    }

    @Test
    @DisplayName("Statistics track chunking behavior")
    void testChunkingStatistics() {
        // Process sequence to form chunks
        for (int i = 0; i < 10; i++) {
            var pattern = createTestPattern(0.8);
            chunkingLayer.processWithChunking(pattern, 0.01);

            if (chunkingLayer.shouldFormChunk()) {
                var chunk = chunkingLayer.formChunk();
                if (chunk != null) {
                    chunkingLayer.getChunkingState().addChunk(chunk);
                }
            }
        }

        var stats = chunkingLayer.getChunkingStatistics();
        assertNotNull(stats, "Statistics should not be null");

        assertTrue(stats.totalChunks() >= 0, "Total chunks should be non-negative");
        assertTrue(stats.activeChunks() <= stats.totalChunks(),
                  "Active chunks should not exceed total");
        assertTrue(stats.averageChunkSize() >= 0, "Average size should be non-negative");
        assertTrue(stats.averageCoherence() >= 0 && stats.averageCoherence() <= 1,
                  "Average coherence should be in [0,1]");
    }

    @Test
    @DisplayName("Chunk types match size ranges")
    void testChunkTypes() {
        // Test each size range
        testChunkType(2, TemporalChunk.ChunkType.SMALL);
        testChunkType(4, TemporalChunk.ChunkType.MEDIUM);
        testChunkType(6, TemporalChunk.ChunkType.LARGE);
        testChunkType(8, TemporalChunk.ChunkType.SUPER);
    }

    private void testChunkType(int size, TemporalChunk.ChunkType expectedType) {
        var items = createChunkItems(size);
        var chunk = new TemporalChunk(items, 0.0, 0);

        assertEquals(expectedType, chunk.getType(),
                    String.format("Chunk of size %d should be %s", size, expectedType));
    }

    @Test
    @DisplayName("Chunk merge combines items correctly")
    void testChunkMerge() {
        var items1 = createChunkItems(3);
        var chunk1 = new TemporalChunk(items1, 0.0, 0);

        var items2 = createChunkItems(3);
        var chunk2 = new TemporalChunk(items2, 0.01, 1);

        var merged = chunk1.merge(chunk2);

        assertNotNull(merged, "Merged chunk should not be null");
        assertTrue(merged.size() >= chunk1.size(),
                  "Merged chunk should contain at least first chunk's items");
    }

    @Test
    @DisplayName("Temporal span computed correctly")
    void testTemporalSpan() {
        List<TemporalChunk.ChunkItem> items = new ArrayList<>();
        items.add(new TemporalChunk.ChunkItem(
            createTestPattern(0.8), 0.8, 0.0, 0
        ));
        items.add(new TemporalChunk.ChunkItem(
            createTestPattern(0.7), 0.7, 0.5, 1
        ));
        items.add(new TemporalChunk.ChunkItem(
            createTestPattern(0.6), 0.6, 1.0, 2
        ));

        var chunk = new TemporalChunk(items, 0.0, 0);
        double span = chunk.getTemporalSpan();

        assertEquals(1.0, span, 1e-10,
                    "Temporal span should be 1.0 (from 0.0 to 1.0)");
    }

    @Test
    @DisplayName("Chunking can be disabled")
    void testChunkingToggle() {
        chunkingLayer.setChunkingEnabled(false);

        // Process patterns with chunking disabled
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(0.8);
            var result = chunkingLayer.processWithChunking(pattern, 0.01);
            assertNotNull(result, "Should still process patterns");
        }

        // No chunks should form
        assertTrue(chunkingLayer.getTemporalChunks().isEmpty(),
                  "No chunks should form when chunking is disabled");

        // Re-enable
        chunkingLayer.setChunkingEnabled(true);
        assertTrue(chunkingLayer.isChunkingEnabled(),
                  "Chunking should be re-enabled");
    }

    @Test
    @DisplayName("Reset clears all chunking state")
    void testReset() {
        // Build up state
        for (int i = 0; i < 10; i++) {
            var pattern = createTestPattern(0.8);
            chunkingLayer.processWithChunking(pattern, 0.01);
        }

        // Reset
        chunkingLayer.resetChunking();

        var state = chunkingLayer.getChunkingState();
        assertEquals(0, state.getHistorySize(),
                    "History should be cleared");
        assertTrue(state.getActiveChunks().isEmpty(),
                  "Chunks should be cleared");
        assertEquals(0.0, state.getCurrentTime(), 1e-10,
                    "Time should be reset");
    }

    @Test
    @DisplayName("Layer state includes activation and temporal context")
    void testLayerState() {
        // Process patterns to build chunks
        baseLayer.setActivation(createTestPattern(0.8));
        for (int i = 0; i < 5; i++) {
            var pattern = createTestPattern(0.8);
            chunkingLayer.processWithChunking(pattern, 0.01);
        }

        // Form chunk
        if (chunkingLayer.shouldFormChunk()) {
            var chunk = chunkingLayer.formChunk();
            if (chunk != null) {
                chunkingLayer.getChunkingState().addChunk(chunk);
            }
        }

        var layerState = chunkingLayer.getLayerState();

        assertNotNull(layerState, "Layer state should not be null");
        assertNotNull(layerState.currentActivation(), "Should have current activation");
        assertTrue(layerState.hasTemporalContext(), "Should have temporal context");
        assertNotNull(layerState.temporalContext(), "Temporal context should not be null");
        assertTrue(layerState.timestamp() >= 0, "Timestamp should be non-negative");
    }

    @Test
    @DisplayName("Context weight controls activation/context blend")
    void testContextWeight() {
        // Default weight
        assertEquals(0.3, chunkingLayer.getContextWeight(), 1e-10,
                    "Default context weight should be 0.3");

        // Set weight
        chunkingLayer.setContextWeight(0.5);
        assertEquals(0.5, chunkingLayer.getContextWeight(), 1e-10,
                    "Context weight should update");

        // Invalid weights
        assertThrows(IllegalArgumentException.class,
                    () -> chunkingLayer.setContextWeight(-0.1),
                    "Negative weight should be rejected");
        assertThrows(IllegalArgumentException.class,
                    () -> chunkingLayer.setContextWeight(1.5),
                    "Weight > 1.0 should be rejected");
    }

    @Test
    @DisplayName("Combined pattern uses context weight")
    void testCombinedPattern() {
        // Set up layer with known activation and context
        baseLayer.setActivation(new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

        // Process patterns and form chunks
        for (int i = 0; i < 5; i++) {
            var pattern = new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            chunkingLayer.processWithChunking(pattern, 0.01);
        }

        if (chunkingLayer.shouldFormChunk()) {
            var chunk = chunkingLayer.formChunk();
            if (chunk != null) {
                chunkingLayer.getChunkingState().addChunk(chunk);
            }
        }

        // Get layer state and combine with different weights
        var layerState = chunkingLayer.getLayerState();

        if (layerState.hasTemporalContext()) {
            // Weight 0.0 - pure activation
            var pure = layerState.combine(0.0);
            assertTrue(pure.get(0) > 0.9, "Should be mostly from activation");

            // Weight 1.0 - pure context
            var context = layerState.combine(1.0);
            assertTrue(context.get(1) > 0.9, "Should be mostly from context");

            // Weight 0.5 - balanced
            var balanced = layerState.combine(0.5);
            assertTrue(balanced.get(0) > 0.4, "Should have activation component");
            assertTrue(balanced.get(1) > 0.4, "Should have context component");
        }
    }

    // ============ Helper Methods ============

    private Pattern createTestPattern(double value) {
        double[] values = new double[baseLayer.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
        return new DenseVector(values);
    }

    /**
     * Create orthogonal patterns where each pattern activates different dimensions.
     * This guarantees near-zero cosine similarity between consecutive patterns.
     */
    private Pattern createOrthogonalPattern(int index) {
        double[] values = new double[baseLayer.size()];
        // Each pattern activates 2 dimensions: index and index+1 (mod size)
        // This ensures consecutive patterns have zero overlap
        int dim1 = (index * 2) % baseLayer.size();
        int dim2 = (index * 2 + 1) % baseLayer.size();
        values[dim1] = 0.7;  // Strong activation
        values[dim2] = 0.7;  // Strong activation
        return new DenseVector(values);
    }

    private Pattern createRandomPattern() {
        double[] values = new double[baseLayer.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble();
        }
        return new DenseVector(values);
    }

    private Pattern createNormalizedRandomPattern() {
        double[] values = new double[baseLayer.size()];
        double sumSquares = 0.0;

        // Generate random values
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble();
            sumSquares += values[i] * values[i];
        }

        // Normalize to unit length
        double norm = Math.sqrt(sumSquares);
        if (norm > 0) {
            for (int i = 0; i < values.length; i++) {
                values[i] /= norm;
            }
        }

        return new DenseVector(values);
    }

    private Pattern addNoise(Pattern base, double noiseLevel) {
        double[] values = new double[base.dimension()];
        for (int i = 0; i < values.length; i++) {
            values[i] = base.get(i) + (random.nextDouble() - 0.5) * 2 * noiseLevel;
            values[i] = Math.max(0.0, Math.min(1.0, values[i]));
        }
        return new DenseVector(values);
    }

    private List<TemporalChunk.ChunkItem> createChunkItems(int count) {
        List<TemporalChunk.ChunkItem> items = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            items.add(new TemporalChunk.ChunkItem(
                createTestPattern(0.7 + i * 0.05),
                0.7 + i * 0.05,
                i * 0.01,
                i
            ));
        }
        return items;
    }

    private double computeMagnitude(Pattern pattern) {
        double sum = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            sum += pattern.get(i) * pattern.get(i);
        }
        return Math.sqrt(sum);
    }

    // ============ Mock Layer for Testing ============

    private static class MockLayer implements Layer {
        private final int size;
        private Pattern activation;

        MockLayer(int size) {
            this.size = size;
            this.activation = new DenseVector(new double[size]);
        }

        @Override
        public String getId() {
            return "mock-layer";
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public LayerType getType() {
            return LayerType.INPUT;
        }

        @Override
        public Pattern getActivation() {
            return activation;
        }

        @Override
        public void setActivation(Pattern activation) {
            this.activation = activation;
        }

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

        @Override
        public com.hellblazer.art.laminar.core.WeightMatrix getWeights() {
            return null;
        }

        @Override
        public void setWeights(com.hellblazer.art.laminar.core.WeightMatrix weights) {
        }

        @Override
        public void updateWeights(Pattern input, double learningRate) {
        }

        @Override
        public void reset() {
            activation = new DenseVector(new double[size]);
        }

        @Override
        public void addActivationListener(com.hellblazer.art.laminar.events.LayerActivationListener listener) {
        }
    }
}