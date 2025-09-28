package com.hellblazer.art.temporal.masking;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Unit tests for ListChunk class.
 */
public class ListChunkTest {

    @Test
    public void testListChunkCreation() {
        List<ItemNode> items = createTestItems(3);
        double formationTime = 100.0;
        int chunkId = 1;

        var chunk = new ListChunk(items, formationTime, chunkId);

        assertEquals(3, chunk.size());
        assertEquals(formationTime, chunk.getFormationTime());
        assertEquals(chunkId, chunk.getChunkId());
        assertEquals(ListChunk.ChunkType.SMALL, chunk.getType());
        assertTrue(chunk.getChunkStrength() > 0);
    }

    @Test
    public void testChunkTypes() {
        // Test SMALL chunk (1-3 items)
        var smallChunk = new ListChunk(createTestItems(2), 0.0, 0);
        assertEquals(ListChunk.ChunkType.SMALL, smallChunk.getType());

        // Test MEDIUM chunk (4-5 items)
        var mediumChunk = new ListChunk(createTestItems(4), 0.0, 1);
        assertEquals(ListChunk.ChunkType.MEDIUM, mediumChunk.getType());

        // Test LARGE chunk (6-7 items)
        var largeChunk = new ListChunk(createTestItems(7), 0.0, 2);
        assertEquals(ListChunk.ChunkType.LARGE, largeChunk.getType());

        // Test SUPER chunk (8+ items)
        var superChunk = new ListChunk(createTestItems(10), 0.0, 3);
        assertEquals(ListChunk.ChunkType.SUPER, superChunk.getType());
    }

    @Test
    public void testChunkPattern() {
        // Create items with specific patterns
        List<ItemNode> items = new ArrayList<>();
        double[][] patterns = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        for (int i = 0; i < patterns.length; i++) {
            items.add(new ItemNode(patterns[i], 1.0, i, 0.0));
        }

        var chunk = new ListChunk(items, 0.0, 0);
        double[] chunkPattern = chunk.getChunkPattern();

        // Average should be [1/3, 1/3, 1/3]
        assertEquals(1.0/3.0, chunkPattern[0], 0.001);
        assertEquals(1.0/3.0, chunkPattern[1], 0.001);
        assertEquals(1.0/3.0, chunkPattern[2], 0.001);
    }

    @Test
    public void testTemporalSpan() {
        List<ItemNode> items = new ArrayList<>();
        // Create items at positions 2, 5, 8
        items.add(new ItemNode(new double[]{1.0}, 1.0, 2, 0.0));
        items.add(new ItemNode(new double[]{1.0}, 1.0, 5, 0.0));
        items.add(new ItemNode(new double[]{1.0}, 1.0, 8, 0.0));

        var chunk = new ListChunk(items, 0.0, 0);

        // Span should be 8 - 2 + 1 = 7
        assertEquals(7, chunk.getTemporalSpan());
    }

    @Test
    public void testChunkOverlap() {
        // Create first chunk with positions 0, 1, 2
        List<ItemNode> items1 = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            items1.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk1 = new ListChunk(items1, 0.0, 0);

        // Create non-overlapping chunk with positions 3, 4, 5
        List<ItemNode> items2 = new ArrayList<>();
        for (int i = 3; i < 6; i++) {
            items2.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk2 = new ListChunk(items2, 0.0, 1);

        assertFalse(chunk1.overlaps(chunk2));

        // Create overlapping chunk with positions 2, 3, 4
        List<ItemNode> items3 = new ArrayList<>();
        for (int i = 2; i < 5; i++) {
            items3.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk3 = new ListChunk(items3, 0.0, 2);

        assertTrue(chunk1.overlaps(chunk3));
    }

    @Test
    public void testChunkMerging() {
        // Create first chunk
        List<ItemNode> items1 = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            items1.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk1 = new ListChunk(items1, 0.0, 0);

        // Create second chunk with some overlap
        List<ItemNode> items2 = new ArrayList<>();
        for (int i = 2; i < 5; i++) {
            items2.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk2 = new ListChunk(items2, 10.0, 1);

        // Merge chunks
        var merged = chunk1.merge(chunk2);

        // Should have 5 unique items (0, 1, 2, 3, 4)
        assertEquals(5, merged.size());

        // Check items are sorted by position
        var mergedItems = merged.getItems();
        for (int i = 0; i < mergedItems.size() - 1; i++) {
            assertTrue(mergedItems.get(i).getPosition() < mergedItems.get(i + 1).getPosition());
        }
    }

    @Test
    public void testChunkDecay() {
        var items = createTestItems(3);
        var chunk = new ListChunk(items, 0.0, 0);

        double initialStrength = chunk.getChunkStrength();
        double decayRate = 0.1;
        double currentTime = 10.0;

        chunk.decay(decayRate, currentTime);

        // Strength should decrease
        assertTrue(chunk.getChunkStrength() < initialStrength);
        assertTrue(chunk.getChunkStrength() > 0);
    }

    @Test
    public void testEmptyChunk() {
        var emptyChunk = new ListChunk(new ArrayList<>(), 0.0, 0);

        assertEquals(0, emptyChunk.size());
        assertEquals(0, emptyChunk.getTemporalSpan());
        assertEquals(0, emptyChunk.getChunkPattern().length);
        assertEquals(ListChunk.ChunkType.SMALL, emptyChunk.getType());
    }

    @Test
    public void testChunkStrength() {
        // Create items with different strengths
        List<ItemNode> items = new ArrayList<>();
        double[] strengths = {1.0, 0.5, 0.75};

        for (int i = 0; i < strengths.length; i++) {
            var node = new ItemNode(new double[]{i}, strengths[i], i, 0.0);
            items.add(node);
        }

        var chunk = new ListChunk(items, 0.0, 0);

        // Initial chunk strength should be average of initial item strengths
        double expectedStrength = (strengths[0] + strengths[1] + strengths[2]) / 3.0;
        assertEquals(expectedStrength, chunk.getChunkStrength(), 0.1);
    }

    @Test
    public void testChunkImmutability() {
        List<ItemNode> items = createTestItems(3);
        var chunk = new ListChunk(items, 0.0, 0);

        // Modify original list
        items.add(new ItemNode(new double[]{9}, 1.0, 9, 0.0));

        // Chunk should be unchanged
        assertEquals(3, chunk.size());

        // Get items and modify
        var chunkItems = chunk.getItems();
        int originalSize = chunkItems.size();
        chunkItems.clear();

        // Chunk should still be unchanged
        assertEquals(originalSize, chunk.size());
    }

    // Helper methods

    private List<ItemNode> createTestItems(int count) {
        List<ItemNode> items = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double[] pattern = new double[]{i * 0.1, i * 0.2, i * 0.3};
            items.add(new ItemNode(pattern, 0.5 + i * 0.1, i, i * 10.0));
        }
        return items;
    }
}