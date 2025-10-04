package com.hellblazer.art.cortical.temporal;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Unit tests for ListChunk class.
 * Part of LIST PARSE temporal chunking system.
 */
class ListChunkTest {

    @Test
    void testListChunkCreation() {
        var items = createTestItems(3);
        var formationTime = 100.0;
        var chunkId = 1;

        var chunk = new ListChunk(items, formationTime, chunkId);

        assertEquals(3, chunk.size());
        assertEquals(formationTime, chunk.getFormationTime());
        assertEquals(chunkId, chunk.getChunkId());
        assertEquals(ListChunk.ChunkType.SMALL, chunk.getType());
        assertTrue(chunk.getChunkStrength() > 0);
    }

    @Test
    void testChunkTypes() {
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
    void testChunkPattern() {
        // Create items with specific patterns
        var items = new ArrayList<ItemNode>();
        var patterns = new double[][]{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        for (int i = 0; i < patterns.length; i++) {
            items.add(new ItemNode(patterns[i], 1.0, i, 0.0));
        }

        var chunk = new ListChunk(items, 0.0, 0);
        var chunkPattern = chunk.getChunkPattern();

        // Average should be [1/3, 1/3, 1/3]
        assertEquals(1.0/3.0, chunkPattern[0], 0.001);
        assertEquals(1.0/3.0, chunkPattern[1], 0.001);
        assertEquals(1.0/3.0, chunkPattern[2], 0.001);
    }

    @Test
    void testTemporalSpan() {
        var items = new ArrayList<ItemNode>();
        // Create items at positions 2, 5, 8
        items.add(new ItemNode(new double[]{1.0}, 1.0, 2, 0.0));
        items.add(new ItemNode(new double[]{1.0}, 1.0, 5, 0.0));
        items.add(new ItemNode(new double[]{1.0}, 1.0, 8, 0.0));

        var chunk = new ListChunk(items, 0.0, 0);

        // Span should be 8 - 2 + 1 = 7
        assertEquals(7, chunk.getTemporalSpan());
    }

    @Test
    void testChunkOverlap() {
        // Create first chunk with positions 0, 1, 2
        var items1 = new ArrayList<ItemNode>();
        for (int i = 0; i < 3; i++) {
            items1.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk1 = new ListChunk(items1, 0.0, 0);

        // Create non-overlapping chunk with positions 3, 4, 5
        var items2 = new ArrayList<ItemNode>();
        for (int i = 3; i < 6; i++) {
            items2.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk2 = new ListChunk(items2, 0.0, 1);

        assertFalse(chunk1.overlaps(chunk2));

        // Create overlapping chunk with positions 2, 3, 4
        var items3 = new ArrayList<ItemNode>();
        for (int i = 2; i < 5; i++) {
            items3.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk3 = new ListChunk(items3, 0.0, 2);

        assertTrue(chunk1.overlaps(chunk3));
    }

    @Test
    void testChunkMerging() {
        // Create first chunk
        var items1 = new ArrayList<ItemNode>();
        for (int i = 0; i < 3; i++) {
            items1.add(new ItemNode(new double[]{i}, 1.0, i, 0.0));
        }
        var chunk1 = new ListChunk(items1, 0.0, 0);

        // Create second chunk with some overlap
        var items2 = new ArrayList<ItemNode>();
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
    void testChunkDecay() {
        var items = createTestItems(3);
        var chunk = new ListChunk(items, 0.0, 0);

        var initialStrength = chunk.getChunkStrength();
        var decayRate = 0.1;
        var currentTime = 10.0;

        chunk.decay(decayRate, currentTime);

        // Strength should decrease
        assertTrue(chunk.getChunkStrength() < initialStrength);
        assertTrue(chunk.getChunkStrength() > 0);
    }

    @Test
    void testEmptyChunk() {
        var emptyChunk = new ListChunk(new ArrayList<>(), 0.0, 0);

        assertEquals(0, emptyChunk.size());
        assertEquals(0, emptyChunk.getTemporalSpan());
        assertEquals(0, emptyChunk.getChunkPattern().length);
        assertEquals(ListChunk.ChunkType.SMALL, emptyChunk.getType());
    }

    @Test
    void testChunkStrength() {
        // Create items with different strengths
        var items = new ArrayList<ItemNode>();
        var strengths = new double[]{1.0, 0.5, 0.75};

        for (int i = 0; i < strengths.length; i++) {
            var node = new ItemNode(new double[]{i}, strengths[i], i, 0.0);
            items.add(node);
        }

        var chunk = new ListChunk(items, 0.0, 0);

        // Initial chunk strength should be average of initial item strengths
        var expectedStrength = (strengths[0] + strengths[1] + strengths[2]) / 3.0;
        assertEquals(expectedStrength, chunk.getChunkStrength(), 0.1);
    }

    @Test
    void testChunkImmutability() {
        var items = createTestItems(3);
        var chunk = new ListChunk(items, 0.0, 0);

        // Modify original list
        items.add(new ItemNode(new double[]{9}, 1.0, 9, 0.0));

        // Chunk should be unchanged
        assertEquals(3, chunk.size());

        // Get items and modify
        var chunkItems = chunk.getItems();
        var originalSize = chunkItems.size();
        chunkItems.clear();

        // Chunk should still be unchanged
        assertEquals(originalSize, chunk.size());
    }

    // Helper methods

    private List<ItemNode> createTestItems(int count) {
        var items = new ArrayList<ItemNode>();
        for (int i = 0; i < count; i++) {
            var pattern = new double[]{i * 0.1, i * 0.2, i * 0.3};
            items.add(new ItemNode(pattern, 0.5 + i * 0.1, i, i * 10.0));
        }
        return items;
    }
}
