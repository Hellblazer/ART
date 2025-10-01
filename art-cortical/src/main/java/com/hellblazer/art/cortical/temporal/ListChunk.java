package com.hellblazer.art.cortical.temporal;

import java.util.ArrayList;
import java.util.List;

/**
 * List chunk representation in the masking field.
 * Represents a coherent group of item nodes that form a temporal chunk.
 *
 * Part of the LIST PARSE multi-scale temporal chunking system
 * (Kazerounian & Grossberg, 2014).
 *
 * Chunk types follow working memory capacity constraints (Miller's 7±2):
 * - SMALL: 1-3 items (e.g., area code "555")
 * - MEDIUM: 4-5 items (e.g., prefix "1234")
 * - LARGE: 6-7 items (e.g., full phone number)
 * - SUPER: 8+ items (complete sequences)
 *
 * @author Hal Hildebrand
 */
public class ListChunk {
    private final List<ItemNode> items;
    private final double formationTime;
    private final int chunkId;
    private double chunkStrength;
    private final ChunkType type;

    public ListChunk(List<ItemNode> items, double formationTime, int chunkId) {
        this.items = new ArrayList<>(items);
        this.formationTime = formationTime;
        this.chunkId = chunkId;
        this.chunkStrength = computeInitialStrength();
        this.type = determineChunkType();
    }

    /**
     * Compute initial chunk strength from constituent items.
     */
    private double computeInitialStrength() {
        var totalStrength = 0.0;
        for (var item : items) {
            totalStrength += item.getStrength();
        }
        return totalStrength / items.size();
    }

    /**
     * Determine chunk type based on size and pattern.
     */
    private ChunkType determineChunkType() {
        if (items.size() <= 3) {
            return ChunkType.SMALL;
        } else if (items.size() <= 5) {
            return ChunkType.MEDIUM;
        } else if (items.size() <= 7) {
            return ChunkType.LARGE;
        } else {
            return ChunkType.SUPER;
        }
    }

    /**
     * Get chunk pattern by averaging item patterns.
     */
    public double[] getChunkPattern() {
        if (items.isEmpty()) {
            return new double[0];
        }

        var patternSize = items.get(0).getPattern().length;
        var chunkPattern = new double[patternSize];

        for (var item : items) {
            var pattern = item.getPattern();
            for (int i = 0; i < patternSize; i++) {
                chunkPattern[i] += pattern[i];
            }
        }

        // Normalize
        for (int i = 0; i < patternSize; i++) {
            chunkPattern[i] /= items.size();
        }

        return chunkPattern;
    }

    /**
     * Get temporal span of the chunk.
     */
    public int getTemporalSpan() {
        if (items.isEmpty()) return 0;

        var minPos = items.get(0).getPosition();
        var maxPos = items.get(0).getPosition();

        for (var item : items) {
            minPos = Math.min(minPos, item.getPosition());
            maxPos = Math.max(maxPos, item.getPosition());
        }

        return maxPos - minPos + 1;
    }

    /**
     * Check if this chunk overlaps with another.
     */
    public boolean overlaps(ListChunk other) {
        for (var item1 : items) {
            for (var item2 : other.items) {
                if (item1.getPosition() == item2.getPosition()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Merge with another chunk.
     */
    public ListChunk merge(ListChunk other) {
        var mergedItems = new ArrayList<>(items);

        // Add non-overlapping items from other chunk
        for (var otherItem : other.items) {
            var exists = false;
            for (var item : items) {
                if (item.getPosition() == otherItem.getPosition()) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                mergedItems.add(otherItem);
            }
        }

        // Sort by position
        mergedItems.sort((a, b) -> Integer.compare(a.getPosition(), b.getPosition()));

        return new ListChunk(mergedItems, formationTime, chunkId);
    }

    /**
     * Apply decay to chunk strength.
     */
    public void decay(double decayRate, double currentTime) {
        var timeSinceFormation = currentTime - formationTime;
        chunkStrength *= Math.exp(-decayRate * timeSinceFormation);
    }

    // Getters
    public List<ItemNode> getItems() {
        return new ArrayList<>(items);
    }

    public int size() {
        return items.size();
    }

    public double getFormationTime() {
        return formationTime;
    }

    public int getChunkId() {
        return chunkId;
    }

    public double getChunkStrength() {
        return chunkStrength;
    }

    public ChunkType getType() {
        return type;
    }

    public void setChunkStrength(double strength) {
        this.chunkStrength = strength;
    }

    /**
     * Chunk type enumeration based on Miller's 7±2 working memory capacity.
     */
    public enum ChunkType {
        SMALL(1, 3),    // 1-3 items (e.g., area code)
        MEDIUM(4, 5),   // 4-5 items (e.g., prefix)
        LARGE(6, 7),    // 6-7 items (e.g., full phone number)
        SUPER(8, 12);   // 8+ items (e.g., complete sequence)

        private final int minSize;
        private final int maxSize;

        ChunkType(int minSize, int maxSize) {
            this.minSize = minSize;
            this.maxSize = maxSize;
        }

        public int getMinSize() {
            return minSize;
        }

        public int getMaxSize() {
            return maxSize;
        }
    }
}
