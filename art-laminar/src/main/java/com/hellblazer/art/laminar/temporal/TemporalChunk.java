package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a temporal chunk of layer activations.
 * Chunks group sequential patterns that form coherent temporal units.
 *
 * Inspired by ListChunk from the MaskingField implementation,
 * adapted for laminar layer processing.
 *
 * @author Hal Hildebrand
 */
public class TemporalChunk {

    private final List<ChunkItem> items;
    private final double formationTime;
    private final int chunkId;
    private final ChunkType type;
    private double strength;
    private double coherence;

    public TemporalChunk(List<ChunkItem> items, double formationTime, int chunkId) {
        this.items = new ArrayList<>(items);
        this.formationTime = formationTime;
        this.chunkId = chunkId;
        this.strength = computeInitialStrength();
        this.coherence = computeCoherence();
        this.type = determineType();
    }

    /**
     * Compute initial strength from constituent items.
     */
    private double computeInitialStrength() {
        if (items.isEmpty()) return 0.0;

        double totalStrength = 0.0;
        for (var item : items) {
            totalStrength += item.activation();
        }
        return totalStrength / items.size();
    }

    /**
     * Compute temporal coherence - how well items fit together.
     */
    private double computeCoherence() {
        if (items.size() < 2) return 1.0;

        // Measure similarity between consecutive items
        double totalSimilarity = 0.0;
        for (int i = 0; i < items.size() - 1; i++) {
            totalSimilarity += computeSimilarity(items.get(i).pattern(),
                                                 items.get(i + 1).pattern());
        }

        return totalSimilarity / (items.size() - 1);
    }

    /**
     * Compute similarity between two patterns.
     */
    private double computeSimilarity(Pattern p1, Pattern p2) {
        if (p1.dimension() != p2.dimension()) return 0.0;

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < p1.dimension(); i++) {
            dotProduct += p1.get(i) * p2.get(i);
            norm1 += p1.get(i) * p1.get(i);
            norm2 += p2.get(i) * p2.get(i);
        }

        if (norm1 == 0 || norm2 == 0) return 0.0;
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Determine chunk type based on size.
     */
    private ChunkType determineType() {
        int size = items.size();
        if (size <= 3) return ChunkType.SMALL;
        if (size <= 5) return ChunkType.MEDIUM;
        if (size <= 7) return ChunkType.LARGE;
        return ChunkType.SUPER;
    }

    /**
     * Get representative pattern for this chunk.
     * Computed as weighted average of item patterns.
     */
    public Pattern getRepresentativePattern() {
        if (items.isEmpty()) return null;

        int dimension = items.get(0).pattern().dimension();
        double[] chunkPattern = new double[dimension];
        double totalWeight = 0.0;

        for (var item : items) {
            double weight = item.activation();
            totalWeight += weight;

            for (int i = 0; i < dimension; i++) {
                chunkPattern[i] += item.pattern().get(i) * weight;
            }
        }

        // Normalize
        if (totalWeight > 0) {
            for (int i = 0; i < dimension; i++) {
                chunkPattern[i] /= totalWeight;
            }
        }

        return new com.hellblazer.art.core.DenseVector(chunkPattern);
    }

    /**
     * Get temporal span of chunk (time from first to last item).
     */
    public double getTemporalSpan() {
        if (items.size() < 2) return 0.0;

        double firstTime = items.get(0).time();
        double lastTime = items.get(items.size() - 1).time();
        return lastTime - firstTime;
    }

    /**
     * Apply decay to chunk strength over time.
     */
    public void decay(double decayRate, double currentTime) {
        double timeSinceFormation = currentTime - formationTime;
        strength *= Math.exp(-decayRate * timeSinceFormation);
    }

    /**
     * Check if this chunk is still active (strength above threshold).
     */
    public boolean isActive(double threshold) {
        return strength >= threshold;
    }

    /**
     * Merge with another chunk if they overlap or are contiguous.
     */
    public TemporalChunk merge(TemporalChunk other) {
        List<ChunkItem> mergedItems = new ArrayList<>(items);

        // Add items from other chunk
        for (var otherItem : other.items) {
            // Check for duplicate times
            boolean exists = items.stream()
                .anyMatch(item -> Math.abs(item.time() - otherItem.time()) < 1e-6);

            if (!exists) {
                mergedItems.add(otherItem);
            }
        }

        // Sort by time
        mergedItems.sort((a, b) -> Double.compare(a.time(), b.time()));

        return new TemporalChunk(mergedItems,
                                Math.min(formationTime, other.formationTime),
                                chunkId);
    }

    // Getters

    public List<ChunkItem> getItems() {
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

    public double getStrength() {
        return strength;
    }

    public void setStrength(double strength) {
        this.strength = strength;
    }

    public double getCoherence() {
        return coherence;
    }

    public ChunkType getType() {
        return type;
    }

    /**
     * Individual item within a temporal chunk.
     */
    public record ChunkItem(
        Pattern pattern,
        double activation,
        double time,
        int sequencePosition
    ) {}

    /**
     * Chunk type based on size.
     */
    public enum ChunkType {
        SMALL(1, 3, "Short sequence"),
        MEDIUM(4, 5, "Phrase-like"),
        LARGE(6, 7, "Miller's magical number"),
        SUPER(8, 12, "Extended sequence");

        private final int minSize;
        private final int maxSize;
        private final String description;

        ChunkType(int minSize, int maxSize, String description) {
            this.minSize = minSize;
            this.maxSize = maxSize;
            this.description = description;
        }

        public int getMinSize() {
            return minSize;
        }

        public int getMaxSize() {
            return maxSize;
        }

        public String getDescription() {
            return description;
        }
    }
}