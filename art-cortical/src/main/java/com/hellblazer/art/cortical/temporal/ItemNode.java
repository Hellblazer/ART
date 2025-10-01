package com.hellblazer.art.cortical.temporal;

/**
 * Item node representation in the masking field.
 * Represents a single item in the temporal sequence with its pattern and metadata.
 *
 * Part of the LIST PARSE multi-scale temporal chunking system
 * (Kazerounian & Grossberg, 2014).
 *
 * @author Hal Hildebrand
 */
public class ItemNode {
    private final double[] pattern;
    private double strength;
    private final int position;      // Position in original sequence
    private final double creationTime;
    private double lastAccessTime;
    private int accessCount;

    public ItemNode(double[] pattern, double initialStrength, int position, double creationTime) {
        this.pattern = pattern.clone();
        this.strength = initialStrength;
        this.position = position;
        this.creationTime = creationTime;
        this.lastAccessTime = creationTime;
        this.accessCount = 1;
    }

    /**
     * Check if this node matches the given pattern.
     */
    public boolean matches(double[] testPattern, double threshold) {
        if (testPattern.length != pattern.length) {
            return false;
        }

        var similarity = computeSimilarity(testPattern);
        return similarity >= threshold;
    }

    /**
     * Compute similarity between patterns using cosine similarity.
     */
    private double computeSimilarity(double[] testPattern) {
        var dotProduct = 0.0;
        var norm1 = 0.0;
        var norm2 = 0.0;

        for (int i = 0; i < pattern.length; i++) {
            dotProduct += pattern[i] * testPattern[i];
            norm1 += pattern[i] * pattern[i];
            norm2 += testPattern[i] * testPattern[i];
        }

        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Strengthen this node.
     */
    public void strengthen(double amount) {
        strength += amount;
        accessCount++;
        lastAccessTime = System.currentTimeMillis() / 1000.0; // Convert to seconds
    }

    /**
     * Apply decay to node strength.
     */
    public void decay(double decayRate, double currentTime) {
        var timeSinceAccess = currentTime - lastAccessTime;
        strength *= Math.exp(-decayRate * timeSinceAccess);
    }

    // Getters
    public double[] getPattern() {
        return pattern.clone();
    }

    public double getStrength() {
        return strength;
    }

    public int getPosition() {
        return position;
    }

    public double getCreationTime() {
        return creationTime;
    }

    public double getLastAccessTime() {
        return lastAccessTime;
    }

    public int getAccessCount() {
        return accessCount;
    }
}
