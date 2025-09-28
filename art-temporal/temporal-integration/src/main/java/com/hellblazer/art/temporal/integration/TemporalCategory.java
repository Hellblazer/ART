package com.hellblazer.art.temporal.integration;

import java.util.Arrays;

/**
 * Represents a temporal category in the ART network.
 * Categories encode temporal sequences with their patterns and dynamics.
 */
public class TemporalCategory {

    private double[] prototype;
    private int sequenceLength;
    private int temporalSpan;
    private double creationTime;
    private double lastAccessTime;
    private int accessCount;
    private double strength;

    public TemporalCategory(double[] pattern, int sequenceLength, int temporalSpan, double creationTime) {
        this.prototype = pattern.clone();
        this.sequenceLength = sequenceLength;
        this.temporalSpan = temporalSpan;
        this.creationTime = creationTime;
        this.lastAccessTime = creationTime;
        this.accessCount = 1;
        this.strength = 1.0;
    }

    /**
     * Compute match between this category and an input pattern.
     * Uses fuzzy ART choice function.
     */
    public double computeMatch(double[] pattern) {
        if (pattern.length != prototype.length) {
            return 0.0;
        }

        // Fuzzy ART choice function: |pattern ∧ prototype| / (α + |prototype|)
        double intersection = 0.0;
        double prototypeNorm = 0.0;

        for (int i = 0; i < pattern.length; i++) {
            intersection += Math.min(pattern[i], prototype[i]);
            prototypeNorm += prototype[i];
        }

        double alpha = 0.001;  // Small constant for numerical stability
        return intersection / (alpha + prototypeNorm);
    }

    /**
     * Update category prototype with new pattern.
     */
    public void update(double[] pattern, double learningRate) {
        if (pattern.length != prototype.length) {
            return;
        }

        // Fast-slow learning: prototype = β * (pattern ∧ prototype) + (1 - β) * prototype
        for (int i = 0; i < prototype.length; i++) {
            double minValue = Math.min(pattern[i], prototype[i]);
            prototype[i] = learningRate * minValue + (1.0 - learningRate) * prototype[i];
        }

        accessCount++;
        strength = Math.min(1.0, strength + 0.1);  // Strengthen on access
    }

    /**
     * Apply temporal decay to category strength.
     */
    public void decay(double decayRate, double currentTime) {
        double timeSinceAccess = currentTime - lastAccessTime;
        strength *= Math.exp(-decayRate * timeSinceAccess);
        lastAccessTime = currentTime;
    }

    /**
     * Check if category matches temporal characteristics.
     */
    public boolean matchesTemporalCharacteristics(int seqLength, int span, double tolerance) {
        double seqDiff = Math.abs(sequenceLength - seqLength) / (double) Math.max(sequenceLength, seqLength);
        double spanDiff = Math.abs(temporalSpan - span) / (double) Math.max(temporalSpan, span);
        return seqDiff <= tolerance && spanDiff <= tolerance;
    }

    /**
     * Increment access count for vectorized performance tracking.
     */
    public void incrementAccessCount() {
        accessCount++;
    }

    /**
     * Update last access time for vectorized performance tracking.
     */
    public void updateLastAccess() {
        lastAccessTime = System.currentTimeMillis() / 1000.0;
    }

    /**
     * Add temporal pattern for vectorized performance (simplified).
     */
    public void addTemporalPattern(com.hellblazer.art.temporal.memory.TemporalPattern pattern) {
        // For performance version, we can just update the access time
        updateLastAccess();
        incrementAccessCount();
    }

    // Getters
    public double[] getPrototype() {
        return prototype.clone();
    }

    public int getSequenceLength() {
        return sequenceLength;
    }

    public int getTemporalSpan() {
        return temporalSpan;
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

    public double getStrength() {
        return strength;
    }

    @Override
    public String toString() {
        return String.format("TemporalCategory[seqLen=%d, span=%d, strength=%.3f, accesses=%d]",
                           sequenceLength, temporalSpan, strength, accessCount);
    }
}