package com.hellblazer.art.cortical.temporal;

/**
 * Parameters for masking field dynamics.
 * Controls multi-scale temporal integration and chunking behavior.
 *
 * Part of the LIST PARSE multi-scale temporal chunking system
 * (Kazerounian & Grossberg, 2014).
 *
 * Time scale: 50-500ms (intermediate between working memory and long-term memory)
 *
 * @author Hal Hildebrand
 */
public record MaskingFieldParameters(
    // Capacity parameters
    int maxItemNodes,
    int maxChunks,
    int minChunkSize,
    int maxChunkSize,

    // Temporal parameters (in seconds)
    double integrationTimeStep,     // 50ms default
    double minChunkInterval,        // Minimum time between chunks
    double maxTemporalGap,          // Maximum gap for coherent sequence

    // Competition parameters
    double spatialScale,            // Spatial extent of competition
    double excitationRange,         // Range of excitatory connections
    double inhibitionRange,         // Range of inhibitory connections
    double competitionStrength,     // Strength of competition
    double winnerThreshold,         // Threshold for winner selection

    // Dynamics parameters
    double itemDecayRate,           // Decay rate for item activations
    double chunkDecayRate,          // Decay rate for chunk activations
    double maxActivation,           // Maximum activation level
    double initialActivation,       // Initial activation for new items
    double activationBoost,         // Boost for matching items
    double activeChunkBoost,        // Boost for active chunk

    // Learning parameters
    double learningRate,            // Rate of strengthening
    double matchingThreshold,       // Threshold for pattern matching
    double selfExcitation,          // Self-excitation strength

    // Control flags
    boolean normalizationEnabled,   // Enable activation normalization
    boolean resetAfterChunk,        // Reset items after chunk formation
    double resetDecayFactor         // Decay factor for reset
) {
    /**
     * Canonical constructor with validation.
     */
    public MaskingFieldParameters {
        validate(maxItemNodes, maxChunks, minChunkSize, maxChunkSize,
                integrationTimeStep, spatialScale, winnerThreshold,
                learningRate, matchingThreshold);
    }

    private static void validate(int maxItemNodes, int maxChunks,
                                 int minChunkSize, int maxChunkSize,
                                 double integrationTimeStep, double spatialScale,
                                 double winnerThreshold, double learningRate,
                                 double matchingThreshold) {
        if (maxItemNodes < 10 || maxItemNodes > 100) {
            throw new IllegalArgumentException("Max item nodes should be in [10, 100]");
        }
        if (minChunkSize < 2 || minChunkSize > maxChunkSize) {
            throw new IllegalArgumentException("Invalid chunk size range");
        }
        if (integrationTimeStep <= 0 || integrationTimeStep > 0.1) {
            throw new IllegalArgumentException("Integration time step should be in (0, 0.1]");
        }
        if (spatialScale <= 0) {
            throw new IllegalArgumentException("Spatial scale must be positive");
        }
        if (winnerThreshold < 0 || winnerThreshold > 1.0) {
            throw new IllegalArgumentException("Winner threshold must be in [0, 1]");
        }
        if (learningRate <= 0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in (0, 1]");
        }
        if (matchingThreshold < 0 || matchingThreshold > 1.0) {
            throw new IllegalArgumentException("Matching threshold must be in [0, 1]");
        }
    }

    /**
     * Create builder for constructing parameters.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create parameters for phone number chunking (3-3-4 pattern).
     */
    public static MaskingFieldParameters phoneNumberDefaults() {
        return builder()
            .maxItemNodes(50)
            .minChunkSize(3)
            .maxChunkSize(4)
            .integrationTimeStep(0.05)     // 50ms time scale
            .minChunkInterval(0.3)         // 300ms between chunks
            .spatialScale(2.0)
            .competitionStrength(0.6)
            .winnerThreshold(0.35)
            .resetAfterChunk(true)
            .build();
    }

    /**
     * Create parameters for general list learning.
     */
    public static MaskingFieldParameters listLearningDefaults() {
        return builder()
            .maxItemNodes(50)
            .minChunkSize(2)
            .maxChunkSize(7)              // Miller's 7±2
            .integrationTimeStep(0.05)
            .minChunkInterval(0.2)
            .maxTemporalGap(5.0)          // Allow larger gaps
            .competitionStrength(0.5)
            .resetAfterChunk(false)       // Keep items for overlapping chunks
            .build();
    }

    /**
     * Builder for MaskingFieldParameters.
     */
    public static class Builder {
        private int maxItemNodes = 50;
        private int maxChunks = 10;
        private int minChunkSize = 3;
        private int maxChunkSize = 7;
        private double integrationTimeStep = 0.05;    // 50ms
        private double minChunkInterval = 0.2;        // 200ms
        private double maxTemporalGap = 3.0;          // Max 3 position gap
        private double spatialScale = 2.0;
        private double excitationRange = 1.0;
        private double inhibitionRange = 3.0;
        private double competitionStrength = 0.5;
        private double winnerThreshold = 0.3;
        private double itemDecayRate = 0.05;
        private double chunkDecayRate = 0.02;
        private double maxActivation = 1.0;
        private double initialActivation = 0.5;
        private double activationBoost = 0.2;
        private double activeChunkBoost = 0.3;
        private double learningRate = 0.1;
        private double matchingThreshold = 0.8;
        private double selfExcitation = 0.1;
        private boolean normalizationEnabled = true;
        private boolean resetAfterChunk = true;
        private double resetDecayFactor = 0.3;

        public Builder maxItemNodes(int val) { maxItemNodes = val; return this; }
        public Builder maxChunks(int val) { maxChunks = val; return this; }
        public Builder minChunkSize(int val) { minChunkSize = val; return this; }
        public Builder maxChunkSize(int val) { maxChunkSize = val; return this; }
        public Builder integrationTimeStep(double val) { integrationTimeStep = val; return this; }
        public Builder minChunkInterval(double val) { minChunkInterval = val; return this; }
        public Builder maxTemporalGap(double val) { maxTemporalGap = val; return this; }
        public Builder spatialScale(double val) { spatialScale = val; return this; }
        public Builder excitationRange(double val) { excitationRange = val; return this; }
        public Builder inhibitionRange(double val) { inhibitionRange = val; return this; }
        public Builder competitionStrength(double val) { competitionStrength = val; return this; }
        public Builder winnerThreshold(double val) { winnerThreshold = val; return this; }
        public Builder itemDecayRate(double val) { itemDecayRate = val; return this; }
        public Builder chunkDecayRate(double val) { chunkDecayRate = val; return this; }
        public Builder maxActivation(double val) { maxActivation = val; return this; }
        public Builder initialActivation(double val) { initialActivation = val; return this; }
        public Builder activationBoost(double val) { activationBoost = val; return this; }
        public Builder activeChunkBoost(double val) { activeChunkBoost = val; return this; }
        public Builder learningRate(double val) { learningRate = val; return this; }
        public Builder matchingThreshold(double val) { matchingThreshold = val; return this; }
        public Builder selfExcitation(double val) { selfExcitation = val; return this; }
        public Builder normalizationEnabled(boolean val) { normalizationEnabled = val; return this; }
        public Builder resetAfterChunk(boolean val) { resetAfterChunk = val; return this; }
        public Builder resetDecayFactor(double val) { resetDecayFactor = val; return this; }

        public MaskingFieldParameters build() {
            return new MaskingFieldParameters(
                maxItemNodes, maxChunks, minChunkSize, maxChunkSize,
                integrationTimeStep, minChunkInterval, maxTemporalGap,
                spatialScale, excitationRange, inhibitionRange,
                competitionStrength, winnerThreshold,
                itemDecayRate, chunkDecayRate, maxActivation,
                initialActivation, activationBoost, activeChunkBoost,
                learningRate, matchingThreshold, selfExcitation,
                normalizationEnabled, resetAfterChunk, resetDecayFactor
            );
        }
    }
}
