package com.hellblazer.art.laminar.temporal;

/**
 * Parameters controlling temporal chunking behavior in layers.
 *
 * Based on LIST PARSE model parameters from Grossberg & Kazerounian (2016).
 *
 * @author Hal Hildebrand
 */
public class ChunkingParameters {

    private final int maxHistorySize;
    private final double chunkFormationThreshold;
    private final double chunkCoherenceThreshold;
    private final double chunkDecayRate;
    private final double activityThreshold;
    private final int minChunkSize;
    private final int maxChunkSize;
    private final double temporalWindowSize;

    private ChunkingParameters(Builder builder) {
        this.maxHistorySize = builder.maxHistorySize;
        this.chunkFormationThreshold = builder.chunkFormationThreshold;
        this.chunkCoherenceThreshold = builder.chunkCoherenceThreshold;
        this.chunkDecayRate = builder.chunkDecayRate;
        this.activityThreshold = builder.activityThreshold;
        this.minChunkSize = builder.minChunkSize;
        this.maxChunkSize = builder.maxChunkSize;
        this.temporalWindowSize = builder.temporalWindowSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Default parameters based on LIST PARSE model.
     */
    public static ChunkingParameters paperDefaults() {
        return builder()
            .maxHistorySize(12)  // Miller's 7±2 plus buffer
            .chunkFormationThreshold(0.5)
            .chunkCoherenceThreshold(0.6)
            .chunkDecayRate(0.01)  // Slow decay (1% per time unit)
            .activityThreshold(0.1)
            .minChunkSize(2)
            .maxChunkSize(7)  // Miller's magical number
            .temporalWindowSize(0.5)  // 500ms window
            .build();
    }

    /**
     * Fast chunking for real-time processing.
     */
    public static ChunkingParameters fastChunking() {
        return builder()
            .maxHistorySize(5)
            .chunkFormationThreshold(0.3)
            .chunkCoherenceThreshold(0.5)
            .chunkDecayRate(0.05)
            .activityThreshold(0.05)
            .minChunkSize(2)
            .maxChunkSize(5)
            .temporalWindowSize(0.2)
            .build();
    }

    // Getters

    public int getMaxHistorySize() {
        return maxHistorySize;
    }

    public double getChunkFormationThreshold() {
        return chunkFormationThreshold;
    }

    public double getChunkCoherenceThreshold() {
        return chunkCoherenceThreshold;
    }

    public double getChunkDecayRate() {
        return chunkDecayRate;
    }

    public double getActivityThreshold() {
        return activityThreshold;
    }

    public int getMinChunkSize() {
        return minChunkSize;
    }

    public int getMaxChunkSize() {
        return maxChunkSize;
    }

    public double getTemporalWindowSize() {
        return temporalWindowSize;
    }

    /**
     * Builder for chunking parameters.
     */
    public static class Builder {
        private int maxHistorySize = 12;
        private double chunkFormationThreshold = 0.5;
        private double chunkCoherenceThreshold = 0.6;
        private double chunkDecayRate = 0.01;
        private double activityThreshold = 0.1;
        private int minChunkSize = 2;
        private int maxChunkSize = 7;
        private double temporalWindowSize = 0.5;

        public Builder maxHistorySize(int size) {
            this.maxHistorySize = size;
            return this;
        }

        public Builder chunkFormationThreshold(double threshold) {
            this.chunkFormationThreshold = threshold;
            return this;
        }

        public Builder chunkCoherenceThreshold(double threshold) {
            this.chunkCoherenceThreshold = threshold;
            return this;
        }

        public Builder chunkDecayRate(double rate) {
            this.chunkDecayRate = rate;
            return this;
        }

        public Builder activityThreshold(double threshold) {
            this.activityThreshold = threshold;
            return this;
        }

        public Builder minChunkSize(int size) {
            this.minChunkSize = size;
            return this;
        }

        public Builder maxChunkSize(int size) {
            this.maxChunkSize = size;
            return this;
        }

        public Builder temporalWindowSize(double size) {
            this.temporalWindowSize = size;
            return this;
        }

        public ChunkingParameters build() {
            validate();
            return new ChunkingParameters(this);
        }

        private void validate() {
            if (maxHistorySize < 2) {
                throw new IllegalArgumentException("maxHistorySize must be >= 2");
            }
            if (chunkFormationThreshold < 0 || chunkFormationThreshold > 1) {
                throw new IllegalArgumentException("chunkFormationThreshold must be in [0,1]");
            }
            if (chunkCoherenceThreshold < 0 || chunkCoherenceThreshold > 1) {
                throw new IllegalArgumentException("chunkCoherenceThreshold must be in [0,1]");
            }
            if (chunkDecayRate < 0) {
                throw new IllegalArgumentException("chunkDecayRate must be >= 0");
            }
            if (activityThreshold < 0) {
                throw new IllegalArgumentException("activityThreshold must be >= 0");
            }
            if (minChunkSize < 2) {
                throw new IllegalArgumentException("minChunkSize must be >= 2");
            }
            if (maxChunkSize < minChunkSize) {
                throw new IllegalArgumentException("maxChunkSize must be >= minChunkSize");
            }
            if (temporalWindowSize <= 0) {
                throw new IllegalArgumentException("temporalWindowSize must be > 0");
            }
        }
    }
}