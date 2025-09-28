package com.hellblazer.art.laminar.core;

import java.io.Serializable;

/**
 * Statistics for monitoring layer activity and performance.
 *
 * @author Hal Hildebrand
 */
public class LayerStatistics implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String layerId;
    private final double meanActivation;
    private final double maxActivation;
    private final double minActivation;
    private final double activationVariance;
    private final int activeNeurons;
    private final int totalNeurons;
    private final double sparsity;
    private final long updateCount;
    private final long timestamp;

    private LayerStatistics(Builder builder) {
        this.layerId = builder.layerId;
        this.meanActivation = builder.meanActivation;
        this.maxActivation = builder.maxActivation;
        this.minActivation = builder.minActivation;
        this.activationVariance = builder.activationVariance;
        this.activeNeurons = builder.activeNeurons;
        this.totalNeurons = builder.totalNeurons;
        this.sparsity = builder.sparsity;
        this.updateCount = builder.updateCount;
        this.timestamp = builder.timestamp;
    }

    public String getLayerId() {
        return layerId;
    }

    public double getMeanActivation() {
        return meanActivation;
    }

    public double getMaxActivation() {
        return maxActivation;
    }

    public double getMinActivation() {
        return minActivation;
    }

    public double getActivationVariance() {
        return activationVariance;
    }

    public int getActiveNeurons() {
        return activeNeurons;
    }

    public int getTotalNeurons() {
        return totalNeurons;
    }

    public double getSparsity() {
        return sparsity;
    }

    public long getUpdateCount() {
        return updateCount;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String layerId;
        private double meanActivation = 0.0;
        private double maxActivation = 0.0;
        private double minActivation = 0.0;
        private double activationVariance = 0.0;
        private int activeNeurons = 0;
        private int totalNeurons = 0;
        private double sparsity = 0.0;
        private long updateCount = 0;
        private long timestamp = System.currentTimeMillis();

        public Builder withLayerId(String layerId) {
            this.layerId = layerId;
            return this;
        }

        public Builder withMeanActivation(double mean) {
            this.meanActivation = mean;
            return this;
        }

        public Builder withMaxActivation(double max) {
            this.maxActivation = max;
            return this;
        }

        public Builder withMinActivation(double min) {
            this.minActivation = min;
            return this;
        }

        public Builder withActivationVariance(double variance) {
            this.activationVariance = variance;
            return this;
        }

        public Builder withActiveNeurons(int active) {
            this.activeNeurons = active;
            return this;
        }

        public Builder withTotalNeurons(int total) {
            this.totalNeurons = total;
            return this;
        }

        public Builder withSparsity(double sparsity) {
            this.sparsity = sparsity;
            return this;
        }

        public Builder withUpdateCount(long count) {
            this.updateCount = count;
            return this;
        }

        public Builder withTimestamp(long timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public LayerStatistics build() {
            return new LayerStatistics(this);
        }
    }

    @Override
    public String toString() {
        return String.format("LayerStatistics[%s: mean=%.3f, active=%d/%d, sparsity=%.3f]",
                layerId, meanActivation, activeNeurons, totalNeurons, sparsity);
    }
}