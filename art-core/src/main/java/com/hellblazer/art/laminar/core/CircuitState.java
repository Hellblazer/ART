package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Snapshot of laminar circuit state for monitoring and visualization.
 *
 * @author Hal Hildebrand
 */
public class CircuitState implements Serializable {
    private static final long serialVersionUID = 1L;

    private final Map<String, Pattern> layerActivations;
    private final Map<String, Double> pathwayGains;
    private final double resonanceScore;
    private final boolean isResonant;
    private final int currentCategory;
    private final int cycleNumber;
    private final long timestamp;
    private final Map<String, Object> metadata;

    private CircuitState(Builder builder) {
        this.layerActivations = new HashMap<>(builder.layerActivations);
        this.pathwayGains = new HashMap<>(builder.pathwayGains);
        this.resonanceScore = builder.resonanceScore;
        this.isResonant = builder.isResonant;
        this.currentCategory = builder.currentCategory;
        this.cycleNumber = builder.cycleNumber;
        this.timestamp = builder.timestamp;
        this.metadata = new HashMap<>(builder.metadata);
    }

    public Map<String, Pattern> getLayerActivations() {
        return new HashMap<>(layerActivations);
    }

    public Map<String, Double> getPathwayGains() {
        return new HashMap<>(pathwayGains);
    }

    public double getResonanceScore() {
        return resonanceScore;
    }

    public boolean isResonant() {
        return isResonant;
    }

    public int getCurrentCategory() {
        return currentCategory;
    }

    public int getCycleNumber() {
        return cycleNumber;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final Map<String, Pattern> layerActivations = new HashMap<>();
        private final Map<String, Double> pathwayGains = new HashMap<>();
        private double resonanceScore = 0.0;
        private boolean isResonant = false;
        private int currentCategory = -1;
        private int cycleNumber = 0;
        private long timestamp = System.currentTimeMillis();
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder withLayerActivation(String layerId, Pattern activation) {
            layerActivations.put(layerId, activation);
            return this;
        }

        public Builder withPathwayGain(String pathwayId, double gain) {
            pathwayGains.put(pathwayId, gain);
            return this;
        }

        public Builder withResonanceScore(double score) {
            this.resonanceScore = score;
            return this;
        }

        public Builder withResonant(boolean resonant) {
            this.isResonant = resonant;
            return this;
        }

        public Builder withCurrentCategory(int category) {
            this.currentCategory = category;
            return this;
        }

        public Builder withCycleNumber(int cycle) {
            this.cycleNumber = cycle;
            return this;
        }

        public Builder withTimestamp(long timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder withMetadata(String key, Object value) {
            metadata.put(key, value);
            return this;
        }

        public CircuitState build() {
            return new CircuitState(this);
        }
    }

    @Override
    public String toString() {
        return String.format("CircuitState[cycle=%d, resonant=%b, score=%.3f, category=%d, layers=%d]",
                cycleNumber, isResonant, resonanceScore, currentCategory, layerActivations.size());
    }
}