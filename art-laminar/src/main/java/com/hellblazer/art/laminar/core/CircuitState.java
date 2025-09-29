package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents the current state of a laminar circuit.
 *
 * @author Hal Hildebrand
 */
public class CircuitState {
    private final Map<String, Pattern> layerActivations;
    private final Map<String, Double> pathwayGains;
    private final double resonanceScore;
    private final boolean resonant;
    private final int currentCategory;
    private final int cycleNumber;

    private CircuitState(Builder builder) {
        this.layerActivations = new HashMap<>(builder.layerActivations);
        this.pathwayGains = new HashMap<>(builder.pathwayGains);
        this.resonanceScore = builder.resonanceScore;
        this.resonant = builder.resonant;
        this.currentCategory = builder.currentCategory;
        this.cycleNumber = builder.cycleNumber;
    }

    public static Builder builder() {
        return new Builder();
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
        return resonant;
    }

    public int getCurrentCategory() {
        return currentCategory;
    }

    public int getCycleNumber() {
        return cycleNumber;
    }

    public static class Builder {
        private final Map<String, Pattern> layerActivations = new HashMap<>();
        private final Map<String, Double> pathwayGains = new HashMap<>();
        private double resonanceScore = 0.0;
        private boolean resonant = false;
        private int currentCategory = -1;
        private int cycleNumber = 0;

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
            this.resonant = resonant;
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

        public CircuitState build() {
            return new CircuitState(this);
        }
    }
}