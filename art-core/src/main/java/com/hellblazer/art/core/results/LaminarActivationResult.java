package com.hellblazer.art.core.results;

import com.hellblazer.art.core.Pattern;
import java.util.HashMap;
import java.util.Map;

/**
 * Result of laminar circuit activation with layer-specific information.
 * Non-sealed class implementing ActivationResult for laminar circuits.
 *
 * @author Hal Hildebrand
 */
public non-sealed class LaminarActivationResult implements ActivationResult {

    private final Map<String, Pattern> layerActivations;
    private final double resonanceScore;
    private final boolean resonant;
    private final int processingCycles;

    private final int categoryIndex;
    private final Pattern bottomUpActivation;
    private final Pattern topDownExpectation;

    protected LaminarActivationResult(Builder builder) {
        this.categoryIndex = builder.categoryIndex;
        this.bottomUpActivation = builder.bottomUpActivation;
        this.topDownExpectation = builder.topDownExpectation;
        this.layerActivations = new HashMap<>(builder.layerActivations);
        this.resonanceScore = builder.resonanceScore;
        this.resonant = builder.resonant;
        this.processingCycles = builder.processingCycles;
    }

    public int getCategoryIndex() {
        return categoryIndex;
    }

    public Pattern getBottomUpActivation() {
        return bottomUpActivation;
    }

    public Pattern getTopDownExpectation() {
        return topDownExpectation;
    }

    public Map<String, Pattern> getLayerActivations() {
        return new HashMap<>(layerActivations);
    }

    public double getResonanceScore() {
        return resonanceScore;
    }

    public boolean isResonant() {
        return resonant;
    }

    public int getProcessingCycles() {
        return processingCycles;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int categoryIndex = -1;
        private Pattern bottomUpActivation;
        private Pattern topDownExpectation;
        private final Map<String, Pattern> layerActivations = new HashMap<>();
        private double resonanceScore = 0.0;
        private boolean resonant = false;
        private int processingCycles = 0;

        public Builder withCategoryIndex(int index) {
            this.categoryIndex = index;
            return this;
        }

        public Builder withBottomUpActivation(Pattern pattern) {
            this.bottomUpActivation = pattern;
            return this;
        }

        public Builder withTopDownExpectation(Pattern pattern) {
            this.topDownExpectation = pattern;
            return this;
        }

        public Builder withLayerActivation(String layerId, Pattern activation) {
            layerActivations.put(layerId, activation);
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

        public Builder withProcessingCycles(int cycles) {
            this.processingCycles = cycles;
            return this;
        }

        public LaminarActivationResult build() {
            return new LaminarActivationResult(this);
        }
    }

    /**
     * Success result when resonance is achieved.
     */
    public static class Success extends LaminarActivationResult {
        public Success(int categoryIndex, Pattern bottomUpActivation, Pattern topDownExpectation,
                      Map<String, Pattern> layerActivations, double resonanceScore, int cycles) {
            super(createSuccessBuilder(categoryIndex, bottomUpActivation, topDownExpectation,
                    layerActivations, resonanceScore, cycles));
        }

        private static Builder createSuccessBuilder(int categoryIndex, Pattern bottomUpActivation,
                Pattern topDownExpectation, Map<String, Pattern> layerActivations,
                double resonanceScore, int cycles) {
            var builder = builder()
                    .withCategoryIndex(categoryIndex)
                    .withBottomUpActivation(bottomUpActivation)
                    .withTopDownExpectation(topDownExpectation)
                    .withResonanceScore(resonanceScore)
                    .withResonant(true)
                    .withProcessingCycles(cycles);
            layerActivations.forEach(builder::withLayerActivation);
            return builder;
        }
    }

    /**
     * Failure result when no resonance achieved.
     */
    public static class Failure extends LaminarActivationResult {
        private final String reason;

        public Failure(String reason, int cycles) {
            super(builder()
                    .withCategoryIndex(-1)
                    .withResonant(false)
                    .withProcessingCycles(cycles));
            this.reason = reason;
        }

        public String getReason() {
            return reason;
        }
    }

    public String toString() {
        return String.format("LaminarActivationResult[category=%d, resonant=%b, score=%.3f, cycles=%d, layers=%d]",
                getCategoryIndex(), resonant, resonanceScore, processingCycles, layerActivations.size());
    }
}