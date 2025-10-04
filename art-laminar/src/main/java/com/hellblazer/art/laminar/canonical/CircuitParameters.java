package com.hellblazer.art.laminar.canonical;

/**
 * Parameters for complete laminar circuit with all 6 layers and resonance dynamics.
 *
 * Integrates parameters for:
 * - All 6 laminar layers (Layer 1, 2/3, 4, 5, 6)
 * - PredictionGenerator (top-down expectations)
 * - PredictionErrorProcessor (vigilance testing)
 * - Category search and resonance dynamics
 *
 * Based on canonical neocortical circuit architecture and ART theory.
 *
 * @param inputSize dimensionality of input patterns
 * @param categorySize maximum number of categories
 * @param vigilance match threshold [0,1] for resonance
 * @param learningRate template update rate [0,1]
 * @param maxSearchIterations maximum category search attempts
 * @param timeStep integration time step for dynamics
 * @param topDownGain strength of top-down modulation [0,1]
 * @param expectationThreshold minimum category activation [0,1]
 * @param resetThreshold error magnitude for reset
 *
 * @see "A Canonical Laminar Neocortical Circuit..." Raizada & Grossberg (2003)
 * @see "Adaptive Resonance Theory" Carpenter & Grossberg (1987)
 * @author Hal Hildebrand
 */
public record CircuitParameters(
    int inputSize,
    int categorySize,
    double vigilance,
    double learningRate,
    int maxSearchIterations,
    double timeStep,
    double topDownGain,
    double expectationThreshold,
    double resetThreshold
) {
    /**
     * Compact validation constructor.
     */
    public CircuitParameters {
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive");
        }
        if (categorySize < 1) {
            throw new IllegalArgumentException("categorySize must be positive");
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("vigilance must be in [0,1]");
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("learningRate must be in [0,1]");
        }
        if (maxSearchIterations < 1) {
            throw new IllegalArgumentException("maxSearchIterations must be positive");
        }
        if (timeStep <= 0.0) {
            throw new IllegalArgumentException("timeStep must be positive");
        }
        if (topDownGain < 0.0 || topDownGain > 1.0) {
            throw new IllegalArgumentException("topDownGain must be in [0,1]");
        }
        if (expectationThreshold < 0.0 || expectationThreshold > 1.0) {
            throw new IllegalArgumentException("expectationThreshold must be in [0,1]");
        }
        if (resetThreshold < 0.0) {
            throw new IllegalArgumentException("resetThreshold must be non-negative");
        }
    }

    /**
     * Create builder for fluent parameter construction.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for CircuitParameters with sensible defaults.
     */
    public static class Builder {
        private int inputSize = 10;
        private int categorySize = 10;
        private double vigilance = 0.7;  // Moderate specificity
        private double learningRate = 0.1;  // Gradual learning
        private int maxSearchIterations = 10;
        private double timeStep = 0.01;  // 10ms
        private double topDownGain = 0.5;  // 50% modulation
        private double expectationThreshold = 0.1;  // Ignore weak categories
        private double resetThreshold = 0.3;  // Moderate error tolerance

        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        public Builder categorySize(int categorySize) {
            this.categorySize = categorySize;
            return this;
        }

        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder maxSearchIterations(int maxSearchIterations) {
            this.maxSearchIterations = maxSearchIterations;
            return this;
        }

        public Builder timeStep(double timeStep) {
            this.timeStep = timeStep;
            return this;
        }

        public Builder topDownGain(double topDownGain) {
            this.topDownGain = topDownGain;
            return this;
        }

        public Builder expectationThreshold(double expectationThreshold) {
            this.expectationThreshold = expectationThreshold;
            return this;
        }

        public Builder resetThreshold(double resetThreshold) {
            this.resetThreshold = resetThreshold;
            return this;
        }

        public CircuitParameters build() {
            return new CircuitParameters(
                inputSize,
                categorySize,
                vigilance,
                learningRate,
                maxSearchIterations,
                timeStep,
                topDownGain,
                expectationThreshold,
                resetThreshold
            );
        }
    }
}
