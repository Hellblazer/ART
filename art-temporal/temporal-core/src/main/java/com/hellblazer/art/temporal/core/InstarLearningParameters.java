package com.hellblazer.art.temporal.core;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for competitive instar learning.
 * Based on Equation 12 from Kazerounian & Grossberg (2014):
 * dW_ij/dt = L * Y_j * (X_i * Z_i - W_ij)
 */
public class InstarLearningParameters implements Parameters {
    private final double learningRate;     // L: learning rate (0.1)
    private final int numCategories;       // Number of categories/neurons
    private final int inputDimension;      // Input vector dimension
    private final double vigilance;        // Vigilance parameter for matching
    private final double resetThreshold;   // Threshold for category reset
    private final boolean enableNormalization; // Maintain weight normalization
    private final boolean enableCompetition;   // Enable competitive learning

    private InstarLearningParameters(Builder builder) {
        this.learningRate = builder.learningRate;
        this.numCategories = builder.numCategories;
        this.inputDimension = builder.inputDimension;
        this.vigilance = builder.vigilance;
        this.resetThreshold = builder.resetThreshold;
        this.enableNormalization = builder.enableNormalization;
        this.enableCompetition = builder.enableCompetition;
        validate();
    }

    @Override
    public void validate() {
        if (learningRate <= 0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in (0, 1], got: " + learningRate);
        }
        if (numCategories < 1 || numCategories > 10000) {
            throw new IllegalArgumentException("Number of categories must be in [1, 10000], got: " + numCategories);
        }
        if (inputDimension < 1 || inputDimension > 100000) {
            throw new IllegalArgumentException("Input dimension must be in [1, 100000], got: " + inputDimension);
        }
        if (vigilance < 0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        if (resetThreshold < 0 || resetThreshold > 1.0) {
            throw new IllegalArgumentException("Reset threshold must be in [0, 1], got: " + resetThreshold);
        }

        // Check learning stability condition
        if (learningRate > 0.5 && enableCompetition) {
            System.err.println("Warning: High learning rate with competition may cause instability");
        }
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("learningRate", learningRate);
        params.put("numCategories", (double) numCategories);
        params.put("inputDimension", (double) inputDimension);
        params.put("vigilance", vigilance);
        params.put("resetThreshold", resetThreshold);
        params.put("enableNormalization", enableNormalization ? 1.0 : 0.0);
        params.put("enableCompetition", enableCompetition ? 1.0 : 0.0);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = new Builder()
            .learningRate(learningRate)
            .numCategories(numCategories)
            .inputDimension(inputDimension)
            .vigilance(vigilance)
            .resetThreshold(resetThreshold)
            .enableNormalization(enableNormalization)
            .enableCompetition(enableCompetition);

        switch (name) {
            case "learningRate" -> builder.learningRate(value);
            case "numCategories" -> builder.numCategories((int) value);
            case "inputDimension" -> builder.inputDimension((int) value);
            case "vigilance" -> builder.vigilance(value);
            case "resetThreshold" -> builder.resetThreshold(value);
            case "enableNormalization" -> builder.enableNormalization(value > 0.5);
            case "enableCompetition" -> builder.enableCompetition(value > 0.5);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public double getLearningRate() { return learningRate; }
    public int getNumCategories() { return numCategories; }
    public int getInputDimension() { return inputDimension; }
    public double getVigilance() { return vigilance; }
    public double getResetThreshold() { return resetThreshold; }
    public boolean isNormalizationEnabled() { return enableNormalization; }
    public boolean isCompetitionEnabled() { return enableCompetition; }

    /**
     * Check if a match value passes the vigilance test.
     */
    public boolean passesVigilance(double match, double inputNorm) {
        if (inputNorm == 0) {
            return false;
        }
        return (match / inputNorm) >= vigilance;
    }

    /**
     * Compute effective learning rate based on competition.
     */
    public double getEffectiveLearningRate(boolean isWinner) {
        if (!enableCompetition) {
            return learningRate;
        }
        return isWinner ? learningRate : 0.0;
    }

    /**
     * Compute time scale of weight adaptation.
     * Weights operate on 1000-10000 ms scale.
     */
    public double getTimeScale() {
        // Inverse of learning rate gives characteristic time
        return 1.0 / learningRate; // ~10 time units for L=0.1
    }

    /**
     * Estimate memory requirements in bytes.
     */
    public long estimateMemoryRequirements() {
        // Each weight is a double (8 bytes)
        // Plus overhead for arrays and objects
        long weightMemory = (long) numCategories * inputDimension * 8;
        long overhead = numCategories * 32 + inputDimension * 16;
        return weightMemory + overhead;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double learningRate = 0.1;
        private int numCategories = 100;
        private int inputDimension = 100;
        private double vigilance = 0.7;
        private double resetThreshold = 0.1;
        private boolean enableNormalization = true;
        private boolean enableCompetition = true;

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder numCategories(int numCategories) {
            this.numCategories = numCategories;
            return this;
        }

        public Builder inputDimension(int inputDimension) {
            this.inputDimension = inputDimension;
            return this;
        }

        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }

        public Builder resetThreshold(double threshold) {
            this.resetThreshold = threshold;
            return this;
        }

        public Builder enableNormalization(boolean enable) {
            this.enableNormalization = enable;
            return this;
        }

        public Builder enableCompetition(boolean enable) {
            this.enableCompetition = enable;
            return this;
        }

        public InstarLearningParameters build() {
            return new InstarLearningParameters(this);
        }
    }

    /**
     * Create default parameters from paper.
     */
    public static InstarLearningParameters paperDefaults() {
        return builder()
            .learningRate(0.1)        // From paper
            .numCategories(100)       // Sufficient for chunking
            .inputDimension(100)      // Matches masking field size
            .vigilance(0.7)           // Moderate selectivity
            .resetThreshold(0.1)      // Low activity triggers reset
            .enableNormalization(true) // Maintain Σw_ij = 1
            .enableCompetition(true)   // Winner-take-all learning
            .build();
    }

    /**
     * Create parameters for fast learning experiments.
     */
    public static InstarLearningParameters fastLearning() {
        return builder()
            .learningRate(0.5)        // 5x faster
            .numCategories(50)        // Fewer categories
            .inputDimension(100)
            .vigilance(0.5)           // Less selective
            .resetThreshold(0.1)
            .enableNormalization(true)
            .enableCompetition(true)
            .build();
    }

    /**
     * Create parameters for high-capacity learning.
     */
    public static InstarLearningParameters highCapacity() {
        return builder()
            .learningRate(0.05)       // Slower but more stable
            .numCategories(500)       // Many categories
            .inputDimension(100)
            .vigilance(0.9)           // Very selective
            .resetThreshold(0.05)     // Sensitive reset
            .enableNormalization(true)
            .enableCompetition(true)
            .build();
    }
}