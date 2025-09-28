package com.hellblazer.art.temporal.masking;

import com.hellblazer.art.temporal.core.Parameters;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for masking field dynamics.
 * Controls multi-scale temporal integration and chunking behavior.
 *
 * Time scale: 50-500ms (intermediate between working memory and long-term memory)
 */
public class MaskingFieldParameters implements Parameters {
    // Capacity parameters
    private final int maxItemNodes;
    private final int maxChunks;
    private final int minChunkSize;
    private final int maxChunkSize;

    // Temporal parameters (in seconds)
    private final double integrationTimeStep;     // 50ms default
    private final double minChunkInterval;        // Minimum time between chunks
    private final double maxTemporalGap;          // Maximum gap for coherent sequence

    // Competition parameters
    private final double spatialScale;            // Spatial extent of competition
    private final double excitationRange;         // Range of excitatory connections
    private final double inhibitionRange;         // Range of inhibitory connections
    private final double competitionStrength;     // Strength of competition
    private final double winnerThreshold;         // Threshold for winner selection

    // Dynamics parameters
    private final double itemDecayRate;           // Decay rate for item activations
    private final double chunkDecayRate;          // Decay rate for chunk activations
    private final double maxActivation;           // Maximum activation level
    private final double initialActivation;       // Initial activation for new items
    private final double activationBoost;         // Boost for matching items
    private final double activeChunkBoost;        // Boost for active chunk

    // Learning parameters
    private final double learningRate;            // Rate of strengthening
    private final double matchingThreshold;       // Threshold for pattern matching
    private final double selfExcitation;          // Self-excitation strength

    // Control flags
    private final boolean normalizationEnabled;   // Enable activation normalization
    private final boolean resetAfterChunk;        // Reset items after chunk formation
    private final double resetDecayFactor;        // Decay factor for reset

    private MaskingFieldParameters(Builder builder) {
        this.maxItemNodes = builder.maxItemNodes;
        this.maxChunks = builder.maxChunks;
        this.minChunkSize = builder.minChunkSize;
        this.maxChunkSize = builder.maxChunkSize;
        this.integrationTimeStep = builder.integrationTimeStep;
        this.minChunkInterval = builder.minChunkInterval;
        this.maxTemporalGap = builder.maxTemporalGap;
        this.spatialScale = builder.spatialScale;
        this.excitationRange = builder.excitationRange;
        this.inhibitionRange = builder.inhibitionRange;
        this.competitionStrength = builder.competitionStrength;
        this.winnerThreshold = builder.winnerThreshold;
        this.itemDecayRate = builder.itemDecayRate;
        this.chunkDecayRate = builder.chunkDecayRate;
        this.maxActivation = builder.maxActivation;
        this.initialActivation = builder.initialActivation;
        this.activationBoost = builder.activationBoost;
        this.activeChunkBoost = builder.activeChunkBoost;
        this.learningRate = builder.learningRate;
        this.matchingThreshold = builder.matchingThreshold;
        this.selfExcitation = builder.selfExcitation;
        this.normalizationEnabled = builder.normalizationEnabled;
        this.resetAfterChunk = builder.resetAfterChunk;
        this.resetDecayFactor = builder.resetDecayFactor;
        validate();
    }

    @Override
    public void validate() {
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

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("maxItemNodes", (double) maxItemNodes);
        params.put("maxChunks", (double) maxChunks);
        params.put("minChunkSize", (double) minChunkSize);
        params.put("maxChunkSize", (double) maxChunkSize);
        params.put("integrationTimeStep", integrationTimeStep);
        params.put("minChunkInterval", minChunkInterval);
        params.put("maxTemporalGap", maxTemporalGap);
        params.put("spatialScale", spatialScale);
        params.put("excitationRange", excitationRange);
        params.put("inhibitionRange", inhibitionRange);
        params.put("competitionStrength", competitionStrength);
        params.put("winnerThreshold", winnerThreshold);
        params.put("itemDecayRate", itemDecayRate);
        params.put("chunkDecayRate", chunkDecayRate);
        params.put("maxActivation", maxActivation);
        params.put("initialActivation", initialActivation);
        params.put("activationBoost", activationBoost);
        params.put("activeChunkBoost", activeChunkBoost);
        params.put("learningRate", learningRate);
        params.put("matchingThreshold", matchingThreshold);
        params.put("selfExcitation", selfExcitation);
        params.put("normalizationEnabled", normalizationEnabled ? 1.0 : 0.0);
        params.put("resetAfterChunk", resetAfterChunk ? 1.0 : 0.0);
        params.put("resetDecayFactor", resetDecayFactor);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = toBuilder();

        switch (name) {
            case "maxItemNodes" -> builder.maxItemNodes((int) value);
            case "maxChunks" -> builder.maxChunks((int) value);
            case "minChunkSize" -> builder.minChunkSize((int) value);
            case "maxChunkSize" -> builder.maxChunkSize((int) value);
            case "integrationTimeStep" -> builder.integrationTimeStep(value);
            case "minChunkInterval" -> builder.minChunkInterval(value);
            case "maxTemporalGap" -> builder.maxTemporalGap(value);
            case "spatialScale" -> builder.spatialScale(value);
            case "excitationRange" -> builder.excitationRange(value);
            case "inhibitionRange" -> builder.inhibitionRange(value);
            case "competitionStrength" -> builder.competitionStrength(value);
            case "winnerThreshold" -> builder.winnerThreshold(value);
            case "itemDecayRate" -> builder.itemDecayRate(value);
            case "chunkDecayRate" -> builder.chunkDecayRate(value);
            case "maxActivation" -> builder.maxActivation(value);
            case "initialActivation" -> builder.initialActivation(value);
            case "activationBoost" -> builder.activationBoost(value);
            case "activeChunkBoost" -> builder.activeChunkBoost(value);
            case "learningRate" -> builder.learningRate(value);
            case "matchingThreshold" -> builder.matchingThreshold(value);
            case "selfExcitation" -> builder.selfExcitation(value);
            case "normalizationEnabled" -> builder.normalizationEnabled(value > 0.5);
            case "resetAfterChunk" -> builder.resetAfterChunk(value > 0.5);
            case "resetDecayFactor" -> builder.resetDecayFactor(value);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public int getMaxItemNodes() { return maxItemNodes; }
    public int getMaxChunks() { return maxChunks; }
    public int getMinChunkSize() { return minChunkSize; }
    public int getMaxChunkSize() { return maxChunkSize; }
    public double getIntegrationTimeStep() { return integrationTimeStep; }
    public double getMinChunkInterval() { return minChunkInterval; }
    public double getMaxTemporalGap() { return maxTemporalGap; }
    public double getSpatialScale() { return spatialScale; }
    public double getExcitationRange() { return excitationRange; }
    public double getInhibitionRange() { return inhibitionRange; }
    public double getCompetitionStrength() { return competitionStrength; }
    public double getWinnerThreshold() { return winnerThreshold; }
    public double getItemDecayRate() { return itemDecayRate; }
    public double getChunkDecayRate() { return chunkDecayRate; }
    public double getMaxActivation() { return maxActivation; }
    public double getInitialActivation() { return initialActivation; }
    public double getActivationBoost() { return activationBoost; }
    public double getActiveChunkBoost() { return activeChunkBoost; }
    public double getLearningRate() { return learningRate; }
    public double getMatchingThreshold() { return matchingThreshold; }
    public double getSelfExcitation() { return selfExcitation; }
    public boolean isNormalizationEnabled() { return normalizationEnabled; }
    public boolean isResetAfterChunk() { return resetAfterChunk; }
    public double getResetDecayFactor() { return resetDecayFactor; }

    private Builder toBuilder() {
        return new Builder()
            .maxItemNodes(maxItemNodes)
            .maxChunks(maxChunks)
            .minChunkSize(minChunkSize)
            .maxChunkSize(maxChunkSize)
            .integrationTimeStep(integrationTimeStep)
            .minChunkInterval(minChunkInterval)
            .maxTemporalGap(maxTemporalGap)
            .spatialScale(spatialScale)
            .excitationRange(excitationRange)
            .inhibitionRange(inhibitionRange)
            .competitionStrength(competitionStrength)
            .winnerThreshold(winnerThreshold)
            .itemDecayRate(itemDecayRate)
            .chunkDecayRate(chunkDecayRate)
            .maxActivation(maxActivation)
            .initialActivation(initialActivation)
            .activationBoost(activationBoost)
            .activeChunkBoost(activeChunkBoost)
            .learningRate(learningRate)
            .matchingThreshold(matchingThreshold)
            .selfExcitation(selfExcitation)
            .normalizationEnabled(normalizationEnabled)
            .resetAfterChunk(resetAfterChunk)
            .resetDecayFactor(resetDecayFactor);
    }

    public static Builder builder() {
        return new Builder();
    }

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
            return new MaskingFieldParameters(this);
        }
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
}