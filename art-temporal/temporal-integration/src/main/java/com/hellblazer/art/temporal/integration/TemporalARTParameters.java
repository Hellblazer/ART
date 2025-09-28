package com.hellblazer.art.temporal.integration;

import com.hellblazer.art.temporal.core.Parameters;
import com.hellblazer.art.temporal.memory.WorkingMemoryParameters;
import com.hellblazer.art.temporal.masking.MaskingFieldParameters;

import java.util.Map;
import java.util.HashMap;
import java.util.Optional;

/**
 * Parameters for the integrated temporal ART system.
 */
public class TemporalARTParameters implements Parameters {

    private final WorkingMemoryParameters workingMemoryParameters;
    private final MaskingFieldParameters maskingFieldParameters;
    private final double vigilance;
    private final double learningRate;
    private final double timeStep;
    private final int maxCategories;
    private final double matchThreshold;
    private final boolean fastLearning;

    private TemporalARTParameters(Builder builder) {
        this.workingMemoryParameters = builder.workingMemoryParameters;
        this.maskingFieldParameters = builder.maskingFieldParameters;
        this.vigilance = builder.vigilance;
        this.learningRate = builder.learningRate;
        this.timeStep = builder.timeStep;
        this.maxCategories = builder.maxCategories;
        this.matchThreshold = builder.matchThreshold;
        this.fastLearning = builder.fastLearning;

        validate();
    }

    /**
     * Create default parameters for standard temporal ART.
     */
    public static TemporalARTParameters defaults() {
        return builder().build();
    }

    /**
     * Create parameters optimized for speech processing.
     */
    public static TemporalARTParameters speechDefaults() {
        return builder()
            .workingMemoryParameters(WorkingMemoryParameters.paperDefaults())
            .maskingFieldParameters(MaskingFieldParameters.phoneNumberDefaults())
            .vigilance(0.7)
            .learningRate(0.5)
            .timeStep(0.01)  // 10ms time step for speech
            .maxCategories(200)
            .build();
    }

    /**
     * Create parameters optimized for list learning.
     */
    public static TemporalARTParameters listLearningDefaults() {
        return builder()
            .workingMemoryParameters(WorkingMemoryParameters.cowansCapacity())
            .maskingFieldParameters(MaskingFieldParameters.listLearningDefaults())
            .vigilance(0.8)
            .learningRate(0.3)
            .timeStep(0.05)  // 50ms time step for list items
            .maxCategories(100)
            .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private WorkingMemoryParameters workingMemoryParameters = WorkingMemoryParameters.paperDefaults();
        private MaskingFieldParameters maskingFieldParameters = MaskingFieldParameters.listLearningDefaults();
        private double vigilance = 0.75;
        private double learningRate = 0.1;
        private double timeStep = 0.01;  // 10ms default
        private int maxCategories = 100;
        private double matchThreshold = 0.5;
        private boolean fastLearning = false;

        public Builder workingMemoryParameters(WorkingMemoryParameters params) {
            this.workingMemoryParameters = params;
            return this;
        }

        public Builder maskingFieldParameters(MaskingFieldParameters params) {
            this.maskingFieldParameters = params;
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

        public Builder timeStep(double timeStep) {
            this.timeStep = timeStep;
            return this;
        }

        public Builder maxCategories(int maxCategories) {
            this.maxCategories = maxCategories;
            return this;
        }

        public Builder matchThreshold(double matchThreshold) {
            this.matchThreshold = matchThreshold;
            return this;
        }

        public Builder fastLearning(boolean fastLearning) {
            this.fastLearning = fastLearning;
            return this;
        }

        public TemporalARTParameters build() {
            return new TemporalARTParameters(this);
        }
    }


    // Getters
    public WorkingMemoryParameters getWorkingMemoryParameters() {
        return workingMemoryParameters;
    }

    public MaskingFieldParameters getMaskingFieldParameters() {
        return maskingFieldParameters;
    }

    public double getVigilance() {
        return vigilance;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getTimeStep() {
        return timeStep;
    }

    public int getMaxCategories() {
        return maxCategories;
    }

    public double getMatchThreshold() {
        return matchThreshold;
    }

    public boolean isFastLearning() {
        return fastLearning;
    }

    /**
     * Get parallelism level for vectorized operations.
     */
    public int getParallelismLevel() {
        return Runtime.getRuntime().availableProcessors();
    }

    /**
     * Get integration time step for multi-scale dynamics.
     */
    public double getIntegrationTimeStep() {
        return timeStep;
    }

    /**
     * Get weight increase factor for category strengthening.
     */
    public double getWeightIncrease() {
        return 0.1; // Default weight increase
    }

    /**
     * Get initial weight for new categories.
     */
    public double getInitialWeight() {
        return 0.8; // Default initial weight
    }

    /**
     * Get memory parameters for compatibility.
     */
    public WorkingMemoryParameters getMemoryParameters() {
        return workingMemoryParameters;
    }

    /**
     * Get masking parameters for compatibility.
     */
    public MaskingFieldParameters getMaskingParameters() {
        return maskingFieldParameters;
    }

    /**
     * Get input dimension for vectorized operations.
     */
    public int getInputDimension() {
        return workingMemoryParameters.getItemDimension();
    }

    @Override
    public void validate() {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be between 0 and 1");
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be between 0 and 1");
        }
        if (timeStep <= 0.0) {
            throw new IllegalArgumentException("Time step must be positive");
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive");
        }
        if (matchThreshold < 0.0 || matchThreshold > 1.0) {
            throw new IllegalArgumentException("Match threshold must be between 0 and 1");
        }
        workingMemoryParameters.validate();
        maskingFieldParameters.validate();
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();

        // Add direct parameters
        params.put("vigilance", vigilance);
        params.put("learningRate", learningRate);
        params.put("timeStep", timeStep);
        params.put("maxCategories", (double) maxCategories);
        params.put("matchThreshold", matchThreshold);
        params.put("fastLearning", fastLearning ? 1.0 : 0.0);

        // Add working memory parameters with prefix
        var wmParams = workingMemoryParameters.getAllParameters();
        for (var entry : wmParams.entrySet()) {
            params.put("wm." + entry.getKey(), entry.getValue());
        }

        // Add masking field parameters with prefix
        var mfParams = maskingFieldParameters.getAllParameters();
        for (var entry : mfParams.entrySet()) {
            params.put("mf." + entry.getKey(), entry.getValue());
        }

        return params;
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return switch (name) {
            case "vigilance" -> Optional.of(vigilance);
            case "learningRate" -> Optional.of(learningRate);
            case "timeStep" -> Optional.of(timeStep);
            case "maxCategories" -> Optional.of((double) maxCategories);
            case "matchThreshold" -> Optional.of(matchThreshold);
            case "fastLearning" -> Optional.of(fastLearning ? 1.0 : 0.0);
            default -> {
                // Check prefixed parameters
                if (name.startsWith("wm.")) {
                    yield workingMemoryParameters.getParameter(name.substring(3));
                } else if (name.startsWith("mf.")) {
                    yield maskingFieldParameters.getParameter(name.substring(3));
                } else {
                    yield Optional.empty();
                }
            }
        };
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = new Builder()
            .workingMemoryParameters(workingMemoryParameters)
            .maskingFieldParameters(maskingFieldParameters)
            .vigilance(vigilance)
            .learningRate(learningRate)
            .timeStep(timeStep)
            .maxCategories(maxCategories)
            .matchThreshold(matchThreshold)
            .fastLearning(fastLearning);

        switch (name) {
            case "vigilance" -> builder.vigilance(value);
            case "learningRate" -> builder.learningRate(value);
            case "timeStep" -> builder.timeStep(value);
            case "maxCategories" -> builder.maxCategories((int) value);
            case "matchThreshold" -> builder.matchThreshold(value);
            case "fastLearning" -> builder.fastLearning(value > 0.5);
            default -> {
                if (name.startsWith("wm.")) {
                    builder.workingMemoryParameters(
                        (WorkingMemoryParameters) workingMemoryParameters.withParameter(
                            name.substring(3), value
                        )
                    );
                } else if (name.startsWith("mf.")) {
                    builder.maskingFieldParameters(
                        (MaskingFieldParameters) maskingFieldParameters.withParameter(
                            name.substring(3), value
                        )
                    );
                } else {
                    throw new IllegalArgumentException("Unknown parameter: " + name);
                }
            }
        }

        return builder.build();
    }
}