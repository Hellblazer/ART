/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.temporal.parameters;

/**
 * Combined parameters for temporal ART algorithms.
 * Includes parameters for both working memory and masking field components.
 *
 * @param workingMemoryParameters parameters for item-and-order working memory
 * @param maskingParameters parameters for masking field network
 * @param vigilance vigilance threshold for category matching (0.0 to 1.0)
 * @param learningRate learning rate for weight updates (0.0 to 1.0)
 * @param maxCategories maximum number of categories that can be created
 * @param enableLearning whether learning is enabled
 * @param enableChunking whether automatic sequence chunking is enabled
 *
 * @author Hal Hildebrand
 */
public record TemporalParameters(
    WorkingMemoryParameters workingMemoryParameters,
    MaskingParameters maskingParameters,
    float vigilance,
    float learningRate,
    int maxCategories,
    boolean enableLearning,
    boolean enableChunking
) {

    /**
     * Default temporal parameters for standard sequence processing.
     */
    public static final TemporalParameters DEFAULT = new TemporalParameters(
        WorkingMemoryParameters.DEFAULT,
        MaskingParameters.DEFAULT,
        0.9f,    // High vigilance for good category separation
        0.1f,    // Moderate learning rate
        100,     // Reasonable maximum categories
        true,    // Learning enabled
        true     // Chunking enabled
    );

    /**
     * Parameters optimized for short sequence processing.
     */
    public static final TemporalParameters SHORT_SEQUENCES = new TemporalParameters(
        WorkingMemoryParameters.SHORT_SEQUENCES,
        MaskingParameters.SHORT_SEQUENCES,
        0.85f,   // Slightly lower vigilance for short sequences
        0.15f,   // Faster learning for short sequences
        50,      // Fewer categories expected
        true,
        true
    );

    /**
     * Parameters optimized for long sequence processing.
     */
    public static final TemporalParameters LONG_SEQUENCES = new TemporalParameters(
        WorkingMemoryParameters.LONG_SEQUENCES,
        MaskingParameters.LONG_SEQUENCES,
        0.95f,   // Higher vigilance for better separation
        0.05f,   // Slower learning for stability
        200,     // More categories expected
        true,
        true
    );

    /**
     * Parameters for real-time processing with minimal delay.
     */
    public static final TemporalParameters REAL_TIME = new TemporalParameters(
        WorkingMemoryParameters.REAL_TIME,
        MaskingParameters.REAL_TIME,
        0.88f,   // Balanced vigilance
        0.1f,    // Standard learning rate
        100,
        true,
        false    // Disable chunking for speed
    );

    /**
     * Create a builder for temporal parameters.
     *
     * @return new parameter builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Validate parameter values.
     *
     * @throws IllegalArgumentException if parameters are invalid
     */
    public void validate() {
        if (vigilance < 0 || vigilance > 1) {
            throw new IllegalArgumentException("Vigilance must be in [0,1]: " + vigilance);
        }
        if (learningRate < 0 || learningRate > 1) {
            throw new IllegalArgumentException("Learning rate must be in [0,1]: " + learningRate);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive: " + maxCategories);
        }

        // Validate sub-parameters
        workingMemoryParameters.validate();
        maskingParameters.validate();
    }

    /**
     * Create a copy with learning disabled (for testing/prediction).
     *
     * @return new parameters with learning disabled
     */
    public TemporalParameters withLearningDisabled() {
        return new TemporalParameters(
            workingMemoryParameters, maskingParameters,
            vigilance, learningRate, maxCategories,
            false, enableChunking
        );
    }

    /**
     * Create a copy with modified vigilance.
     *
     * @param newVigilance the new vigilance value
     * @return new parameters with updated vigilance
     */
    public TemporalParameters withVigilance(float newVigilance) {
        return new TemporalParameters(
            workingMemoryParameters, maskingParameters,
            newVigilance, learningRate, maxCategories,
            enableLearning, enableChunking
        );
    }

    /**
     * Builder class for temporal parameters.
     */
    public static class Builder {
        private WorkingMemoryParameters workingMemoryParameters = DEFAULT.workingMemoryParameters;
        private MaskingParameters maskingParameters = DEFAULT.maskingParameters;
        private float vigilance = DEFAULT.vigilance;
        private float learningRate = DEFAULT.learningRate;
        private int maxCategories = DEFAULT.maxCategories;
        private boolean enableLearning = DEFAULT.enableLearning;
        private boolean enableChunking = DEFAULT.enableChunking;

        public Builder workingMemoryParameters(WorkingMemoryParameters params) {
            this.workingMemoryParameters = params;
            return this;
        }

        public Builder maskingParameters(MaskingParameters params) {
            this.maskingParameters = params;
            return this;
        }

        public Builder vigilance(float vigilance) {
            this.vigilance = vigilance;
            return this;
        }

        public Builder learningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder maxCategories(int maxCategories) {
            this.maxCategories = maxCategories;
            return this;
        }

        public Builder enableLearning(boolean enableLearning) {
            this.enableLearning = enableLearning;
            return this;
        }

        public Builder enableChunking(boolean enableChunking) {
            this.enableChunking = enableChunking;
            return this;
        }

        /**
         * Build and validate the temporal parameters.
         *
         * @return validated parameters
         */
        public TemporalParameters build() {
            var params = new TemporalParameters(
                workingMemoryParameters, maskingParameters,
                vigilance, learningRate, maxCategories,
                enableLearning, enableChunking
            );
            params.validate();
            return params;
        }
    }
}