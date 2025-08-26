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
package com.hellblazer.art.core;

import java.util.List;

/**
 * Sealed interface representing the result of DeepARTMAP operations.
 * 
 * This interface provides a type-safe way to handle the various outcomes
 * of DeepARTMAP training and prediction operations. Results can be either
 * successful or represent various failure conditions.
 * 
 * @author Hal Hildebrand
 */
public sealed interface DeepARTMAPResult 
    permits DeepARTMAPResult.Success, 
            DeepARTMAPResult.ValidationFailure,
            DeepARTMAPResult.TrainingFailure {

    /**
     * Successful DeepARTMAP operation result.
     * 
     * @param layerResults     the results from each hierarchical layer
     * @param deepLabels      the concatenated labels from all layers
     * @param supervisedMode  whether the operation was in supervised mode
     * @param categoryCount   the total number of categories created
     */
    record Success(
        List<Object> layerResults,
        int[][] deepLabels,
        boolean supervisedMode,
        int categoryCount
    ) implements DeepARTMAPResult {
        
        public Success {
            if (layerResults == null) {
                throw new IllegalArgumentException("layerResults cannot be null");
            }
            if (deepLabels == null) {
                throw new IllegalArgumentException("deepLabels cannot be null");
            }
            if (categoryCount < 0) {
                throw new IllegalArgumentException("categoryCount cannot be negative");
            }
        }
        
        /**
         * Get the number of hierarchical layers that were trained.
         * 
         * @return the number of layers
         */
        public int getLayerCount() {
            return layerResults.size();
        }
        
        /**
         * Get the number of samples processed.
         * 
         * @return the number of samples
         */
        public int getSampleCount() {
            return deepLabels.length;
        }
    }

    /**
     * Validation failure result indicating invalid input data.
     * 
     * @param reason      the specific validation failure reason
     * @param parameter   the parameter that caused the validation failure
     * @param value       the invalid value that was provided
     */
    record ValidationFailure(
        String reason,
        String parameter,
        Object value
    ) implements DeepARTMAPResult {
        
        public ValidationFailure {
            if (reason == null || reason.isBlank()) {
                throw new IllegalArgumentException("reason cannot be null or blank");
            }
            if (parameter == null || parameter.isBlank()) {
                throw new IllegalArgumentException("parameter cannot be null or blank");
            }
        }
        
        /**
         * Create a validation failure for inconsistent sample counts.
         * 
         * @param expectedCount the expected sample count
         * @param actualCount   the actual sample count
         * @return validation failure result
         */
        public static ValidationFailure inconsistentSampleCount(int expectedCount, int actualCount) {
            return new ValidationFailure(
                "Inconsistent sample number in input matrices",
                "sample_count",
                "expected=" + expectedCount + ", actual=" + actualCount
            );
        }
        
        /**
         * Create a validation failure for wrong number of input channels.
         * 
         * @param expectedChannels the expected number of channels
         * @param actualChannels   the actual number of channels
         * @return validation failure result
         */
        public static ValidationFailure wrongChannelCount(int expectedChannels, int actualChannels) {
            return new ValidationFailure(
                "Must provide " + expectedChannels + " input matrices for " + expectedChannels + " ART modules",
                "channel_count",
                "expected=" + expectedChannels + ", actual=" + actualChannels
            );
        }
        
        /**
         * Create a validation failure for empty input data.
         * 
         * @return validation failure result
         */
        public static ValidationFailure emptyData() {
            return new ValidationFailure(
                "Cannot fit with empty data",
                "data",
                "empty"
            );
        }
        
        /**
         * Create a validation failure for null channel data.
         * 
         * @param channelIndex the index of the null channel
         * @return validation failure result
         */
        public static ValidationFailure nullChannelData(int channelIndex) {
            return new ValidationFailure(
                "channel data cannot be null",
                "channel_" + channelIndex,
                null
            );
        }
    }

    /**
     * Training failure result indicating an error during the learning process.
     * 
     * @param stage       the training stage where the failure occurred
     * @param layer       the layer index where the failure occurred (-1 if not layer-specific)
     * @param cause       the underlying exception that caused the failure
     * @param message     human-readable description of the failure
     */
    record TrainingFailure(
        TrainingStage stage,
        int layer,
        Throwable cause,
        String message
    ) implements DeepARTMAPResult {
        
        public TrainingFailure {
            if (stage == null) {
                throw new IllegalArgumentException("stage cannot be null");
            }
            if (message == null || message.isBlank()) {
                throw new IllegalArgumentException("message cannot be null or blank");
            }
        }
        
        /**
         * Create a training failure for layer initialization.
         * 
         * @param layer the layer index
         * @param cause the underlying cause
         * @return training failure result
         */
        public static TrainingFailure layerInitializationFailed(int layer, Throwable cause) {
            return new TrainingFailure(
                TrainingStage.LAYER_INITIALIZATION,
                layer,
                cause,
                "Failed to initialize layer " + layer + ": " + cause.getMessage()
            );
        }
        
        /**
         * Create a training failure for label propagation.
         * 
         * @param fromLayer the source layer
         * @param toLayer   the target layer
         * @param cause     the underlying cause
         * @return training failure result
         */
        public static TrainingFailure labelPropagationFailed(int fromLayer, int toLayer, Throwable cause) {
            return new TrainingFailure(
                TrainingStage.LABEL_PROPAGATION,
                toLayer,
                cause,
                "Failed to propagate labels from layer " + fromLayer + " to layer " + toLayer + ": " + cause.getMessage()
            );
        }
        
        /**
         * Create a training failure for convergence issues.
         * 
         * @param layer the layer where convergence failed
         * @return training failure result
         */
        public static TrainingFailure convergenceFailed(int layer) {
            return new TrainingFailure(
                TrainingStage.CONVERGENCE,
                layer,
                null,
                "Training failed to converge at layer " + layer
            );
        }
    }

    /**
     * Enumeration of training stages where failures can occur.
     */
    enum TrainingStage {
        DATA_PREPARATION,
        LAYER_INITIALIZATION,
        SUPERVISED_TRAINING,
        UNSUPERVISED_TRAINING,
        LABEL_PROPAGATION,
        DEEP_MAPPING,
        CONVERGENCE,
        FINALIZATION
    }
}