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
package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.State;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for hybrid predictors that combine ART neural networks with other ML approaches.
 *
 * HybridPredictor integrates traditional ART pattern matching with modern machine
 * learning techniques such as deep neural networks, gradient boosting, or ensemble
 * methods. This enables leveraging the strengths of both ART (fast learning,
 * incremental adaptation) and other ML models (complex pattern recognition,
 * non-linear relationships).
 *
 * Key capabilities:
 * - Multi-model ensemble prediction
 * - ART-guided feature selection for other ML models
 * - Confidence-weighted prediction combination
 * - Online adaptation and model selection
 * - Interpretable hybrid decisions
 * - Performance monitoring and model switching
 *
 * @param <S> the type of states for prediction
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface HybridPredictor<S extends State<?>> extends AutoCloseable {

    /**
     * Hybrid prediction strategies for combining multiple models.
     */
    enum HybridStrategy {
        /** Use ART as primary, others as fallback */
        ART_PRIMARY,
        /** Use other ML as primary, ART as fallback */
        ML_PRIMARY,
        /** Weighted combination of all models */
        WEIGHTED_ENSEMBLE,
        /** Dynamic selection based on confidence */
        CONFIDENCE_BASED,
        /** Majority voting among models */
        MAJORITY_VOTE,
        /** Adaptive switching based on performance */
        ADAPTIVE_SWITCHING,
        /** Hierarchical delegation (ART filters, ML refines) */
        HIERARCHICAL
    }

    /**
     * Types of ML models that can be integrated with ART.
     */
    enum ModelType {
        /** Traditional ART neural network */
        ART,
        /** Deep neural network */
        DEEP_NN,
        /** Gradient boosting (XGBoost, LightGBM) */
        GRADIENT_BOOSTING,
        /** Random Forest */
        RANDOM_FOREST,
        /** Support Vector Machine */
        SVM,
        /** K-Nearest Neighbors */
        KNN,
        /** Linear/Logistic Regression */
        LINEAR,
        /** Gaussian Process */
        GAUSSIAN_PROCESS,
        /** Custom model implementation */
        CUSTOM
    }

    /**
     * Make a prediction combining multiple models.
     *
     * @param state the state to predict for
     * @return hybrid prediction result
     * @throws IllegalArgumentException if state is null or invalid
     */
    HybridPrediction<S> predict(S state);

    /**
     * Make predictions for multiple states efficiently.
     *
     * @param states the states to predict for
     * @return list of hybrid predictions in the same order
     */
    default List<HybridPrediction<S>> predictBatch(List<S> states) {
        return states.stream()
                    .map(this::predict)
                    .toList();
    }

    /**
     * Add a new model to the hybrid ensemble.
     *
     * @param modelId unique identifier for the model
     * @param model the model implementation
     * @param weight initial weight for ensemble combination
     * @param modelType the type of ML model
     */
    void addModel(String modelId, Object model, double weight, ModelType modelType);

    /**
     * Remove a model from the ensemble.
     *
     * @param modelId the model identifier to remove
     * @return true if model was found and removed
     */
    boolean removeModel(String modelId);

    /**
     * Update the weight of a model in the ensemble.
     *
     * @param modelId the model identifier
     * @param newWeight the new weight value
     * @return true if model was found and updated
     */
    boolean updateModelWeight(String modelId, double newWeight);

    /**
     * Get the current hybrid strategy.
     *
     * @return the hybrid strategy being used
     */
    HybridStrategy getStrategy();

    /**
     * Set the hybrid strategy for combining model predictions.
     *
     * @param strategy the new strategy to use
     */
    void setStrategy(HybridStrategy strategy);

    /**
     * Get information about all models in the ensemble.
     *
     * @return map of model ID to model information
     */
    Map<String, ModelInfo> getModelInfo();

    /**
     * Get the performance statistics for each model.
     *
     * @return map of model ID to performance metrics
     */
    Map<String, ModelPerformance> getModelPerformance();

    /**
     * Train/update models with new data.
     *
     * @param trainingData the data to train on
     * @param labels optional labels for supervised learning
     */
    void train(List<S> trainingData, Optional<List<Object>> labels);

    /**
     * Get the confidence threshold for predictions.
     * Predictions below this threshold may trigger special handling.
     *
     * @return confidence threshold [0.0, 1.0]
     */
    double getConfidenceThreshold();

    /**
     * Set the confidence threshold for predictions.
     *
     * @param threshold new confidence threshold [0.0, 1.0]
     */
    void setConfidenceThreshold(double threshold);

    /**
     * Check if the hybrid predictor is ready to make predictions.
     *
     * @return true if at least one model is trained and ready
     */
    default boolean isReady() {
        return !getModelInfo().isEmpty() &&
               getModelInfo().values().stream().anyMatch(ModelInfo::isReady);
    }

    /**
     * Get explanation for the last prediction made.
     * Useful for interpretability and debugging.
     *
     * @return optional explanation of prediction reasoning
     */
    Optional<PredictionExplanation> getLastPredictionExplanation();

    /**
     * Enable or disable explanation generation.
     * Explanations may have performance overhead.
     *
     * @param enabled whether to generate explanations
     */
    void setExplanationEnabled(boolean enabled);

    /**
     * Validate the hybrid predictor configuration.
     *
     * @return list of validation issues (empty if valid)
     */
    default List<String> validate() {
        if (getModelInfo().isEmpty()) {
            return List.of("No models configured in hybrid predictor");
        }

        var issues = new java.util.ArrayList<String>();
        double totalWeight = getModelInfo().values().stream()
                                          .mapToDouble(ModelInfo::getWeight)
                                          .sum();
        if (Math.abs(totalWeight - 1.0) > 0.01) {
            issues.add("Model weights do not sum to 1.0: " + totalWeight);
        }

        return issues;
    }

    /**
     * Reset all models to their initial state.
     */
    void reset();

    /**
     * Get memory usage of the hybrid predictor.
     *
     * @return optional memory usage in bytes
     */
    default Optional<Long> getMemoryUsage() {
        return Optional.empty();
    }

    /**
     * Release resources used by the hybrid predictor.
     */
    @Override
    void close();

    /**
     * Represents a hybrid prediction result combining multiple models.
     *
     * @param <S> the state type
     */
    interface HybridPrediction<S extends State<?>> {
        /** Get the final combined prediction */
        ActivationResult getFinalPrediction();

        /** Get individual predictions from each model */
        Map<String, ActivationResult> getModelPredictions();

        /** Get the confidence in the final prediction */
        double getConfidence();

        /** Get the strategy used for combination */
        HybridStrategy getStrategy();

        /** Get the state that was predicted for */
        S getInputState();

        /** Check if this is a high-confidence prediction */
        default boolean isHighConfidence() {
            return getConfidence() > 0.8;
        }

        /** Check if models agreed on the prediction */
        default boolean hasConsensus() {
            var predictions = getModelPredictions().values();
            if (predictions.size() < 2) return true;

            var firstResult = predictions.iterator().next();
            if (!(firstResult instanceof ActivationResult.Success first)) return false;

            return predictions.stream()
                             .allMatch(p -> p instanceof ActivationResult.Success s &&
                                          s.categoryIndex() == first.categoryIndex());
        }

        /** Get the disagreement level between models */
        default double getDisagreement() {
            var predictions = getModelPredictions().values();
            if (predictions.size() < 2) return 0.0;

            // Calculate variance in activation values
            var activations = predictions.stream()
                                        .filter(p -> p instanceof ActivationResult.Success)
                                        .map(p -> ((ActivationResult.Success) p).activationValue())
                                        .toList();

            if (activations.size() < 2) return 0.0;

            double mean = activations.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double variance = activations.stream()
                                        .mapToDouble(a -> Math.pow(a - mean, 2))
                                        .average().orElse(0.0);

            return Math.sqrt(variance);
        }
    }

    /**
     * Information about a model in the hybrid ensemble.
     */
    interface ModelInfo {
        /** Get the model identifier */
        String getId();

        /** Get the model type */
        ModelType getType();

        /** Get the current weight in ensemble */
        double getWeight();

        /** Check if model is ready for predictions */
        boolean isReady();

        /** Get model-specific configuration */
        Map<String, Object> getConfiguration();

        /** Get when the model was last updated */
        long getLastUpdateTime();

        /** Get number of training samples seen */
        long getTrainingSampleCount();
    }

    /**
     * Performance metrics for a model in the ensemble.
     */
    interface ModelPerformance {
        /** Get prediction accuracy */
        double getAccuracy();

        /** Get average prediction time in milliseconds */
        double getAveragePredictionTime();

        /** Get confidence calibration error */
        double getCalibrationError();

        /** Get number of predictions made */
        long getPredictionCount();

        /** Get memory usage in bytes */
        long getMemoryUsage();

        /** Get recent performance trend */
        PerformanceTrend getTrend();

        /** Performance trend indicators */
        enum PerformanceTrend {
            IMPROVING, STABLE, DEGRADING, UNKNOWN
        }
    }

    /**
     * Explanation for a hybrid prediction decision.
     */
    interface PredictionExplanation {
        /** Get the explanation strategy used */
        HybridStrategy getStrategy();

        /** Get individual model contributions */
        Map<String, ModelContribution> getModelContributions();

        /** Get the reasoning for the final decision */
        String getDecisionReasoning();

        /** Get feature importance if available */
        Optional<Map<String, Double>> getFeatureImportance();

        /** Check if explanation is available */
        default boolean isAvailable() {
            return !getModelContributions().isEmpty();
        }

        /** Model contribution to the final prediction */
        record ModelContribution(String modelId, double weight, double confidence,
                               ActivationResult prediction, String reasoning) {
        }
    }
}