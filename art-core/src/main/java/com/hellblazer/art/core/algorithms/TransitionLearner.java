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

import com.hellblazer.art.core.State;
import com.hellblazer.art.core.Transition;
import com.hellblazer.art.core.TransitionMemory;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for learning from state transitions in hybrid ART neural networks.
 *
 * TransitionLearner captures the dynamics of state transitions, enabling
 * predictive modeling, temporal pattern recognition, and sequential learning.
 * It integrates with ART's pattern matching while adding temporal awareness
 * and transition-based learning capabilities.
 *
 * Key capabilities:
 * - Learning transition patterns and probabilities
 * - Temporal sequence modeling
 * - Transition prediction and forecasting
 * - Reinforcement learning from transition rewards
 * - Memory consolidation and replay
 * - Causal relationship inference
 *
 * @param <S> the type of states involved in transitions
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface TransitionLearner<S extends State<?>> extends AutoCloseable {

    /**
     * Learning modes for transition learning.
     */
    enum LearningMode {
        /** Learn from individual transitions as they occur */
        ONLINE,
        /** Learn from batches of transitions */
        BATCH,
        /** Learn from stored transition sequences */
        OFFLINE,
        /** Combine online and offline learning */
        HYBRID,
        /** Learn through experience replay */
        REPLAY
    }

    /**
     * Types of transition models that can be learned.
     */
    enum ModelType {
        /** Simple frequency-based transition probabilities */
        FREQUENCY,
        /** Markov chain transition model */
        MARKOV,
        /** Hidden Markov Model */
        HMM,
        /** Recurrent neural network model */
        RNN,
        /** Transformer-based sequence model */
        TRANSFORMER,
        /** Hybrid ART-based temporal model */
        HYBRID_ART
    }

    /**
     * Learn from a single transition.
     *
     * @param transition the transition to learn from
     * @throws IllegalArgumentException if transition is null or invalid
     */
    void learn(Transition<S, ?> transition);

    /**
     * Learn from a batch of transitions.
     *
     * @param transitions the transitions to learn from
     * @throws IllegalArgumentException if transitions is null or contains invalid transitions
     */
    default void learnBatch(List<Transition<S, ?>> transitions) {
        if (transitions == null) {
            throw new IllegalArgumentException("Transition list cannot be null");
        }
        transitions.forEach(this::learn);
    }

    /**
     * Learn from a sequence of transitions preserving temporal order.
     *
     * @param sequence the ordered sequence of transitions
     * @param preserveOrder whether to enforce temporal ordering constraints
     */
    void learnSequence(List<Transition<S, ?>> sequence, boolean preserveOrder);

    /**
     * Predict the most likely next state given a current state.
     *
     * @param currentState the current state
     * @return optional predicted next state
     */
    Optional<S> predictNextState(S currentState);

    /**
     * Predict multiple possible next states with their probabilities.
     *
     * @param currentState the current state
     * @param maxPredictions maximum number of predictions to return
     * @return list of state predictions with probabilities, ordered by likelihood
     */
    List<StatePrediction<S>> predictNextStates(S currentState, int maxPredictions);

    /**
     * Predict a sequence of future states.
     *
     * @param currentState the current state
     * @param sequenceLength number of future states to predict
     * @return predicted sequence of states
     */
    List<S> predictSequence(S currentState, int sequenceLength);

    /**
     * Get the probability of a specific transition occurring.
     *
     * @param sourceState the source state
     * @param targetState the target state
     * @return transition probability [0.0, 1.0]
     */
    double getTransitionProbability(S sourceState, S targetState);

    /**
     * Get all learned transitions from a given state.
     *
     * @param sourceState the source state
     * @return list of possible transitions with their probabilities
     */
    List<TransitionProbability<S>> getTransitionsFrom(S sourceState);

    /**
     * Get the learning mode used by this learner.
     *
     * @return the learning mode, never null
     */
    LearningMode getLearningMode();

    /**
     * Get the type of transition model being learned.
     *
     * @return the model type, never null
     */
    ModelType getModelType();

    /**
     * Get the number of transitions learned so far.
     *
     * @return total number of transitions processed
     */
    long getTransitionCount();

    /**
     * Get the number of unique states encountered.
     *
     * @return number of distinct states
     */
    int getStateCount();

    /**
     * Check if the learner has sufficient data to make reliable predictions.
     *
     * @return true if enough transitions have been learned
     */
    default boolean hasTrainedModel() {
        return getTransitionCount() > 0 && getStateCount() > 1;
    }

    /**
     * Get confidence in predictions from this learner.
     * Higher values indicate more reliable predictions.
     *
     * @return confidence level [0.0, 1.0]
     */
    default double getModelConfidence() {
        return hasTrainedModel() ? Math.min(getTransitionCount() / 100.0, 1.0) : 0.0;
    }

    /**
     * Update the learner with experience replay from memory.
     *
     * @param memory the transition memory to replay from
     * @param replaySize number of transitions to replay
     */
    default void performExperienceReplay(TransitionMemory<S> memory, int replaySize) {
        var transitions = memory.sample(replaySize, TransitionMemory.RetrievalStrategy.RANDOM);
        learnBatch(transitions);
    }

    /**
     * Consolidate learned transitions for long-term memory.
     * This may involve compressing frequent transitions or removing outliers.
     *
     * @param consolidationThreshold transitions older than this age are consolidated
     * @return number of transitions consolidated
     */
    default int consolidate(Duration consolidationThreshold) {
        return 0; // Default: no consolidation
    }

    /**
     * Forget old or infrequent transitions to maintain model efficiency.
     *
     * @param forgettingRate rate at which old transitions are forgotten [0.0, 1.0]
     * @return number of transitions forgotten
     */
    default int forget(double forgettingRate) {
        return 0; // Default: no forgetting
    }

    /**
     * Get learning statistics for monitoring performance.
     *
     * @return learning statistics
     */
    LearningStatistics getStatistics();

    /**
     * Validate the learned model for consistency and correctness.
     *
     * @return list of validation issues (empty if valid)
     */
    default List<String> validateModel() {
        return List.of(); // Default: assume valid
    }

    /**
     * Reset the learner to its initial state, clearing all learned transitions.
     */
    void reset();

    /**
     * Export the learned model for persistence or analysis.
     *
     * @param format the export format (implementation-specific)
     * @return optional exported model data
     */
    default Optional<Object> exportModel(String format) {
        return Optional.empty(); // Default: not supported
    }

    /**
     * Import a previously learned model.
     *
     * @param modelData the model data to import
     * @param format the data format
     * @return true if import was successful
     */
    default boolean importModel(Object modelData, String format) {
        return false; // Default: not supported
    }

    /**
     * Get memory usage of the learned model.
     *
     * @return optional memory usage in bytes
     */
    default Optional<Long> getMemoryUsage() {
        return Optional.empty();
    }

    /**
     * Release resources used by the learner.
     */
    @Override
    void close();

    /**
     * Represents a state prediction with associated probability.
     *
     * @param <S> the state type
     */
    record StatePrediction<S extends State<?>>(S state, double probability, double confidence) {
        public StatePrediction {
            if (state == null) {
                throw new IllegalArgumentException("State cannot be null");
            }
            if (probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException("Probability must be in range [0.0, 1.0]");
            }
            if (confidence < 0.0 || confidence > 1.0) {
                throw new IllegalArgumentException("Confidence must be in range [0.0, 1.0]");
            }
        }

        /**
         * Check if this prediction is considered reliable.
         */
        public boolean isReliable() {
            return confidence > 0.5 && probability > 0.1;
        }

        /**
         * Get a score combining probability and confidence.
         */
        public double getScore() {
            return probability * confidence;
        }
    }

    /**
     * Represents a transition with its learned probability.
     *
     * @param <S> the state type
     */
    record TransitionProbability<S extends State<?>>(S targetState, double probability,
                                                    int frequency, long lastSeen) {
        public TransitionProbability {
            if (targetState == null) {
                throw new IllegalArgumentException("Target state cannot be null");
            }
            if (probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException("Probability must be in range [0.0, 1.0]");
            }
            if (frequency < 0) {
                throw new IllegalArgumentException("Frequency cannot be negative");
            }
        }

        /**
         * Check if this transition is frequently observed.
         */
        public boolean isFrequent() {
            return frequency > 5 && probability > 0.1;
        }

        /**
         * Check if this transition was observed recently.
         */
        public boolean isRecent(long currentTime, Duration threshold) {
            return currentTime - lastSeen < threshold.toMillis();
        }
    }

    /**
     * Learning statistics for monitoring transition learner performance.
     */
    interface LearningStatistics {
        /** Get total number of transitions processed */
        long getTotalTransitions();

        /** Get number of unique source states */
        int getUniqueSourceStates();

        /** Get number of unique target states */
        int getUniqueTargetStates();

        /** Get average transition probability */
        double getAverageTransitionProbability();

        /** Get model entropy (measure of uncertainty) */
        double getModelEntropy();

        /** Get prediction accuracy on recent transitions */
        double getPredictionAccuracy();

        /** Get learning rate (transitions per second) */
        double getLearningRate();

        /** Get memory usage statistics */
        Map<String, Long> getMemoryUsage();

        /** Get consolidation statistics */
        Map<String, Long> getConsolidationStats();

        /** Get forgetting statistics */
        Map<String, Long> getForgettingStats();

        /** Check if learning is converging */
        default boolean isConverging() {
            return getPredictionAccuracy() > 0.8 && getModelEntropy() < 2.0;
        }

        /** Get learning efficiency (accuracy / memory usage) */
        default double getLearningEfficiency() {
            var memUsage = getMemoryUsage().values().stream().mapToLong(Long::longValue).sum();
            return memUsage > 0 ? getPredictionAccuracy() / memUsage * 1000000 : 0.0;
        }
    }
}