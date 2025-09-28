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
package com.hellblazer.art.core.parameters;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Enumeration and configuration for learning modes in hybrid ART neural networks.
 *
 * LearningMode defines the various learning paradigms supported by hybrid ART systems,
 * from traditional unsupervised ART learning to modern hybrid approaches that combine
 * multiple learning strategies. Each mode has specific characteristics, requirements,
 * and optimization parameters.
 *
 * This sealed interface provides type safety and enables pattern matching for
 * learning mode-specific behavior while maintaining extensibility for future modes.
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public sealed interface LearningMode
    permits LearningMode.Unsupervised, LearningMode.Supervised, LearningMode.SemiSupervised,
            LearningMode.Reinforcement, LearningMode.Transfer, LearningMode.Continual,
            LearningMode.MetaLearning, LearningMode.SelfSupervised, LearningMode.Hybrid,
            LearningMode.Custom {

    /**
     * Get the unique identifier for this learning mode.
     *
     * @return learning mode identifier
     */
    String getId();

    /**
     * Get a human-readable name for this learning mode.
     *
     * @return learning mode name
     */
    String getName();

    /**
     * Get a description of this learning mode.
     *
     * @return learning mode description
     */
    String getDescription();

    /**
     * Get the learning paradigm category.
     *
     * @return learning paradigm
     */
    LearningParadigm getParadigm();

    /**
     * Check if this learning mode requires labeled data.
     *
     * @return true if labeled data is required
     */
    boolean requiresLabels();

    /**
     * Check if this learning mode supports online learning.
     *
     * @return true if online learning is supported
     */
    boolean supportsOnlineLearning();

    /**
     * Check if this learning mode supports batch learning.
     *
     * @return true if batch learning is supported
     */
    boolean supportsBatchLearning();

    /**
     * Get the typical use cases for this learning mode.
     *
     * @return set of use cases
     */
    Set<UseCase> getUseCases();

    /**
     * Get the advantages of this learning mode.
     *
     * @return list of advantages
     */
    List<String> getAdvantages();

    /**
     * Get the limitations of this learning mode.
     *
     * @return list of limitations
     */
    List<String> getLimitations();

    /**
     * Get recommended parameter settings for this learning mode.
     *
     * @return map of parameter names to recommended values
     */
    Map<String, Object> getRecommendedParameters();

    /**
     * Get performance metrics relevant to this learning mode.
     *
     * @return list of relevant metrics
     */
    List<String> getRelevantMetrics();

    /**
     * Check if this learning mode is compatible with another mode.
     * Useful for hybrid approaches that combine multiple modes.
     *
     * @param other the other learning mode
     * @return true if modes are compatible
     */
    default boolean isCompatibleWith(LearningMode other) {
        return this.getParadigm() == other.getParadigm() ||
               this instanceof Hybrid ||
               other instanceof Hybrid;
    }

    /**
     * Learning paradigm categories.
     */
    enum LearningParadigm {
        /** Learning without labeled examples */
        UNSUPERVISED,
        /** Learning with labeled examples */
        SUPERVISED,
        /** Learning with limited labeled examples */
        SEMI_SUPERVISED,
        /** Learning through interaction and rewards */
        REINFORCEMENT,
        /** Learning by transferring knowledge */
        TRANSFER,
        /** Learning continuously over time */
        CONTINUAL,
        /** Learning to learn efficiently */
        META_LEARNING,
        /** Learning from self-generated supervision */
        SELF_SUPERVISED,
        /** Combining multiple paradigms */
        HYBRID,
        /** Custom or domain-specific paradigm */
        CUSTOM
    }

    /**
     * Common use cases for learning modes.
     */
    enum UseCase {
        CLUSTERING, CLASSIFICATION, REGRESSION, PATTERN_RECOGNITION,
        ANOMALY_DETECTION, FEATURE_LEARNING, REPRESENTATION_LEARNING,
        SEQUENCE_MODELING, FORECASTING, RECOMMENDATION, OPTIMIZATION,
        CONTROL, GAME_PLAYING, ROBOTICS, NATURAL_LANGUAGE_PROCESSING,
        COMPUTER_VISION, SPEECH_RECOGNITION, DRUG_DISCOVERY, FINANCE
    }

    /**
     * Traditional unsupervised ART learning.
     * Discovers patterns and creates categories without labeled data.
     */
    record Unsupervised(double vigilance, int maxCategories, boolean complementCoding) implements LearningMode {
        public Unsupervised {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in range [0,1]");
            }
            if (maxCategories <= 0) {
                throw new IllegalArgumentException("Max categories must be positive");
            }
        }

        @Override
        public String getId() { return "unsupervised"; }

        @Override
        public String getName() { return "Unsupervised Learning"; }

        @Override
        public String getDescription() {
            return "Traditional ART unsupervised learning that discovers patterns and creates categories " +
                   "without labeled examples, using vigilance-based category formation.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.UNSUPERVISED; }

        @Override
        public boolean requiresLabels() { return false; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.CLUSTERING, UseCase.PATTERN_RECOGNITION, UseCase.ANOMALY_DETECTION,
                         UseCase.FEATURE_LEARNING, UseCase.REPRESENTATION_LEARNING);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "No labeled data required",
                "Fast online learning",
                "Stable category formation",
                "Handles arbitrary input distributions",
                "Adaptive vigilance control"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "No direct class prediction",
                "Category interpretation can be challenging",
                "Vigilance parameter tuning required",
                "Limited to similarity-based clustering"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "vigilance", 0.7,
                "learningRate", 0.1,
                "maxCategories", 1000,
                "complementCoding", true
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("categoryCount", "averageActivation", "categoryUtilization",
                          "vigilanceViolations", "compressionRatio");
        }
    }

    /**
     * Supervised learning with labeled examples.
     * Learns to map inputs to specific target outputs.
     */
    record Supervised(double vigilance, boolean useARTMAP, double mapfieldVigilance,
                     boolean enableErrorCorrection) implements LearningMode {
        public Supervised {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in range [0,1]");
            }
            if (mapfieldVigilance < 0.0 || mapfieldVigilance > 1.0) {
                throw new IllegalArgumentException("Mapfield vigilance must be in range [0,1]");
            }
        }

        @Override
        public String getId() { return "supervised"; }

        @Override
        public String getName() { return "Supervised Learning"; }

        @Override
        public String getDescription() {
            return "Supervised learning using ARTMAP or similar architectures to learn " +
                   "input-output mappings with labeled training data.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.SUPERVISED; }

        @Override
        public boolean requiresLabels() { return true; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.CLASSIFICATION, UseCase.REGRESSION, UseCase.PATTERN_RECOGNITION,
                         UseCase.COMPUTER_VISION, UseCase.NATURAL_LANGUAGE_PROCESSING);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Direct class prediction capability",
                "Fast online learning with labels",
                "Good generalization with minimal data",
                "Interpretable category-class mappings",
                "Handles incremental class addition"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Requires labeled training data",
                "May overfit with noisy labels",
                "Limited to classification/regression tasks",
                "Mapfield parameter tuning needed"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "vigilance", 0.8,
                "mapfieldVigilance", 0.9,
                "learningRate", 0.1,
                "useARTMAP", true,
                "enableErrorCorrection", true
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("accuracy", "precision", "recall", "f1Score", "confusionMatrix",
                          "categoryCount", "mapfieldActivations");
        }
    }

    /**
     * Semi-supervised learning with limited labeled data.
     * Combines labeled and unlabeled examples for improved learning.
     */
    record SemiSupervised(double vigilance, double labeledDataRatio, boolean usePseudoLabeling,
                         double confidenceThreshold) implements LearningMode {
        public SemiSupervised {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in range [0,1]");
            }
            if (labeledDataRatio < 0.0 || labeledDataRatio > 1.0) {
                throw new IllegalArgumentException("Labeled data ratio must be in range [0,1]");
            }
            if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
                throw new IllegalArgumentException("Confidence threshold must be in range [0,1]");
            }
        }

        @Override
        public String getId() { return "semi_supervised"; }

        @Override
        public String getName() { return "Semi-Supervised Learning"; }

        @Override
        public String getDescription() {
            return "Semi-supervised learning that leverages both labeled and unlabeled data " +
                   "to improve learning performance with limited supervision.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.SEMI_SUPERVISED; }

        @Override
        public boolean requiresLabels() { return true; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.CLASSIFICATION, UseCase.CLUSTERING, UseCase.FEATURE_LEARNING,
                         UseCase.NATURAL_LANGUAGE_PROCESSING, UseCase.COMPUTER_VISION);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Leverages abundant unlabeled data",
                "Better performance than pure supervised with limited labels",
                "Can discover hidden structure in data",
                "Cost-effective learning approach",
                "Robust to label noise"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Still requires some labeled data",
                "Complex parameter tuning",
                "May propagate label errors",
                "Performance depends on data distribution assumptions"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "vigilance", 0.75,
                "labeledDataRatio", 0.1,
                "confidenceThreshold", 0.8,
                "usePseudoLabeling", true,
                "maxIterations", 100
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("labeledAccuracy", "unlabeledAccuracy", "pseudoLabelAccuracy",
                          "confidenceCalibration", "labelPropagationRate");
        }
    }

    /**
     * Reinforcement learning through interaction and rewards.
     * Learns optimal actions through trial and error.
     */
    record Reinforcement(double explorationRate, double discountFactor, boolean useExperienceReplay,
                        int replayBufferSize, double targetUpdateRate) implements LearningMode {
        public Reinforcement {
            if (explorationRate < 0.0 || explorationRate > 1.0) {
                throw new IllegalArgumentException("Exploration rate must be in range [0,1]");
            }
            if (discountFactor < 0.0 || discountFactor > 1.0) {
                throw new IllegalArgumentException("Discount factor must be in range [0,1]");
            }
            if (replayBufferSize <= 0) {
                throw new IllegalArgumentException("Replay buffer size must be positive");
            }
            if (targetUpdateRate < 0.0 || targetUpdateRate > 1.0) {
                throw new IllegalArgumentException("Target update rate must be in range [0,1]");
            }
        }

        @Override
        public String getId() { return "reinforcement"; }

        @Override
        public String getName() { return "Reinforcement Learning"; }

        @Override
        public String getDescription() {
            return "Reinforcement learning that discovers optimal actions through interaction " +
                   "with the environment and reward signals.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.REINFORCEMENT; }

        @Override
        public boolean requiresLabels() { return false; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.CONTROL, UseCase.OPTIMIZATION, UseCase.GAME_PLAYING,
                         UseCase.ROBOTICS, UseCase.RECOMMENDATION);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Learns optimal policies without explicit supervision",
                "Adapts to changing environments",
                "Handles sequential decision making",
                "Maximizes long-term rewards",
                "Applicable to complex control problems"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Requires reward signal design",
                "Can be sample inefficient",
                "Exploration-exploitation tradeoff",
                "May converge to local optima",
                "Credit assignment problem"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "explorationRate", 0.1,
                "discountFactor", 0.95,
                "learningRate", 0.01,
                "replayBufferSize", 10000,
                "targetUpdateRate", 0.005
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("averageReward", "episodeLength", "explorationRate",
                          "valueError", "policyGradientNorm", "qValueDistribution");
        }
    }

    /**
     * Transfer learning from pre-trained models.
     * Leverages knowledge from related tasks or domains.
     */
    record Transfer(String sourceModel, boolean freezeFeatures, double transferRate,
                   List<String> transferableLayers, boolean useFineTuning) implements LearningMode {
        public Transfer {
            if (transferRate < 0.0 || transferRate > 1.0) {
                throw new IllegalArgumentException("Transfer rate must be in range [0,1]");
            }
            if (sourceModel == null || sourceModel.isBlank()) {
                throw new IllegalArgumentException("Source model must be specified");
            }
        }

        @Override
        public String getId() { return "transfer"; }

        @Override
        public String getName() { return "Transfer Learning"; }

        @Override
        public String getDescription() {
            return "Transfer learning that leverages knowledge from pre-trained models " +
                   "to accelerate learning on new but related tasks.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.TRANSFER; }

        @Override
        public boolean requiresLabels() { return true; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.COMPUTER_VISION, UseCase.NATURAL_LANGUAGE_PROCESSING,
                         UseCase.SPEECH_RECOGNITION, UseCase.CLASSIFICATION, UseCase.FEATURE_LEARNING);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Faster convergence on new tasks",
                "Requires less training data",
                "Leverages existing knowledge",
                "Better performance on small datasets",
                "Reduced computational requirements"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Requires suitable source model",
                "May suffer from negative transfer",
                "Domain mismatch can hurt performance",
                "Limited to similar task types",
                "Feature transferability not always clear"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "transferRate", 0.5,
                "learningRate", 0.001,
                "freezeFeatures", true,
                "useFineTuning", true,
                "warmupEpochs", 5
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("transferAccuracy", "convergenceSpeed", "featureReuse",
                          "negativeTransferDetection", "layerActivationSimilarity");
        }
    }

    /**
     * Continual learning over time with memory retention.
     * Learns new tasks while retaining knowledge of previous tasks.
     */
    record Continual(boolean useExperienceReplay, int memorySize, double forgettingRate,
                    String memoryManagementStrategy, boolean catastrophicForgettingPrevention) implements LearningMode {
        public Continual {
            if (forgettingRate < 0.0 || forgettingRate > 1.0) {
                throw new IllegalArgumentException("Forgetting rate must be in range [0,1]");
            }
            if (memorySize <= 0) {
                throw new IllegalArgumentException("Memory size must be positive");
            }
        }

        @Override
        public String getId() { return "continual"; }

        @Override
        public String getName() { return "Continual Learning"; }

        @Override
        public String getDescription() {
            return "Continual learning that accumulates knowledge over time while " +
                   "preventing catastrophic forgetting of previous tasks.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.CONTINUAL; }

        @Override
        public boolean requiresLabels() { return true; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return false; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.SEQUENCE_MODELING, UseCase.ROBOTICS, UseCase.RECOMMENDATION,
                         UseCase.NATURAL_LANGUAGE_PROCESSING, UseCase.COMPUTER_VISION);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Learns continuously without forgetting",
                "Adapts to changing environments",
                "Accumulates knowledge over time",
                "Handles task sequence learning",
                "Memory-efficient knowledge retention"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Complex memory management",
                "May still suffer from some forgetting",
                "Computational overhead for memory replay",
                "Task boundary detection challenges",
                "Scalability with number of tasks"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "memorySize", 5000,
                "forgettingRate", 0.01,
                "replayFrequency", 10,
                "memoryManagementStrategy", "reservoir_sampling",
                "catastrophicForgettingPrevention", true
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("backwardTransfer", "forwardTransfer", "forgettingMeasure",
                          "memoryEfficiency", "taskAccuracy", "interferenceLevel");
        }
    }

    /**
     * Meta-learning for rapid adaptation to new tasks.
     * Learns to learn efficiently across different tasks.
     */
    record MetaLearning(int supportSetSize, int querySetSize, int metaIterations,
                       double innerLearningRate, double outerLearningRate, String algorithm) implements LearningMode {
        public MetaLearning {
            if (supportSetSize <= 0) {
                throw new IllegalArgumentException("Support set size must be positive");
            }
            if (querySetSize <= 0) {
                throw new IllegalArgumentException("Query set size must be positive");
            }
            if (metaIterations <= 0) {
                throw new IllegalArgumentException("Meta iterations must be positive");
            }
            if (innerLearningRate <= 0.0) {
                throw new IllegalArgumentException("Inner learning rate must be positive");
            }
            if (outerLearningRate <= 0.0) {
                throw new IllegalArgumentException("Outer learning rate must be positive");
            }
        }

        @Override
        public String getId() { return "meta_learning"; }

        @Override
        public String getName() { return "Meta-Learning"; }

        @Override
        public String getDescription() {
            return "Meta-learning that learns to adapt quickly to new tasks with " +
                   "minimal training examples by learning across task distributions.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.META_LEARNING; }

        @Override
        public boolean requiresLabels() { return true; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.CLASSIFICATION, UseCase.REGRESSION, UseCase.OPTIMIZATION,
                         UseCase.DRUG_DISCOVERY, UseCase.ROBOTICS);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Rapid adaptation to new tasks",
                "Learns generalizable learning strategies",
                "Effective with limited data per task",
                "Transfers learning algorithms not just features",
                "Improves few-shot learning performance"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Requires distribution of related tasks",
                "Computationally expensive training",
                "Complex hyperparameter tuning",
                "May not generalize to very different domains",
                "Implementation complexity"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "supportSetSize", 5,
                "querySetSize", 15,
                "metaIterations", 1000,
                "innerLearningRate", 0.01,
                "outerLearningRate", 0.001,
                "algorithm", "MAML"
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("fewShotAccuracy", "adaptationSpeed", "metaGradientNorm",
                          "taskGeneralization", "innerLoopConvergence");
        }
    }

    /**
     * Self-supervised learning without manual labels.
     * Creates supervision signals from the data itself.
     */
    record SelfSupervised(String pretext, boolean useContrastiveLearning, double temperature,
                         int negativeReward, boolean useMasking) implements LearningMode {
        @Override
        public String getId() { return "self_supervised"; }

        @Override
        public String getName() { return "Self-Supervised Learning"; }

        @Override
        public String getDescription() {
            return "Self-supervised learning that creates supervision signals from " +
                   "the data itself without requiring manual annotations.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.SELF_SUPERVISED; }

        @Override
        public boolean requiresLabels() { return false; }

        @Override
        public boolean supportsOnlineLearning() { return true; }

        @Override
        public boolean supportsBatchLearning() { return true; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.REPRESENTATION_LEARNING, UseCase.FEATURE_LEARNING,
                         UseCase.COMPUTER_VISION, UseCase.NATURAL_LANGUAGE_PROCESSING,
                         UseCase.SPEECH_RECOGNITION);
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "No manual annotation required",
                "Leverages large amounts of unlabeled data",
                "Learns rich representations",
                "Good transferability to downstream tasks",
                "Cost-effective learning approach"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Pretext task design is crucial",
                "May learn irrelevant features",
                "Computationally intensive",
                "Requires careful hyperparameter tuning",
                "Quality depends on data diversity"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "temperature", 0.1,
                "negativeReward", 1024,
                "batchSize", 256,
                "useMasking", true,
                "maskingRatio", 0.15
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("contrastiveLoss", "representationQuality", "linearEvaluation",
                          "downstreamPerformance", "featureDiversity");
        }
    }

    /**
     * Hybrid learning combining multiple paradigms.
     * Integrates different learning approaches for enhanced performance.
     */
    record Hybrid(List<LearningMode> modes, Map<String, Double> weights, String combinationStrategy,
                 boolean dynamicWeighting, double adaptationRate) implements LearningMode {
        public Hybrid {
            if (modes == null || modes.isEmpty()) {
                throw new IllegalArgumentException("Hybrid mode must include at least one learning mode");
            }
            if (weights != null && !weights.keySet().equals(
                modes.stream().map(LearningMode::getId).collect(java.util.stream.Collectors.toSet()))) {
                throw new IllegalArgumentException("Weights must be provided for all learning modes");
            }
            if (adaptationRate < 0.0 || adaptationRate > 1.0) {
                throw new IllegalArgumentException("Adaptation rate must be in range [0,1]");
            }
        }

        @Override
        public String getId() { return "hybrid"; }

        @Override
        public String getName() { return "Hybrid Learning"; }

        @Override
        public String getDescription() {
            return "Hybrid learning that combines multiple learning paradigms " +
                   "to leverage the strengths of different approaches.";
        }

        @Override
        public LearningParadigm getParadigm() { return LearningParadigm.HYBRID; }

        @Override
        public boolean requiresLabels() {
            return modes.stream().anyMatch(LearningMode::requiresLabels);
        }

        @Override
        public boolean supportsOnlineLearning() {
            return modes.stream().allMatch(LearningMode::supportsOnlineLearning);
        }

        @Override
        public boolean supportsBatchLearning() {
            return modes.stream().allMatch(LearningMode::supportsBatchLearning);
        }

        @Override
        public Set<UseCase> getUseCases() {
            return modes.stream()
                       .flatMap(mode -> mode.getUseCases().stream())
                       .collect(java.util.stream.Collectors.toSet());
        }

        @Override
        public List<String> getAdvantages() {
            return List.of(
                "Combines strengths of multiple approaches",
                "More robust and versatile",
                "Can handle diverse data types",
                "Adaptive to changing requirements",
                "Better overall performance"
            );
        }

        @Override
        public List<String> getLimitations() {
            return List.of(
                "Increased complexity",
                "More hyperparameters to tune",
                "Higher computational requirements",
                "Potential conflicts between modes",
                "Difficult to interpret results"
            );
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return Map.of(
                "combinationStrategy", "weighted_average",
                "adaptationRate", 0.1,
                "dynamicWeighting", true,
                "convergenceThreshold", 0.01
            );
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("hybridPerformance", "modeContributions", "weightEvolution",
                          "convergenceSpeed", "robustness", "adaptability");
        }
    }

    /**
     * Custom learning mode for domain-specific requirements.
     * Allows users to define their own learning paradigms.
     */
    record Custom(String customId, String customName, String customDescription,
                 LearningParadigm paradigm, Map<String, Object> parameters,
                 boolean requiresLabels, boolean supportsOnline, boolean supportsBatch) implements LearningMode {
        public Custom {
            if (customId == null || customId.isBlank()) {
                throw new IllegalArgumentException("Custom ID cannot be null or blank");
            }
            if (customName == null || customName.isBlank()) {
                throw new IllegalArgumentException("Custom name cannot be null or blank");
            }
        }

        @Override
        public String getId() { return customId; }

        @Override
        public String getName() { return customName; }

        @Override
        public String getDescription() { return customDescription; }

        @Override
        public LearningParadigm getParadigm() { return paradigm; }

        @Override
        public boolean requiresLabels() { return requiresLabels; }

        @Override
        public boolean supportsOnlineLearning() { return supportsOnline; }

        @Override
        public boolean supportsBatchLearning() { return supportsBatch; }

        @Override
        public Set<UseCase> getUseCases() {
            return Set.of(UseCase.OPTIMIZATION); // Default for custom modes
        }

        @Override
        public List<String> getAdvantages() {
            return List.of("Customized for specific domain", "Flexible implementation");
        }

        @Override
        public List<String> getLimitations() {
            return List.of("Requires domain expertise", "May lack generalizability");
        }

        @Override
        public Map<String, Object> getRecommendedParameters() {
            return parameters != null ? Map.copyOf(parameters) : Map.of();
        }

        @Override
        public List<String> getRelevantMetrics() {
            return List.of("customMetric1", "customMetric2"); // Should be overridden
        }
    }

    /**
     * Factory methods for creating common learning modes.
     */
    static LearningMode unsupervised() {
        return new Unsupervised(0.7, 1000, true);
    }

    static LearningMode supervised() {
        return new Supervised(0.8, true, 0.9, true);
    }

    static LearningMode semiSupervised() {
        return new SemiSupervised(0.75, 0.1, true, 0.8);
    }

    static LearningMode reinforcement() {
        return new Reinforcement(0.1, 0.95, true, 10000, 0.005);
    }

    static LearningMode transfer(String sourceModel) {
        return new Transfer(sourceModel, true, 0.5, List.of(), true);
    }

    static LearningMode continual() {
        return new Continual(true, 5000, 0.01, "reservoir_sampling", true);
    }

    static LearningMode metaLearning() {
        return new MetaLearning(5, 15, 1000, 0.01, 0.001, "MAML");
    }

    static LearningMode selfSupervised(String pretext) {
        return new SelfSupervised(pretext, true, 0.1, 1024, true);
    }

    static LearningMode hybrid(LearningMode... modes) {
        return new Hybrid(List.of(modes), null, "weighted_average", true, 0.1);
    }

    static LearningMode custom(String id, String name, String description) {
        return new Custom(id, name, description, LearningParadigm.CUSTOM,
                         Map.of(), false, true, true);
    }
}