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
package com.hellblazer.art.core.utils;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for validating probability distributions and stochastic properties
 * in hybrid ART neural networks.
 *
 * ProbabilityValidator ensures that probability values, distributions, and
 * stochastic processes maintain mathematical consistency and validity. This
 * is crucial for hybrid systems that combine deterministic ART algorithms
 * with probabilistic machine learning approaches.
 *
 * Key validation capabilities:
 * - Probability value range checking [0,1]
 * - Distribution normalization validation (sum to 1.0)
 * - Independence and conditional probability testing
 * - Bayesian consistency verification
 * - Entropy and information-theoretic measures
 * - Statistical significance testing
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface ProbabilityValidator {

    /**
     * Validation severity levels.
     */
    enum Severity {
        /** Informational message, no action needed */
        INFO,
        /** Warning that should be addressed */
        WARNING,
        /** Error that must be fixed */
        ERROR,
        /** Critical error that breaks system assumptions */
        CRITICAL
    }

    /**
     * Types of probability validations.
     */
    enum ValidationType {
        /** Basic range and bounds checking */
        BASIC,
        /** Distribution normalization */
        NORMALIZATION,
        /** Independence assumptions */
        INDEPENDENCE,
        /** Conditional probability consistency */
        CONDITIONAL,
        /** Bayesian inference validity */
        BAYESIAN,
        /** Information-theoretic properties */
        INFORMATION_THEORY,
        /** Statistical significance */
        STATISTICAL
    }

    /**
     * Validate a single probability value.
     *
     * @param probability the probability value to validate
     * @param context optional context description for error reporting
     * @return validation result
     */
    ValidationResult validateProbability(double probability, String context);

    /**
     * Validate a probability distribution (array of probabilities that should sum to 1.0).
     *
     * @param distribution the probability distribution
     * @param tolerance acceptable tolerance for sum deviation from 1.0
     * @param context optional context description
     * @return validation result
     */
    ValidationResult validateDistribution(double[] distribution, double tolerance, String context);

    /**
     * Validate a probability distribution with default tolerance.
     *
     * @param distribution the probability distribution
     * @param context optional context description
     * @return validation result
     */
    default ValidationResult validateDistribution(double[] distribution, String context) {
        return validateDistribution(distribution, 1e-9, context);
    }

    /**
     * Validate conditional probability consistency: P(A|B) * P(B) = P(A,B).
     *
     * @param pAGivenB probability of A given B
     * @param pB probability of B
     * @param pAAndB joint probability of A and B
     * @param tolerance acceptable tolerance for validation
     * @param context optional context description
     * @return validation result
     */
    ValidationResult validateConditionalProbability(double pAGivenB, double pB, double pAAndB,
                                                   double tolerance, String context);

    /**
     * Validate independence assumption: P(A,B) = P(A) * P(B).
     *
     * @param pA probability of event A
     * @param pB probability of event B
     * @param pAAndB joint probability of A and B
     * @param tolerance acceptable tolerance for validation
     * @param context optional context description
     * @return validation result
     */
    ValidationResult validateIndependence(double pA, double pB, double pAAndB,
                                         double tolerance, String context);

    /**
     * Validate Bayes' theorem consistency: P(A|B) = P(B|A) * P(A) / P(B).
     *
     * @param pAGivenB posterior probability P(A|B)
     * @param pBGivenA likelihood P(B|A)
     * @param pA prior probability P(A)
     * @param pB evidence probability P(B)
     * @param tolerance acceptable tolerance for validation
     * @param context optional context description
     * @return validation result
     */
    ValidationResult validateBayesTheorem(double pAGivenB, double pBGivenA, double pA, double pB,
                                         double tolerance, String context);

    /**
     * Validate probability transition matrix (rows sum to 1.0).
     *
     * @param transitionMatrix the transition probability matrix
     * @param tolerance acceptable tolerance for row sum validation
     * @param context optional context description
     * @return validation result
     */
    ValidationResult validateTransitionMatrix(double[][] transitionMatrix, double tolerance, String context);

    /**
     * Calculate and validate entropy of a probability distribution.
     *
     * @param distribution the probability distribution
     * @param expectedEntropy optional expected entropy value for comparison
     * @param tolerance acceptable tolerance for entropy validation
     * @param context optional context description
     * @return validation result including calculated entropy
     */
    EntropyValidationResult validateEntropy(double[] distribution, Optional<Double> expectedEntropy,
                                          double tolerance, String context);

    /**
     * Validate Kullback-Leibler divergence between two distributions.
     *
     * @param p the first probability distribution
     * @param q the second probability distribution
     * @param maxDivergence maximum acceptable KL divergence
     * @param context optional context description
     * @return validation result including calculated KL divergence
     */
    DivergenceValidationResult validateKLDivergence(double[] p, double[] q, double maxDivergence, String context);

    /**
     * Validate that probabilities are statistically significant.
     *
     * @param observedProbability the observed probability
     * @param sampleSize the sample size used to estimate the probability
     * @param confidenceLevel desired confidence level (e.g., 0.95 for 95%)
     * @param context optional context description
     * @return validation result including confidence interval
     */
    StatisticalValidationResult validateStatisticalSignificance(double observedProbability, long sampleSize,
                                                               double confidenceLevel, String context);

    /**
     * Perform comprehensive validation of a probabilistic model.
     *
     * @param model the probabilistic model to validate
     * @param validationTypes types of validations to perform
     * @return comprehensive validation report
     */
    ValidationReport validateModel(ProbabilisticModel model, List<ValidationType> validationTypes);

    /**
     * Get the validation tolerance settings.
     *
     * @return current tolerance settings for different validation types
     */
    Map<ValidationType, Double> getToleranceSettings();

    /**
     * Set validation tolerance for a specific validation type.
     *
     * @param validationType the type of validation
     * @param tolerance the tolerance value
     */
    void setTolerance(ValidationType validationType, double tolerance);

    /**
     * Check if strict validation mode is enabled.
     * In strict mode, warnings are treated as errors.
     *
     * @return true if strict mode is enabled
     */
    boolean isStrictMode();

    /**
     * Enable or disable strict validation mode.
     *
     * @param strict whether to enable strict mode
     */
    void setStrictMode(boolean strict);

    /**
     * Basic validation result interface.
     */
    interface ValidationResult {
        /** Check if validation passed */
        boolean isValid();

        /** Get validation severity */
        Severity getSeverity();

        /** Get validation message */
        String getMessage();

        /** Get validation context */
        Optional<String> getContext();

        /** Get validation type */
        ValidationType getType();

        /** Get any computed values during validation */
        Map<String, Object> getMetrics();

        /** Create a successful validation result */
        static ValidationResult success(ValidationType type, String context) {
            return new ValidationResult() {
                @Override
                public boolean isValid() { return true; }
                @Override
                public Severity getSeverity() { return Severity.INFO; }
                @Override
                public String getMessage() { return "Validation passed"; }
                @Override
                public Optional<String> getContext() { return Optional.ofNullable(context); }
                @Override
                public ValidationType getType() { return type; }
                @Override
                public Map<String, Object> getMetrics() { return Map.of(); }
            };
        }

        /** Create a failed validation result */
        static ValidationResult failure(ValidationType type, Severity severity, String message, String context) {
            return new ValidationResult() {
                @Override
                public boolean isValid() { return false; }
                @Override
                public Severity getSeverity() { return severity; }
                @Override
                public String getMessage() { return message; }
                @Override
                public Optional<String> getContext() { return Optional.ofNullable(context); }
                @Override
                public ValidationType getType() { return type; }
                @Override
                public Map<String, Object> getMetrics() { return Map.of(); }
            };
        }
    }

    /**
     * Entropy validation result with calculated entropy value.
     */
    interface EntropyValidationResult extends ValidationResult {
        /** Get the calculated entropy */
        double getCalculatedEntropy();

        /** Get the expected entropy if provided */
        Optional<Double> getExpectedEntropy();

        /** Get the entropy difference from expected */
        default Optional<Double> getEntropyDifference() {
            return getExpectedEntropy().map(expected -> Math.abs(getCalculatedEntropy() - expected));
        }
    }

    /**
     * Divergence validation result with calculated divergence value.
     */
    interface DivergenceValidationResult extends ValidationResult {
        /** Get the calculated KL divergence */
        double getCalculatedDivergence();

        /** Get the maximum acceptable divergence */
        double getMaxDivergence();

        /** Check if divergence is within acceptable limits */
        default boolean isDivergenceAcceptable() {
            return getCalculatedDivergence() <= getMaxDivergence();
        }
    }

    /**
     * Statistical validation result with confidence intervals.
     */
    interface StatisticalValidationResult extends ValidationResult {
        /** Get the observed probability */
        double getObservedProbability();

        /** Get the sample size */
        long getSampleSize();

        /** Get the confidence level */
        double getConfidenceLevel();

        /** Get the lower bound of confidence interval */
        double getConfidenceLowerBound();

        /** Get the upper bound of confidence interval */
        double getConfidenceUpperBound();

        /** Get the margin of error */
        default double getMarginOfError() {
            return (getConfidenceUpperBound() - getConfidenceLowerBound()) / 2.0;
        }

        /** Check if the probability is statistically significant */
        default boolean isStatisticallySignificant() {
            return getMarginOfError() < 0.05; // 5% margin of error threshold
        }
    }

    /**
     * Comprehensive validation report for a probabilistic model.
     */
    interface ValidationReport {
        /** Get all validation results */
        List<ValidationResult> getResults();

        /** Get results filtered by validation type */
        List<ValidationResult> getResults(ValidationType type);

        /** Get results filtered by severity */
        List<ValidationResult> getResults(Severity severity);

        /** Check if overall validation passed */
        boolean isValid();

        /** Get the highest severity found */
        Severity getHighestSeverity();

        /** Get summary statistics */
        Map<String, Object> getSummary();

        /** Check if there are any critical errors */
        default boolean hasCriticalErrors() {
            return getResults(Severity.CRITICAL).size() > 0;
        }

        /** Check if there are any errors */
        default boolean hasErrors() {
            return getResults(Severity.ERROR).size() > 0 || hasCriticalErrors();
        }

        /** Check if there are any warnings */
        default boolean hasWarnings() {
            return getResults(Severity.WARNING).size() > 0;
        }
    }

    /**
     * Interface for probabilistic models that can be validated.
     */
    interface ProbabilisticModel {
        /** Get probability distributions used by the model */
        Map<String, double[]> getDistributions();

        /** Get transition matrices if applicable */
        Map<String, double[][]> getTransitionMatrices();

        /** Get individual probability values */
        Map<String, Double> getProbabilities();

        /** Get conditional probabilities if applicable */
        Map<String, ConditionalProbability> getConditionalProbabilities();

        /** Get model metadata for validation context */
        Map<String, Object> getMetadata();

        /** Conditional probability representation */
        record ConditionalProbability(double pAGivenB, double pA, double pB, String description) {
            public ConditionalProbability {
                if (pAGivenB < 0.0 || pAGivenB > 1.0) {
                    throw new IllegalArgumentException("Conditional probability must be in [0,1]");
                }
                if (pA < 0.0 || pA > 1.0) {
                    throw new IllegalArgumentException("Prior probability must be in [0,1]");
                }
                if (pB < 0.0 || pB > 1.0) {
                    throw new IllegalArgumentException("Evidence probability must be in [0,1]");
                }
            }

            /** Calculate joint probability P(A,B) = P(A|B) * P(B) */
            public double getJointProbability() {
                return pAGivenB * pB;
            }
        }
    }
}