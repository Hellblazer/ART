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

import com.hellblazer.art.core.State;
import com.hellblazer.art.core.Transition;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for validating Markov property assumptions in hybrid ART neural networks.
 *
 * MarkovPropertyValidator ensures that state transition sequences satisfy the
 * Markov property: the probability of the next state depends only on the current
 * state, not on the sequence of events that preceded it. This is fundamental
 * for many probabilistic models and temporal learning algorithms.
 *
 * Key validation capabilities:
 * - First-order Markov property testing
 * - Higher-order Markov model validation
 * - Memory depth analysis
 * - Independence testing for transition sequences
 * - Stationarity assumption verification
 * - Ergodicity property checking
 *
 * @param <S> the type of states in the Markov process
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface MarkovPropertyValidator<S extends State<?>> {

    /**
     * Types of Markov property validations.
     */
    enum MarkovTest {
        /** Basic first-order Markov property */
        FIRST_ORDER,
        /** Higher-order Markov dependencies */
        HIGHER_ORDER,
        /** Memory depth analysis */
        MEMORY_DEPTH,
        /** Transition independence testing */
        INDEPENDENCE,
        /** Stationarity of transition probabilities */
        STATIONARITY,
        /** Ergodicity properties */
        ERGODICITY,
        /** Time-homogeneity assumption */
        TIME_HOMOGENEITY
    }

    /**
     * Markov model orders for higher-order testing.
     */
    enum MarkovOrder {
        ZERO(0),    // Memoryless (independent transitions)
        FIRST(1),   // Standard first-order Markov
        SECOND(2),  // Second-order (depends on last 2 states)
        THIRD(3),   // Third-order (depends on last 3 states)
        VARIABLE(-1); // Variable order

        private final int order;

        MarkovOrder(int order) {
            this.order = order;
        }

        public int getOrder() {
            return order;
        }
    }

    /**
     * Validate the first-order Markov property for a sequence of transitions.
     * Tests if P(S_t+1 | S_t, S_t-1, ..., S_0) = P(S_t+1 | S_t)
     *
     * @param transitions the sequence of transitions to analyze
     * @param significanceLevel statistical significance level (e.g., 0.05)
     * @return validation result with statistical test results
     */
    MarkovValidationResult validateFirstOrderMarkov(List<Transition<S, ?>> transitions, double significanceLevel);

    /**
     * Validate higher-order Markov properties.
     * Tests if the process follows an n-th order Markov model.
     *
     * @param transitions the sequence of transitions to analyze
     * @param order the Markov order to test
     * @param significanceLevel statistical significance level
     * @return validation result with order-specific analysis
     */
    MarkovValidationResult validateHigherOrderMarkov(List<Transition<S, ?>> transitions,
                                                     MarkovOrder order, double significanceLevel);

    /**
     * Determine the optimal Markov order for a transition sequence.
     * Uses information criteria (AIC, BIC) to select the best model order.
     *
     * @param transitions the sequence of transitions to analyze
     * @param maxOrder maximum order to consider
     * @return optimal order determination result
     */
    OrderSelectionResult determineOptimalOrder(List<Transition<S, ?>> transitions, int maxOrder);

    /**
     * Test the memory depth of the Markov process.
     * Determines how far back in history affects future states.
     *
     * @param transitions the sequence of transitions to analyze
     * @param maxDepth maximum memory depth to test
     * @param significanceLevel statistical significance level
     * @return memory depth analysis result
     */
    MemoryDepthResult analyzeMemoryDepth(List<Transition<S, ?>> transitions,
                                        int maxDepth, double significanceLevel);

    /**
     * Validate stationarity of transition probabilities over time.
     * Tests if transition probabilities remain constant across different time periods.
     *
     * @param transitions the sequence of transitions to analyze
     * @param timeWindow window size for partitioning the sequence
     * @param significanceLevel statistical significance level
     * @return stationarity validation result
     */
    StationarityResult validateStationarity(List<Transition<S, ?>> transitions,
                                          Duration timeWindow, double significanceLevel);

    /**
     * Test ergodicity properties of the Markov chain.
     * Checks if long-run frequencies converge to stationary distribution.
     *
     * @param transitions the sequence of transitions to analyze
     * @param convergenceThreshold threshold for convergence testing
     * @return ergodicity test result
     */
    ErgodicityResult testErgodicity(List<Transition<S, ?>> transitions, double convergenceThreshold);

    /**
     * Validate time-homogeneity assumption.
     * Tests if transition probabilities are independent of absolute time.
     *
     * @param transitions the sequence of transitions to analyze
     * @param timePartitions number of time partitions for testing
     * @param significanceLevel statistical significance level
     * @return time-homogeneity validation result
     */
    TimeHomogeneityResult validateTimeHomogeneity(List<Transition<S, ?>> transitions,
                                                 int timePartitions, double significanceLevel);

    /**
     * Perform comprehensive Markov analysis including all validation types.
     *
     * @param transitions the sequence of transitions to analyze
     * @param tests the types of Markov tests to perform
     * @param significanceLevel statistical significance level
     * @return comprehensive Markov analysis report
     */
    MarkovAnalysisReport analyzeMarkovProperties(List<Transition<S, ?>> transitions,
                                               List<MarkovTest> tests, double significanceLevel);

    /**
     * Calculate transition probability matrix from observed transitions.
     *
     * @param transitions the sequence of transitions
     * @return transition probability matrix with state mappings
     */
    TransitionMatrix<S> calculateTransitionMatrix(List<Transition<S, ?>> transitions);

    /**
     * Calculate stationary distribution of the Markov chain.
     *
     * @param transitionMatrix the transition probability matrix
     * @return optional stationary distribution (may not exist or converge)
     */
    Optional<StationaryDistribution<S>> calculateStationaryDistribution(TransitionMatrix<S> transitionMatrix);

    /**
     * Validate that transition matrix satisfies Markov chain properties.
     *
     * @param transitionMatrix the transition matrix to validate
     * @return transition matrix validation result
     */
    TransitionMatrixValidationResult validateTransitionMatrix(TransitionMatrix<S> transitionMatrix);

    /**
     * Get current validation settings and thresholds.
     *
     * @return map of validation parameters
     */
    Map<String, Object> getValidationSettings();

    /**
     * Update validation settings.
     *
     * @param settings new validation parameters
     */
    void updateValidationSettings(Map<String, Object> settings);

    /**
     * Base interface for Markov validation results.
     */
    interface MarkovValidationResult {
        /** Check if Markov property is satisfied */
        boolean satisfiesMarkovProperty();

        /** Get the test type performed */
        MarkovTest getTestType();

        /** Get statistical test p-value */
        double getPValue();

        /** Get significance level used */
        double getSignificanceLevel();

        /** Get test statistic value */
        double getTestStatistic();

        /** Get validation message */
        String getMessage();

        /** Get additional metrics */
        Map<String, Object> getMetrics();

        /** Check if result is statistically significant */
        default boolean isStatisticallySignificant() {
            return getPValue() < getSignificanceLevel();
        }
    }

    /**
     * Result of optimal Markov order determination.
     */
    interface OrderSelectionResult extends MarkovValidationResult {
        /** Get the selected optimal order */
        MarkovOrder getOptimalOrder();

        /** Get information criteria values for different orders */
        Map<MarkovOrder, Double> getInformationCriteria();

        /** Get likelihood values for different orders */
        Map<MarkovOrder, Double> getLikelihoods();

        /** Get the information criterion used for selection */
        InformationCriterion getCriterion();

        /** Information criteria for model selection */
        enum InformationCriterion {
            AIC, BIC, HQC, MDL
        }
    }

    /**
     * Result of memory depth analysis.
     */
    interface MemoryDepthResult extends MarkovValidationResult {
        /** Get the determined memory depth */
        int getMemoryDepth();

        /** Get mutual information at different depths */
        Map<Integer, Double> getMutualInformation();

        /** Get significance test results for each depth */
        Map<Integer, Boolean> getSignificanceResults();

        /** Check if memory depth is bounded */
        boolean isBoundedMemory();
    }

    /**
     * Result of stationarity validation.
     */
    interface StationarityResult extends MarkovValidationResult {
        /** Check if transition probabilities are stationary */
        boolean isStationary();

        /** Get transition probability changes over time */
        Map<Duration, Double> getProbabilityChanges();

        /** Get the stationarity test statistic */
        double getStationarityStatistic();

        /** Get time windows analyzed */
        List<Duration> getTimeWindows();
    }

    /**
     * Result of ergodicity testing.
     */
    interface ErgodicityResult extends MarkovValidationResult {
        /** Check if the Markov chain is ergodic */
        boolean isErgodic();

        /** Check if the chain is irreducible */
        boolean isIrreducible();

        /** Check if the chain is aperiodic */
        boolean isAperiodic();

        /** Get convergence rate to stationary distribution */
        Optional<Double> getConvergenceRate();

        /** Get mixing time estimate */
        Optional<Duration> getMixingTime();
    }

    /**
     * Result of time-homogeneity validation.
     */
    interface TimeHomogeneityResult extends MarkovValidationResult {
        /** Check if transition probabilities are time-homogeneous */
        boolean isTimeHomogeneous();

        /** Get transition probability variations across time */
        Map<String, Double> getTemporalVariations();

        /** Get homogeneity test results for each state pair */
        Map<String, Boolean> getHomogeneityTests();
    }

    /**
     * Comprehensive Markov analysis report.
     */
    interface MarkovAnalysisReport {
        /** Get all validation results */
        List<MarkovValidationResult> getResults();

        /** Get results by test type */
        List<MarkovValidationResult> getResults(MarkovTest testType);

        /** Check if overall Markov assumptions are satisfied */
        boolean satisfiesMarkovAssumptions();

        /** Get the recommended Markov order */
        MarkovOrder getRecommendedOrder();

        /** Get summary statistics */
        Map<String, Object> getSummary();

        /** Get identified violations */
        List<String> getViolations();

        /** Get recommendations for addressing violations */
        List<String> getRecommendations();
    }

    /**
     * Transition probability matrix with state mappings.
     */
    interface TransitionMatrix<S extends State<?>> {
        /** Get the probability matrix */
        double[][] getMatrix();

        /** Get state index mapping */
        Map<S, Integer> getStateIndices();

        /** Get states by index */
        List<S> getStates();

        /** Get transition probability between specific states */
        double getTransitionProbability(S from, S to);

        /** Get number of states */
        int getStateCount();

        /** Check if matrix is doubly stochastic */
        boolean isDoublyStochastic();

        /** Get row sums (should be 1.0 for valid transition matrix) */
        double[] getRowSums();
    }

    /**
     * Stationary distribution of a Markov chain.
     */
    interface StationaryDistribution<S extends State<?>> {
        /** Get the stationary probabilities */
        Map<S, Double> getProbabilities();

        /** Get convergence properties */
        ConvergenceProperties getConvergenceProperties();

        /** Check if distribution is unique */
        boolean isUnique();

        /** Get entropy of stationary distribution */
        double getEntropy();

        /** Convergence properties of stationary distribution */
        interface ConvergenceProperties {
            /** Check if distribution converged */
            boolean hasConverged();

            /** Get convergence tolerance achieved */
            double getToleranceAchieved();

            /** Get number of iterations required */
            int getIterations();

            /** Get convergence rate */
            double getConvergenceRate();
        }
    }

    /**
     * Validation result for transition matrices.
     */
    interface TransitionMatrixValidationResult extends MarkovValidationResult {
        /** Check if matrix is a valid stochastic matrix */
        boolean isValidStochasticMatrix();

        /** Check if matrix is irreducible */
        boolean isIrreducible();

        /** Check if matrix is aperiodic */
        boolean isAperiodic();

        /** Get row sum validation results */
        Map<Integer, Double> getRowSumErrors();

        /** Get eigenvalue analysis */
        EigenvalueAnalysis getEigenvalueAnalysis();

        /** Eigenvalue analysis for transition matrix */
        interface EigenvalueAnalysis {
            /** Get all eigenvalues */
            List<Double> getEigenvalues();

            /** Get the dominant eigenvalue */
            double getDominantEigenvalue();

            /** Get second-largest eigenvalue magnitude */
            double getSecondLargestEigenvalue();

            /** Check if dominant eigenvalue is 1.0 */
            boolean hasDominantEigenvalueOne();

            /** Get spectral gap */
            double getSpectralGap();
        }
    }
}