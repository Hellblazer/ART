package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.laminar.canonical.CircuitParameters;

/**
 * Unified parameters for ARTLaminarCircuit combining laminar and FuzzyART configuration.
 *
 * <p>Bridges between laminar circuit parameters and FuzzyART parameters,
 * providing type-safe configuration with validation and conversion methods.
 *
 * <h2>Parameter Categories</h2>
 * <ul>
 *   <li><b>Dimensional:</b> inputSize, maxCategories</li>
 *   <li><b>Shared ART/Laminar:</b> vigilance, learningRate</li>
 *   <li><b>FuzzyART specific:</b> alpha (choice parameter)</li>
 *   <li><b>Laminar specific:</b> topDownGain, timeStep</li>
 * </ul>
 *
 * <h2>Typical Values</h2>
 * <ul>
 *   <li>vigilance: 0.7 (moderate specificity)</li>
 *   <li>learningRate: 0.1 (10% adaptation per iteration)</li>
 *   <li>alpha: 0.001 (small choice parameter, typical for FuzzyART)</li>
 *   <li>topDownGain: 0.8 (strong top-down modulation)</li>
 *   <li>timeStep: 0.01 (temporal integration step)</li>
 * </ul>
 *
 * @param inputSize dimensionality of input patterns
 * @param maxCategories maximum number of categories
 * @param vigilance match threshold ρ ∈ [0,1] for resonance
 * @param learningRate template learning rate β ∈ [0,1]
 * @param alpha choice parameter for FuzzyART activation ≥ 0, typically 0.001
 * @param topDownGain expectation modulation strength ∈ [0,1]
 * @param timeStep temporal integration step for dynamics > 0
 * @param expectationThreshold minimum activation for expectation ∈ [0,1]
 * @param maxSearchIterations category search limit
 *
 * @see FuzzyParameters
 * @see CircuitParameters
 * @author Claude Code
 */
public record ARTCircuitParameters(
    int inputSize,
    int maxCategories,
    double vigilance,
    double learningRate,
    double alpha,
    double topDownGain,
    double timeStep,
    double expectationThreshold,
    int maxSearchIterations
) {
    /**
     * Canonical constructor with validation.
     */
    public ARTCircuitParameters {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive, got: " + inputSize);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("maxCategories must be positive, got: " + maxCategories);
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("vigilance must be [0,1], got: " + vigilance);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("learningRate must be [0,1], got: " + learningRate);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("alpha must be non-negative, got: " + alpha);
        }
        if (topDownGain < 0.0 || topDownGain > 1.0) {
            throw new IllegalArgumentException("topDownGain must be [0,1], got: " + topDownGain);
        }
        if (timeStep <= 0.0 || timeStep > 1.0) {
            throw new IllegalArgumentException("timeStep must be (0,1], got: " + timeStep);
        }
        if (expectationThreshold < 0.0 || expectationThreshold > 1.0) {
            throw new IllegalArgumentException("expectationThreshold must be [0,1], got: " + expectationThreshold);
        }
        if (maxSearchIterations <= 0) {
            throw new IllegalArgumentException("maxSearchIterations must be positive, got: " + maxSearchIterations);
        }
    }

    /**
     * Convert to FuzzyParameters for FuzzyART.
     *
     * <p>Maps shared parameters:
     * <ul>
     *   <li>vigilance → vigilance (ρ)</li>
     *   <li>alpha → alpha (choice parameter)</li>
     *   <li>learningRate → beta (β)</li>
     * </ul>
     *
     * @return FuzzyParameters with mapped values
     */
    public FuzzyParameters toFuzzyParameters() {
        return new FuzzyParameters(vigilance, alpha, learningRate);
    }

    /**
     * Convert to VectorizedParameters for VectorizedFuzzyART.
     *
     * <p>Maps shared parameters and adds performance optimization settings:
     * <ul>
     *   <li>vigilance → vigilanceThreshold (ρ)</li>
     *   <li>learningRate → learningRate (β)</li>
     *   <li>alpha → alpha (choice parameter)</li>
     *   <li>Parallelism: auto-configured based on system</li>
     *   <li>SIMD: enabled by default</li>
     *   <li>Cache: configured for expected category count</li>
     * </ul>
     *
     * @return VectorizedParameters with optimized settings
     */
    public com.hellblazer.art.performance.algorithms.VectorizedParameters toVectorizedParameters() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new com.hellblazer.art.performance.algorithms.VectorizedParameters(
            vigilance,                     // vigilanceThreshold
            learningRate,                  // learningRate
            alpha,                         // alpha
            Math.max(2, processors / 2),   // parallelismLevel - use half available cores
            50,                            // parallelThreshold - use parallel for >50 categories
            1000,                          // maxCacheSize
            true,                          // enableSIMD
            true,                          // enableJOML
            0.8                            // memoryOptimizationThreshold
        );
    }

    /**
     * Convert to CircuitParameters for laminar circuit.
     *
     * <p>Maps parameters including laminar-specific settings.
     *
     * @return CircuitParameters with mapped values
     */
    public CircuitParameters toCircuitParameters() {
        return new CircuitParameters(
            inputSize,
            maxCategories,
            vigilance,
            learningRate,
            maxSearchIterations,
            timeStep,
            topDownGain,
            expectationThreshold,
            0.5  // resetThreshold (not used in ART integration)
        );
    }

    /**
     * Create default parameters for given input size.
     *
     * <p>Defaults:
     * <ul>
     *   <li>vigilance: 0.7 (moderate)</li>
     *   <li>learningRate: 0.1 (10% per iteration)</li>
     *   <li>alpha: 0.001 (standard FuzzyART)</li>
     *   <li>topDownGain: 0.8 (strong modulation)</li>
     *   <li>timeStep: 0.01 (fine-grained)</li>
     * </ul>
     *
     * @param inputSize input pattern dimension
     * @return default parameters
     */
    public static ARTCircuitParameters createDefault(int inputSize) {
        return new ARTCircuitParameters(
            inputSize,
            100,     // maxCategories
            0.7,     // vigilance
            0.1,     // learningRate
            0.001,   // alpha
            0.8,     // topDownGain
            0.01,    // timeStep
            0.05,    // expectationThreshold
            100      // maxSearchIterations
        );
    }

    /**
     * Create high vigilance parameters (specific categories).
     *
     * <p>Use for fine-grained discrimination where many specific categories are desired.
     *
     * @param inputSize input pattern dimension
     * @return high vigilance parameters (ρ = 0.9)
     */
    public static ARTCircuitParameters forHighVigilance(int inputSize) {
        return new ARTCircuitParameters(
            inputSize,
            100,
            0.9,     // HIGH vigilance - very specific
            0.1,
            0.001,
            0.8,
            0.01,
            0.05,
            100
        );
    }

    /**
     * Create low vigilance parameters (broad categories).
     *
     * <p>Use for coarse grouping where fewer, more general categories are desired.
     *
     * @param inputSize input pattern dimension
     * @return low vigilance parameters (ρ = 0.5)
     */
    public static ARTCircuitParameters forLowVigilance(int inputSize) {
        return new ARTCircuitParameters(
            inputSize,
            100,
            0.5,     // LOW vigilance - very general
            0.1,
            0.001,
            0.8,
            0.01,
            0.05,
            100
        );
    }

    /**
     * Create a builder for custom parameter configuration.
     *
     * @param inputSize input pattern dimension
     * @return builder instance
     */
    public static Builder builder(int inputSize) {
        return new Builder(inputSize);
    }

    /**
     * Builder for ARTCircuitParameters with fluent API.
     */
    public static class Builder {
        private final int inputSize;
        private int maxCategories = 100;
        private double vigilance = 0.7;
        private double learningRate = 0.1;
        private double alpha = 0.001;
        private double topDownGain = 0.8;
        private double timeStep = 0.01;
        private double expectationThreshold = 0.05;
        private int maxSearchIterations = 100;

        private Builder(int inputSize) {
            this.inputSize = inputSize;
        }

        public Builder maxCategories(int maxCategories) {
            this.maxCategories = maxCategories;
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

        public Builder choiceParameter(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder topDownGain(double topDownGain) {
            this.topDownGain = topDownGain;
            return this;
        }

        public Builder timeStep(double timeStep) {
            this.timeStep = timeStep;
            return this;
        }

        public Builder expectationThreshold(double expectationThreshold) {
            this.expectationThreshold = expectationThreshold;
            return this;
        }

        public Builder maxSearchIterations(int maxSearchIterations) {
            this.maxSearchIterations = maxSearchIterations;
            return this;
        }

        public ARTCircuitParameters build() {
            return new ARTCircuitParameters(
                inputSize,
                maxCategories,
                vigilance,
                learningRate,
                alpha,
                topDownGain,
                timeStep,
                expectationThreshold,
                maxSearchIterations
            );
        }
    }

    @Override
    public String toString() {
        return String.format(
            "ARTCircuitParameters[inputSize=%d, maxCategories=%d, ρ=%.3f, β=%.3f, α=%.4f]",
            inputSize, maxCategories, vigilance, learningRate, alpha
        );
    }
}