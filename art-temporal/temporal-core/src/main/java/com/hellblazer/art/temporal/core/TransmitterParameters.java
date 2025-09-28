package com.hellblazer.art.temporal.core;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for habituative transmitter gates.
 * Based on Equation 7 from Kazerounian & Grossberg (2014):
 * dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 */
public class TransmitterParameters implements Parameters {
    private final double epsilon;          // ε: recovery rate (0.005 - very slow)
    private final double lambda;           // λ: linear depletion rate (0.1)
    private final double mu;               // μ: quadratic depletion rate (0.05)
    private final double depletionThreshold; // Threshold for triggering reset
    private final double initialLevel;     // Initial transmitter level (typically 1.0)
    private final boolean enableQuadratic; // Enable quadratic depletion term

    private TransmitterParameters(Builder builder) {
        this.epsilon = builder.epsilon;
        this.lambda = builder.lambda;
        this.mu = builder.mu;
        this.depletionThreshold = builder.depletionThreshold;
        this.initialLevel = builder.initialLevel;
        this.enableQuadratic = builder.enableQuadratic;
        validate();
    }

    @Override
    public void validate() {
        if (epsilon <= 0 || epsilon > 0.1) {
            throw new IllegalArgumentException("Recovery rate ε must be in (0, 0.1], got: " + epsilon);
        }
        if (lambda < 0 || lambda > 1.0) {
            throw new IllegalArgumentException("Linear depletion λ must be in [0, 1], got: " + lambda);
        }
        if (mu < 0 || mu > 1.0) {
            throw new IllegalArgumentException("Quadratic depletion μ must be in [0, 1], got: " + mu);
        }
        if (depletionThreshold <= 0 || depletionThreshold > 0.5) {
            throw new IllegalArgumentException("Depletion threshold must be in (0, 0.5], got: " + depletionThreshold);
        }
        if (initialLevel <= 0 || initialLevel > 1.0) {
            throw new IllegalArgumentException("Initial level must be in (0, 1], got: " + initialLevel);
        }

        // Check stability condition: ε + λ + μ should be reasonable
        double totalRate = epsilon + lambda + mu;
        if (totalRate > 2.0) {
            throw new IllegalArgumentException("Total rate (ε + λ + μ) too high for stability: " + totalRate);
        }
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("epsilon", epsilon);
        params.put("lambda", lambda);
        params.put("mu", mu);
        params.put("depletionThreshold", depletionThreshold);
        params.put("initialLevel", initialLevel);
        params.put("enableQuadratic", enableQuadratic ? 1.0 : 0.0);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = new Builder()
            .epsilon(epsilon)
            .lambda(lambda)
            .mu(mu)
            .depletionThreshold(depletionThreshold)
            .initialLevel(initialLevel)
            .enableQuadratic(enableQuadratic);

        switch (name) {
            case "epsilon" -> builder.epsilon(value);
            case "lambda" -> builder.lambda(value);
            case "mu" -> builder.mu(value);
            case "depletionThreshold" -> builder.depletionThreshold(value);
            case "initialLevel" -> builder.initialLevel(value);
            case "enableQuadratic" -> builder.enableQuadratic(value > 0.5);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public double getEpsilon() { return epsilon; }
    public double getLambda() { return lambda; }
    public double getMu() { return enableQuadratic ? mu : 0.0; }
    public double getDepletionThreshold() { return depletionThreshold; }
    public double getInitialLevel() { return initialLevel; }
    public boolean isQuadraticEnabled() { return enableQuadratic; }

    /**
     * Compute equilibrium transmitter level for given signal.
     * Z_eq = ε / (ε + λS + μS²)
     */
    public double computeEquilibrium(double signal) {
        double denominator = epsilon + lambda * signal;
        if (enableQuadratic) {
            denominator += mu * signal * signal;
        }
        return epsilon / denominator;
    }

    /**
     * Compute depletion rate for given signal.
     */
    public double computeDepletionRate(double signal) {
        double rate = lambda * signal;
        if (enableQuadratic) {
            rate += mu * signal * signal;
        }
        return rate;
    }

    /**
     * Compute time scale of transmitter dynamics.
     * Transmitters operate on 500-5000 ms scale.
     */
    public double getTimeScale() {
        // Inverse of recovery rate gives characteristic time
        return 1.0 / epsilon; // ~200 time units for ε=0.005
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double epsilon = 0.005;
        private double lambda = 0.1;
        private double mu = 0.05;
        private double depletionThreshold = 0.2;
        private double initialLevel = 1.0;
        private boolean enableQuadratic = true;

        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }

        public Builder mu(double mu) {
            this.mu = mu;
            return this;
        }

        public Builder depletionThreshold(double threshold) {
            this.depletionThreshold = threshold;
            return this;
        }

        public Builder initialLevel(double level) {
            this.initialLevel = level;
            return this;
        }

        public Builder enableQuadratic(boolean enable) {
            this.enableQuadratic = enable;
            return this;
        }

        public TransmitterParameters build() {
            return new TransmitterParameters(this);
        }
    }

    /**
     * Create default parameters from paper.
     */
    public static TransmitterParameters paperDefaults() {
        return builder()
            .epsilon(0.005)      // Very slow recovery
            .lambda(0.1)         // Linear depletion
            .mu(0.05)           // Quadratic depletion
            .depletionThreshold(0.2)
            .initialLevel(1.0)
            .enableQuadratic(true)
            .build();
    }

    /**
     * Create fast recovery parameters for testing.
     */
    public static TransmitterParameters fastRecovery() {
        return builder()
            .epsilon(0.05)       // 10x faster recovery
            .lambda(0.1)
            .mu(0.05)
            .depletionThreshold(0.2)
            .initialLevel(1.0)
            .enableQuadratic(true)
            .build();
    }
}