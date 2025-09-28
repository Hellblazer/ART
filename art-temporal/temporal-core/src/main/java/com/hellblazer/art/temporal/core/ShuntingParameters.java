package com.hellblazer.art.temporal.core;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for shunting dynamics.
 * Based on Kazerounian & Grossberg (2014).
 */
public class ShuntingParameters implements Parameters {
    private final double decayRate;        // A: passive decay (0.05-0.2)
    private final double upperBound;       // B: upper bound (typically 1.0)
    private final double lowerBound;       // C: lower bound (typically 0.0)
    private final double selfExcitation;   // Self-excitation strength
    private final double lateralInhibition; // Lateral inhibition strength
    private final boolean enableNormalization;

    private ShuntingParameters(Builder builder) {
        this.decayRate = builder.decayRate;
        this.upperBound = builder.upperBound;
        this.lowerBound = builder.lowerBound;
        this.selfExcitation = builder.selfExcitation;
        this.lateralInhibition = builder.lateralInhibition;
        this.enableNormalization = builder.enableNormalization;
        validate();
    }

    @Override
    public void validate() {
        if (decayRate <= 0 || decayRate > 1.0) {
            throw new IllegalArgumentException("Decay rate must be in (0, 1], got: " + decayRate);
        }
        if (upperBound <= lowerBound) {
            throw new IllegalArgumentException("Upper bound must be greater than lower bound");
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException("Self-excitation must be non-negative");
        }
        if (lateralInhibition < 0) {
            throw new IllegalArgumentException("Lateral inhibition must be non-negative");
        }
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("decayRate", decayRate);
        params.put("upperBound", upperBound);
        params.put("lowerBound", lowerBound);
        params.put("selfExcitation", selfExcitation);
        params.put("lateralInhibition", lateralInhibition);
        params.put("enableNormalization", enableNormalization ? 1.0 : 0.0);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = new Builder()
            .decayRate(decayRate)
            .upperBound(upperBound)
            .lowerBound(lowerBound)
            .selfExcitation(selfExcitation)
            .lateralInhibition(lateralInhibition)
            .enableNormalization(enableNormalization);

        switch (name) {
            case "decayRate" -> builder.decayRate(value);
            case "upperBound" -> builder.upperBound(value);
            case "lowerBound" -> builder.lowerBound(value);
            case "selfExcitation" -> builder.selfExcitation(value);
            case "lateralInhibition" -> builder.lateralInhibition(value);
            case "enableNormalization" -> builder.enableNormalization(value > 0.5);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public double getDecayRate() { return decayRate; }
    public double getUpperBound() { return upperBound; }
    public double getLowerBound() { return lowerBound; }
    public double getSelfExcitation() { return selfExcitation; }
    public double getLateralInhibition() { return lateralInhibition; }
    public boolean isNormalizationEnabled() { return enableNormalization; }

    /**
     * Compute stability condition for time step.
     * dt < 2 / |λ_max| where λ_max is max eigenvalue
     */
    public double computeMaxStableTimeStep() {
        // Conservative estimate based on decay rate and inhibition
        double maxEigenvalue = decayRate + selfExcitation + lateralInhibition;
        return 2.0 / maxEigenvalue * 0.9; // 0.9 for safety margin
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double decayRate = 0.1;
        private double upperBound = 1.0;
        private double lowerBound = 0.0;
        private double selfExcitation = 0.2;
        private double lateralInhibition = 0.3;
        private boolean enableNormalization = true;

        public Builder decayRate(double decayRate) {
            this.decayRate = decayRate;
            return this;
        }

        public Builder upperBound(double upperBound) {
            this.upperBound = upperBound;
            return this;
        }

        public Builder lowerBound(double lowerBound) {
            this.lowerBound = lowerBound;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder enableNormalization(boolean enable) {
            this.enableNormalization = enable;
            return this;
        }

        public ShuntingParameters build() {
            return new ShuntingParameters(this);
        }
    }

    /**
     * Create default parameters from paper.
     */
    public static ShuntingParameters paperDefaults() {
        return builder()
            .decayRate(0.1)
            .upperBound(1.0)
            .lowerBound(0.0)
            .selfExcitation(0.2)
            .lateralInhibition(0.3)
            .enableNormalization(true)
            .build();
    }
}