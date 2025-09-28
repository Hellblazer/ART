package com.hellblazer.art.temporal.core;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for multi-scale masking field.
 * Based on Equation 5 from Kazerounian & Grossberg (2014):
 * dY_jk/dt = -α * Y_jk + (β - Y_jk) * [f(Y_jk) + I_jk] - Y_jk * Σ g(Y_lm)
 */
public class MaskingFieldParameters implements Parameters {
    private final double alpha;            // α: passive decay rate (0.1)
    private final double beta;             // β: upper bound (1.0)
    private final int numScales;           // Number of scales (typically 3-5)
    private final int fieldSize;           // Size of each scale field (typically 100)
    private final double selfExcitation;   // f(Y) function strength
    private final double lateralInhibition; // g(Y) function strength
    private final double asymmetryFactor;  // Asymmetric inhibition between scales
    private final double convergenceThreshold; // For convergence detection
    private final boolean enableMultiScale; // Enable multi-scale processing

    private MaskingFieldParameters(Builder builder) {
        this.alpha = builder.alpha;
        this.beta = builder.beta;
        this.numScales = builder.numScales;
        this.fieldSize = builder.fieldSize;
        this.selfExcitation = builder.selfExcitation;
        this.lateralInhibition = builder.lateralInhibition;
        this.asymmetryFactor = builder.asymmetryFactor;
        this.convergenceThreshold = builder.convergenceThreshold;
        this.enableMultiScale = builder.enableMultiScale;
        validate();
    }

    @Override
    public void validate() {
        if (alpha <= 0 || alpha > 1.0) {
            throw new IllegalArgumentException("Passive decay α must be in (0, 1], got: " + alpha);
        }
        if (beta <= 0) {
            throw new IllegalArgumentException("Upper bound β must be positive, got: " + beta);
        }
        if (numScales < 1 || numScales > 10) {
            throw new IllegalArgumentException("Number of scales must be in [1, 10], got: " + numScales);
        }
        if (fieldSize < 10 || fieldSize > 1000) {
            throw new IllegalArgumentException("Field size must be in [10, 1000], got: " + fieldSize);
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException("Self-excitation must be non-negative, got: " + selfExcitation);
        }
        if (lateralInhibition < 0) {
            throw new IllegalArgumentException("Lateral inhibition must be non-negative, got: " + lateralInhibition);
        }
        if (asymmetryFactor < 1.0 || asymmetryFactor > 5.0) {
            throw new IllegalArgumentException("Asymmetry factor must be in [1, 5], got: " + asymmetryFactor);
        }
        if (convergenceThreshold <= 0 || convergenceThreshold > 0.1) {
            throw new IllegalArgumentException("Convergence threshold must be in (0, 0.1], got: " + convergenceThreshold);
        }
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("alpha", alpha);
        params.put("beta", beta);
        params.put("numScales", (double) numScales);
        params.put("fieldSize", (double) fieldSize);
        params.put("selfExcitation", selfExcitation);
        params.put("lateralInhibition", lateralInhibition);
        params.put("asymmetryFactor", asymmetryFactor);
        params.put("convergenceThreshold", convergenceThreshold);
        params.put("enableMultiScale", enableMultiScale ? 1.0 : 0.0);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = new Builder()
            .alpha(alpha)
            .beta(beta)
            .numScales(numScales)
            .fieldSize(fieldSize)
            .selfExcitation(selfExcitation)
            .lateralInhibition(lateralInhibition)
            .asymmetryFactor(asymmetryFactor)
            .convergenceThreshold(convergenceThreshold)
            .enableMultiScale(enableMultiScale);

        switch (name) {
            case "alpha" -> builder.alpha(value);
            case "beta" -> builder.beta(value);
            case "numScales" -> builder.numScales((int) value);
            case "fieldSize" -> builder.fieldSize((int) value);
            case "selfExcitation" -> builder.selfExcitation(value);
            case "lateralInhibition" -> builder.lateralInhibition(value);
            case "asymmetryFactor" -> builder.asymmetryFactor(value);
            case "convergenceThreshold" -> builder.convergenceThreshold(value);
            case "enableMultiScale" -> builder.enableMultiScale(value > 0.5);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public double getAlpha() { return alpha; }
    public double getBeta() { return beta; }
    public int getNumScales() { return enableMultiScale ? numScales : 1; }
    public int getFieldSize() { return fieldSize; }
    public double getSelfExcitation() { return selfExcitation; }
    public double getLateralInhibition() { return lateralInhibition; }
    public double getAsymmetryFactor() { return asymmetryFactor; }
    public double getConvergenceThreshold() { return convergenceThreshold; }
    public boolean isMultiScaleEnabled() { return enableMultiScale; }

    /**
     * Compute inhibition strength from scale i to scale j.
     * Larger scales inhibit smaller scales more strongly.
     */
    public double computeInterScaleInhibition(int fromScale, int toScale) {
        if (!enableMultiScale || fromScale == toScale) {
            return 0.0;
        }

        if (fromScale > toScale) {
            // Larger scale inhibiting smaller
            return lateralInhibition * asymmetryFactor;
        } else {
            // Smaller scale inhibiting larger
            return lateralInhibition / asymmetryFactor;
        }
    }

    /**
     * Get preferred length for a given scale.
     * Scale 0: length 1, Scale 1: length 3, Scale 2: length 5, etc.
     */
    public int getPreferredLength(int scale) {
        return 2 * scale + 1;
    }

    /**
     * Compute time scale of masking field dynamics.
     * Masking field operates on 50-500 ms scale.
     */
    public double getTimeScale() {
        // Inverse of decay rate gives characteristic time
        return 1.0 / alpha; // ~10 time units for α=0.1
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double alpha = 0.1;
        private double beta = 1.0;
        private int numScales = 3;
        private int fieldSize = 100;
        private double selfExcitation = 0.2;
        private double lateralInhibition = 0.3;
        private double asymmetryFactor = 2.0;
        private double convergenceThreshold = 0.01;
        private boolean enableMultiScale = true;

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        public Builder numScales(int numScales) {
            this.numScales = numScales;
            return this;
        }

        public Builder fieldSize(int fieldSize) {
            this.fieldSize = fieldSize;
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

        public Builder asymmetryFactor(double factor) {
            this.asymmetryFactor = factor;
            return this;
        }

        public Builder convergenceThreshold(double threshold) {
            this.convergenceThreshold = threshold;
            return this;
        }

        public Builder enableMultiScale(boolean enable) {
            this.enableMultiScale = enable;
            return this;
        }

        public MaskingFieldParameters build() {
            return new MaskingFieldParameters(this);
        }
    }

    /**
     * Create default parameters from paper.
     */
    public static MaskingFieldParameters paperDefaults() {
        return builder()
            .alpha(0.1)               // Passive decay
            .beta(1.0)                // Upper bound
            .numScales(3)             // 3 scales: item, chunk, sequence
            .fieldSize(100)           // 100 cells per scale
            .selfExcitation(0.2)
            .lateralInhibition(0.3)
            .asymmetryFactor(2.0)     // 2x stronger inhibition from larger scales
            .convergenceThreshold(0.01)
            .enableMultiScale(true)
            .build();
    }

    /**
     * Create single-scale parameters for testing.
     */
    public static MaskingFieldParameters singleScale() {
        return builder()
            .alpha(0.1)
            .beta(1.0)
            .numScales(1)
            .fieldSize(100)
            .selfExcitation(0.2)
            .lateralInhibition(0.3)
            .asymmetryFactor(1.0)
            .convergenceThreshold(0.01)
            .enableMultiScale(false)
            .build();
    }
}