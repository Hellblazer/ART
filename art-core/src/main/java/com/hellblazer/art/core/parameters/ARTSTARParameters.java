package com.hellblazer.art.core.parameters;

import java.util.Objects;

/**
 * Parameters for ARTSTAR (ART with STability and Adaptability Regulation).
 * 
 * ARTSTAR extends traditional ART with dynamic regulation mechanisms that
 * automatically balance stability (preserving learned knowledge) and 
 * adaptability (learning new patterns). The regulation system monitors
 * network performance and adjusts parameters accordingly.
 * 
 * Key regulation mechanisms:
 * - Stability regulation: Prevents catastrophic forgetting
 * - Adaptability regulation: Ensures continued learning capacity
 * - Dynamic vigilance: Adjusts vigilance based on learning success
 * - Category health monitoring: Tracks category usage and decay
 * 
 * @param vigilance Base vigilance parameter (0 < vigilance <= 1)
 * @param alpha Choice parameter (alpha >= 0) 
 * @param beta Learning rate (0 <= beta <= 1)
 * @param stabilityFactor Controls resistance to forgetting (0 < stabilityFactor <= 1)
 * @param adaptabilityFactor Controls learning aggressiveness (0 < adaptabilityFactor <= 1) 
 * @param regulationRate Rate of parameter adjustment (0 < regulationRate <= 1)
 * @param categoryDecayRate Rate of category decay when unused (0 <= categoryDecayRate < 1)
 * @param vigilanceRange Maximum vigilance adjustment range (0 < vigilanceRange <= 0.5)
 * @param minCategoryStrength Minimum strength before category removal (0 < minCategoryStrength <= 1)
 * @param maxCategories Maximum number of categories (0 = unlimited)
 */
public record ARTSTARParameters(
    double vigilance,
    double alpha,
    double beta,
    double stabilityFactor,
    double adaptabilityFactor,
    double regulationRate,
    double categoryDecayRate,
    double vigilanceRange,
    double minCategoryStrength,
    int maxCategories
) {
    
    /**
     * Create ARTSTAR parameters with validation.
     */
    public ARTSTARParameters {
        if (vigilance <= 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in (0, 1], got: " + vigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be >= 0, got: " + alpha);
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException("Beta must be in [0, 1], got: " + beta);
        }
        if (stabilityFactor <= 0.0 || stabilityFactor > 1.0) {
            throw new IllegalArgumentException("Stability factor must be in (0, 1], got: " + stabilityFactor);
        }
        if (adaptabilityFactor <= 0.0 || adaptabilityFactor > 1.0) {
            throw new IllegalArgumentException("Adaptability factor must be in (0, 1], got: " + adaptabilityFactor);
        }
        if (regulationRate <= 0.0 || regulationRate > 1.0) {
            throw new IllegalArgumentException("Regulation rate must be in (0, 1], got: " + regulationRate);
        }
        if (categoryDecayRate < 0.0 || categoryDecayRate >= 1.0) {
            throw new IllegalArgumentException("Category decay rate must be in [0, 1), got: " + categoryDecayRate);
        }
        if (vigilanceRange <= 0.0 || vigilanceRange > 0.5) {
            throw new IllegalArgumentException("Vigilance range must be in (0, 0.5], got: " + vigilanceRange);
        }
        if (minCategoryStrength <= 0.0 || minCategoryStrength > 1.0) {
            throw new IllegalArgumentException("Min category strength must be in (0, 1], got: " + minCategoryStrength);
        }
        if (maxCategories < 0) {
            throw new IllegalArgumentException("Max categories must be >= 0, got: " + maxCategories);
        }
    }
    
    /**
     * Create ARTSTAR parameters from individual values.
     */
    public static ARTSTARParameters of(double vigilance, double alpha, double beta,
                                       double stabilityFactor, double adaptabilityFactor,
                                       double regulationRate, double categoryDecayRate,
                                       double vigilanceRange, double minCategoryStrength,
                                       int maxCategories) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    /**
     * Create ARTSTAR parameters with sensible defaults.
     */
    public static ARTSTARParameters defaults() {
        return new ARTSTARParameters(
            0.7,    // vigilance
            0.0,    // alpha  
            1.0,    // beta
            0.8,    // stabilityFactor - high stability
            0.6,    // adaptabilityFactor - moderate adaptability
            0.1,    // regulationRate - slow adaptation
            0.01,   // categoryDecayRate - slow decay
            0.2,    // vigilanceRange - moderate adjustment range
            0.1,    // minCategoryStrength - low threshold
            0       // maxCategories - unlimited
        );
    }
    
    /**
     * Create builder for fluent parameter construction.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder class for fluent parameter construction.
     */
    public static class Builder {
        private double vigilance = 0.7;
        private double alpha = 0.0;
        private double beta = 1.0;
        private double stabilityFactor = 0.8;
        private double adaptabilityFactor = 0.6;
        private double regulationRate = 0.1;
        private double categoryDecayRate = 0.01;
        private double vigilanceRange = 0.2;
        private double minCategoryStrength = 0.1;
        private int maxCategories = 0;
        
        public Builder vigilance(double vigilance) { this.vigilance = vigilance; return this; }
        public Builder choiceParameter(double alpha) { this.alpha = alpha; return this; }
        public Builder learningRate(double beta) { this.beta = beta; return this; }
        public Builder stabilityFactor(double stabilityFactor) { this.stabilityFactor = stabilityFactor; return this; }
        public Builder adaptabilityFactor(double adaptabilityFactor) { this.adaptabilityFactor = adaptabilityFactor; return this; }
        public Builder regulationRate(double regulationRate) { this.regulationRate = regulationRate; return this; }
        public Builder categoryDecayRate(double categoryDecayRate) { this.categoryDecayRate = categoryDecayRate; return this; }
        public Builder vigilanceRange(double vigilanceRange) { this.vigilanceRange = vigilanceRange; return this; }
        public Builder minCategoryStrength(double minCategoryStrength) { this.minCategoryStrength = minCategoryStrength; return this; }
        public Builder maxCategories(int maxCategories) { this.maxCategories = maxCategories; return this; }
        
        public ARTSTARParameters build() {
            return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                        adaptabilityFactor, regulationRate, categoryDecayRate,
                                        vigilanceRange, minCategoryStrength, maxCategories);
        }
    }
    
    // Immutable update methods
    public ARTSTARParameters withVigilance(double vigilance) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withAlpha(double alpha) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withBeta(double beta) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withStabilityFactor(double stabilityFactor) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withAdaptabilityFactor(double adaptabilityFactor) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withRegulationRate(double regulationRate) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withCategoryDecayRate(double categoryDecayRate) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withVigilanceRange(double vigilanceRange) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withMinCategoryStrength(double minCategoryStrength) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    public ARTSTARParameters withMaxCategories(int maxCategories) {
        return new ARTSTARParameters(vigilance, alpha, beta, stabilityFactor,
                                    adaptabilityFactor, regulationRate, categoryDecayRate,
                                    vigilanceRange, minCategoryStrength, maxCategories);
    }
    
    /**
     * Calculate current effective vigilance based on regulation state.
     */
    public double getEffectiveVigilance(double stabilityMeasure, double adaptabilityMeasure) {
        // Balance stability and adaptability to adjust vigilance
        double balance = (stabilityMeasure - adaptabilityMeasure) * regulationRate;
        double adjustment = balance * vigilanceRange;
        return Math.max(vigilance - vigilanceRange, 
                       Math.min(vigilance + vigilanceRange, vigilance + adjustment));
    }
    
    @Override
    public String toString() {
        return String.format("ARTSTARParameters{vigilance=%.3f, alpha=%.3f, beta=%.3f, " +
                           "stability=%.3f, adaptability=%.3f, regulation=%.3f, " +
                           "decay=%.4f, vRange=%.3f, minStrength=%.3f, maxCat=%d}",
                           vigilance, alpha, beta, stabilityFactor, adaptabilityFactor,
                           regulationRate, categoryDecayRate, vigilanceRange, 
                           minCategoryStrength, maxCategories);
    }
}