package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * Weight vector for ARTSTAR (ART with STability and Adaptability Regulation).
 * 
 * ARTSTARWeight extends traditional category weights with regulation mechanisms:
 * - Category weights: The learned pattern representation
 * - Stability measure: Tracks resistance to modification
 * - Adaptability measure: Tracks learning responsiveness  
 * - Usage count: Number of times this category has been activated
 * - Last update time: Timestamp of most recent update for decay calculation
 * - Strength: Overall category health measure
 * 
 * The regulation system uses these measures to dynamically adjust learning
 * parameters and maintain optimal stability-adaptability balance.
 */
public final class ARTSTARWeight implements WeightVector {
    
    private final double[] categoryWeights;
    private final double stabilityMeasure;
    private final double adaptabilityMeasure;
    private final long usageCount;
    private final long lastUpdateTime;
    private final double strength;
    
    /**
     * Create ARTSTAR weight with all regulation data.
     */
    public ARTSTARWeight(double[] categoryWeights, double stabilityMeasure,
                        double adaptabilityMeasure, long usageCount,
                        long lastUpdateTime, double strength) {
        this.categoryWeights = Objects.requireNonNull(categoryWeights, "Category weights cannot be null").clone();
        
        if (categoryWeights.length == 0) {
            throw new IllegalArgumentException("Category weights cannot be empty");
        }
        
        // Validate regulation measures
        if (stabilityMeasure < 0.0 || stabilityMeasure > 1.0 || 
            Double.isNaN(stabilityMeasure) || Double.isInfinite(stabilityMeasure)) {
            throw new IllegalArgumentException("Stability measure must be in [0, 1], got: " + stabilityMeasure);
        }
        if (adaptabilityMeasure < 0.0 || adaptabilityMeasure > 1.0 ||
            Double.isNaN(adaptabilityMeasure) || Double.isInfinite(adaptabilityMeasure)) {
            throw new IllegalArgumentException("Adaptability measure must be in [0, 1], got: " + adaptabilityMeasure);
        }
        if (strength <= 0.0 || strength > 1.0 || Double.isNaN(strength) || Double.isInfinite(strength)) {
            throw new IllegalArgumentException("Strength must be in (0, 1], got: " + strength);
        }
        if (usageCount < 0) {
            throw new IllegalArgumentException("Usage count must be >= 0, got: " + usageCount);
        }
        if (lastUpdateTime < 0) {
            throw new IllegalArgumentException("Last update time must be >= 0, got: " + lastUpdateTime);
        }
        
        // Validate category weights
        for (int i = 0; i < categoryWeights.length; i++) {
            if (Double.isNaN(categoryWeights[i]) || Double.isInfinite(categoryWeights[i])) {
                throw new IllegalArgumentException("Category weight at index " + i + " is invalid: " + categoryWeights[i]);
            }
        }
        
        this.stabilityMeasure = stabilityMeasure;
        this.adaptabilityMeasure = adaptabilityMeasure;
        this.usageCount = usageCount;
        this.lastUpdateTime = lastUpdateTime;
        this.strength = strength;
    }
    
    /**
     * Create initial ARTSTAR weight from input vector.
     */
    public static ARTSTARWeight fromVector(Pattern input, double initialStrength) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        var weights = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            weights[i] = input.get(i);
        }
        
        long currentTime = System.currentTimeMillis();
        return new ARTSTARWeight(weights, 0.5, 0.5, 1, currentTime, initialStrength);
    }
    
    /**
     * Create ARTSTAR weight with uniform initialization.
     */
    public static ARTSTARWeight withUniformWeights(int dimension, double value, double initialStrength) {
        var weights = new double[dimension];
        Arrays.fill(weights, value);
        long currentTime = System.currentTimeMillis();
        return new ARTSTARWeight(weights, 0.5, 0.5, 1, currentTime, initialStrength);
    }
    
    @Override
    public int dimension() {
        return categoryWeights.length;
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= categoryWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + categoryWeights.length);
        }
        return categoryWeights[index];
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double value : categoryWeights) {
            sum += Math.abs(value);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ARTSTARParameters artstarParams)) {
            throw new IllegalArgumentException("Parameters must be ARTSTARParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match weight dimension " + dimension());
        }
        
        var newCategoryWeights = new double[categoryWeights.length];
        
        // Update category weights using fuzzy ART learning rule with stability regulation
        double effectiveLearningRate = artstarParams.beta() * 
            (1.0 - stabilityMeasure * 0.5); // Stability reduces learning rate
        
        for (int i = 0; i < newCategoryWeights.length; i++) {
            double fuzzyMin = Math.min(input.get(i), categoryWeights[i]);
            newCategoryWeights[i] = effectiveLearningRate * fuzzyMin + 
                                   (1.0 - effectiveLearningRate) * categoryWeights[i];
        }
        
        // Update regulation measures based on learning success
        double similarity = calculateSimilarity(input);
        double stabilityUpdate = similarity - 0.5; // High similarity increases stability
        double adaptabilityUpdate = (1.0 - similarity) - 0.5; // High novelty increases adaptability
        
        double newStability = Math.max(0.0, Math.min(1.0, 
            stabilityMeasure + artstarParams.regulationRate() * stabilityUpdate));
        double newAdaptability = Math.max(0.0, Math.min(1.0,
            adaptabilityMeasure + artstarParams.regulationRate() * adaptabilityUpdate));
        
        // Update strength based on successful learning
        double strengthIncrease = 0.01; // Small boost for successful learning
        double newStrength = Math.min(1.0, strength + strengthIncrease);
        
        // Update usage and timestamp
        long currentTime = System.currentTimeMillis();
        
        return new ARTSTARWeight(newCategoryWeights, newStability, newAdaptability,
                                usageCount + 1, currentTime, newStrength);
    }
    
    // Getters for all regulation data
    public double[] getCategoryWeights() {
        return categoryWeights.clone();
    }
    
    public double getCategoryWeight(int index) {
        if (index < 0 || index >= categoryWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + categoryWeights.length);
        }
        return categoryWeights[index];
    }
    
    public double getStabilityMeasure() {
        return stabilityMeasure;
    }
    
    public double getAdaptabilityMeasure() {
        return adaptabilityMeasure;
    }
    
    public long getUsageCount() {
        return usageCount;
    }
    
    public long getLastUpdateTime() {
        return lastUpdateTime;
    }
    
    public double getStrength() {
        return strength;
    }
    
    /**
     * Calculate regulation balance (-1 = high stability, +1 = high adaptability).
     */
    public double getRegulationBalance() {
        return adaptabilityMeasure - stabilityMeasure;
    }
    
    /**
     * Check if category should be considered for removal.
     */
    public boolean isWeak(double threshold) {
        return strength < threshold;
    }
    
    /**
     * Calculate time since last update in milliseconds.
     */
    public long getTimeSinceLastUpdate() {
        return Math.max(0, System.currentTimeMillis() - lastUpdateTime);
    }
    
    /**
     * Calculate decay factor based on time and decay rate.
     */
    public double calculateDecayFactor(double decayRate) {
        if (decayRate <= 0.0) return 1.0;
        
        long timeDelta = getTimeSinceLastUpdate();
        double timeInHours = timeDelta / (1000.0 * 60.0 * 60.0); // Convert to hours
        return Math.exp(-decayRate * timeInHours);
    }
    
    /**
     * Apply time-based decay to strength.
     */
    public double getDecayedStrength(double decayRate) {
        return strength * calculateDecayFactor(decayRate);
    }
    
    // Immutable update methods
    
    /**
     * Update category weights with new learning.
     */
    public ARTSTARWeight withCategoryWeights(double[] newWeights) {
        return new ARTSTARWeight(newWeights, stabilityMeasure, adaptabilityMeasure,
                                usageCount, lastUpdateTime, strength);
    }
    
    /**
     * Update stability measure.
     */
    public ARTSTARWeight withStabilityMeasure(double newStability) {
        return new ARTSTARWeight(categoryWeights, newStability, adaptabilityMeasure,
                                usageCount, lastUpdateTime, strength);
    }
    
    /**
     * Update adaptability measure.
     */
    public ARTSTARWeight withAdaptabilityMeasure(double newAdaptability) {
        return new ARTSTARWeight(categoryWeights, stabilityMeasure, newAdaptability,
                                usageCount, lastUpdateTime, strength);
    }
    
    /**
     * Update regulation measures (stability and adaptability).
     */
    public ARTSTARWeight withRegulationMeasures(double newStability, double newAdaptability) {
        return new ARTSTARWeight(categoryWeights, newStability, newAdaptability,
                                usageCount, lastUpdateTime, strength);
    }
    
    /**
     * Increment usage count and update timestamp.
     */
    public ARTSTARWeight withUsage() {
        long currentTime = System.currentTimeMillis();
        return new ARTSTARWeight(categoryWeights, stabilityMeasure, adaptabilityMeasure,
                                usageCount + 1, currentTime, strength);
    }
    
    /**
     * Update strength value.
     */
    public ARTSTARWeight withStrength(double newStrength) {
        return new ARTSTARWeight(categoryWeights, stabilityMeasure, adaptabilityMeasure,
                                usageCount, lastUpdateTime, newStrength);
    }
    
    /**
     * Comprehensive update with all regulation data.
     */
    public ARTSTARWeight withUpdate(double[] newWeights, double newStability, double newAdaptability,
                                   long newUsageCount, long newUpdateTime, double newStrength) {
        return new ARTSTARWeight(newWeights, newStability, newAdaptability,
                                newUsageCount, newUpdateTime, newStrength);
    }
    
    /**
     * Calculate similarity to input vector using standard fuzzy ART measure.
     */
    public double calculateSimilarity(Pattern input) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                                             " does not match category dimension " + dimension());
        }
        
        double intersection = 0.0;
        double inputMagnitude = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double fuzzyMin = Math.min(input.get(i), categoryWeights[i]);
            intersection += fuzzyMin;
            inputMagnitude += input.get(i);
        }
        
        return inputMagnitude > 0.0 ? intersection / inputMagnitude : 0.0;
    }
    
    /**
     * Calculate stability-weighted similarity that considers regulation state.
     */
    public double calculateRegulatedSimilarity(Pattern input) {
        double baseSimilarity = calculateSimilarity(input);
        
        // Higher stability increases similarity (resistance to change)
        // Higher adaptability decreases similarity (willingness to change)
        double regulationWeight = stabilityMeasure - (adaptabilityMeasure * 0.5);
        regulationWeight = Math.max(0.0, Math.min(1.0, regulationWeight));
        
        return baseSimilarity * (0.5 + 0.5 * regulationWeight);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        var other = (ARTSTARWeight) obj;
        return Arrays.equals(categoryWeights, other.categoryWeights) &&
               Double.compare(stabilityMeasure, other.stabilityMeasure) == 0 &&
               Double.compare(adaptabilityMeasure, other.adaptabilityMeasure) == 0 &&
               usageCount == other.usageCount &&
               lastUpdateTime == other.lastUpdateTime &&
               Double.compare(strength, other.strength) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(categoryWeights), stabilityMeasure, 
                           adaptabilityMeasure, usageCount, lastUpdateTime, strength);
    }
    
    @Override
    public String toString() {
        return String.format("ARTSTARWeight{dim=%d, stability=%.3f, adaptability=%.3f, " +
                           "usage=%d, strength=%.3f, balance=%.3f}",
                           dimension(), stabilityMeasure, adaptabilityMeasure,
                           usageCount, strength, getRegulationBalance());
    }
    
    /**
     * Create compact string representation.
     */
    public String toCompactString() {
        return String.format("ARTSTAR{d=%d,s=%.2f,a=%.2f,u=%d,st=%.2f}",
                           dimension(), stabilityMeasure, adaptabilityMeasure,
                           usageCount, strength);
    }
    
    /**
     * Create detailed string with full data.
     */
    public String toDetailedString() {
        return String.format("ARTSTARWeight{weights=%s, stability=%.4f, adaptability=%.4f, " +
                           "usage=%d, lastUpdate=%d, strength=%.4f, balance=%.4f}",
                           Arrays.toString(categoryWeights), stabilityMeasure, adaptabilityMeasure,
                           usageCount, lastUpdateTime, strength, getRegulationBalance());
    }
}