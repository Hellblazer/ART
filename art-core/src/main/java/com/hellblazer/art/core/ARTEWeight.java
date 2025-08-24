package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Weight vector for ART-E (Enhanced ART) with adaptive learning features.
 * 
 * ARTEWeight extends traditional ART weight vectors with:
 * - Feature importance tracking for dynamic weighting
 * - Performance metrics for category quality assessment
 * - Adaptation history for learning rate adjustment
 * - Context sensitivity for environmental awareness
 * - Convergence detection for optimization
 * 
 * The weight vector maintains both standard category weights and enhancement features
 * that enable ART-E's advanced learning capabilities.
 */
public final class ARTEWeight implements WeightVector {
    
    private final double[] categoryWeights;
    private final double[] featureImportances;
    private final List<Double> performanceHistory;
    private final double familiarityScore;
    private final double contextAdaptation;
    private final long lastUpdateTime;
    private final long creationTime;
    private final int updateCount;
    private final double convergenceMetric;
    
    /**
     * Create ARTEWeight with all enhancement features.
     */
    public ARTEWeight(double[] categoryWeights,
                      double[] featureImportances,
                      List<Double> performanceHistory,
                      double familiarityScore,
                      double contextAdaptation,
                      long lastUpdateTime,
                      long creationTime,
                      int updateCount,
                      double convergenceMetric) {
        
        this.categoryWeights = Objects.requireNonNull(categoryWeights, "Category weights cannot be null").clone();
        this.featureImportances = Objects.requireNonNull(featureImportances, "Feature importances cannot be null").clone();
        this.performanceHistory = List.copyOf(Objects.requireNonNull(performanceHistory, "Performance history cannot be null"));
        
        if (categoryWeights.length != featureImportances.length) {
            throw new IllegalArgumentException("Category weights and feature importances must have same length");
        }
        if (familiarityScore < 0.0 || familiarityScore > 1.0) {
            throw new IllegalArgumentException("Familiarity score must be in [0,1], got: " + familiarityScore);
        }
        if (contextAdaptation < 0.0 || contextAdaptation > 1.0) {
            throw new IllegalArgumentException("Context adaptation must be in [0,1], got: " + contextAdaptation);
        }
        if (updateCount < 0) {
            throw new IllegalArgumentException("Update count must be >= 0, got: " + updateCount);
        }
        if (convergenceMetric < 0.0) {
            throw new IllegalArgumentException("Convergence metric must be >= 0, got: " + convergenceMetric);
        }
        if (creationTime > lastUpdateTime) {
            throw new IllegalArgumentException("Creation time cannot be after last update time");
        }
        
        this.familiarityScore = familiarityScore;
        this.contextAdaptation = contextAdaptation;
        this.lastUpdateTime = lastUpdateTime;
        this.creationTime = creationTime;
        this.updateCount = updateCount;
        this.convergenceMetric = convergenceMetric;
    }
    
    /**
     * Create initial ARTEWeight from input vector.
     */
    public static ARTEWeight fromInput(Pattern input, ARTEParameters params) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        
        var categoryWeights = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            categoryWeights[i] = input.get(i);
        }
        
        // Initialize feature importances from parameters or uniform
        var featureImportances = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            featureImportances[i] = params.getFeatureWeight(i);
        }
        
        long currentTime = System.currentTimeMillis();
        
        return new ARTEWeight(
            categoryWeights,
            featureImportances,
            List.of(1.0), // Perfect performance for initial category
            0.0,          // No familiarity initially
            0.5,          // Neutral context adaptation
            currentTime,
            currentTime,
            0,            // No updates yet
            1.0           // Initial high change for new category
        );
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= categoryWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + categoryWeights.length);
        }
        return categoryWeights[index];
    }
    
    @Override
    public int dimension() {
        return categoryWeights.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double weight : categoryWeights) {
            sum += Math.abs(weight);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ARTEParameters arteParams)) {
            throw new IllegalArgumentException("Parameters must be ARTEParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension (" + input.dimension() + 
                                             ") must match weight dimension (" + dimension() + ")");
        }
        
        // Calculate new familiarity based on similarity to input
        double newFamiliarity = calculateFamiliarity(input);
        
        // Get adaptive learning rate based on familiarity
        double adaptiveLearningRate = arteParams.getAdaptiveLearningRate(newFamiliarity);
        
        // Update category weights using enhanced fuzzy ART learning with feature weighting
        var newCategoryWeights = new double[dimension()];
        for (int i = 0; i < dimension(); i++) {
            double featureWeight = arteParams.getFeatureWeight(i);
            double weightedInput = input.get(i) * featureWeight;
            double fuzzyMin = Math.min(weightedInput, categoryWeights[i]);
            
            newCategoryWeights[i] = adaptiveLearningRate * fuzzyMin + 
                                   (1.0 - adaptiveLearningRate) * categoryWeights[i];
        }
        
        // Update feature importances based on input variability
        var newFeatureImportances = updateFeatureImportances(input, arteParams);
        
        // Calculate performance metric for this update
        double currentPerformance = calculatePerformanceMetric(input, newCategoryWeights);
        
        // Update performance history (keep window size)
        var newPerformanceHistory = updatePerformanceHistory(currentPerformance, arteParams.performanceWindowSize());
        
        // Update context adaptation based on performance trend
        double newContextAdaptation = updateContextAdaptation(newPerformanceHistory);
        
        // Calculate convergence metric
        double newConvergenceMetric = calculateConvergenceMetric(newCategoryWeights);
        
        return new ARTEWeight(
            newCategoryWeights,
            newFeatureImportances,
            newPerformanceHistory,
            newFamiliarity,
            newContextAdaptation,
            System.currentTimeMillis(),
            creationTime,
            updateCount + 1,
            newConvergenceMetric
        );
    }
    
    /**
     * Calculate familiarity score based on input similarity.
     */
    public double calculateFamiliarity(Pattern input) {
        double intersection = 0.0;
        double union = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            intersection += Math.min(input.get(i), categoryWeights[i]);
            union += Math.max(input.get(i), categoryWeights[i]);
        }
        
        return union > 0.0 ? intersection / union : 0.0;
    }
    
    /**
     * Update feature importances based on input variability.
     */
    private double[] updateFeatureImportances(Pattern input, ARTEParameters params) {
        if (!params.featureWeightingEnabled()) {
            return featureImportances.clone();
        }
        
        var newImportances = new double[dimension()];
        double adjustmentRate = params.topologyAdjustmentRate();
        
        for (int i = 0; i < dimension(); i++) {
            // Calculate feature relevance based on difference from current weight
            double relevance = Math.abs(input.get(i) - categoryWeights[i]);
            
            // Update importance with exponential smoothing
            newImportances[i] = (1.0 - adjustmentRate) * featureImportances[i] + 
                               adjustmentRate * relevance;
        }
        
        // Normalize importances to sum to 1.0
        double sum = Arrays.stream(newImportances).sum();
        if (sum > 0.0) {
            for (int i = 0; i < newImportances.length; i++) {
                newImportances[i] /= sum;
            }
        }
        
        return newImportances;
    }
    
    /**
     * Calculate performance metric for current update.
     */
    private double calculatePerformanceMetric(Pattern input, double[] newWeights) {
        // Performance based on how well the updated weights match the input
        double matchQuality = 0.0;
        double totalPossible = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            matchQuality += Math.min(input.get(i), newWeights[i]);
            totalPossible += Math.max(input.get(i), newWeights[i]);
        }
        
        return totalPossible > 0.0 ? matchQuality / totalPossible : 0.0;
    }
    
    /**
     * Update performance history maintaining window size.
     */
    private List<Double> updatePerformanceHistory(double newPerformance, int windowSize) {
        var newHistory = new java.util.ArrayList<>(performanceHistory);
        newHistory.add(newPerformance);
        
        // Maintain window size
        while (newHistory.size() > windowSize) {
            newHistory.remove(0);
        }
        
        return List.copyOf(newHistory);
    }
    
    /**
     * Update context adaptation based on performance trend.
     */
    private double updateContextAdaptation(List<Double> newPerformanceHistory) {
        if (newPerformanceHistory.size() < 2) {
            return contextAdaptation; // Not enough history
        }
        
        // Calculate performance trend (recent vs older performance)
        double recentPerformance = newPerformanceHistory.get(newPerformanceHistory.size() - 1);
        double olderPerformance = newPerformanceHistory.get(0);
        
        // Positive trend increases adaptation, negative trend decreases it
        double trend = recentPerformance - olderPerformance;
        double newAdaptation = contextAdaptation + 0.1 * trend;
        
        return Math.max(0.0, Math.min(1.0, newAdaptation));
    }
    
    /**
     * Calculate convergence metric based on weight stability.
     */
    private double calculateConvergenceMetric(double[] newWeights) {
        if (updateCount == 0) {
            return 1.0; // High change for first update
        }
        
        double totalChange = 0.0;
        for (int i = 0; i < dimension(); i++) {
            totalChange += Math.abs(newWeights[i] - categoryWeights[i]);
        }
        
        return totalChange / dimension(); // Average change per dimension
    }
    
    // Accessors for enhanced features
    
    public double[] getCategoryWeights() {
        return categoryWeights.clone();
    }
    
    public double[] getFeatureImportances() {
        return featureImportances.clone();
    }
    
    public List<Double> getPerformanceHistory() {
        return performanceHistory;
    }
    
    public double getFamiliarityScore() {
        return familiarityScore;
    }
    
    public double getContextAdaptation() {
        return contextAdaptation;
    }
    
    public long getLastUpdateTime() {
        return lastUpdateTime;
    }
    
    public long getCreationTime() {
        return creationTime;
    }
    
    public int getUpdateCount() {
        return updateCount;
    }
    
    public double getConvergenceMetric() {
        return convergenceMetric;
    }
    
    public long getAge() {
        return lastUpdateTime - creationTime;
    }
    
    /**
     * Get average performance over recent history.
     */
    public double getAveragePerformance() {
        if (performanceHistory.isEmpty()) {
            return 0.0;
        }
        return performanceHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Check if category has converged based on recent changes.
     */
    public boolean hasConverged(double threshold) {
        return convergenceMetric < threshold;
    }
    
    /**
     * Check if category performance is above threshold.
     */
    public boolean isPerforming(double threshold) {
        return getAveragePerformance() >= threshold;
    }
    
    /**
     * Get the most important feature dimension.
     */
    public int getMostImportantFeature() {
        int maxIndex = 0;
        for (int i = 1; i < featureImportances.length; i++) {
            if (featureImportances[i] > featureImportances[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * Calculate weighted similarity to input using feature importances.
     */
    public double calculateWeightedSimilarity(Pattern input) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.dimension() != dimension()) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }
        
        double weightedIntersection = 0.0;
        double weightedUnion = 0.0;
        
        for (int i = 0; i < dimension(); i++) {
            double intersection = Math.min(input.get(i), categoryWeights[i]);
            double union = Math.max(input.get(i), categoryWeights[i]);
            
            // Weight both intersection and union by feature importance
            weightedIntersection += intersection * featureImportances[i];
            weightedUnion += union * featureImportances[i];
        }
        
        return weightedUnion > 0.0 ? weightedIntersection / weightedUnion : 0.0;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof ARTEWeight other)) return false;
        
        return Arrays.equals(categoryWeights, other.categoryWeights) &&
               Arrays.equals(featureImportances, other.featureImportances) &&
               Objects.equals(performanceHistory, other.performanceHistory) &&
               Double.compare(familiarityScore, other.familiarityScore) == 0 &&
               Double.compare(contextAdaptation, other.contextAdaptation) == 0 &&
               lastUpdateTime == other.lastUpdateTime &&
               creationTime == other.creationTime &&
               updateCount == other.updateCount &&
               Double.compare(convergenceMetric, other.convergenceMetric) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(
            Arrays.hashCode(categoryWeights),
            Arrays.hashCode(featureImportances),
            performanceHistory,
            familiarityScore,
            contextAdaptation,
            lastUpdateTime,
            creationTime,
            updateCount,
            convergenceMetric
        );
    }
    
    @Override
    public String toString() {
        return String.format("ARTEWeight{dim=%d, familiarity=%.3f, performance=%.3f, " +
                           "updates=%d, convergence=%.6f, age=%dms}",
                           dimension(), familiarityScore, getAveragePerformance(),
                           updateCount, convergenceMetric, getAge());
    }
}