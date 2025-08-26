package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.parameters.ARTEParameters;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.ARTEWeight;import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * ART-E (Enhanced ART) implementation with adaptive learning features.
 */
public final class ARTE extends BaseART {
    
    private final Random random;
    private long totalLearningSteps;
    private long convergenceDetectionSteps;
    private double networkPerformance;
    private long lastOptimizationTime;
    
    /**
     * Create a new ART-E instance with no initial categories.
     */
    public ARTE() {
        super();
        this.random = new Random();
        initializeNetworkState();
    }
    
    /**
     * Create a new ART-E instance with initial categories.
     * @param initialCategories the initial ART-E weight categories (will be copied)
     */
    public ARTE(List<ARTEWeight> initialCategories) {
        super(Objects.requireNonNull(initialCategories, "Initial categories cannot be null"));
        this.random = new Random();
        initializeNetworkState();
    }
    
    /**
     * Create a new ART-E instance with specified random seed.
     */
    public ARTE(long seed) {
        super();
        this.random = new Random(seed);
        initializeNetworkState();
    }
    
    private void initializeNetworkState() {
        this.totalLearningSteps = 0;
        this.convergenceDetectionSteps = 0;
        this.networkPerformance = 0.0;
        this.lastOptimizationTime = System.currentTimeMillis();
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTEWeight arteWeight)) {
            throw new IllegalArgumentException("Weight must be ARTEWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTEParameters arteParams)) {
            throw new IllegalArgumentException("Parameters must be ARTEParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var categoryWeights = arteWeight.getCategoryWeights();
        var featureImportances = arteWeight.getFeatureImportances();
        
        // Enhanced choice function with feature weighting
        double weightedIntersection = 0.0;
        double weightedCategoryMagnitude = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double featureWeight = arteParams.featureWeightingEnabled() ? 
                featureImportances[i] : 1.0;
            
            // Calculate fuzzy min with feature weighting
            double weightedInput = input.get(i) * featureWeight;
            double weightedCategory = categoryWeights[i] * featureWeight;
            
            weightedIntersection += Math.min(weightedInput, weightedCategory);
            weightedCategoryMagnitude += weightedCategory;
        }
        
        // Base choice function
        double baseActivation = weightedIntersection / 
            (arteParams.alpha() + weightedCategoryMagnitude);
        
        // Apply performance boost for well-performing categories
        double performanceBoost = arteWeight.getAveragePerformance() * 0.1;
        
        // Apply familiarity regulation (balance exploration vs exploitation)
        double familiarityRegulation = 1.0 + (arteWeight.getFamiliarityScore() - 0.5) * 0.2;
        
        return baseActivation * familiarityRegulation + performanceBoost;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTEWeight arteWeight)) {
            throw new IllegalArgumentException("Weight must be ARTEWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTEParameters arteParams)) {
            throw new IllegalArgumentException("Parameters must be ARTEParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var categoryWeights = arteWeight.getCategoryWeights();
        var featureImportances = arteWeight.getFeatureImportances();
        
        // Calculate weighted intersection and input magnitude
        double weightedIntersection = 0.0;
        double weightedInputMagnitude = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double featureWeight = arteParams.featureWeightingEnabled() ? 
                featureImportances[i] : 1.0;
            
            double weightedInput = input.get(i) * featureWeight;
            double weightedCategory = categoryWeights[i] * featureWeight;
            
            weightedIntersection += Math.min(weightedInput, weightedCategory);
            weightedInputMagnitude += weightedInput;
        }
        
        // Avoid division by zero
        if (weightedInputMagnitude == 0.0) {
            boolean isEmpty = true;
            for (double w : categoryWeights) {
                if (w != 0.0) {
                    isEmpty = false;
                    break;
                }
            }
            double matchValue = isEmpty ? 1.0 : 0.0;
            boolean isAccepted = matchValue >= arteParams.vigilance();
            return isAccepted ? new MatchResult.Accepted(matchValue, arteParams.vigilance()) :
                               new MatchResult.Rejected(matchValue, arteParams.vigilance());
        }
        
        // Calculate match ratio
        double matchRatio = weightedIntersection / weightedInputMagnitude;
        
        // Context-sensitive vigilance adjustment
        double contextFactor = arteWeight.getContextAdaptation();
        double effectiveVigilance = arteParams.getEffectiveVigilance(contextFactor);
        
        // Additional adjustments based on category performance and familiarity
        double performanceAdjustment = (arteWeight.getAveragePerformance() - 0.5) * 0.1;
        double familiarityAdjustment = (arteWeight.getFamiliarityScore() - 0.5) * 0.05;
        
        double finalVigilance = Math.max(0.0, Math.min(1.0, 
            effectiveVigilance + performanceAdjustment + familiarityAdjustment));
        
        // Test against adjusted vigilance threshold
        boolean isAccepted = matchRatio >= finalVigilance;
        return isAccepted ? new MatchResult.Accepted(matchRatio, finalVigilance) :
                           new MatchResult.Rejected(matchRatio, finalVigilance);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(currentWeight instanceof ARTEWeight arteWeight)) {
            throw new IllegalArgumentException("Weight must be ARTEWeight, got: " + currentWeight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTEParameters arteParams)) {
            throw new IllegalArgumentException("Parameters must be ARTEParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        // Update using ARTEWeight's enhanced update method
        return arteWeight.update(input, parameters);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ARTEParameters arteParams)) {
            throw new IllegalArgumentException("Parameters must be ARTEParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        return ARTEWeight.fromInput(input, arteParams);
    }
    
    /**
     * Enhanced learning step with ART-E optimizations.
     */
    public ActivationResult stepFitEnhanced(Pattern input, ARTEParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        totalLearningSteps++;
        
        // Perform standard ART learning step
        var result = stepFit(input, parameters);
        
        // Update network performance metrics
        updateNetworkPerformance(result, parameters);
        
        // Periodic optimization and pruning
        if (shouldOptimizeNetwork(parameters)) {
            optimizeNetwork(parameters);
        }
        
        return result;
    }
    
    /**
     * Update network-level performance metrics.
     */
    private void updateNetworkPerformance(ActivationResult result, ARTEParameters params) {
        if (result instanceof ActivationResult.Success success) {
            // Track successful learning
            double successRate = 1.0;
            networkPerformance = 0.9 * networkPerformance + 0.1 * successRate;
        } else {
            // Track failed learning (shouldn't happen with current BaseART)
            double failureRate = 0.0;
            networkPerformance = 0.9 * networkPerformance + 0.1 * failureRate;
        }
        
        // Detect convergence patterns
        if (result instanceof ActivationResult.Success success) {
            var weight = success.updatedWeight();
            if (weight instanceof ARTEWeight arteWeight) {
                if (arteWeight.hasConverged(params.convergenceThreshold())) {
                    convergenceDetectionSteps++;
                }
            }
        }
    }
    
    /**
     * Check if network optimization should be performed.
     */
    private boolean shouldOptimizeNetwork(ARTEParameters params) {
        long timeSinceLastOptimization = System.currentTimeMillis() - lastOptimizationTime;
        
        // Optimize every 100 learning steps or every 10 seconds
        return (totalLearningSteps % 100 == 0) || (timeSinceLastOptimization > 10000);
    }
    
    /**
     * Perform network optimization including pruning and topology adjustment.
     */
    public void optimizeNetwork(ARTEParameters params) {
        // Prune underperforming categories
        pruneUnderperformingCategories(params);
        
        // Apply topology adjustments
        if (params.shouldAdjustTopology(random.nextDouble())) {
            adjustNetworkTopology(params);
        }
        
        // Update optimization timestamp
        lastOptimizationTime = System.currentTimeMillis();
    }
    
    /**
     * Remove categories that consistently underperform.
     */
    private void pruneUnderperformingCategories(ARTEParameters params) {
        var performingCategories = new ArrayList<WeightVector>();
        
        for (var category : getCategories()) {
            if (category instanceof ARTEWeight arteWeight) {
                // Keep categories that meet performance threshold or are too new to evaluate
                if (arteWeight.isPerforming(params.performanceThreshold()) || 
                    arteWeight.getUpdateCount() < params.performanceWindowSize()) {
                    performingCategories.add(category);
                }
            } else {
                // Keep non-ARTE weights
                performingCategories.add(category);
            }
        }
        
        // Only prune if we have categories remaining
        if (!performingCategories.isEmpty() && performingCategories.size() < getCategoryCount()) {
            replaceAllCategories(performingCategories);
        }
    }
    
    /**
     * Adjust network topology based on performance and convergence patterns.
     * Implements feature weight adaptation to optimize category discrimination.
     */
    private void adjustNetworkTopology(ARTEParameters params) {
        // Adjust global feature weights based on category feature importances
        // This helps the network focus on the most discriminative features
        adaptGlobalFeatureWeights(params);
    }
    
    /**
     * Adapt global feature weights based on category feature importances.
     */
    private void adaptGlobalFeatureWeights(ARTEParameters params) {
        if (!params.featureWeightingEnabled() || getCategories().isEmpty()) {
            return;
        }
        
        // Calculate average feature importances across all categories
        var arteCategories = getCategories().stream()
            .filter(ARTEWeight.class::isInstance)
            .map(ARTEWeight.class::cast)
            .toList();
        
        if (arteCategories.isEmpty()) {
            return;
        }
        
        int dimension = arteCategories.get(0).dimension();
        var averageImportances = new double[dimension];
        
        for (var category : arteCategories) {
            var importances = category.getFeatureImportances();
            for (int i = 0; i < dimension; i++) {
                averageImportances[i] += importances[i];
            }
        }
        
        // Normalize by number of categories
        for (int i = 0; i < dimension; i++) {
            averageImportances[i] /= arteCategories.size();
        }
        
        // This could be used to update global parameters, but we keep params immutable
        // In a full implementation, this might trigger parameter adaptation
    }
    
    /**
     * Get enhanced network analysis including ART-E specific metrics.
     */
    public NetworkAnalysis analyzeNetwork() {
        var arteCategories = getCategories().stream()
            .filter(ARTEWeight.class::isInstance)
            .map(ARTEWeight.class::cast)
            .toList();
        
        if (arteCategories.isEmpty()) {
            return new NetworkAnalysis(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0);
        }
        
        double averageFamiliarity = arteCategories.stream()
            .mapToDouble(ARTEWeight::getFamiliarityScore)
            .average().orElse(0.0);
            
        double averagePerformance = arteCategories.stream()
            .mapToDouble(ARTEWeight::getAveragePerformance)
            .average().orElse(0.0);
            
        double averageContextAdaptation = arteCategories.stream()
            .mapToDouble(ARTEWeight::getContextAdaptation)
            .average().orElse(0.0);
            
        double averageConvergence = arteCategories.stream()
            .mapToDouble(ARTEWeight::getConvergenceMetric)
            .average().orElse(0.0);
            
        int totalUpdates = arteCategories.stream()
            .mapToInt(ARTEWeight::getUpdateCount)
            .sum();
            
        long averageAge = arteCategories.stream()
            .mapToLong(ARTEWeight::getAge)
            .sum() / arteCategories.size();
            
        double convergenceRate = totalLearningSteps > 0 ? 
            (double) convergenceDetectionSteps / totalLearningSteps : 0.0;
        
        return new NetworkAnalysis(
            getCategoryCount(),
            networkPerformance,
            averageFamiliarity,
            averagePerformance,
            averageContextAdaptation,
            totalUpdates,
            averageConvergence,
            convergenceRate,
            averageAge
        );
    }
    
    /**
     * Get specific ART-E category.
     */
    public ARTEWeight getARTECategory(int index) {
        var category = getCategory(index);
        if (!(category instanceof ARTEWeight arteWeight)) {
            throw new IllegalArgumentException("Category " + index + " is not an ARTEWeight");
        }
        return arteWeight;
    }
    
    /**
     * Get network performance metrics.
     */
    public double getNetworkPerformance() {
        return networkPerformance;
    }
    
    /**
     * Get total learning steps performed.
     */
    public long getTotalLearningSteps() {
        return totalLearningSteps;
    }
    
    /**
     * Get convergence detection rate.
     */
    public double getConvergenceRate() {
        return totalLearningSteps > 0 ? (double) convergenceDetectionSteps / totalLearningSteps : 0.0;
    }
    
    @Override
    public String toString() {
        return String.format("ARTE{categories=%d, performance=%.3f, steps=%d, convergence=%.3f}",
                           getCategoryCount(), networkPerformance, totalLearningSteps, getConvergenceRate());
    }
    
    /**
     * Enhanced network analysis result for ART-E.
     */
    public record NetworkAnalysis(
        int totalCategories,
        double networkPerformance,
        double averageFamiliarity,
        double averagePerformance,
        double averageContextAdaptation,
        int totalUpdates,
        double averageConvergence,
        double convergenceRate,
        long averageAge
    ) {
        @Override
        public String toString() {
            return String.format("NetworkAnalysis{categories=%d, netPerf=%.3f, " +
                               "avgFam=%.3f, avgPerf=%.3f, avgCtx=%.3f, updates=%d, " +
                               "avgConv=%.6f, convRate=%.3f, avgAge=%dms}",
                               totalCategories, networkPerformance,
                               averageFamiliarity, averagePerformance, averageContextAdaptation,
                               totalUpdates, averageConvergence, convergenceRate, averageAge);
        }
    }
}