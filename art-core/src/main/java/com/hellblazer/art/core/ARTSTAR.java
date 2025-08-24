package com.hellblazer.art.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * ARTSTAR (ART with STability and Adaptability Regulation) implementation.
 * 
 * ARTSTAR extends traditional ART with dynamic regulation mechanisms that
 * automatically balance stability (preserving learned knowledge) and 
 * adaptability (learning new patterns). The system continuously monitors
 * network performance and adjusts parameters to maintain optimal learning.
 * 
 * Key features:
 * - Dynamic vigilance adjustment based on stability/adaptability balance
 * - Category strength tracking with time-based decay
 * - Automatic category pruning of weak categories
 * - Stability/adaptability regulation based on learning success
 * - Maximum category limits with intelligent pruning
 * 
 * Algorithm components:
 * 1. Regulated choice function with stability bias
 * 2. Dynamic vigilance test with adaptability adjustment
 * 3. Stability-aware weight updates with regulation learning
 * 4. Category health monitoring and pruning
 * 5. Network-level regulation parameter adjustment
 */
public final class ARTSTAR extends BaseART {
    
    // Network-level regulation state
    private double networkStability;
    private double networkAdaptability;
    private long totalLearningEvents;
    private long successfulLearningEvents;
    private long lastRegulationUpdate;
    
    /**
     * Create a new ARTSTAR instance with no initial categories.
     */
    public ARTSTAR() {
        super();
        initializeRegulationState();
    }
    
    /**
     * Create a new ARTSTAR instance with initial categories.
     * @param initialCategories the initial ARTSTAR weight categories (will be copied)
     */
    public ARTSTAR(List<ARTSTARWeight> initialCategories) {
        super(Objects.requireNonNull(initialCategories, "Initial categories cannot be null"));
        initializeRegulationState();
    }
    
    private void initializeRegulationState() {
        this.networkStability = 0.5;
        this.networkAdaptability = 0.5;
        this.totalLearningEvents = 0;
        this.successfulLearningEvents = 0;
        this.lastRegulationUpdate = System.currentTimeMillis();
    }
    
    @Override
    protected double calculateActivation(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTSTARWeight artstarWeight)) {
            throw new IllegalArgumentException("Weight must be ARTSTARWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTSTARParameters artstarParams)) {
            throw new IllegalArgumentException("Parameters must be ARTSTARParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var categoryWeights = artstarWeight.getCategoryWeights();
        
        // Calculate fuzzy min (intersection)
        double intersection = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            intersection += Math.min(input.get(i), categoryWeights[i]);
        }
        
        // Calculate category magnitude
        double categoryMagnitude = 0.0;
        for (int i = 0; i < categoryWeights.length; i++) {
            categoryMagnitude += categoryWeights[i];
        }
        
        // Base activation using choice function
        double baseActivation = intersection / (artstarParams.alpha() + categoryMagnitude);
        
        // Apply stability regulation - more stable categories get boosted activation
        double stabilityBoost = artstarWeight.getStabilityMeasure() * artstarParams.stabilityFactor();
        double regulatedActivation = baseActivation * (1.0 + stabilityBoost);
        
        // Apply category strength weighting
        return regulatedActivation * artstarWeight.getStrength();
    }
    
    @Override
    protected MatchResult checkVigilance(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTSTARWeight artstarWeight)) {
            throw new IllegalArgumentException("Weight must be ARTSTARWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTSTARParameters artstarParams)) {
            throw new IllegalArgumentException("Parameters must be ARTSTARParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var categoryWeights = artstarWeight.getCategoryWeights();
        
        // Calculate fuzzy min (intersection)
        double intersection = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            intersection += Math.min(input.get(i), categoryWeights[i]);
        }
        
        // Calculate input magnitude
        double inputMagnitude = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            inputMagnitude += input.get(i);
        }
        
        // Avoid division by zero
        if (inputMagnitude == 0.0) {
            double categorySum = 0.0;
            for (double w : categoryWeights) {
                categorySum += w;
            }
            boolean isAccepted = categorySum == 0.0;
            double matchValue = isAccepted ? 1.0 : 0.0;
            return isAccepted ? new MatchResult.Accepted(matchValue, artstarParams.vigilance()) :
                               new MatchResult.Rejected(matchValue, artstarParams.vigilance());
        }
        
        // Calculate match ratio
        double matchRatio = intersection / inputMagnitude;
        
        // Dynamic vigilance based on regulation state
        double effectiveVigilance = artstarParams.getEffectiveVigilance(
            networkStability, networkAdaptability);
        
        // Apply stability/adaptability bias to vigilance test
        double stabilityBias = artstarWeight.getStabilityMeasure() * artstarParams.stabilityFactor();
        double adaptabilityBias = artstarWeight.getAdaptabilityMeasure() * artstarParams.adaptabilityFactor();
        double regulationBias = (stabilityBias - adaptabilityBias) * 0.1; // Small adjustment
        
        double adjustedVigilance = Math.max(0.0, Math.min(1.0, effectiveVigilance + regulationBias));
        
        // Test against adjusted vigilance threshold
        boolean isAccepted = matchRatio >= adjustedVigilance;
        return isAccepted ? new MatchResult.Accepted(matchRatio, adjustedVigilance) :
                           new MatchResult.Rejected(matchRatio, adjustedVigilance);
    }
    
    @Override
    protected WeightVector updateWeights(Vector input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(currentWeight instanceof ARTSTARWeight artstarWeight)) {
            throw new IllegalArgumentException("Weight must be ARTSTARWeight, got: " + currentWeight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTSTARParameters artstarParams)) {
            throw new IllegalArgumentException("Parameters must be ARTSTARParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var currentCategoryWeights = artstarWeight.getCategoryWeights();
        var newCategoryWeights = new double[currentCategoryWeights.length];
        
        // Update category weights using fuzzy ART learning rule with stability regulation
        double effectiveLearningRate = artstarParams.beta() * 
            (1.0 - artstarWeight.getStabilityMeasure() * 0.5); // Stability reduces learning rate
        
        for (int i = 0; i < newCategoryWeights.length; i++) {
            double fuzzyMin = Math.min(input.get(i), currentCategoryWeights[i]);
            newCategoryWeights[i] = effectiveLearningRate * fuzzyMin + 
                                   (1.0 - effectiveLearningRate) * currentCategoryWeights[i];
        }
        
        // Update regulation measures based on learning success
        double stabilityUpdate = calculateStabilityUpdate(input, artstarWeight, artstarParams);
        double adaptabilityUpdate = calculateAdaptabilityUpdate(input, artstarWeight, artstarParams);
        
        double newStability = updateRegulationMeasure(artstarWeight.getStabilityMeasure(), 
                                                     stabilityUpdate, artstarParams.regulationRate());
        double newAdaptability = updateRegulationMeasure(artstarWeight.getAdaptabilityMeasure(),
                                                        adaptabilityUpdate, artstarParams.regulationRate());
        
        // Update strength based on successful learning
        double strengthIncrease = 0.01; // Small boost for successful learning
        double newStrength = Math.min(1.0, artstarWeight.getStrength() + strengthIncrease);
        
        // Update usage and timestamp
        long currentTime = System.currentTimeMillis();
        
        return new ARTSTARWeight(newCategoryWeights, newStability, newAdaptability,
                                artstarWeight.getUsageCount() + 1, currentTime, newStrength);
    }
    
    @Override
    protected WeightVector createInitialWeight(Vector input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ARTSTARParameters artstarParams)) {
            throw new IllegalArgumentException("Parameters must be ARTSTARParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        // Initialize category weights to input (complement coded if needed)
        var categoryWeights = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            categoryWeights[i] = input.get(i);
        }
        
        // Initialize regulation measures to balanced state
        double initialStability = networkStability;
        double initialAdaptability = networkAdaptability;
        double initialStrength = 1.0; // New categories start strong
        
        long currentTime = System.currentTimeMillis();
        
        return new ARTSTARWeight(categoryWeights, initialStability, initialAdaptability,
                                1, currentTime, initialStrength);
    }
    
    /**
     * Calculate stability measure update based on learning consistency.
     */
    private double calculateStabilityUpdate(Vector input, ARTSTARWeight weight, ARTSTARParameters params) {
        // Calculate how similar the input is to the current category
        double similarity = weight.calculateSimilarity(input);
        
        // High similarity increases stability (pattern is consistent)
        // Low similarity decreases stability (pattern is changing)
        return similarity - 0.5; // Range: [-0.5, +0.5]
    }
    
    /**
     * Calculate adaptability measure update based on learning novelty.
     */
    private double calculateAdaptabilityUpdate(Vector input, ARTSTARWeight weight, ARTSTARParameters params) {
        // Calculate pattern novelty (inverse of similarity)
        double similarity = weight.calculateSimilarity(input);
        double novelty = 1.0 - similarity;
        
        // High novelty increases adaptability (willing to learn new patterns)
        // Low novelty decreases adaptability (pattern is familiar)
        return novelty - 0.5; // Range: [-0.5, +0.5]
    }
    
    /**
     * Update regulation measure with exponential smoothing.
     */
    private double updateRegulationMeasure(double current, double update, double rate) {
        double newValue = current + rate * update;
        return Math.max(0.0, Math.min(1.0, newValue));
    }
    
    /**
     * Perform network-level regulation updates.
     */
    public void updateNetworkRegulation(ARTSTARParameters params) {
        long currentTime = System.currentTimeMillis();
        
        // Update network stability/adaptability based on learning success
        if (totalLearningEvents > 0) {
            double successRate = (double) successfulLearningEvents / totalLearningEvents;
            
            // High success rate increases stability (network is learning well)
            networkStability = updateRegulationMeasure(networkStability, 
                                                      successRate - 0.5, params.regulationRate());
            
            // Moderate success rate maintains adaptability
            double adaptabilityTarget = 0.8 - (successRate * 0.3); // Inverse relationship
            networkAdaptability = updateRegulationMeasure(networkAdaptability,
                                                         adaptabilityTarget - networkAdaptability, 
                                                         params.regulationRate());
        }
        
        // Apply category decay and pruning
        applyCategoryDecay(params);
        pruneLowStrengthCategories(params);
        enforceCategoryLimits(params);
        
        lastRegulationUpdate = currentTime;
    }
    
    /**
     * Apply time-based decay to all categories.
     */
    private void applyCategoryDecay(ARTSTARParameters params) {
        if (params.categoryDecayRate() <= 0.0) return;
        
        var decayedCategories = new ArrayList<WeightVector>();
        
        for (var category : getCategories()) {
            if (category instanceof ARTSTARWeight artstarWeight) {
                double decayedStrength = artstarWeight.getDecayedStrength(params.categoryDecayRate());
                if (decayedStrength >= params.minCategoryStrength()) {
                    decayedCategories.add(artstarWeight.withStrength(decayedStrength));
                }
                // Categories below minimum strength are excluded (effectively pruned)
            } else {
                decayedCategories.add(category); // Non-ARTSTAR weights pass through
            }
        }
        
        // Replace categories with decayed versions
        replaceCategories(decayedCategories);
    }
    
    /**
     * Remove categories with strength below threshold.
     */
    private void pruneLowStrengthCategories(ARTSTARParameters params) {
        var strongCategories = new ArrayList<WeightVector>();
        
        for (var category : getCategories()) {
            if (category instanceof ARTSTARWeight artstarWeight) {
                if (!artstarWeight.isWeak(params.minCategoryStrength())) {
                    strongCategories.add(category);
                }
            } else {
                strongCategories.add(category); // Non-ARTSTAR weights pass through
            }
        }
        
        if (strongCategories.size() < getCategories().size()) {
            replaceCategories(strongCategories);
        }
    }
    
    /**
     * Enforce maximum category limits by pruning weakest categories.
     */
    private void enforceCategoryLimits(ARTSTARParameters params) {
        if (params.maxCategories() <= 0 || getCategoryCount() <= params.maxCategories()) {
            return; // No limit or within limit
        }
        
        // Sort categories by strength (descending) and keep the strongest
        var sortedCategories = new ArrayList<>(getCategories());
        sortedCategories.sort((a, b) -> {
            if (a instanceof ARTSTARWeight wa && b instanceof ARTSTARWeight wb) {
                return Double.compare(wb.getStrength(), wa.getStrength());
            }
            return 0; // Non-ARTSTAR weights maintain order
        });
        
        var limitedCategories = new ArrayList<>(sortedCategories.subList(0, params.maxCategories()));
        replaceCategories(limitedCategories);
    }
    
    /**
     * Replace all categories with new list.
     */
    private void replaceCategories(List<WeightVector> newCategories) {
        replaceAllCategories(newCategories);
    }
    
    /**
     * Perform a learning step with regulation tracking.
     * This method should be called instead of stepFit to get ARTSTAR regulation benefits.
     */
    public ActivationResult stepFitWithRegulation(Vector input, ARTSTARParameters parameters) {
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        totalLearningEvents++;
        
        // Perform standard ART learning step
        var result = stepFit(input, parameters);
        
        // Track learning success
        if (result instanceof ActivationResult.Success) {
            successfulLearningEvents++;
        }
        
        // Periodically update network regulation
        long timeSinceLastUpdate = System.currentTimeMillis() - lastRegulationUpdate;
        if (timeSinceLastUpdate > 60000) { // Update every minute
            updateNetworkRegulation(parameters);
        }
        
        return result;
    }
    
    // Analysis and introspection methods
    
    /**
     * Get current network regulation state.
     */
    public RegulationAnalysis analyzeRegulationState() {
        double averageStability = getCategories().stream()
            .filter(ARTSTARWeight.class::isInstance)
            .mapToDouble(w -> ((ARTSTARWeight) w).getStabilityMeasure())
            .average().orElse(0.0);
            
        double averageAdaptability = getCategories().stream()
            .filter(ARTSTARWeight.class::isInstance)
            .mapToDouble(w -> ((ARTSTARWeight) w).getAdaptabilityMeasure())
            .average().orElse(0.0);
            
        double averageStrength = getCategories().stream()
            .filter(ARTSTARWeight.class::isInstance)
            .mapToDouble(w -> ((ARTSTARWeight) w).getStrength())
            .average().orElse(0.0);
            
        long averageUsage = Math.round(getCategories().stream()
            .filter(ARTSTARWeight.class::isInstance)
            .mapToLong(w -> ((ARTSTARWeight) w).getUsageCount())
            .average().orElse(0.0));
        
        double learningSuccessRate = totalLearningEvents > 0 ? 
            (double) successfulLearningEvents / totalLearningEvents : 0.0;
        
        return new RegulationAnalysis(networkStability, networkAdaptability,
                                     averageStability, averageAdaptability, averageStrength,
                                     averageUsage, learningSuccessRate, getCategoryCount());
    }
    
    /**
     * Get regulation data for specific category.
     */
    public ARTSTARWeight getARTSTARCategory(int index) {
        var category = getCategory(index);
        if (!(category instanceof ARTSTARWeight artstarWeight)) {
            throw new IllegalArgumentException("Category " + index + " is not an ARTSTARWeight");
        }
        return artstarWeight;
    }
    
    @Override
    public String toString() {
        return String.format("ARTSTAR{categories=%d, stability=%.3f, adaptability=%.3f, success=%.3f}",
                           getCategoryCount(), networkStability, networkAdaptability,
                           totalLearningEvents > 0 ? (double) successfulLearningEvents / totalLearningEvents : 0.0);
    }
    
    /**
     * Analysis result for ARTSTAR regulation state.
     */
    public record RegulationAnalysis(
        double networkStability,
        double networkAdaptability,
        double averageCategoryStability,
        double averageCategoryAdaptability,
        double averageCategoryStrength,
        long averageUsageCount,
        double learningSuccessRate,
        int totalCategories
    ) {
        @Override
        public String toString() {
            return String.format("RegulationAnalysis{netStab=%.3f, netAdapt=%.3f, " +
                               "avgStab=%.3f, avgAdapt=%.3f, avgStr=%.3f, avgUsage=%d, " +
                               "success=%.3f, categories=%d}",
                               networkStability, networkAdaptability,
                               averageCategoryStability, averageCategoryAdaptability,
                               averageCategoryStrength, averageUsageCount,
                               learningSuccessRate, totalCategories);
        }
    }
}