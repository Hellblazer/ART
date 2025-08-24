package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * ART-A (Attentional ART) implementation with dynamic attention weighting.
 * 
 * ART-A extends traditional ART with attention mechanisms that dynamically
 * weight input features based on their discriminative power for each category.
 * This allows the network to focus on the most important features and ignore
 * irrelevant ones, improving learning efficiency and category separation.
 * 
 * Key features:
 * - Attention-weighted activation computation
 * - Dynamic attention weight learning based on category discrimination
 * - Attention-based vigilance testing  
 * - Feature importance analysis through attention weights
 * 
 * Algorithm components:
 * 1. Attention-weighted choice function for category activation
 * 2. Attention-modified vigilance test for category acceptance
 * 3. Joint learning of category weights and attention weights
 * 4. Attention weight evolution based on discriminative features
 */
public final class ARTA extends BaseART {
    
    /**
     * Create a new ART-A instance with no initial categories.
     */
    public ARTA() {
        super();
    }
    
    /**
     * Create a new ART-A instance with initial categories.
     * @param initialCategories the initial ART-A weight categories (will be copied)
     */
    public ARTA(java.util.List<ARTAWeight> initialCategories) {
        super(Objects.requireNonNull(initialCategories, "Initial categories cannot be null"));
    }
    
    @Override
    protected double calculateActivation(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null"); 
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTAWeight artaWeight)) {
            throw new IllegalArgumentException("Weight must be ARTAWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTAParameters artaParams)) {
            throw new IllegalArgumentException("Parameters must be ARTAParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        // Validate dimensions
        if (input.dimension() != artaWeight.dimension()) {
            throw new IllegalArgumentException("Input dimension (" + input.dimension() + 
                ") must match weight dimension (" + artaWeight.dimension() + ")");
        }
        
        // ART-A attention-weighted choice function:
        // T_j = |I ∧ w_j| / (α + |w_j|) where ∧ uses attention weighting
        
        var categoryWeights = artaWeight.getCategoryWeights();
        var attentionWeights = artaWeight.getAttentionWeights();
        
        // Calculate attention-weighted fuzzy min (intersection)
        double intersectionSum = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            double fuzzyMin = Math.min(input.get(i), categoryWeights[i]);
            intersectionSum += attentionWeights[i] * fuzzyMin;  // Attention weighting
        }
        
        // Calculate attention-weighted category magnitude
        double categoryMagnitude = 0.0;
        for (int i = 0; i < categoryWeights.length; i++) {
            categoryMagnitude += attentionWeights[i] * categoryWeights[i];  // Attention weighting  
        }
        
        // Choice function with attention weighting
        double activation = intersectionSum / (artaParams.alpha() + categoryMagnitude);
        
        return activation;
    }
    
    @Override
    protected MatchResult checkVigilance(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof ARTAWeight artaWeight)) {
            throw new IllegalArgumentException("Weight must be ARTAWeight, got: " + weight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTAParameters artaParams)) {
            throw new IllegalArgumentException("Parameters must be ARTAParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var categoryWeights = artaWeight.getCategoryWeights();
        var attentionWeights = artaWeight.getAttentionWeights();
        
        // ART-A attention-weighted match function:
        // ρ_j = |I ∧ w_j| / |I| where both operations use attention weighting
        
        // Calculate attention-weighted fuzzy min (intersection)
        double intersectionSum = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            double fuzzyMin = Math.min(input.get(i), categoryWeights[i]);
            intersectionSum += attentionWeights[i] * fuzzyMin;
        }
        
        // Calculate attention-weighted input magnitude
        double inputMagnitude = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            inputMagnitude += attentionWeights[i] * input.get(i);
        }
        
        // Avoid division by zero
        if (inputMagnitude == 0.0) {
            // If attention-weighted input is zero, accept if category is also zero
            double categorySum = Arrays.stream(categoryWeights).sum();
            boolean isAccepted = categorySum == 0.0;
            double matchValue = isAccepted ? 1.0 : 0.0;
            return isAccepted ? new MatchResult.Accepted(matchValue, artaParams.vigilance()) : 
                               new MatchResult.Rejected(matchValue, artaParams.vigilance());
        }
        
        // Calculate match ratio with attention weighting
        double matchRatio = intersectionSum / inputMagnitude;
        
        // Test against vigilance threshold
        boolean isAccepted = matchRatio >= artaParams.vigilance();
        return isAccepted ? new MatchResult.Accepted(matchRatio, artaParams.vigilance()) : 
                           new MatchResult.Rejected(matchRatio, artaParams.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Vector input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(currentWeight instanceof ARTAWeight artaWeight)) {
            throw new IllegalArgumentException("Weight must be ARTAWeight, got: " + currentWeight.getClass().getSimpleName());
        }
        if (!(parameters instanceof ARTAParameters artaParams)) {
            throw new IllegalArgumentException("Parameters must be ARTAParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        var currentCategoryWeights = artaWeight.getCategoryWeights();
        var currentAttentionWeights = artaWeight.getAttentionWeights();
        
        // Update category weights using fuzzy min learning rule (like FuzzyART)
        var newCategoryWeights = new double[currentCategoryWeights.length];
        for (int i = 0; i < currentCategoryWeights.length; i++) {
            double fuzzyMin = Math.min(input.get(i), currentCategoryWeights[i]);
            newCategoryWeights[i] = artaParams.beta() * fuzzyMin + 
                                   (1.0 - artaParams.beta()) * currentCategoryWeights[i];
        }
        
        // Update attention weights based on discriminative power
        var newAttentionWeights = updateAttentionWeights(
            input, artaWeight, artaParams, newCategoryWeights
        );
        
        return new ARTAWeight(newCategoryWeights, newAttentionWeights);
    }
    
    @Override
    protected WeightVector createInitialWeight(Vector input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ARTAParameters artaParams)) {
            throw new IllegalArgumentException("Parameters must be ARTAParameters, got: " + parameters.getClass().getSimpleName());
        }
        
        // Initialize category weights to input (complement coded)
        var categoryWeights = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            categoryWeights[i] = input.get(i);
        }
        
        // Initialize attention weights uniformly (equal attention to all features initially)
        var initialAttentionWeight = Math.max(artaParams.minAttentionWeight(), 0.5);
        var attentionWeights = new double[input.dimension()];
        Arrays.fill(attentionWeights, initialAttentionWeight);
        
        return new ARTAWeight(categoryWeights, attentionWeights);
    }
    
    /**
     * Update attention weights based on discriminative power of features.
     * Features that help distinguish this category from others get higher attention.
     */
    private double[] updateAttentionWeights(Vector input, ARTAWeight currentWeight, 
                                           ARTAParameters parameters, double[] newCategoryWeights) {
        
        var currentAttentionWeights = currentWeight.getAttentionWeights();
        var newAttentionWeights = new double[currentAttentionWeights.length];
        
        // For each dimension, calculate how discriminative it is
        for (int i = 0; i < currentAttentionWeights.length; i++) {
            
            // Calculate feature discriminability:
            // Features with high variance across categories should get more attention
            // Features that are consistent within this category should get more attention
            
            // Simple discriminability measure: difference between input and category prototype
            double featureDifference = Math.abs(input.get(i) - newCategoryWeights[i]);
            
            // Discriminability: features with small differences are more characteristic
            double discriminability = 1.0 / (1.0 + featureDifference);
            
            // Update attention weight using learning rule
            double targetAttentionWeight = discriminability;
            
            // Ensure minimum attention weight
            targetAttentionWeight = Math.max(targetAttentionWeight, parameters.minAttentionWeight());
            
            // Apply attention learning rate
            newAttentionWeights[i] = parameters.attentionLearningRate() * targetAttentionWeight + 
                                    (1.0 - parameters.attentionLearningRate()) * currentAttentionWeights[i];
            
            // Clamp to valid range
            newAttentionWeights[i] = Math.max(parameters.minAttentionWeight(), 
                                            Math.min(1.0, newAttentionWeights[i]));
        }
        
        return newAttentionWeights;
    }
    
    /**
     * Calculate attention-weighted similarity between input and category.
     * This provides insight into how well the input matches the category
     * when attention weighting is applied.
     * 
     * @param input the input vector
     * @param categoryIndex the category index to compare against
     * @return attention-weighted similarity (0 to 1)
     */
    public double getAttentionWeightedSimilarity(Vector input, int categoryIndex) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        
        if (categoryIndex < 0 || categoryIndex >= getCategoryCount()) {
            throw new IndexOutOfBoundsException("Category index " + categoryIndex + 
                " out of bounds for " + getCategoryCount() + " categories");
        }
        
        var weight = getCategory(categoryIndex);
        if (!(weight instanceof ARTAWeight artaWeight)) {
            throw new IllegalStateException("Expected ARTAWeight, got: " + weight.getClass().getSimpleName());
        }
        
        return artaWeight.attentionWeightedSimilarity(input);
    }
    
    /**
     * Get the attention weights for a specific category.
     * This shows which features the network is focusing on for this category.
     * 
     * @param categoryIndex the category index
     * @return copy of the attention weights array
     */
    public double[] getAttentionWeights(int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= getCategoryCount()) {
            throw new IndexOutOfBoundsException("Category index " + categoryIndex + 
                " out of bounds for " + getCategoryCount() + " categories");
        }
        
        var weight = getCategory(categoryIndex);
        if (!(weight instanceof ARTAWeight artaWeight)) {
            throw new IllegalStateException("Expected ARTAWeight, got: " + weight.getClass().getSimpleName());
        }
        
        return artaWeight.getAttentionWeights();
    }
    
    /**
     * Analyze the attention distribution across all categories.
     * Returns statistics about which features are most attended to globally.
     * 
     * @return attention analysis with feature importance rankings
     */
    public AttentionAnalysis analyzeAttentionDistribution() {
        if (getCategoryCount() == 0) {
            return new AttentionAnalysis(new double[0], new int[0], new double[0]);
        }
        
        var firstCategory = getCategory(0);
        if (!(firstCategory instanceof ARTAWeight)) {
            throw new IllegalStateException("Expected ARTAWeight categories");
        }
        
        int dimensions = firstCategory.dimension();
        var meanAttention = new double[dimensions];
        var maxAttention = new double[dimensions];
        
        // Calculate mean and max attention per dimension
        for (int categoryIdx = 0; categoryIdx < getCategoryCount(); categoryIdx++) {
            var weight = (ARTAWeight) getCategory(categoryIdx);
            var attention = weight.getAttentionWeights();
            
            for (int dim = 0; dim < dimensions; dim++) {
                meanAttention[dim] += attention[dim];
                maxAttention[dim] = Math.max(maxAttention[dim], attention[dim]);
            }
        }
        
        // Normalize mean attention
        for (int dim = 0; dim < dimensions; dim++) {
            meanAttention[dim] /= getCategoryCount();
        }
        
        // Rank features by mean attention (descending order)
        var featureRanking = new int[dimensions];
        for (int i = 0; i < dimensions; i++) {
            featureRanking[i] = i;
        }
        
        // Sort by mean attention (bubble sort for simplicity)
        for (int i = 0; i < dimensions - 1; i++) {
            for (int j = 0; j < dimensions - i - 1; j++) {
                if (meanAttention[featureRanking[j]] < meanAttention[featureRanking[j + 1]]) {
                    int temp = featureRanking[j];
                    featureRanking[j] = featureRanking[j + 1];
                    featureRanking[j + 1] = temp;
                }
            }
        }
        
        return new AttentionAnalysis(meanAttention, featureRanking, maxAttention);
    }
    
    /**
     * Record containing attention analysis results.
     */
    public record AttentionAnalysis(
        double[] meanAttentionPerFeature,
        int[] featureRanking,  // Features ranked by mean attention (descending)
        double[] maxAttentionPerFeature
    ) {
        public AttentionAnalysis {
            Objects.requireNonNull(meanAttentionPerFeature);
            Objects.requireNonNull(featureRanking);
            Objects.requireNonNull(maxAttentionPerFeature);
        }
        
        @Override
        public String toString() {
            var sb = new StringBuilder();
            sb.append("AttentionAnalysis{features=").append(meanAttentionPerFeature.length);
            sb.append(", top3=[");
            
            int showFeatures = Math.min(3, featureRanking.length);
            for (int i = 0; i < showFeatures; i++) {
                if (i > 0) sb.append(", ");
                int featureIdx = featureRanking[i];
                sb.append(String.format("f%d:%.3f", featureIdx, meanAttentionPerFeature[featureIdx]));
            }
            sb.append("]}");
            
            return sb.toString();
        }
    }
}