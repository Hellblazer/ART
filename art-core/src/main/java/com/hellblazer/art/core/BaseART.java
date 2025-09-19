package com.hellblazer.art.core;

import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.CategoryResult;
import com.hellblazer.art.core.results.MatchResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Abstract base class implementing the template method pattern for ART algorithms.
 * Provides the common ART algorithm structure while allowing specific implementations
 * to customize activation, vigilance, and learning behaviors.
 * 
 * @param <P> the type of parameters used by this ART algorithm
 */
public abstract class BaseART<P> implements ARTAlgorithm<P> {
    
    protected final List<WeightVector> categories;
    protected final List<Long> categoryUsageCount;
    protected final List<Long> categoryLastUsedTimestamp;
    protected long totalActivations = 0;
    
    /**
     * Create a new BaseART instance with no initial categories.
     */
    protected BaseART() {
        this.categories = new ArrayList<>();
        this.categoryUsageCount = new ArrayList<>();
        this.categoryLastUsedTimestamp = new ArrayList<>();
    }
    
    /**
     * Create a new BaseART instance with initial categories.
     * @param initialCategories the initial categories (will be copied)
     */
    protected BaseART(List<? extends WeightVector> initialCategories) {
        Objects.requireNonNull(initialCategories, "Initial categories cannot be null");
        this.categories = new ArrayList<>(initialCategories);
        this.categoryUsageCount = new ArrayList<>();
        this.categoryLastUsedTimestamp = new ArrayList<>();
        var currentTime = System.currentTimeMillis();
        for (int i = 0; i < initialCategories.size(); i++) {
            this.categoryUsageCount.add(0L);
            this.categoryLastUsedTimestamp.add(currentTime);
        }
    }
    
    /**
     * Main template method implementing the complete ART algorithm (Reference-compatible).
     * This method orchestrates the complete ART learning cycle with full reference parity compatibility:
     * 1. Handle empty category case
     * 2. Calculate activations with optional match reset filtering
     * 3. Test categories in activation order with NaN marking
     * 4. Apply match tracking on vigilance failures
     * 5. Update weights or create new category
     * 
     * @param input the input vector
     * @param parameters the algorithm parameters
     * @return the result of the activation process
     */
    public final ActivationResult stepFit(Pattern input, P parameters) {
        return stepFit(input, parameters, null, MatchTrackingMode.MT_PLUS, 0.0);
    }

    /**
     * Enhanced stepFit method for vectorized algorithms that need additional processing.
     * This is a convenience method that delegates to the standard stepFit method.
     * 
     * @param input the input vector
     * @param parameters the algorithm parameters
     * @return the result of the activation process
     */
    public final ActivationResult stepFitEnhanced(Pattern input, P parameters) {
        return stepFit(input, parameters);
    }
    
    /**
     * Train the network with a single pattern (modern API).
     * This is an alias for stepFit() to match the ARTAlgorithm interface.
     * 
     * @param input the input pattern to learn
     * @param parameters the algorithm-specific parameters
     * @return the result of the learning step
     */
    @Override
    public final ActivationResult learn(Pattern input, P parameters) {
        return stepFit(input, parameters);
    }
    
    /**
     * Predict the category for a pattern (modern API).
     * This is an alias for stepPredict() to match the ARTAlgorithm interface.
     * 
     * @param input the input pattern to classify
     * @param parameters the algorithm-specific parameters  
     * @return the prediction result
     */
    @Override
    public final ActivationResult predict(Pattern input, P parameters) {
        return stepPredict(input, parameters);
    }

    /**
     * Complete step_fit implementation matching reference parity exactly.
     * 
     * @param input the input vector
     * @param parameters the algorithm parameters  
     * @param matchResetFunc optional match reset function
     * @param matchTracking match tracking mode
     * @param epsilon epsilon parameter for match tracking
     * @return the result of the activation process
     */
    public final ActivationResult stepFit(
            Pattern input, 
            P parameters, 
            MatchResetFunction matchResetFunc,
            MatchTrackingMode matchTracking,
            double epsilon) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Step 1: Handle empty categories - create first category
        if (categories.isEmpty()) {
            synchronized (categories) {
                // Double-check after acquiring lock
                if (categories.isEmpty()) {
                    var newWeight = createInitialWeight(input, parameters);
                    categories.add(newWeight);
                    categoryUsageCount.add(1L);
                    categoryLastUsedTimestamp.add(System.currentTimeMillis());
                    totalActivations++;
                    return new ActivationResult.Success(0, 1.0, newWeight);
                }
            }
        }
        
        // Step 2: Calculate activations, applying match reset filtering if specified
        var activations = new Double[categories.size()];
        var caches = new ActivationCache[categories.size()];
        
        for (int i = 0; i < categories.size(); i++) {
            var weight = categories.get(i);
            
            // Apply match reset function if provided (Python MT~ mode)
            if (matchTracking == MatchTrackingMode.MT_COMPLEMENT && matchResetFunc != null) {
                boolean shouldConsider = matchResetFunc.shouldConsiderCategory(
                    input, weight, i, parameters, java.util.Optional.empty());
                if (!shouldConsider) {
                    activations[i] = Double.NaN;
                    caches[i] = ActivationCache.empty("skipped");
                    continue;
                }
            }
            
            // Calculate activation with caching
            var result = calculateActivationWithCache(input, weight, parameters);
            activations[i] = result.activation();
            caches[i] = result.cache();
        }
        
        // Step 3: Python-style iterative category testing with NaN marking
        var baseParams = deepCopyParams(parameters);
        var mtOperator = matchTracking.getOperator();
        
        while (hasValidActivation(activations)) {
            // Find category with highest valid activation (nanargmax equivalent)
            int bestCategory = findBestValidCategory(activations);
            var weight = categories.get(bestCategory);
            var cache = caches[bestCategory];
            
            // Test match criterion (vigilance) with caching
            var matchResult = checkVigilanceWithCache(input, weight, parameters, cache, mtOperator);
            
            // Apply match reset logic
            boolean noMatchReset = matchResetFunc == null || 
                (matchTracking != MatchTrackingMode.MT_COMPLEMENT && 
                 matchResetFunc.shouldConsiderCategory(input, weight, bestCategory, parameters, 
                     matchResult.cache().getData()));
            
            if (matchResult.result().isAccepted() && noMatchReset) {
                // Success: update weight and return
                var updatedWeight = updateWeightsWithCache(input, weight, parameters, matchResult.cache());
                categories.set(bestCategory, updatedWeight);
                // Update usage statistics
                categoryUsageCount.set(bestCategory, categoryUsageCount.get(bestCategory) + 1);
                categoryLastUsedTimestamp.set(bestCategory, System.currentTimeMillis());
                totalActivations++;
                restoreParams(baseParams, parameters);
                return new ActivationResult.Success(bestCategory, activations[bestCategory], updatedWeight);
            } else {
                // Mark this category as tested (Python: T[c_] = np.nan)
                activations[bestCategory] = Double.NaN;
                
                // Apply match tracking if vigilance passed but match reset failed
                if (matchResult.result().isAccepted() && !noMatchReset) {
                    boolean keepSearching = applyMatchTracking(
                        matchResult.cache(), epsilon, parameters, matchTracking);
                    if (!keepSearching) {
                        // Stop searching all categories (Python: T[:] = np.nan)
                        for (int i = 0; i < activations.length; i++) {
                            activations[i] = Double.NaN;
                        }
                    }
                }
            }
        }
        
        // Step 4: All categories failed - create new category
        synchronized (categories) {
            var newWeight = createInitialWeight(input, parameters);
            categories.add(newWeight);
            categoryUsageCount.add(1L);
            categoryLastUsedTimestamp.add(System.currentTimeMillis());
            totalActivations++;
            var newIndex = categories.size() - 1;
            restoreParams(baseParams, parameters);
            return new ActivationResult.Success(newIndex, 1.0, newWeight);
        }
    }
    
    // ==================== PYTHON-COMPATIBLE HELPER METHODS ====================
    
    /**
     * Check if there are any valid (non-NaN) activations remaining.
     */
    private boolean hasValidActivation(Double[] activations) {
        for (Double activation : activations) {
            if (!Double.isNaN(activation)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Find the index of the highest valid (non-NaN) activation (nanargmax equivalent).
     */
    private int findBestValidCategory(Double[] activations) {
        int bestIndex = -1;
        double bestActivation = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < activations.length; i++) {
            if (!Double.isNaN(activations[i]) && activations[i] > bestActivation) {
                bestActivation = activations[i];
                bestIndex = i;
            }
        }
        
        if (bestIndex == -1) {
            throw new IllegalStateException("No valid activation found");
        }
        
        return bestIndex;
    }
    
    /**
     * Record for activation result with cache.
     */
    public record ActivationWithCache(double activation, ActivationCache cache) {}
    
    /**
     * Record for vigilance result with cache.
     */
    public record VigilanceWithCache(MatchResult result, ActivationCache cache) {}
    
    // ==================== CACHED COMPUTATION METHODS ====================
    
    /**
     * Calculate activation with caching support.
     */
    protected ActivationWithCache calculateActivationWithCache(
            Pattern input, WeightVector weight, P parameters) {
        double activation = calculateActivation(input, weight, parameters);
        var cache = ActivationCache.empty(getAlgorithmName());
        return new ActivationWithCache(activation, cache);
    }
    
    /**
     * Check vigilance with caching and match tracking operator.
     */
    protected VigilanceWithCache checkVigilanceWithCache(
            Pattern input, WeightVector weight, P parameters, 
            ActivationCache cache, java.util.function.BinaryOperator<Double> mtOperator) {
        var result = checkVigilance(input, weight, parameters);
        return new VigilanceWithCache(result, cache);
    }
    
    /**
     * Update weights with caching support.
     */
    protected WeightVector updateWeightsWithCache(
            Pattern input, WeightVector currentWeight, P parameters, ActivationCache cache) {
        return updateWeights(input, currentWeight, parameters);
    }
    
    // ==================== PARAMETER MANAGEMENT ====================
    
    /**
     * Create a deep copy of parameters for restoration after match tracking.
     */
    protected P deepCopyParams(P parameters) {
        // Default implementation returns the same object (assuming immutable)
        // Subclasses should override if parameters are mutable
        return parameters;
    }
    
    /**
     * Restore parameters from a saved copy.
     */
    protected void restoreParams(P savedParams, P currentParams) {
        // Default implementation does nothing (assuming immutable parameters)
        // Subclasses should override if parameters are mutable
    }
    
    /**
     * Apply match tracking logic.
     */
    protected boolean applyMatchTracking(
            ActivationCache cache, double epsilon, P parameters, MatchTrackingMode mode) {
        // Default implementation: always continue searching
        // Subclasses can override for specific match tracking behavior
        return true;
    }
    
    /**
     * Get the algorithm name for cache identification.
     */
    protected String getAlgorithmName() {
        return getClass().getSimpleName();
    }
    
    // ==================== ABSTRACT METHODS ====================
    
    /**
     * Public method to get activation value for a specific category.
     * This allows external algorithms (like SMART) to access proper activation calculations.
     * 
     * @param input the input vector
     * @param categoryIndex the index of the category to calculate activation for
     * @param parameters the algorithm-specific parameters
     * @return the activation value (higher means better match)
     * @throws IndexOutOfBoundsException if categoryIndex is invalid
     */
    public double getActivationValue(Pattern input, int categoryIndex, P parameters) {
        if (categoryIndex < 0 || categoryIndex >= categories.size()) {
            throw new IndexOutOfBoundsException("Category index " + categoryIndex + 
                " out of bounds for " + categories.size() + " categories");
        }
        return calculateActivation(input, categories.get(categoryIndex), parameters);
    }
    
    /**
     * Calculate the activation value for a specific category given an input.
     * This is algorithm-specific (e.g., choice function in FuzzyART).
     * 
     * @param input the input vector
     * @param weight the category weight vector
     * @param parameters the algorithm parameters
     * @return the activation value for this category
     */
    protected abstract double calculateActivation(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Test whether the input matches the category well enough according to vigilance.
     * This is algorithm-specific (e.g., vigilance test in FuzzyART).
     * 
     * @param input the input vector
     * @param weight the category weight vector
     * @param parameters the algorithm parameters
     * @return the match result (accepted or rejected)
     */
    protected abstract MatchResult checkVigilance(Pattern input, WeightVector weight, P parameters);
    
    /**
     * Update the category weight based on the input using the learning rule.
     * This is algorithm-specific (e.g., fuzzy min learning in FuzzyART).
     * 
     * @param input the input vector
     * @param currentWeight the current category weight
     * @param parameters the algorithm parameters
     * @return the updated weight vector
     */
    protected abstract WeightVector updateWeights(Pattern input, WeightVector currentWeight, P parameters);
    
    /**
     * Create an initial weight vector for a new category based on the input.
     * This is algorithm-specific (e.g., complement coding initialization in FuzzyART).
     * 
     * @param input the input vector that will become the first example of this category
     * @param parameters the algorithm parameters
     * @return the initial weight vector for the new category
     */
    protected abstract WeightVector createInitialWeight(Pattern input, P parameters);
    
    /**
     * Find the winner among categories using winner-take-all competition.
     * Default implementation selects the category with highest activation.
     * 
     * @param activations the activation values for all categories
     * @return the winner result with index and activation value
     */
    protected CategoryResult findWinner(double[] activations) {
        if (activations.length == 0) {
            throw new IllegalArgumentException("Cannot find winner with no categories");
        }
        
        int winnerIndex = 0;
        double maxActivation = activations[0];
        
        for (int i = 1; i < activations.length; i++) {
            if (activations[i] > maxActivation) {
                maxActivation = activations[i];
                winnerIndex = i;
            }
        }
        
        return CategoryResult.of(winnerIndex, categories, activations);
    }
    
    /**
     * Get an unmodifiable view of the categories.
     * @return unmodifiable list of category weight vectors
     */
    public final List<WeightVector> getCategories() {
        return Collections.unmodifiableList(categories);
    }
    
    /**
     * Get the number of categories.
     * @return the number of categories
     */
    public final int getCategoryCount() {
        return categories.size();
    }
    
    /**
     * Get a specific category weight by index.
     * @param index the category index
     * @return the weight vector for that category
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public final WeightVector getCategory(int index) {
        if (index < 0 || index >= categories.size()) {
            throw new IndexOutOfBoundsException("Category index " + index + 
                " out of bounds for " + categories.size() + " categories");
        }
        return categories.get(index);
    }
    
    /**
     * Clear all categories (reset the network).
     */
    public final void clear() {
        categories.clear();
        categoryUsageCount.clear();
        categoryLastUsedTimestamp.clear();
        totalActivations = 0;
    }
    
    /**
     * Clear all categories (alias for sklearn compatibility).
     */
    public final void clearCategories() {
        clear();
    }
    
    /**
     * Predict the category for a pattern without learning.
     * Returns the category with highest activation that passes vigilance.
     * 
     * @param input the input pattern
     * @param parameters the algorithm parameters
     * @return the prediction result
     */
    public final ActivationResult stepPredict(Pattern input, P parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (categories.isEmpty()) {
            return ActivationResult.NoMatch.instance();
        }
        
        // Calculate activations for all categories
        var activations = new Double[categories.size()];
        for (int i = 0; i < categories.size(); i++) {
            var weight = categories.get(i);
            var result = calculateActivationWithCache(input, weight, parameters);
            activations[i] = result.activation();
        }
        
        // Find best category
        int bestCategory = -1;
        double bestActivation = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < activations.length; i++) {
            if (!Double.isNaN(activations[i]) && activations[i] > bestActivation) {
                bestActivation = activations[i];
                bestCategory = i;
            }
        }
        
        if (bestCategory >= 0) {
            return new ActivationResult.Success(bestCategory, bestActivation, categories.get(bestCategory));
        } else {
            return ActivationResult.NoMatch.instance();
        }
    }
    
    /**
     * Replace all categories with a new list (for subclass use).
     * @param newCategories the new categories to replace with
     */
    protected final void replaceAllCategories(List<WeightVector> newCategories) {
        Objects.requireNonNull(newCategories, "New categories cannot be null");
        categories.clear();
        categories.addAll(newCategories);
        // Reset usage stats
        categoryUsageCount.clear();
        categoryLastUsedTimestamp.clear();
        var currentTime = System.currentTimeMillis();
        for (int i = 0; i < newCategories.size(); i++) {
            categoryUsageCount.add(0L);
            categoryLastUsedTimestamp.add(currentTime);
        }
    }
    
    /**
     * Get a string representation showing the number of categories.
     * @return string representation
     */
    @Override
    public String toString() {
        return getClass().getSimpleName() + "{categories=" + categories.size() + "}";
    }
    
    // ==================== CATEGORY PRUNING METHODS ====================
    
    /**
     * Prune categories based on usage frequency.
     * Removes categories that have been used less than the threshold percentage.
     * 
     * @param minUsageRatio minimum usage ratio (0.0 to 1.0) relative to mean usage
     * @return number of categories pruned
     */
    public int pruneByUsageFrequency(double minUsageRatio) {
        if (minUsageRatio < 0.0 || minUsageRatio > 1.0) {
            throw new IllegalArgumentException("minUsageRatio must be between 0.0 and 1.0");
        }
        if (categories.isEmpty()) {
            return 0;
        }
        
        // Calculate mean usage
        var meanUsage = categoryUsageCount.stream()
            .mapToLong(Long::longValue)
            .average()
            .orElse(0.0);
        
        var threshold = (long)(meanUsage * minUsageRatio);
        
        // Find indices to remove
        var indicesToRemove = new ArrayList<Integer>();
        for (int i = 0; i < categoryUsageCount.size(); i++) {
            if (categoryUsageCount.get(i) < threshold) {
                indicesToRemove.add(i);
            }
        }
        
        // Remove in reverse order to maintain indices
        for (int i = indicesToRemove.size() - 1; i >= 0; i--) {
            int idx = indicesToRemove.get(i);
            categories.remove(idx);
            categoryUsageCount.remove(idx);
            categoryLastUsedTimestamp.remove(idx);
        }
        
        return indicesToRemove.size();
    }
    
    /**
     * Prune categories based on age (time since last use).
     * 
     * @param maxAgeMillis maximum age in milliseconds since last use
     * @return number of categories pruned
     */
    public int pruneByAge(long maxAgeMillis) {
        if (maxAgeMillis <= 0) {
            throw new IllegalArgumentException("maxAgeMillis must be positive");
        }
        if (categories.isEmpty()) {
            return 0;
        }
        
        var currentTime = System.currentTimeMillis();
        var cutoffTime = currentTime - maxAgeMillis;
        
        // Find indices to remove
        var indicesToRemove = new ArrayList<Integer>();
        for (int i = 0; i < categoryLastUsedTimestamp.size(); i++) {
            if (categoryLastUsedTimestamp.get(i) < cutoffTime) {
                indicesToRemove.add(i);
            }
        }
        
        // Remove in reverse order
        for (int i = indicesToRemove.size() - 1; i >= 0; i--) {
            int idx = indicesToRemove.get(i);
            categories.remove(idx);
            categoryUsageCount.remove(idx);
            categoryLastUsedTimestamp.remove(idx);
        }
        
        return indicesToRemove.size();
    }
    
    /**
     * Prune categories to maintain a maximum count.
     * Keeps the most frequently used categories.
     * 
     * @param maxCategories maximum number of categories to keep
     * @return number of categories pruned
     */
    public int pruneToMaxSize(int maxCategories) {
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("maxCategories must be positive");
        }
        if (categories.size() <= maxCategories) {
            return 0;
        }
        
        // Create indices sorted by usage count
        var indices = new ArrayList<Integer>();
        for (int i = 0; i < categories.size(); i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Long.compare(
            categoryUsageCount.get(b), categoryUsageCount.get(a)));
        
        // Keep top maxCategories, remove the rest
        var toKeep = new ArrayList<WeightVector>();
        var toKeepUsage = new ArrayList<Long>();
        var toKeepTimestamp = new ArrayList<Long>();
        
        for (int i = 0; i < maxCategories; i++) {
            int idx = indices.get(i);
            toKeep.add(categories.get(idx));
            toKeepUsage.add(categoryUsageCount.get(idx));
            toKeepTimestamp.add(categoryLastUsedTimestamp.get(idx));
        }
        
        var pruned = categories.size() - maxCategories;
        
        categories.clear();
        categories.addAll(toKeep);
        categoryUsageCount.clear();
        categoryUsageCount.addAll(toKeepUsage);
        categoryLastUsedTimestamp.clear();
        categoryLastUsedTimestamp.addAll(toKeepTimestamp);
        
        return pruned;
    }
    
    /**
     * Get usage statistics for a specific category.
     * 
     * @param index category index
     * @return usage count for the category
     */
    public long getCategoryUsageCount(int index) {
        if (index < 0 || index >= categories.size()) {
            throw new IndexOutOfBoundsException("Category index out of bounds");
        }
        return categoryUsageCount.get(index);
    }
    
    /**
     * Get last used timestamp for a specific category.
     * 
     * @param index category index
     * @return timestamp when category was last used
     */
    public long getCategoryLastUsedTimestamp(int index) {
        if (index < 0 || index >= categories.size()) {
            throw new IndexOutOfBoundsException("Category index out of bounds");
        }
        return categoryLastUsedTimestamp.get(index);
    }
    
    /**
     * Get total number of activations.
     * 
     * @return total activation count
     */
    public long getTotalActivations() {
        return totalActivations;
    }
}