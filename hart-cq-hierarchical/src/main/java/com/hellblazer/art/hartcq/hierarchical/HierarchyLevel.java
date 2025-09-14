/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.hartcq.hierarchical;

import com.hellblazer.art.core.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ConcurrentHashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a single level in the HART-CQ hierarchical processing architecture.
 * Each level corresponds to different temporal granularities:
 * - Level 1: Token-level processing (individual words/tokens)
 * - Level 2: Window-level aggregation (phrases/sequences) 
 * - Level 3: Document-level synthesis (entire documents)
 * 
 * Each level maintains its own vigilance parameter, category mappings, and processing state.
 * 
 * @author Hal Hildebrand
 */
public class HierarchyLevel {
    private static final Logger logger = LoggerFactory.getLogger(HierarchyLevel.class);
    
    /**
     * Enumeration of hierarchical processing levels.
     */
    public enum Level {
        TOKEN(1, "Token-level processing", 0.7),
        WINDOW(2, "Window-level aggregation", 0.8),
        DOCUMENT(3, "Document-level synthesis", 0.9);
        
        private final int levelNumber;
        private final String description;
        private final double defaultVigilance;
        
        Level(int levelNumber, String description, double defaultVigilance) {
            this.levelNumber = levelNumber;
            this.description = description;
            this.defaultVigilance = defaultVigilance;
        }
        
        public int getLevelNumber() { return levelNumber; }
        public String getDescription() { return description; }
        public double getDefaultVigilance() { return defaultVigilance; }
        
        public static Level fromNumber(int levelNumber) {
            return switch (levelNumber) {
                case 1 -> TOKEN;
                case 2 -> WINDOW;
                case 3 -> DOCUMENT;
                default -> throw new IllegalArgumentException("Invalid level number: " + levelNumber);
            };
        }
    }
    
    private final Level level;
    private final double vigilance;
    private final int maxCategories;
    private final AtomicInteger nextCategoryId;
    private final ConcurrentHashMap<Integer, CategoryState> categories;
    private final ConcurrentHashMap<Pattern, Integer> patternToCategory;
    
    // Processing statistics
    private volatile long totalProcessed = 0;
    private volatile long totalMatches = 0;
    private volatile long totalNewCategories = 0;
    private volatile double averageActivation = 0.0;
    
    /**
     * Create a hierarchy level with specified parameters.
     * 
     * @param level The hierarchical level type
     * @param vigilance Vigilance parameter for this level
     * @param maxCategories Maximum number of categories for this level
     */
    public HierarchyLevel(Level level, double vigilance, int maxCategories) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be between 0.0 and 1.0, got: " + vigilance);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive, got: " + maxCategories);
        }
        
        this.level = level;
        this.vigilance = vigilance;
        this.maxCategories = maxCategories;
        this.nextCategoryId = new AtomicInteger(0);
        this.categories = new ConcurrentHashMap<>();
        this.patternToCategory = new ConcurrentHashMap<>();
        
        logger.debug("Created hierarchy level: {} with vigilance: {}, max categories: {}", 
                    level, vigilance, maxCategories);
    }
    
    /**
     * Create a hierarchy level with default parameters for the level type.
     * 
     * @param level The hierarchical level type
     * @param maxCategories Maximum number of categories for this level
     */
    public HierarchyLevel(Level level, int maxCategories) {
        this(level, level.getDefaultVigilance(), maxCategories);
    }
    
    /**
     * Process a pattern at this hierarchical level.
     * Returns existing category if pattern matches, creates new category if needed.
     * 
     * @param pattern Input pattern to process
     * @param learningEnabled Whether new categories can be created
     * @return Processing result with category assignment
     */
    public LevelProcessingResult processPattern(Pattern pattern, boolean learningEnabled) {
        if (pattern == null) {
            throw new IllegalArgumentException("Pattern cannot be null");
        }
        
        var startTime = System.nanoTime();
        totalProcessed++;
        
        try {
            // Check for exact pattern match first
            var existingCategory = patternToCategory.get(pattern);
            if (existingCategory != null) {
                var categoryState = categories.get(existingCategory);
                if (categoryState != null) {
                    categoryState.incrementActivations();
                    totalMatches++;
                    updateAverageActivation(1.0);
                    
                    var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                    return LevelProcessingResult.createMatch(existingCategory, 1.0, processingTime, true);
                }
            }
            
            // Find best matching category based on vigilance threshold
            var bestMatch = findBestMatchingCategory(pattern);
            
            if (bestMatch != null && bestMatch.activation >= vigilance) {
                // Pattern matches existing category
                var categoryState = categories.get(bestMatch.categoryId);
                if (categoryState != null) {
                    categoryState.incrementActivations();
                    categoryState.updatePattern(pattern, bestMatch.activation);
                    totalMatches++;
                    updateAverageActivation(bestMatch.activation);
                    
                    // Cache pattern-to-category mapping for faster future lookups
                    if (bestMatch.activation > 0.9) { // Only cache high-confidence matches
                        patternToCategory.put(pattern, bestMatch.categoryId);
                    }
                    
                    var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                    return LevelProcessingResult.createMatch(bestMatch.categoryId, bestMatch.activation, 
                                                           processingTime, false);
                }
            }
            
            // No match found - create new category if learning is enabled
            if (learningEnabled && categories.size() < maxCategories) {
                var newCategoryId = createNewCategory(pattern);
                totalNewCategories++;
                
                var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                return LevelProcessingResult.createNewCategory(newCategoryId, processingTime);
            }
            
            // Learning disabled or max categories reached - return no match
            var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
            return LevelProcessingResult.createNoMatch(processingTime, 
                !learningEnabled ? "Learning disabled" : "Max categories reached");
            
        } catch (Exception e) {
            logger.error("Error processing pattern at level {}: {}", level, e.getMessage(), e);
            var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
            return LevelProcessingResult.createError(e.getMessage(), processingTime);
        }
    }
    
    /**
     * Find the best matching category for a given pattern.
     * 
     * @param pattern Input pattern
     * @return Best match information or null if no categories exist
     */
    private CategoryMatch findBestMatchingCategory(Pattern pattern) {
        if (categories.isEmpty()) {
            return null;
        }
        
        CategoryMatch bestMatch = null;
        double bestActivation = 0.0;
        
        for (var entry : categories.entrySet()) {
            var categoryId = entry.getKey();
            var categoryState = entry.getValue();
            
            // Calculate activation based on pattern similarity
            var activation = calculateActivation(pattern, categoryState.getPrototype());
            
            if (activation > bestActivation) {
                bestActivation = activation;
                bestMatch = new CategoryMatch(categoryId, activation);
            }
        }
        
        return bestMatch;
    }
    
    /**
     * Calculate activation between input pattern and category prototype.
     * Uses complement coding compatible similarity measure.
     * 
     * @param input Input pattern
     * @param prototype Category prototype pattern
     * @return Activation value between 0.0 and 1.0
     */
    private double calculateActivation(Pattern input, Pattern prototype) {
        if (input.dimension() != prototype.dimension()) {
            logger.warn("Dimension mismatch: input={}, prototype={}", 
                       input.dimension(), prototype.dimension());
            return 0.0;
        }
        
        // Calculate fuzzy intersection (minimum) and norms
        double intersection = 0.0;
        double inputNorm = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double inputVal = input.get(i);
            double prototypeVal = prototype.get(i);
            
            intersection += Math.min(inputVal, prototypeVal);
            inputNorm += inputVal;
        }
        
        // Avoid division by zero
        if (inputNorm == 0.0) {
            return 0.0;
        }
        
        // Fuzzy ART activation: |intersection| / (alpha + |input|)
        // Using alpha = 0.001 for numerical stability
        double alpha = 0.001;
        return intersection / (alpha + inputNorm);
    }
    
    /**
     * Create a new category with the given pattern as prototype.
     * 
     * @param pattern Prototype pattern for new category
     * @return New category ID
     */
    private int createNewCategory(Pattern pattern) {
        var categoryId = nextCategoryId.getAndIncrement();
        var categoryState = new CategoryState(categoryId, pattern);
        
        categories.put(categoryId, categoryState);
        
        // Cache exact pattern match for fast lookup
        patternToCategory.put(pattern, categoryId);
        
        logger.debug("Created new category {} at level {} with pattern dimension {}", 
                    categoryId, level, pattern.dimension());
        
        return categoryId;
    }
    
    /**
     * Update running average activation.
     */
    private void updateAverageActivation(double newActivation) {
        // Exponential moving average with alpha = 0.1
        averageActivation = 0.9 * averageActivation + 0.1 * newActivation;
    }
    
    /**
     * Get processing statistics for this level.
     * 
     * @return Level statistics
     */
    public LevelStatistics getStatistics() {
        double matchRate = totalProcessed > 0 ? (double) totalMatches / totalProcessed : 0.0;
        double newCategoryRate = totalProcessed > 0 ? (double) totalNewCategories / totalProcessed : 0.0;
        
        return new LevelStatistics(
            level,
            vigilance,
            maxCategories,
            categories.size(),
            totalProcessed,
            totalMatches,
            totalNewCategories,
            matchRate,
            newCategoryRate,
            averageActivation
        );
    }
    
    /**
     * Reset level state and clear all categories.
     */
    public void reset() {
        categories.clear();
        patternToCategory.clear();
        nextCategoryId.set(0);
        totalProcessed = 0;
        totalMatches = 0;
        totalNewCategories = 0;
        averageActivation = 0.0;
        
        logger.debug("Reset hierarchy level: {}", level);
    }
    
    /**
     * Get category states for inspection.
     * 
     * @return Map of category states (defensive copy)
     */
    public Map<Integer, CategoryState> getCategoryStates() {
        return Map.copyOf(categories);
    }
    
    // Getters
    public Level getLevel() { return level; }
    public double getVigilance() { return vigilance; }
    public int getMaxCategories() { return maxCategories; }
    public int getCurrentCategoryCount() { return categories.size(); }
    public long getTotalProcessed() { return totalProcessed; }
    
    /**
     * Information about a category match.
     */
    private record CategoryMatch(int categoryId, double activation) {}
    
    /**
     * State information for a learned category.
     */
    public static class CategoryState {
        private final int categoryId;
        private volatile Pattern prototype;
        private final AtomicInteger activationCount;
        private volatile long lastActivationTime;
        private volatile double totalActivation;
        
        public CategoryState(int categoryId, Pattern prototype) {
            this.categoryId = categoryId;
            this.prototype = prototype;
            this.activationCount = new AtomicInteger(1);
            this.lastActivationTime = System.currentTimeMillis();
            this.totalActivation = 1.0;
        }
        
        public void incrementActivations() {
            activationCount.incrementAndGet();
            lastActivationTime = System.currentTimeMillis();
        }
        
        public void updatePattern(Pattern newPattern, double activation) {
            // Simple prototype update - in practice might use learning rate
            // For now, keep original prototype for stability
            totalActivation += activation;
        }
        
        public int getCategoryId() { return categoryId; }
        public Pattern getPrototype() { return prototype; }
        public int getActivationCount() { return activationCount.get(); }
        public long getLastActivationTime() { return lastActivationTime; }
        public double getAverageActivation() { 
            return activationCount.get() > 0 ? totalActivation / activationCount.get() : 0.0;
        }
    }
    
    /**
     * Result of pattern processing at a hierarchy level.
     */
    public static class LevelProcessingResult {
        private final boolean success;
        private final Integer categoryId;
        private final double activation;
        private final double processingTimeMs;
        private final boolean exactMatch;
        private final boolean newCategory;
        private final String errorMessage;
        
        private LevelProcessingResult(boolean success, Integer categoryId, double activation, 
                                    double processingTimeMs, boolean exactMatch, 
                                    boolean newCategory, String errorMessage) {
            this.success = success;
            this.categoryId = categoryId;
            this.activation = activation;
            this.processingTimeMs = processingTimeMs;
            this.exactMatch = exactMatch;
            this.newCategory = newCategory;
            this.errorMessage = errorMessage;
        }
        
        public static LevelProcessingResult createMatch(int categoryId, double activation, 
                                                      double processingTimeMs, boolean exactMatch) {
            return new LevelProcessingResult(true, categoryId, activation, processingTimeMs, 
                                           exactMatch, false, null);
        }
        
        public static LevelProcessingResult createNewCategory(int categoryId, double processingTimeMs) {
            return new LevelProcessingResult(true, categoryId, 1.0, processingTimeMs, 
                                           false, true, null);
        }
        
        public static LevelProcessingResult createNoMatch(double processingTimeMs, String reason) {
            return new LevelProcessingResult(false, null, 0.0, processingTimeMs, 
                                           false, false, reason);
        }
        
        public static LevelProcessingResult createError(String errorMessage, double processingTimeMs) {
            return new LevelProcessingResult(false, null, 0.0, processingTimeMs, 
                                           false, false, errorMessage);
        }
        
        // Getters
        public boolean isSuccess() { return success; }
        public Integer getCategoryId() { return categoryId; }
        public double getActivation() { return activation; }
        public double getProcessingTimeMs() { return processingTimeMs; }
        public boolean isExactMatch() { return exactMatch; }
        public boolean isNewCategory() { return newCategory; }
        public String getErrorMessage() { return errorMessage; }
    }
    
    /**
     * Statistics for a hierarchy level.
     */
    public record LevelStatistics(
        Level level,
        double vigilance,
        int maxCategories,
        int currentCategories,
        long totalProcessed,
        long totalMatches,
        long totalNewCategories,
        double matchRate,
        double newCategoryRate,
        double averageActivation
    ) {
        @Override
        public String toString() {
            return String.format("LevelStats{level=%s, vigilance=%.2f, categories=%d/%d, " +
                               "processed=%d, matches=%d(%.1f%%), new=%d(%.1f%%), avgActivation=%.3f}",
                level, vigilance, currentCategories, maxCategories, totalProcessed,
                totalMatches, matchRate * 100, totalNewCategories, newCategoryRate * 100, 
                averageActivation);
        }
    }
}