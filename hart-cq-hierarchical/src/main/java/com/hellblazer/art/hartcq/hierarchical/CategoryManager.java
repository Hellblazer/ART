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
import com.hellblazer.art.hartcq.hierarchical.HierarchyLevel.Level;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * Manages learned categories across the hierarchical HART-CQ architecture.
 * Handles category storage and retrieval, category pruning and merging,
 * and provides deterministic category selection for consistent processing.
 * 
 * The CategoryManager maintains a global view of categories across all hierarchy
 * levels while preserving level-specific category semantics and relationships.
 * 
 * @author Hal Hildebrand
 */
public class CategoryManager {
    private static final Logger logger = LoggerFactory.getLogger(CategoryManager.class);
    
    private final CategoryParameters parameters;
    private final ConcurrentHashMap<CategoryKey, ManagedCategory> categories;
    private final ConcurrentHashMap<Level, AtomicInteger> categoryCounters;
    private final ConcurrentHashMap<Level, Set<Integer>> levelCategories;
    private final ReentrantReadWriteLock managementLock;
    
    // Performance tracking
    private final AtomicLong totalAccesses = new AtomicLong(0);
    private final AtomicLong totalPrunings = new AtomicLong(0);
    private final AtomicLong totalMergings = new AtomicLong(0);
    private volatile long lastMaintenanceTime = System.currentTimeMillis();
    
    // Deterministic selection state
    private final Random deterministicRandom;
    private volatile long deterministicSeed = 12345L;
    
    /**
     * Create a CategoryManager with specified parameters.
     * 
     * @param parameters Configuration parameters for category management
     */
    public CategoryManager(CategoryParameters parameters) {
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        
        this.parameters = parameters;
        this.categories = new ConcurrentHashMap<>();
        this.categoryCounters = new ConcurrentHashMap<>();
        this.levelCategories = new ConcurrentHashMap<>();
        this.managementLock = new ReentrantReadWriteLock();
        this.deterministicRandom = new Random(deterministicSeed);
        
        // Initialize category counters for each level
        for (var level : Level.values()) {
            categoryCounters.put(level, new AtomicInteger(0));
            levelCategories.put(level, ConcurrentHashMap.newKeySet());
        }
        
        logger.info("CategoryManager initialized with parameters: {}", parameters);
    }
    
    /**
     * Store a learned category from a hierarchy level.
     * 
     * @param level The hierarchy level
     * @param categoryId The category ID from the level
     * @param prototype The category prototype pattern
     * @param activation The activation strength
     * @param metadata Additional category metadata
     * @return The managed category key for global reference
     */
    public CategoryKey storeCategory(Level level, int categoryId, Pattern prototype, 
                                   double activation, Map<String, Object> metadata) {
        if (level == null) {
            throw new IllegalArgumentException("Level cannot be null");
        }
        if (prototype == null) {
            throw new IllegalArgumentException("Prototype cannot be null");
        }
        
        var categoryKey = new CategoryKey(level, categoryId);

        // Check if category already exists
        var existing = categories.get(categoryKey);

        if (existing == null) {
            // Create new category
            var managedCategory = new ManagedCategory(
                categoryKey,
                prototype,
                activation,
                System.currentTimeMillis(),
                metadata != null ? Map.copyOf(metadata) : Map.of()
            );

            // Store the new category
            categories.put(categoryKey, managedCategory);

            // Update level tracking
            levelCategories.get(level).add(categoryId);

            // Update category counter
            categoryCounters.get(level).incrementAndGet();
            logger.debug("Stored new category: {} with activation {:.3f}",
                        categoryKey, activation);
        } else {
            // Update existing category with new activation
            existing.updateActivation(activation);
            logger.debug("Updated existing category: {} with activation {:.3f}",
                        categoryKey, activation);
        }
        
        totalAccesses.incrementAndGet();
        
        // Trigger maintenance if needed
        if (shouldPerformMaintenance()) {
            performMaintenanceAsync();
        }
        
        return categoryKey;
    }
    
    /**
     * Retrieve a managed category by key.
     * 
     * @param key The category key
     * @return The managed category or null if not found
     */
    public ManagedCategory getCategory(CategoryKey key) {
        if (key == null) {
            return null;
        }
        
        var category = categories.get(key);
        if (category != null) {
            category.recordAccess();
            totalAccesses.incrementAndGet();
        }
        
        return category;
    }
    
    /**
     * Get all categories for a specific hierarchy level.
     * 
     * @param level The hierarchy level
     * @return List of managed categories for the level
     */
    public List<ManagedCategory> getCategoriesForLevel(Level level) {
        if (level == null) {
            return List.of();
        }
        
        return categories.entrySet().stream()
            .filter(entry -> entry.getKey().level() == level)
            .map(Map.Entry::getValue)
            .sorted((a, b) -> Integer.compare(a.getKey().categoryId(), b.getKey().categoryId()))
            .collect(Collectors.toList());
    }
    
    /**
     * Find the most similar category to a given pattern at a specific level.
     * Uses deterministic selection when multiple categories have similar similarity.
     * 
     * @param level The hierarchy level
     * @param pattern The input pattern
     * @param minSimilarity Minimum similarity threshold
     * @return The best matching category or null if none meet threshold
     */
    public ManagedCategory findMostSimilar(Level level, Pattern pattern, double minSimilarity) {
        if (level == null || pattern == null) {
            return null;
        }
        
        var levelCategorySet = levelCategories.get(level);
        if (levelCategorySet.isEmpty()) {
            return null;
        }
        
        var candidates = new ArrayList<SimilarityCandidate>();
        
        // Calculate similarities
        for (var categoryId : levelCategorySet) {
            var key = new CategoryKey(level, categoryId);
            var category = categories.get(key);
            
            if (category != null) {
                double similarity = calculateSimilarity(pattern, category.getPrototype());
                if (similarity >= minSimilarity) {
                    candidates.add(new SimilarityCandidate(category, similarity));
                }
            }
        }
        
        if (candidates.isEmpty()) {
            return null;
        }
        
        // Sort by similarity (descending) and then by deterministic criteria
        candidates.sort((a, b) -> {
            int similarityCompare = Double.compare(b.similarity, a.similarity);
            if (similarityCompare != 0) {
                return similarityCompare;
            }
            // For equal similarities, use deterministic selection
            return deterministicComparison(a.category, b.category);
        });
        
        var bestCandidate = candidates.get(0);
        bestCandidate.category.recordAccess();
        totalAccesses.incrementAndGet();
        
        logger.debug("Found most similar category: {} with similarity {:.3f} from {} candidates",
                    bestCandidate.category.getKey(), bestCandidate.similarity, candidates.size());
        
        return bestCandidate.category;
    }
    
    /**
     * Perform deterministic category selection when learning is disabled.
     * Returns consistent category assignments for the same input patterns.
     * 
     * @param level The hierarchy level
     * @param pattern The input pattern
     * @param availableCategories Set of available category IDs
     * @return Deterministically selected category ID or -1 if none available
     */
    public int selectDeterministic(Level level, Pattern pattern, Set<Integer> availableCategories) {
        if (level == null || pattern == null || availableCategories == null || availableCategories.isEmpty()) {
            return -1;
        }
        
        // Create deterministic hash from pattern
        long patternHash = calculatePatternHash(pattern);
        
        // Combine with level and seed for deterministic selection
        long deterministicValue = patternHash ^ level.getLevelNumber() ^ deterministicSeed;
        
        // Convert to consistent category selection
        var categoryList = new ArrayList<>(availableCategories);
        categoryList.sort(Integer::compare); // Ensure consistent ordering
        
        int selectedIndex = Math.abs((int) (deterministicValue % categoryList.size()));
        int selectedCategory = categoryList.get(selectedIndex);
        
        logger.debug("Deterministic selection: level={}, patternHash={}, selected={}",
                    level, patternHash, selectedCategory);
        
        return selectedCategory;
    }
    
    /**
     * Prune unused or low-activation categories to maintain performance.
     * Uses configurable thresholds for pruning decisions.
     * 
     * @return Number of categories pruned
     */
    public int pruneCategories() {
        managementLock.writeLock().lock();
        try {
            var currentTime = System.currentTimeMillis();
            var pruneThreshold = currentTime - parameters.maxCategoryAge();
            var toRemove = new ArrayList<CategoryKey>();
            
            for (var entry : categories.entrySet()) {
                var category = entry.getValue();
                var shouldPrune = false;
                
                // Prune based on age
                if (category.getCreationTime() < pruneThreshold) {
                    shouldPrune = true;
                }
                
                // Prune based on low usage
                if (category.getAccessCount() < parameters.minAccessCount() && 
                    category.getLastAccessTime() < (currentTime - parameters.inactivityThreshold())) {
                    shouldPrune = true;
                }
                
                // Prune based on low activation
                if (category.getAverageActivation() < parameters.minActivationThreshold()) {
                    shouldPrune = true;
                }
                
                if (shouldPrune) {
                    toRemove.add(entry.getKey());
                }
            }
            
            // Remove identified categories
            int prunedCount = 0;
            for (var key : toRemove) {
                var removed = categories.remove(key);
                if (removed != null) {
                    levelCategories.get(key.level()).remove(key.categoryId());
                    categoryCounters.get(key.level()).decrementAndGet();
                    prunedCount++;
                }
            }
            
            totalPrunings.addAndGet(prunedCount);
            
            if (prunedCount > 0) {
                logger.info("Pruned {} categories based on age, usage, and activation thresholds", 
                           prunedCount);
            }
            
            return prunedCount;
            
        } finally {
            managementLock.writeLock().unlock();
        }
    }
    
    /**
     * Merge similar categories to reduce redundancy.
     * Uses similarity threshold to identify merge candidates.
     * 
     * @return Number of categories merged
     */
    public int mergeCategories() {
        managementLock.writeLock().lock();
        try {
            int mergedCount = 0;
            
            for (var level : Level.values()) {
                var levelCats = getCategoriesForLevel(level);
                if (levelCats.size() < 2) {
                    continue; // Need at least 2 categories to merge
                }
                
                var merged = new HashSet<CategoryKey>();
                
                for (int i = 0; i < levelCats.size(); i++) {
                    var cat1 = levelCats.get(i);
                    if (merged.contains(cat1.getKey())) {
                        continue;
                    }
                    
                    for (int j = i + 1; j < levelCats.size(); j++) {
                        var cat2 = levelCats.get(j);
                        if (merged.contains(cat2.getKey())) {
                            continue;
                        }
                        
                        double similarity = calculateSimilarity(cat1.getPrototype(), cat2.getPrototype());
                        
                        if (similarity >= parameters.mergeSimilarityThreshold()) {
                            // Merge cat2 into cat1 (keep the one with higher activation)
                            var target = cat1.getAverageActivation() >= cat2.getAverageActivation() ? cat1 : cat2;
                            var source = target == cat1 ? cat2 : cat1;
                            
                            // Merge activation histories
                            target.mergeWith(source);
                            
                            // Remove source category
                            categories.remove(source.getKey());
                            levelCategories.get(level).remove(source.getKey().categoryId());
                            categoryCounters.get(level).decrementAndGet();
                            merged.add(source.getKey());
                            mergedCount++;
                            
                            logger.debug("Merged categories: {} into {} with similarity {:.3f}",
                                        source.getKey(), target.getKey(), similarity);
                            break; // Move to next category
                        }
                    }
                }
            }
            
            totalMergings.addAndGet(mergedCount);
            
            if (mergedCount > 0) {
                logger.info("Merged {} categories based on similarity threshold {:.3f}", 
                           mergedCount, parameters.mergeSimilarityThreshold());
            }
            
            return mergedCount;
            
        } finally {
            managementLock.writeLock().unlock();
        }
    }
    
    /**
     * Get comprehensive statistics about managed categories.
     * 
     * @return Category management statistics
     */
    public CategoryStatistics getStatistics() {
        managementLock.readLock().lock();
        try {
            var levelStats = new HashMap<Level, LevelCategoryStats>();
            
            for (var level : Level.values()) {
                var levelCats = getCategoriesForLevel(level);
                var activationStats = levelCats.stream()
                    .mapToDouble(ManagedCategory::getAverageActivation)
                    .summaryStatistics();
                
                var accessStats = levelCats.stream()
                    .mapToLong(ManagedCategory::getAccessCount)
                    .summaryStatistics();
                
                levelStats.put(level, new LevelCategoryStats(
                    level,
                    levelCats.size(),
                    activationStats.getAverage(),
                    activationStats.getMin(),
                    activationStats.getMax(),
                    (long) accessStats.getAverage(),
                    accessStats.getMin(),
                    accessStats.getMax()
                ));
            }
            
            return new CategoryStatistics(
                categories.size(),
                levelStats,
                totalAccesses.get(),
                totalPrunings.get(),
                totalMergings.get(),
                lastMaintenanceTime,
                System.currentTimeMillis()
            );
            
        } finally {
            managementLock.readLock().unlock();
        }
    }
    
    /**
     * Reset all managed categories and counters.
     */
    public void reset() {
        managementLock.writeLock().lock();
        try {
            categories.clear();

            for (var level : Level.values()) {
                categoryCounters.get(level).set(0);
                levelCategories.get(level).clear();
            }

            totalAccesses.set(0);

            // Reset deterministic seed to default
            deterministicSeed = 12345L;
            deterministicRandom.setSeed(deterministicSeed);
            totalPrunings.set(0);
            totalMergings.set(0);
            lastMaintenanceTime = System.currentTimeMillis();
            
            // Reset deterministic random with original seed
            deterministicRandom.setSeed(deterministicSeed);
            
            logger.info("CategoryManager reset completed");
            
        } finally {
            managementLock.writeLock().unlock();
        }
    }
    
    /**
     * Set the seed for deterministic category selection.
     * 
     * @param seed The deterministic seed value
     */
    public void setDeterministicSeed(long seed) {
        this.deterministicSeed = seed;
        deterministicRandom.setSeed(seed);
        logger.debug("Set deterministic seed to: {}", seed);
    }
    
    // Private helper methods
    
    /**
     * Calculate similarity between two patterns using cosine similarity.
     */
    private double calculateSimilarity(Pattern pattern1, Pattern pattern2) {
        if (pattern1.dimension() != pattern2.dimension()) {
            return 0.0;
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < pattern1.dimension(); i++) {
            double val1 = pattern1.get(i);
            double val2 = pattern2.get(i);

            dotProduct += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * Calculate deterministic hash from pattern for consistent selection.
     */
    private long calculatePatternHash(Pattern pattern) {
        long hash = 0;
        for (int i = 0; i < pattern.dimension(); i++) {
            long bits = Double.doubleToLongBits(pattern.get(i));
            hash ^= bits + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
    
    /**
     * Deterministic comparison for consistent category ordering.
     */
    private int deterministicComparison(ManagedCategory a, ManagedCategory b) {
        // Primary: category ID
        int idCompare = Integer.compare(a.getKey().categoryId(), b.getKey().categoryId());
        if (idCompare != 0) {
            return idCompare;
        }
        
        // Secondary: creation time
        return Long.compare(a.getCreationTime(), b.getCreationTime());
    }
    
    /**
     * Check if maintenance should be performed.
     */
    private boolean shouldPerformMaintenance() {
        long currentTime = System.currentTimeMillis();
        return (currentTime - lastMaintenanceTime) > parameters.maintenanceInterval();
    }
    
    /**
     * Perform maintenance operations asynchronously.
     */
    private void performMaintenanceAsync() {
        // Simple async execution - in production might use thread pool
        new Thread(() -> {
            try {
                int pruned = pruneCategories();
                int merged = mergeCategories();
                lastMaintenanceTime = System.currentTimeMillis();
                
                if (pruned > 0 || merged > 0) {
                    logger.info("Maintenance completed: pruned={}, merged={}", pruned, merged);
                }
            } catch (Exception e) {
                logger.error("Error during category maintenance: {}", e.getMessage(), e);
            }
        }, "CategoryManager-Maintenance").start();
    }
    
    // Inner classes and records
    
    /**
     * Parameters for category management.
     */
    public record CategoryParameters(
        long maxCategoryAge,
        int minAccessCount,
        long inactivityThreshold,
        double minActivationThreshold,
        double mergeSimilarityThreshold,
        long maintenanceInterval
    ) {
        public static CategoryParameters defaults() {
            return new CategoryParameters(
                24 * 60 * 60 * 1000L, // 24 hours max age
                5,                    // Minimum 5 accesses
                60 * 60 * 1000L,     // 1 hour inactivity
                0.1,                 // Minimum 10% activation
                0.95,                // 95% similarity for merging
                5 * 60 * 1000L       // 5 minute maintenance interval
            );
        }
        
        @Override
        public String toString() {
            return String.format("CategoryParams{maxAge=%dms, minAccess=%d, inactivity=%dms, " +
                               "minActivation=%.2f, mergeSimilarity=%.2f, maintenance=%dms}",
                maxCategoryAge, minAccessCount, inactivityThreshold, 
                minActivationThreshold, mergeSimilarityThreshold, maintenanceInterval);
        }
    }
    
    /**
     * Key for identifying categories across levels.
     */
    public record CategoryKey(Level level, int categoryId) {
        @Override
        public String toString() {
            return String.format("%s:%d", level.name(), categoryId);
        }
    }
    
    /**
     * Managed category with tracking information.
     */
    public static class ManagedCategory {
        private final CategoryKey key;
        private final Pattern prototype;
        private final long creationTime;
        private final Map<String, Object> metadata;
        
        private volatile double totalActivation;
        private volatile int activationCount;
        private volatile long lastAccessTime;
        private volatile long accessCount;
        
        public ManagedCategory(CategoryKey key, Pattern prototype, double initialActivation,
                             long creationTime, Map<String, Object> metadata) {
            this.key = key;
            this.prototype = prototype;
            this.creationTime = creationTime;
            this.metadata = metadata;
            this.totalActivation = initialActivation;
            this.activationCount = 1;
            this.lastAccessTime = creationTime;
            this.accessCount = 0;
        }
        
        public void updateActivation(double activation) {
            synchronized (this) {
                totalActivation += activation;
                activationCount++;
            }
        }
        
        public void recordAccess() {
            lastAccessTime = System.currentTimeMillis();
            accessCount++;
        }
        
        public void mergeWith(ManagedCategory other) {
            synchronized (this) {
                totalActivation += other.totalActivation;
                activationCount += other.activationCount;
                accessCount += other.accessCount;
                lastAccessTime = Math.max(lastAccessTime, other.lastAccessTime);
            }
        }
        
        public double getAverageActivation() {
            return activationCount > 0 ? totalActivation / activationCount : 0.0;
        }
        
        // Getters
        public CategoryKey getKey() { return key; }
        public Pattern getPrototype() { return prototype; }
        public long getCreationTime() { return creationTime; }
        public Map<String, Object> getMetadata() { return metadata; }
        public long getLastAccessTime() { return lastAccessTime; }
        public long getAccessCount() { return accessCount; }
        public double getTotalActivation() { return totalActivation; }
        public int getActivationCount() { return activationCount; }
        
        @Override
        public String toString() {
            return String.format("ManagedCategory{key=%s, avgActivation=%.3f, accesses=%d}",
                key, getAverageActivation(), accessCount);
        }
    }
    
    /**
     * Similarity candidate for category matching.
     */
    private record SimilarityCandidate(ManagedCategory category, double similarity) {}
    
    /**
     * Statistics for categories at a specific level.
     */
    public record LevelCategoryStats(
        Level level,
        int categoryCount,
        double avgActivation,
        double minActivation,
        double maxActivation,
        long avgAccesses,
        long minAccesses,
        long maxAccesses
    ) {
        @Override
        public String toString() {
            return String.format("Level%s{count=%d, activation=%.3f(%.3f-%.3f), accesses=%d(%d-%d)}",
                level.name(), categoryCount, avgActivation, minActivation, maxActivation,
                avgAccesses, minAccesses, maxAccesses);
        }
    }
    
    /**
     * Comprehensive category management statistics.
     */
    public record CategoryStatistics(
        int totalCategories,
        Map<Level, LevelCategoryStats> levelStats,
        long totalAccesses,
        long totalPrunings,
        long totalMergings,
        long lastMaintenanceTime,
        long currentTime
    ) {
        @Override
        public String toString() {
            return String.format("CategoryStats{total=%d, accesses=%d, pruned=%d, merged=%d, " +
                               "levels=[%s]}",
                totalCategories, totalAccesses, totalPrunings, totalMergings,
                levelStats.values().stream().map(Object::toString).collect(Collectors.joining(", ")));
        }
    }
}