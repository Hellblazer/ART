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

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.artmap.DeepARTMAP;
import com.hellblazer.art.core.artmap.DeepARTMAPParameters;
import com.hellblazer.art.core.artmap.DeepARTMAPResult;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.hartcq.Token;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Adapter to connect HART-CQ multi-channel processing to DeepARTMAP hierarchical learning.
 * Handles conversion between HART-CQ channel outputs and ART-compatible patterns,
 * manages vigilance parameter adaptation, and provides deterministic category formation.
 * 
 * @author Hal Hildebrand
 */
public class ARTAdapter {
    private static final Logger logger = LoggerFactory.getLogger(ARTAdapter.class);
    
    private final DeepARTMAP deepARTMAP;
    private final DeepARTMAPParameters parameters;
    private final AtomicBoolean isLearning;
    private final ConcurrentHashMap<Integer, CategoryInfo> categoryCache;
    
    // Adaptive vigilance parameters
    private final double baseVigilance;
    private final double vigilanceIncrement;
    private final double maxVigilance;
    private final int maxCategories;
    
    // Performance tracking
    private volatile long totalProcessingTime = 0;
    private volatile int totalPatterns = 0;
    
    /**
     * Create an ARTAdapter with specified ART modules and parameters.
     * 
     * @param modules List of BaseART modules for hierarchical levels
     * @param baseVigilance Base vigilance parameter for adaptation
     * @param vigilanceIncrement Increment for adaptive vigilance
     * @param maxCategories Maximum number of categories per level
     */
    public ARTAdapter(List<BaseART> modules, double baseVigilance, 
                      double vigilanceIncrement, int maxCategories) {
        this.baseVigilance = baseVigilance;
        this.vigilanceIncrement = vigilanceIncrement;
        this.maxVigilance = Math.min(0.99, baseVigilance + (vigilanceIncrement * 10));
        this.maxCategories = maxCategories;
        
        this.parameters = new DeepARTMAPParameters(baseVigilance, 0.1, maxCategories, true);
        this.deepARTMAP = new DeepARTMAP(modules, parameters);
        this.isLearning = new AtomicBoolean(false);
        this.categoryCache = new ConcurrentHashMap<>();
        
        logger.info("ARTAdapter initialized with {} modules, base vigilance: {}, max categories: {}", 
                   modules.size(), baseVigilance, maxCategories);
    }
    
    /**
     * Create an ARTAdapter with default FuzzyART modules.
     * 
     * @param numLevels Number of hierarchical levels
     * @param inputDimension Dimension of input patterns
     */
    public static ARTAdapter createDefault(int numLevels, int inputDimension) {
        var modules = new ArrayList<BaseART>();
        for (int i = 0; i < numLevels; i++) {
            modules.add(new FuzzyART());
        }
        return new ARTAdapter(modules, 0.75, 0.05, 1000);
    }
    
    /**
     * Convert HART-CQ channel outputs to ART-compatible pattern format.
     * Applies complement coding for fuzzy ART compatibility.
     * 
     * @param channelOutputs Array of channel output features
     * @return List of Pattern arrays for hierarchical processing
     */
    public List<Pattern[]> convertToARTPatterns(float[][] channelOutputs) {
        if (channelOutputs == null || channelOutputs.length == 0) {
            throw new IllegalArgumentException("Channel outputs cannot be null or empty");
        }
        
        var patterns = new ArrayList<Pattern[]>();
        
        // Create patterns for each hierarchical level
        for (int level = 0; level < channelOutputs.length; level++) {
            var levelOutputs = channelOutputs[level];
            if (levelOutputs == null || levelOutputs.length == 0) {
                throw new IllegalArgumentException("Level output cannot be null or empty at level " + level);
            }
            
            // Apply complement coding: [x, 1-x] for fuzzy ART
            var complementCoded = applyComplementCoding(levelOutputs);
            var pattern = new DenseVector(complementCoded);
            
            // Each level gets a single pattern array
            patterns.add(new Pattern[]{pattern});
        }
        
        return patterns;
    }
    
    /**
     * Convert single channel output to ART patterns for all levels.
     * Replicates the pattern across hierarchical levels with different vigilance.
     * 
     * @param channelOutput Single channel output features
     * @param numLevels Number of hierarchical levels
     * @return List of Pattern arrays for hierarchical processing
     */
    public List<Pattern[]> convertSingleChannelToARTPatterns(float[] channelOutput, int numLevels) {
        if (channelOutput == null || channelOutput.length == 0) {
            throw new IllegalArgumentException("Channel output cannot be null or empty");
        }
        
        var patterns = new ArrayList<Pattern[]>();
        var complementCoded = applyComplementCoding(channelOutput);
        var basePattern = new DenseVector(complementCoded);
        
        // Replicate pattern for each level
        for (int level = 0; level < numLevels; level++) {
            patterns.add(new Pattern[]{basePattern});
        }
        
        return patterns;
    }
    
    /**
     * Apply complement coding transformation for fuzzy ART compatibility.
     * Transforms input [x] to [x, 1-x] format.
     * 
     * @param input Input feature array
     * @return Complement coded features
     */
    public double[] applyComplementCoding(float[] input) {
        var complementCoded = new double[input.length * 2];
        
        for (int i = 0; i < input.length; i++) {
            // Normalize to [0,1] range if needed
            double normalizedValue = Math.max(0.0, Math.min(1.0, input[i]));
            
            complementCoded[i] = normalizedValue;                    // Original value
            complementCoded[i + input.length] = 1.0 - normalizedValue; // Complement
        }
        
        return complementCoded;
    }
    
    /**
     * Enable learning mode for category formation.
     */
    public void enableLearning() {
        isLearning.set(true);
        logger.debug("Learning mode enabled");
    }
    
    /**
     * Disable learning mode for deterministic prediction.
     */
    public void disableLearning() {
        isLearning.set(false);
        logger.debug("Learning mode disabled - entering deterministic mode");
    }
    
    /**
     * Check if adapter is in learning mode.
     * 
     * @return true if learning is enabled
     */
    public boolean isLearning() {
        return isLearning.get();
    }
    
    /**
     * Process patterns through DeepARTMAP with adaptive vigilance.
     * In learning mode, creates new categories as needed.
     * In prediction mode, uses existing categories deterministically.
     * 
     * @param patterns List of Pattern arrays for hierarchical levels
     * @param labels Optional labels for supervised learning (null for unsupervised)
     * @return Processing result with category assignments
     */
    public ARTProcessingResult processPatterns(List<Pattern[]> patterns, int[] labels) {
        var startTime = System.nanoTime();
        
        try {
            DeepARTMAPResult result;
            
            if (isLearning.get()) {
                // Learning mode: train with adaptive vigilance
                result = trainWithAdaptiveVigilance(patterns, labels);
            } else {
                // Prediction mode: deterministic processing
                result = processForPrediction(patterns);
            }
            
            // Create processing result
            var processingResult = createProcessingResult(result, patterns.size());
            
            // Update performance metrics
            var processingTime = System.nanoTime() - startTime;
            updatePerformanceMetrics(processingTime);
            
            return processingResult;
            
        } catch (Exception e) {
            logger.error("Error processing patterns: {}", e.getMessage(), e);
            return ARTProcessingResult.createError(e.getMessage());
        }
    }
    
    /**
     * Train DeepARTMAP with adaptive vigilance parameter.
     * Increases vigilance if too many categories are being created.
     */
    private DeepARTMAPResult trainWithAdaptiveVigilance(List<Pattern[]> patterns, int[] labels) {
        DeepARTMAPResult result;
        double currentVigilance = baseVigilance;
        int attempts = 0;
        
        while (attempts < 5) { // Limit adaptation attempts
            // Update parameters with current vigilance
            var adaptedParams = parameters.copyWithVigilance(currentVigilance);
            var adaptedDeepARTMAP = new DeepARTMAP(deepARTMAP.getModules(), adaptedParams);
            
            // Attempt training
            if (labels != null) {
                result = adaptedDeepARTMAP.fitSupervised(patterns, labels);
            } else {
                result = adaptedDeepARTMAP.fitUnsupervised(patterns);
            }
            
            // Check if result is acceptable
            if (result instanceof DeepARTMAPResult.Success success) {
                int categoryCount = success.categoryCount();
                if (categoryCount <= maxCategories) {
                    logger.debug("Training successful with vigilance: {}, categories: {}", 
                               currentVigilance, categoryCount);
                    return result;
                }
                
                // Too many categories, increase vigilance
                currentVigilance = Math.min(maxVigilance, currentVigilance + vigilanceIncrement);
                logger.debug("Too many categories ({}), increasing vigilance to: {}", 
                           categoryCount, currentVigilance);
            } else {
                // Training failed, try with lower vigilance
                currentVigilance = Math.max(0.1, currentVigilance - vigilanceIncrement);
                logger.debug("Training failed, decreasing vigilance to: {}", currentVigilance);
            }
            
            attempts++;
        }
        
        // Fallback to original parameters
        if (labels != null) {
            return deepARTMAP.fitSupervised(patterns, labels);
        } else {
            return deepARTMAP.fitUnsupervised(patterns);
        }
    }
    
    /**
     * Process patterns for prediction using existing trained model.
     */
    private DeepARTMAPResult processForPrediction(List<Pattern[]> patterns) {
        if (!deepARTMAP.isTrained()) {
            throw new IllegalStateException("DeepARTMAP must be trained before prediction");
        }
        
        // Use predict method for deterministic results
        var predictions = deepARTMAP.predict(patterns);
        var deepPredictions = deepARTMAP.predictDeep(patterns);
        
        // Create success result from predictions
        return new DeepARTMAPResult.Success(
            List.of("Prediction completed"),
            deepPredictions,
            false, // Not supervised prediction
            deepARTMAP.getTrainingCategoryCount()
        );
    }
    
    /**
     * Create ARTProcessingResult from DeepARTMAP result.
     */
    private ARTProcessingResult createProcessingResult(DeepARTMAPResult result, int numLevels) {
        if (result instanceof DeepARTMAPResult.Success success) {
            var deepLabels = success.deepLabels();
            var hierarchicalCategories = deepLabels.length > 0 ? deepLabels[0] : new int[numLevels];
            
            // Cache category information
            for (int i = 0; i < hierarchicalCategories.length; i++) {
                int category = hierarchicalCategories[i];
                categoryCache.computeIfAbsent(category, k -> new CategoryInfo(k, System.currentTimeMillis()));
            }
            
            return ARTProcessingResult.createSuccess(
                hierarchicalCategories,
                success.categoryCount(),
                success.supervisedMode()
            );
        } else if (result instanceof DeepARTMAPResult.TrainingFailure failure) {
            return ARTProcessingResult.createError("Training failed: " + failure.message());
        } else {
            return ARTProcessingResult.createError("Unknown result type: " + result.getClass().getSimpleName());
        }
    }
    
    /**
     * Update performance tracking metrics.
     */
    private void updatePerformanceMetrics(long processingTimeNanos) {
        totalProcessingTime += processingTimeNanos;
        totalPatterns++;
        
        if (totalPatterns % 1000 == 0) {
            double avgTimeMs = (totalProcessingTime / 1_000_000.0) / totalPatterns;
            logger.debug("Processed {} patterns, avg time: {:.2f} ms", totalPatterns, avgTimeMs);
        }
    }
    
    /**
     * Get current performance statistics.
     * 
     * @return Performance statistics
     */
    public PerformanceStats getPerformanceStats() {
        double avgTimeMs = totalPatterns > 0 ? 
            (totalProcessingTime / 1_000_000.0) / totalPatterns : 0.0;
        double throughputPerSec = totalPatterns > 0 ? 
            1000.0 / avgTimeMs : 0.0;
            
        return new PerformanceStats(
            totalPatterns,
            avgTimeMs,
            throughputPerSec,
            categoryCache.size(),
            isLearning.get()
        );
    }
    
    /**
     * Reset adapter state and clear caches.
     */
    public void reset() {
        deepARTMAP.clearDeepARTMAP();
        categoryCache.clear();
        totalProcessingTime = 0;
        totalPatterns = 0;
        isLearning.set(false);
        logger.info("ARTAdapter reset completed");
    }
    
    /**
     * Get the underlying DeepARTMAP instance.
     * 
     * @return DeepARTMAP instance
     */
    public DeepARTMAP getDeepARTMAP() {
        return deepARTMAP;
    }
    
    /**
     * Get current category cache.
     * 
     * @return Map of category information
     */
    public ConcurrentHashMap<Integer, CategoryInfo> getCategoryCache() {
        return categoryCache;
    }
    
    /**
     * Information about a learned category.
     */
    public static class CategoryInfo {
        private final int categoryId;
        private final long creationTime;
        private volatile int accessCount = 0;
        
        public CategoryInfo(int categoryId, long creationTime) {
            this.categoryId = categoryId;
            this.creationTime = creationTime;
        }
        
        public int getCategoryId() { return categoryId; }
        public long getCreationTime() { return creationTime; }
        public int getAccessCount() { return accessCount; }
        
        public void incrementAccess() { accessCount++; }
    }
    
    /**
     * Performance statistics for the adapter.
     */
    public record PerformanceStats(
        int totalPatterns,
        double avgProcessingTimeMs,
        double throughputPerSecond,
        int totalCategories,
        boolean isLearning
    ) {
        @Override
        public String toString() {
            return String.format("PerformanceStats{patterns=%d, avgTime=%.2fms, throughput=%.1f/sec, categories=%d, learning=%s}",
                totalPatterns, avgProcessingTimeMs, throughputPerSecond, totalCategories, isLearning);
        }
    }
    
    /**
     * Result of ART pattern processing.
     */
    public static class ARTProcessingResult {
        private final boolean success;
        private final int[] hierarchicalCategories;
        private final int totalCategories;
        private final boolean supervised;
        private final String errorMessage;
        
        private ARTProcessingResult(boolean success, int[] hierarchicalCategories, 
                                   int totalCategories, boolean supervised, String errorMessage) {
            this.success = success;
            this.hierarchicalCategories = hierarchicalCategories;
            this.totalCategories = totalCategories;
            this.supervised = supervised;
            this.errorMessage = errorMessage;
        }
        
        public static ARTProcessingResult createSuccess(int[] hierarchicalCategories, 
                                                       int totalCategories, boolean supervised) {
            return new ARTProcessingResult(true, hierarchicalCategories, totalCategories, supervised, null);
        }
        
        public static ARTProcessingResult createError(String errorMessage) {
            return new ARTProcessingResult(false, null, 0, false, errorMessage);
        }
        
        public boolean isSuccess() { return success; }
        public int[] getHierarchicalCategories() { return hierarchicalCategories; }
        public int getTotalCategories() { return totalCategories; }
        public boolean isSupervised() { return supervised; }
        public String getErrorMessage() { return errorMessage; }
        
        public int getTopCategory() {
            return hierarchicalCategories != null && hierarchicalCategories.length > 0 
                ? hierarchicalCategories[0] : -1;
        }
    }
}