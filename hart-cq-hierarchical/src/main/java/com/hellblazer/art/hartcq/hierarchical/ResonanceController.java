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

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Controls resonance between hierarchical levels in HART-CQ processing.
 * Manages top-down and bottom-up signals, adaptive vigilance control, and 
 * stability-plasticity balance across the hierarchical architecture.
 * 
 * The resonance controller implements ART's core principle of matching between
 * expectation (top-down) and sensation (bottom-up) to achieve stable learning
 * while maintaining plasticity for new patterns.
 * 
 * @author Hal Hildebrand
 */
public class ResonanceController {
    private static final Logger logger = LoggerFactory.getLogger(ResonanceController.class);
    
    private final List<HierarchyLevel> levels;
    private final ResonanceParameters parameters;
    private final ConcurrentHashMap<Integer, ResonanceState> activeResonances;
    private final ReentrantReadWriteLock resonanceLock;
    
    // Adaptive parameters
    private volatile double currentStabilityFactor = 0.5;
    private volatile double currentPlasticityFactor = 0.5;
    private volatile long totalResonanceEvents = 0;
    private volatile long successfulResonances = 0;
    
    // Performance tracking
    private final AtomicReference<ResonanceMetrics> lastMetrics = new AtomicReference<>();
    
    /**
     * Create a resonance controller for the given hierarchy levels.
     * 
     * @param levels List of hierarchy levels from bottom (token) to top (document)
     * @param parameters Resonance control parameters
     */
    public ResonanceController(List<HierarchyLevel> levels, ResonanceParameters parameters) {
        if (levels == null || levels.isEmpty()) {
            throw new IllegalArgumentException("Levels cannot be null or empty");
        }
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters cannot be null");
        }
        
        this.levels = List.copyOf(levels); // Defensive copy
        this.parameters = parameters;
        this.activeResonances = new ConcurrentHashMap<>();
        this.resonanceLock = new ReentrantReadWriteLock();
        
        logger.info("ResonanceController initialized with {} levels and parameters: {}", 
                   levels.size(), parameters);
    }
    
    /**
     * Process a pattern through the hierarchical resonance system.
     * Coordinates bottom-up and top-down processing to achieve resonance.
     * 
     * @param inputPattern Input pattern to process
     * @param learningEnabled Whether learning is enabled
     * @return Resonance result containing hierarchical category assignments
     */
    public ResonanceResult processPattern(Pattern inputPattern, boolean learningEnabled) {
        if (inputPattern == null) {
            throw new IllegalArgumentException("Input pattern cannot be null");
        }
        
        var startTime = System.nanoTime();
        var resonanceId = (int) (totalResonanceEvents % Integer.MAX_VALUE);
        totalResonanceEvents++;
        
        try {
            // Create initial resonance state
            var resonanceState = new ResonanceState(resonanceId, inputPattern, System.currentTimeMillis());
            activeResonances.put(resonanceId, resonanceState);
            
            // Perform bottom-up processing
            var bottomUpResult = performBottomUpProcessing(inputPattern, learningEnabled, resonanceState);
            
            if (!bottomUpResult.success) {
                return ResonanceResult.createFailure(bottomUpResult.errorMessage, 
                    (System.nanoTime() - startTime) / 1_000_000.0);
            }
            
            // Perform top-down validation
            var topDownResult = performTopDownValidation(bottomUpResult.categoryAssignments, 
                inputPattern, resonanceState);
            
            // Check for resonance achievement
            if (checkResonanceCondition(bottomUpResult, topDownResult, resonanceState)) {
                successfulResonances++;
                updateStabilityPlasticityBalance(true, resonanceState);
                
                var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                return ResonanceResult.createSuccess(
                    bottomUpResult.categoryAssignments,
                    topDownResult.vigilanceAdjustments,
                    resonanceState.getResonanceStrength(),
                    processingTime,
                    resonanceId
                );
            } else {
                // Resonance failed - try adaptive adjustment if learning is enabled
                if (learningEnabled && parameters.enableAdaptiveVigilance) {
                    var adaptedResult = attemptAdaptiveResonance(inputPattern, bottomUpResult, 
                        topDownResult, resonanceState);
                    
                    if (adaptedResult != null) {
                        successfulResonances++;
                        updateStabilityPlasticityBalance(true, resonanceState);
                        
                        var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                        return adaptedResult.withProcessingTime(processingTime);
                    }
                }
                
                updateStabilityPlasticityBalance(false, resonanceState);
                var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
                return ResonanceResult.createFailure("Resonance not achieved", processingTime);
            }
            
        } catch (Exception e) {
            logger.error("Error in resonance processing: {}", e.getMessage(), e);
            var processingTime = (System.nanoTime() - startTime) / 1_000_000.0;
            return ResonanceResult.createFailure("Processing error: " + e.getMessage(), processingTime);
            
        } finally {
            // Cleanup resonance state
            activeResonances.remove(resonanceId);
        }
    }
    
    /**
     * Perform bottom-up processing through all hierarchy levels.
     * Processes patterns from token level upward to document level.
     */
    private BottomUpResult performBottomUpProcessing(Pattern inputPattern, boolean learningEnabled,
                                                    ResonanceState resonanceState) {
        var categoryAssignments = new int[levels.size()];
        var activations = new double[levels.size()];
        var processingTimes = new double[levels.size()];
        
        Pattern currentPattern = inputPattern;
        
        for (int i = 0; i < levels.size(); i++) {
            var level = levels.get(i);
            
            try {
                // Process pattern at current level
                var result = level.processPattern(currentPattern, learningEnabled);
                
                if (result.isSuccess() && result.getCategoryId() != null) {
                    categoryAssignments[i] = result.getCategoryId();
                    activations[i] = result.getActivation();
                    processingTimes[i] = result.getProcessingTimeMs();
                    
                    // Update resonance state
                    resonanceState.addLevelResult(i, result);
                    
                    // Create pattern for next level (in practice, this would be the category prototype)
                    // For now, use the same pattern with slight modification to represent hierarchy
                    currentPattern = createNextLevelPattern(currentPattern, result.getCategoryId(), i);
                    
                } else {
                    // Level processing failed
                    String error = result.getErrorMessage() != null ? 
                        result.getErrorMessage() : "Unknown processing error";
                    return new BottomUpResult(false, categoryAssignments, activations, 
                        processingTimes, "Level " + i + " failed: " + error);
                }
                
            } catch (Exception e) {
                logger.error("Error in bottom-up processing at level {}: {}", i, e.getMessage(), e);
                return new BottomUpResult(false, categoryAssignments, activations, 
                    processingTimes, "Level " + i + " exception: " + e.getMessage());
            }
        }
        
        return new BottomUpResult(true, categoryAssignments, activations, processingTimes, null);
    }
    
    /**
     * Perform top-down validation of category assignments.
     * Checks consistency and applies vigilance adjustments if needed.
     */
    private TopDownResult performTopDownValidation(int[] categoryAssignments, Pattern originalPattern,
                                                  ResonanceState resonanceState) {
        var vigilanceAdjustments = new double[levels.size()];
        var validationScores = new double[levels.size()];
        
        // Start from top level and validate downward
        for (int i = levels.size() - 1; i >= 0; i--) {
            var level = levels.get(i);
            var categoryId = categoryAssignments[i];
            
            // Get category prototype for comparison
            var categoryStates = level.getCategoryStates();
            var categoryState = categoryStates.get(categoryId);
            
            if (categoryState != null) {
                var prototype = categoryState.getPrototype();
                var validationScore = calculateValidationScore(originalPattern, prototype, i);
                validationScores[i] = validationScore;
                
                // Adjust vigilance based on validation score and hierarchy consistency
                var consistencyScore = calculateHierarchyConsistency(categoryAssignments, i);
                vigilanceAdjustments[i] = calculateVigilanceAdjustment(validationScore, 
                    consistencyScore, level.getVigilance());
                
                resonanceState.addValidationResult(i, validationScore, vigilanceAdjustments[i]);
                
            } else {
                // Category not found - this shouldn't happen in normal operation
                logger.warn("Category {} not found at level {}", categoryId, i);
                validationScores[i] = 0.0;
                vigilanceAdjustments[i] = 0.0;
            }
        }
        
        return new TopDownResult(true, vigilanceAdjustments, validationScores, null);
    }
    
    /**
     * Check if resonance condition is achieved across all levels.
     */
    private boolean checkResonanceCondition(BottomUpResult bottomUp, TopDownResult topDown,
                                          ResonanceState resonanceState) {
        // Calculate overall resonance strength
        double totalActivation = 0.0;
        double totalValidation = 0.0;
        
        for (int i = 0; i < levels.size(); i++) {
            totalActivation += bottomUp.activations[i];
            totalValidation += topDown.validationScores[i];
        }
        
        double avgActivation = totalActivation / levels.size();
        double avgValidation = totalValidation / levels.size();
        
        // Resonance achieved if both activation and validation exceed thresholds
        boolean activationOk = avgActivation >= parameters.minActivationThreshold;
        boolean validationOk = avgValidation >= parameters.minValidationThreshold;
        
        // Calculate and store resonance strength
        double resonanceStrength = (avgActivation + avgValidation) / 2.0;
        resonanceState.setResonanceStrength(resonanceStrength);
        
        boolean resonanceAchieved = activationOk && validationOk && 
            resonanceStrength >= parameters.minResonanceThreshold;
        
        logger.debug("Resonance check: activation={:.3f} ({}), validation={:.3f} ({}), " +
                    "strength={:.3f}, achieved={}",
            avgActivation, activationOk, avgValidation, validationOk, 
            resonanceStrength, resonanceAchieved);
        
        return resonanceAchieved;
    }
    
    /**
     * Attempt adaptive resonance by adjusting vigilance parameters.
     */
    private ResonanceResult attemptAdaptiveResonance(Pattern inputPattern, BottomUpResult bottomUp,
                                                   TopDownResult topDown, ResonanceState resonanceState) {
        // Try reducing vigilance to allow more flexible matching
        for (int attempt = 0; attempt < parameters.maxAdaptationAttempts; attempt++) {
            double adaptationFactor = parameters.vigilanceAdaptationRate * (attempt + 1);
            
            // Create adapted processing with reduced vigilance
            // This is a simplified version - in practice would require level-specific adaptation
            var adaptedBottomUp = performAdaptedBottomUpProcessing(inputPattern, adaptationFactor,
                resonanceState);
            
            if (adaptedBottomUp.success) {
                var adaptedTopDown = performTopDownValidation(adaptedBottomUp.categoryAssignments,
                    inputPattern, resonanceState);
                
                if (checkResonanceCondition(adaptedBottomUp, adaptedTopDown, resonanceState)) {
                    logger.debug("Adaptive resonance achieved on attempt {} with adaptation factor {:.3f}",
                        attempt + 1, adaptationFactor);
                    
                    return ResonanceResult.createSuccess(
                        adaptedBottomUp.categoryAssignments,
                        adaptedTopDown.vigilanceAdjustments,
                        resonanceState.getResonanceStrength(),
                        0.0, // Processing time will be set by caller
                        resonanceState.getResonanceId()
                    );
                }
            }
        }
        
        logger.debug("Adaptive resonance failed after {} attempts", parameters.maxAdaptationAttempts);
        return null; // Adaptation failed
    }
    
    /**
     * Perform adapted bottom-up processing with modified vigilance.
     */
    private BottomUpResult performAdaptedBottomUpProcessing(Pattern inputPattern, 
                                                          double adaptationFactor,
                                                          ResonanceState resonanceState) {
        // This is a simplified implementation - in practice would need to create
        // temporary hierarchy levels with adapted vigilance parameters
        
        // For now, just return the original bottom-up result
        // A full implementation would create adapted levels or modify vigilance temporarily
        return performBottomUpProcessing(inputPattern, true, resonanceState);
    }
    
    /**
     * Create pattern for next hierarchical level.
     */
    private Pattern createNextLevelPattern(Pattern currentPattern, int categoryId, int currentLevel) {
        // Simple approach: create a pattern that encodes the category ID
        // In practice, this would be the learned prototype or a transformed representation
        
        var originalData = new double[currentPattern.dimension()];
        for (int i = 0; i < currentPattern.dimension(); i++) {
            originalData[i] = currentPattern.get(i);
        }
        
        // Add category information as additional dimensions
        var extendedData = new double[originalData.length + 2];
        System.arraycopy(originalData, 0, extendedData, 0, originalData.length);
        extendedData[originalData.length] = categoryId / 1000.0; // Normalized category ID
        extendedData[originalData.length + 1] = currentLevel / 10.0; // Normalized level
        
        return Pattern.of(extendedData);
    }
    
    /**
     * Calculate validation score between original pattern and category prototype.
     */
    private double calculateValidationScore(Pattern original, Pattern prototype, int level) {
        if (original.dimension() != prototype.dimension()) {
            return 0.0;
        }
        
        // Use cosine similarity with level weighting
        double dotProduct = 0.0;
        double originalNorm = 0.0;
        double prototypeNorm = 0.0;
        
        for (int i = 0; i < original.dimension(); i++) {
            double origVal = original.get(i);
            double protoVal = prototype.get(i);

            dotProduct += origVal * protoVal;
            originalNorm += origVal * origVal;
            prototypeNorm += protoVal * protoVal;
        }
        
        if (originalNorm == 0.0 || prototypeNorm == 0.0) {
            return 0.0;
        }
        
        double cosineSimilarity = dotProduct / (Math.sqrt(originalNorm) * Math.sqrt(prototypeNorm));
        
        // Weight by level importance (higher levels get more weight)
        double levelWeight = 1.0 + (level * 0.1);
        return Math.max(0.0, Math.min(1.0, cosineSimilarity * levelWeight));
    }
    
    /**
     * Calculate consistency across hierarchical levels.
     */
    private double calculateHierarchyConsistency(int[] categoryAssignments, int currentLevel) {
        if (currentLevel == 0) {
            return 1.0; // Base level is always consistent
        }
        
        // Simple consistency measure: check if categories are related
        // In practice, this would check learned hierarchical relationships
        double consistency = 1.0;
        
        for (int i = 0; i < currentLevel; i++) {
            // Simplified: assume categories are consistent if they're within range
            int categoryDiff = Math.abs(categoryAssignments[currentLevel] - categoryAssignments[i]);
            consistency *= Math.exp(-categoryDiff / 10.0); // Exponential decay
        }
        
        return Math.max(0.0, Math.min(1.0, consistency));
    }
    
    /**
     * Calculate vigilance adjustment based on validation and consistency scores.
     */
    private double calculateVigilanceAdjustment(double validationScore, double consistencyScore,
                                              double currentVigilance) {
        // If validation is high and consistency is high, can reduce vigilance slightly
        // If validation is low, should increase vigilance
        
        double targetAdjustment = 0.0;
        
        if (validationScore > 0.8 && consistencyScore > 0.8) {
            // High confidence - can be more permissive
            targetAdjustment = -parameters.vigilanceAdaptationRate * 0.5;
        } else if (validationScore < 0.6) {
            // Low confidence - need to be more strict
            targetAdjustment = parameters.vigilanceAdaptationRate;
        }
        
        // Ensure adjustment doesn't push vigilance out of bounds
        double newVigilance = currentVigilance + targetAdjustment;
        newVigilance = Math.max(0.1, Math.min(0.99, newVigilance));
        
        return newVigilance - currentVigilance;
    }
    
    /**
     * Update stability-plasticity balance based on resonance outcome.
     */
    private void updateStabilityPlasticityBalance(boolean successful, ResonanceState resonanceState) {
        resonanceLock.writeLock().lock();
        try {
            if (successful) {
                // Successful resonance - increase stability slightly
                currentStabilityFactor = Math.min(0.9, currentStabilityFactor + 0.01);
                currentPlasticityFactor = Math.max(0.1, currentPlasticityFactor - 0.01);
            } else {
                // Failed resonance - increase plasticity slightly
                currentPlasticityFactor = Math.min(0.9, currentPlasticityFactor + 0.01);
                currentStabilityFactor = Math.max(0.1, currentStabilityFactor - 0.01);
            }
            
            // Update metrics
            updateMetrics(resonanceState);
            
        } finally {
            resonanceLock.writeLock().unlock();
        }
    }
    
    /**
     * Update resonance metrics.
     */
    private void updateMetrics(ResonanceState resonanceState) {
        double successRate = totalResonanceEvents > 0 ? 
            (double) successfulResonances / totalResonanceEvents : 0.0;
        
        var metrics = new ResonanceMetrics(
            totalResonanceEvents,
            successfulResonances,
            successRate,
            currentStabilityFactor,
            currentPlasticityFactor,
            resonanceState.getResonanceStrength(),
            System.currentTimeMillis()
        );
        
        lastMetrics.set(metrics);
    }
    
    /**
     * Get current resonance metrics.
     */
    public ResonanceMetrics getMetrics() {
        resonanceLock.readLock().lock();
        try {
            return lastMetrics.get();
        } finally {
            resonanceLock.readLock().unlock();
        }
    }
    
    /**
     * Reset resonance controller state.
     */
    public void reset() {
        resonanceLock.writeLock().lock();
        try {
            activeResonances.clear();
            currentStabilityFactor = 0.5;
            currentPlasticityFactor = 0.5;
            totalResonanceEvents = 0;
            successfulResonances = 0;
            lastMetrics.set(null);
            
            logger.info("ResonanceController reset completed");
        } finally {
            resonanceLock.writeLock().unlock();
        }
    }
    
    /**
     * Get current stability-plasticity balance.
     */
    public StabilityPlasticityBalance getBalance() {
        resonanceLock.readLock().lock();
        try {
            return new StabilityPlasticityBalance(currentStabilityFactor, currentPlasticityFactor);
        } finally {
            resonanceLock.readLock().unlock();
        }
    }
    
    // Inner classes for results and state management
    
    /**
     * Parameters for resonance control.
     */
    public record ResonanceParameters(
        double minActivationThreshold,
        double minValidationThreshold,
        double minResonanceThreshold,
        boolean enableAdaptiveVigilance,
        double vigilanceAdaptationRate,
        int maxAdaptationAttempts
    ) {
        public static ResonanceParameters defaults() {
            return new ResonanceParameters(0.6, 0.6, 0.7, true, 0.05, 3);
        }
        
        @Override
        public String toString() {
            return String.format("ResonanceParams{minAct=%.2f, minVal=%.2f, minRes=%.2f, adaptive=%s}",
                minActivationThreshold, minValidationThreshold, minResonanceThreshold, 
                enableAdaptiveVigilance);
        }
    }
    
    /**
     * State tracking for active resonance processing.
     */
    private static class ResonanceState {
        private final int resonanceId;
        private final Pattern originalPattern;
        private final long startTime;
        private volatile double resonanceStrength = 0.0;
        
        public ResonanceState(int resonanceId, Pattern originalPattern, long startTime) {
            this.resonanceId = resonanceId;
            this.originalPattern = originalPattern;
            this.startTime = startTime;
        }
        
        public void addLevelResult(int level, HierarchyLevel.LevelProcessingResult result) {
            // Could store level-specific results for analysis
        }
        
        public void addValidationResult(int level, double validationScore, double vigilanceAdjustment) {
            // Could store validation results for analysis
        }
        
        public int getResonanceId() { return resonanceId; }
        public Pattern getOriginalPattern() { return originalPattern; }
        public long getStartTime() { return startTime; }
        public double getResonanceStrength() { return resonanceStrength; }
        public void setResonanceStrength(double resonanceStrength) { this.resonanceStrength = resonanceStrength; }
    }
    
    /**
     * Result of bottom-up processing through hierarchy.
     */
    private record BottomUpResult(
        boolean success,
        int[] categoryAssignments,
        double[] activations,
        double[] processingTimes,
        String errorMessage
    ) {}
    
    /**
     * Result of top-down validation.
     */
    private record TopDownResult(
        boolean success,
        double[] vigilanceAdjustments,
        double[] validationScores,
        String errorMessage
    ) {}
    
    /**
     * Result of resonance processing.
     */
    public static class ResonanceResult {
        private final boolean success;
        private final int[] hierarchicalCategories;
        private final double[] vigilanceAdjustments;
        private final double resonanceStrength;
        private final double processingTimeMs;
        private final Integer resonanceId;
        private final String errorMessage;
        
        private ResonanceResult(boolean success, int[] hierarchicalCategories, 
                               double[] vigilanceAdjustments, double resonanceStrength,
                               double processingTimeMs, Integer resonanceId, String errorMessage) {
            this.success = success;
            this.hierarchicalCategories = hierarchicalCategories;
            this.vigilanceAdjustments = vigilanceAdjustments;
            this.resonanceStrength = resonanceStrength;
            this.processingTimeMs = processingTimeMs;
            this.resonanceId = resonanceId;
            this.errorMessage = errorMessage;
        }
        
        public static ResonanceResult createSuccess(int[] hierarchicalCategories, 
                                                  double[] vigilanceAdjustments,
                                                  double resonanceStrength,
                                                  double processingTimeMs,
                                                  int resonanceId) {
            return new ResonanceResult(true, hierarchicalCategories, vigilanceAdjustments,
                resonanceStrength, processingTimeMs, resonanceId, null);
        }
        
        public static ResonanceResult createFailure(String errorMessage, double processingTimeMs) {
            return new ResonanceResult(false, null, null, 0.0, processingTimeMs, null, errorMessage);
        }
        
        public ResonanceResult withProcessingTime(double processingTimeMs) {
            return new ResonanceResult(success, hierarchicalCategories, vigilanceAdjustments,
                resonanceStrength, processingTimeMs, resonanceId, errorMessage);
        }
        
        // Getters
        public boolean isSuccess() { return success; }
        public int[] getHierarchicalCategories() { return hierarchicalCategories; }
        public double[] getVigilanceAdjustments() { return vigilanceAdjustments; }
        public double getResonanceStrength() { return resonanceStrength; }
        public double getProcessingTimeMs() { return processingTimeMs; }
        public Integer getResonanceId() { return resonanceId; }
        public String getErrorMessage() { return errorMessage; }
    }
    
    /**
     * Resonance processing metrics.
     */
    public record ResonanceMetrics(
        long totalEvents,
        long successfulEvents,
        double successRate,
        double stabilityFactor,
        double plasticityFactor,
        double lastResonanceStrength,
        long timestamp
    ) {
        @Override
        public String toString() {
            return String.format("ResonanceMetrics{events=%d, success=%d(%.1f%%), " +
                               "stability=%.2f, plasticity=%.2f, lastStrength=%.3f}",
                totalEvents, successfulEvents, successRate * 100, 
                stabilityFactor, plasticityFactor, lastResonanceStrength);
        }
    }
    
    /**
     * Current stability-plasticity balance.
     */
    public record StabilityPlasticityBalance(double stability, double plasticity) {
        public boolean isBalanced() {
            return Math.abs(stability - plasticity) < 0.2;
        }
        
        public boolean isStabilityDominant() {
            return stability > plasticity + 0.1;
        }
        
        public boolean isPlasticityDominant() {
            return plasticity > stability + 0.1;
        }
        
        @Override
        public String toString() {
            return String.format("Balance{stability=%.2f, plasticity=%.2f, %s}",
                stability, plasticity,
                isBalanced() ? "balanced" : 
                    (isStabilityDominant() ? "stability-dominant" : "plasticity-dominant"));
        }
    }
}