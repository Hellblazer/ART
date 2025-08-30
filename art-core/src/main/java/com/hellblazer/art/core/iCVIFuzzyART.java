package com.hellblazer.art.core;

import com.hellblazer.art.core.cvi.CalinskiHarabaszIndex;
import com.hellblazer.art.core.cvi.ClusterValidityIndex;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * iCVIFuzzyART - FuzzyART with incremental CVI integration.
 * Optimized for streaming data with incremental CVI updates.
 */
public class iCVIFuzzyART extends CVIART {
    
    // Incremental update management
    private boolean forceNonIncremental = false;
    private boolean wasIncrementallyUpdated = false;
    private boolean wasLastUpdateBatch = false;
    private int cviUpdateCount = 0;
    private int patternsSinceLastUpdate = 0;
    
    // Streaming data management
    private final LinkedList<Pattern> patternBuffer = new LinkedList<>();
    private int maxMemoryPatterns = Integer.MAX_VALUE;
    private int storedPatternCount = 0;
    
    // Update coordination
    private UpdateCoordination updateCoordination = UpdateCoordination.INDEPENDENT;
    private boolean wasLastUpdateSynchronized = false;
    
    // FuzzyART specific
    private boolean useComplementCoding = false;
    private double choiceParameter = 0.0;
    private double learningRate = 1.0;
    
    // Update statistics
    private final CVIUpdateStatistics updateStats = new CVIUpdateStatistics();
    
    // CVI update frequency
    private int cviUpdateFrequency = 1;
    
    public iCVIFuzzyART() {
        super();
    }
    
    // Incremental Update Methods
    
    public boolean wasIncrementallyUpdated() {
        return wasIncrementallyUpdated;
    }
    
    public boolean wasLastUpdateBatch() {
        return wasLastUpdateBatch;
    }
    
    public boolean wasLastUpdateSynchronized() {
        return wasLastUpdateSynchronized;
    }
    
    public int getCVIUpdateCount() {
        return cviUpdateCount;
    }
    
    public void setForceNonIncremental(boolean force) {
        this.forceNonIncremental = force;
    }
    
    // Streaming Data Methods
    
    public int getStoredPatternCount() {
        return Math.min(storedPatternCount, maxMemoryPatterns);
    }
    
    public boolean isUsingComplementCoding() {
        return useComplementCoding;
    }
    
    // Statistics Methods
    
    public CVIUpdateStatistics getCVIUpdateStatistics() {
        return updateStats;
    }
    
    // Learning Methods
    
    @Override
    public LearningResult learn(Pattern pattern, CVIARTParameters params) {
        // Handle iCVIFuzzyARTParameters specifically
        if (params instanceof iCVIFuzzyARTParameters fuzzyParams) {
            return learn(pattern, fuzzyParams);
        }
        
        // Fall back to parent implementation for regular CVIARTParameters
        var result = super.learn(pattern, params);
        
        // Update our streaming buffer and statistics
        patternBuffer.add(pattern);
        storedPatternCount++;
        patternsSinceLastUpdate++;
        
        // Manage memory bounds
        if (storedPatternCount > maxMemoryPatterns) {
            pruneOldestPatterns();
        }
        
        // Update CVIs based on frequency
        if (shouldUpdateCVIs()) {
            updateCVIsIncremental();
            patternsSinceLastUpdate = 0;
        }
        
        return result;
    }
    
    public LearningResult learn(Pattern pattern, iCVIFuzzyARTParameters params) {
        // Update FuzzyART specific parameters
        useComplementCoding = params.isUseComplementCoding();
        choiceParameter = params.getChoiceParameter();
        learningRate = params.getLearningRate();
        maxMemoryPatterns = params.getMaxMemoryPatterns();
        cviUpdateFrequency = params.getCVIUpdateFrequency();
        updateCoordination = params.getUpdateCoordination();
        
        // Apply complement coding if enabled
        Pattern processedPattern = pattern;
        if (useComplementCoding) {
            processedPattern = applyComplementCoding(pattern);
        }
        
        // Use parent's learning mechanism with the processed pattern
        var result = super.learn(processedPattern, params);
        
        // Update our streaming buffer and statistics
        patternBuffer.add(processedPattern);
        storedPatternCount++;
        patternsSinceLastUpdate++;
        
        // Manage memory bounds
        if (storedPatternCount > maxMemoryPatterns) {
            pruneOldestPatterns();
        }
        
        // Update CVIs based on frequency
        if (shouldUpdateCVIs()) {
            updateCVIsIncremental();
            patternsSinceLastUpdate = 0;
        }
        
        return result;
    }
    
    private Pattern applyComplementCoding(Pattern pattern) {
        // Apply complement coding: [a, 1-a]
        int originalSize = pattern.dimension();
        double[] complementCoded = new double[originalSize * 2];
        
        for (int i = 0; i < originalSize; i++) {
            complementCoded[i] = pattern.get(i);
            complementCoded[i + originalSize] = 1.0 - pattern.get(i);
        }
        
        return new DenseVector(complementCoded);
    }
    
    
    private void pruneOldestPatterns() {
        // Remove oldest patterns to maintain memory bound
        while (patternBuffer.size() > maxMemoryPatterns && !patternBuffer.isEmpty()) {
            patternBuffer.removeFirst();
        }
        storedPatternCount = patternBuffer.size();
    }
    
    private boolean shouldUpdateCVIs() {
        return patternsSinceLastUpdate >= cviUpdateFrequency;
    }
    
    private void updateCVIsIncremental() {
        var patterns = super.getPatternHistory();
        if (patterns.size() < 2) return;
        
        // Try incremental update for each CVI
        for (var cvi : getCVIs()) {
            String cviName = cvi.getName();
            boolean usedIncremental = false;
            
            // Try incremental update if not forced to batch
            if (!forceNonIncremental && patterns.size() > 0) {
                Pattern lastPattern = patterns.get(patterns.size() - 1);
                int lastLabel = (patterns.size() - 1) % getCategoryCount();
                
                if (cvi.updateIncremental(lastPattern, lastLabel)) {
                    usedIncremental = true;
                    wasIncrementallyUpdated = true;
                    wasLastUpdateBatch = false;
                    updateStats.recordIncrementalUpdate(cviName);
                }
            }
            
            // Fall back to batch if incremental not supported or forced
            if (!usedIncremental) {
                updateCVIBatch(cvi, patterns);
                wasIncrementallyUpdated = false;
                wasLastUpdateBatch = true;
                updateStats.recordBatchUpdate(cviName);
            }
        }
        
        // Handle update coordination
        if (updateCoordination == UpdateCoordination.SYNCHRONIZED) {
            wasLastUpdateSynchronized = true;
        } else {
            wasLastUpdateSynchronized = false;
        }
        
        cviUpdateCount++;
    }
    
    private void updateCVIBatch(ClusterValidityIndex cvi, List<Pattern> patterns) {
        // Generate labels for batch update
        int[] labels = new int[patterns.size()];
        for (int i = 0; i < patterns.size(); i++) {
            labels[i] = i % Math.max(1, getCategoryCount());
        }
        
        // Calculate centroids
        var centroids = calculateCentroidsForBatch(patterns, labels);
        
        try {
            double score = cvi.calculate(patterns, labels, centroids);
            updateCVIScore(cvi.getName(), score);
        } catch (Exception e) {
            // Handle failure gracefully
            System.err.println("Batch CVI update failed for " + cvi.getName());
        }
    }
    
    private List<Pattern> calculateCentroidsForBatch(List<Pattern> patterns, int[] labels) {
        int maxLabel = Arrays.stream(labels).max().orElse(0);
        List<Pattern> centroids = new ArrayList<>();
        
        for (int k = 0; k <= maxLabel; k++) {
            List<Pattern> clusterPatterns = new ArrayList<>();
            for (int i = 0; i < patterns.size(); i++) {
                if (labels[i] == k) {
                    clusterPatterns.add(patterns.get(i));
                }
            }
            
            if (!clusterPatterns.isEmpty()) {
                centroids.add(calculateCentroid(clusterPatterns));
            }
        }
        
        return centroids;
    }
    
    private Pattern calculateCentroid(List<Pattern> patterns) {
        if (patterns.isEmpty()) return null;
        
        int dimensions = patterns.get(0).dimension();
        double[] centroid = new double[dimensions];
        
        for (var pattern : patterns) {
            for (int i = 0; i < dimensions; i++) {
                centroid[i] += pattern.get(i);
            }
        }
        
        for (int i = 0; i < dimensions; i++) {
            centroid[i] /= patterns.size();
        }
        
        return new DenseVector(centroid);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        // Override to include choice parameter in FuzzyART activation
        // T_j = |I ∧ w_j| / (α + |w_j|)
        // where α is the choice parameter
        
        double sumMin = 0.0;
        double sumWeight = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            sumMin += Math.min(input.get(i), weight.get(i));
            sumWeight += weight.get(i);
        }
        
        // Apply choice parameter (alpha)
        double denominator = choiceParameter + sumWeight;
        if (denominator == 0) {
            return 0.0;
        }
        
        return sumMin / denominator;
    }
    
    // getCVIs() is now provided by parent class as a protected method
    // No need to override it here
    
    private void updateCVIScore(String cviName, double score) {
        // Update score in parent's score tracking
        var scores = getCurrentCVIScores();
        scores.put(cviName, score);
    }
    
    /**
     * Parameters for iCVIFuzzyART
     */
    public static class iCVIFuzzyARTParameters extends CVIARTParameters {
        private int cviUpdateFrequency = 1;
        private int maxMemoryPatterns = Integer.MAX_VALUE;
        private UpdateCoordination updateCoordination = UpdateCoordination.INDEPENDENT;
        private boolean useComplementCoding = false;
        private double choiceParameter = 0.0;
        private double learningRate = 1.0;
        private double vigilance = 0.5;
        
        // Getters and setters
        public int getCVIUpdateFrequency() { return cviUpdateFrequency; }
        public void setCVIUpdateFrequency(int freq) { this.cviUpdateFrequency = freq; }
        
        public int getMaxMemoryPatterns() { return maxMemoryPatterns; }
        public void setMaxMemoryPatterns(int max) { this.maxMemoryPatterns = max; }
        
        public boolean isAdaptiveVigilance() { return super.isAdaptiveVigilance(); }
        public void setAdaptiveVigilance(boolean adaptive) { super.setAdaptiveVigilance(adaptive); }
        
        public UpdateCoordination getUpdateCoordination() { return updateCoordination; }
        public void setUpdateCoordination(UpdateCoordination coord) { this.updateCoordination = coord; }
        
        public boolean isUseComplementCoding() { return useComplementCoding; }
        public void setUseComplementCoding(boolean use) { this.useComplementCoding = use; }
        
        public double getChoiceParameter() { return choiceParameter; }
        public void setChoiceParameter(double choice) { this.choiceParameter = choice; }
        
        public double getLearningRate() { return learningRate; }
        public void setLearningRate(double rate) { this.learningRate = rate; }
        
        public double getVigilance() { return vigilance; }
        public void setVigilance(double v) {
            if (Double.isNaN(v) || v < 0) {
                this.vigilance = 0.5;
            } else {
                this.vigilance = Math.min(1.0, v);
            }
            super.setInitialVigilance(this.vigilance);
        }
    }
    
    /**
     * CVI update statistics
     */
    public static class CVIUpdateStatistics {
        private final Map<String, Integer> incrementalUpdates = new ConcurrentHashMap<>();
        private final Map<String, Integer> batchUpdates = new ConcurrentHashMap<>();
        
        public void recordIncrementalUpdate(String cviName) {
            incrementalUpdates.merge(cviName, 1, Integer::sum);
        }
        
        public void recordBatchUpdate(String cviName) {
            batchUpdates.merge(cviName, 1, Integer::sum);
        }
        
        public int getIncrementalUpdates(String cviName) {
            return incrementalUpdates.getOrDefault(cviName, 0);
        }
        
        public int getBatchUpdates(String cviName) {
            return batchUpdates.getOrDefault(cviName, 0);
        }
    }
    
    /**
     * Update coordination strategies
     */
    public enum UpdateCoordination {
        INDEPENDENT,   // Each CVI updates independently
        SYNCHRONIZED,  // All CVIs update together
        ADAPTIVE       // System decides coordination
    }
}