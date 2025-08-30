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
package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.parameters.DualVigilanceParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.FuzzyWeight;

import java.util.*;

/**
 * DualVigilanceART implementation with dual-threshold system for improved noise handling.
 * 
 * This algorithm extends standard ART with two vigilance parameters:
 * - Upper vigilance (rho): Standard matching criterion for category assignment
 * - Lower vigilance (rho_lb): Defines boundary nodes for noise tolerance
 * 
 * Patterns that fail upper vigilance but pass lower vigilance become "boundary nodes"
 * that provide noise tolerance without corrupting core clusters.
 * 
 * Based on: Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
 * "Dual vigilance fuzzy adaptive resonance theory"
 * Neural Networks, 109, 1-5.
 * 
 * @author Hal Hildebrand
 */
public class DualVigilanceART extends BaseART {
    
    // Track which categories are boundary nodes
    private final Set<Integer> boundaryNodes = new HashSet<>();
    
    // Map categories to cluster groups for structure analysis
    private final Map<Integer, Integer> categoryMap = new HashMap<>();
    
    // Track category statistics
    private final Map<Integer, Map<String, Object>> categoryStats = new HashMap<>();
    
    /**
     * Default constructor creates empty ART network.
     */
    public DualVigilanceART() {
        super();
    }
    
    /**
     * Constructor with initial categories.
     * @param initialCategories initial weight vectors
     */
    public DualVigilanceART(List<? extends WeightVector> initialCategories) {
        super(initialCategories);
        for (int i = 0; i < initialCategories.size(); i++) {
            categoryMap.put(i, i);
            initializeCategoryStatistics(i);
        }
    }
    
    // ==================== CORE ART METHODS ====================
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        var params = validateAndCastParameters(parameters);
        
        // Fuzzy ART activation: |input ∧ weight| / |weight|
        var complementCoded = ensureComplementCoding(input, weight);
        var weightVector = Pattern.of(((FuzzyWeight)weight).data());
        var intersection = complementCoded.min(weightVector);
        
        return intersection.l1Norm() / (params.alpha() + weightVector.l1Norm());
    }
    
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        var params = validateAndCastParameters(parameters);
        
        // Calculate match value (fuzzy ART style)
        var complementCoded = ensureComplementCoding(input, weight);
        var weightVector = Pattern.of(((FuzzyWeight)weight).data());
        var intersection = complementCoded.min(weightVector);
        var matchValue = intersection.l1Norm() / complementCoded.l1Norm();
        
        // Find the category index by comparing weight references
        int categoryIndex = findCategoryIndex(weight);
        
        // Check upper vigilance first
        if (params.passesUpperVigilance(matchValue)) {
            // Passes upper vigilance - accept the match
            if (categoryIndex >= 0) {
                // Don't remove from boundaryNodes - once a boundary, always a boundary
                updateCategoryStatistics(categoryIndex, matchValue);
            }
            return new MatchResult.Accepted(matchValue, params.rho());
        }
        
        // Failed upper vigilance, now check lower vigilance for boundary node
        if (params.passesLowerVigilance(matchValue)) {
            // Check if this is an existing category
            if (categoryIndex >= 0) {
                // Existing category - accept if it's already a boundary node
                if (boundaryNodes.contains(categoryIndex)) {
                    updateCategoryStatistics(categoryIndex, matchValue);
                    return new MatchResult.Accepted(matchValue, params.rhoLb());
                }
            }
            
            // New pattern that would create a boundary node
            if (shouldCreateBoundaryNode) {
                shouldCreateBoundaryNode = false; // Reset flag
                nextCategoryIsBoundary = true; // Mark next created category as boundary
                // Return rejected to force new category creation
                return new MatchResult.Rejected(matchValue, params.rho());
            }
        }
        
        // Failed both thresholds - reject
        return new MatchResult.Rejected(matchValue, params.rho());
    }
    
    // Track if the next category created should be a boundary node
    private boolean nextCategoryIsBoundary = false;
    // Track if we should check for boundary node creation
    private boolean shouldCreateBoundaryNode = true;
    // Track the index where boundary node should be created
    private int pendingBoundaryNodeIndex = -1;
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        var params = validateAndCastParameters(parameters);
        var complementCoded = ensureComplementCoding(input, null);
        
        // Initialize new weight as the input (FuzzyART style)
        var initialData = new double[complementCoded.dimension()];
        for (int i = 0; i < initialData.length; i++) {
            initialData[i] = complementCoded.get(i);
        }
        
        var newWeight = new FuzzyWeight(initialData, input.dimension());
        
        // If this should be a boundary node, record the index where it will be added
        if (nextCategoryIsBoundary) {
            pendingBoundaryNodeIndex = getCategoryCount(); // This will be the index after adding
            // Add to boundary nodes immediately since we know the index
            boundaryNodes.add(pendingBoundaryNodeIndex);
            initializeCategoryStatistics(pendingBoundaryNodeIndex);
            nextCategoryIsBoundary = false; // Reset flag
            pendingBoundaryNodeIndex = -1; // Clear pending
        }
        
        // Reset the boundary check flag for next iteration
        shouldCreateBoundaryNode = true;
        
        return newWeight;
    }
    
    /**
     * Find the category index for a given weight vector by reference comparison.
     */
    private int findCategoryIndex(WeightVector weight) {
        for (int i = 0; i < getCategoryCount(); i++) {
            if (getCategory(i) == weight) {
                return i;
            }
        }
        return -1; // Not found
    }
    
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        // Find the category index using the helper method
        int categoryIndex = findCategoryIndex(currentWeight);
        
        // Boundary nodes don't get weight updates
        if (categoryIndex >= 0 && boundaryNodes.contains(categoryIndex)) {
            return currentWeight;
        }
        
        // Normal weight update for core nodes
        var params = validateAndCastParameters(parameters);
        var complementCoded = ensureComplementCoding(input, currentWeight);
        var weightData = ((FuzzyWeight)currentWeight).data();
        var beta = params.beta();
        var newData = new double[weightData.length];
        
        // Apply fuzzy learning rule: β * min(input, weight) + (1-β) * weight
        for (int i = 0; i < weightData.length; i++) {
            var minValue = Math.min(complementCoded.get(i), weightData[i]);
            newData[i] = beta * minValue + (1.0 - beta) * weightData[i];
        }
        
        var originalDim = ((FuzzyWeight)currentWeight).originalDimension();
        return new FuzzyWeight(newData, originalDim);
    }
    
    @Override
    public String getAlgorithmName() {
        return "DualVigilanceART";
    }
    
    
    /**
     * Train the network on multiple patterns.
     * @param patterns array of input patterns
     * @param parameters DualVigilanceParameters
     */
    public void fit(Pattern[] patterns, Object parameters) {
        for (var pattern : patterns) {
            var result = stepFit(pattern, parameters);
            // Check if we need to mark a category as boundary after it was added
            if (pendingBoundaryNodeIndex >= 0 && result instanceof ActivationResult.Success success) {
                if (success.categoryIndex() == pendingBoundaryNodeIndex) {
                    boundaryNodes.add(pendingBoundaryNodeIndex);
                    initializeCategoryStatistics(pendingBoundaryNodeIndex);
                }
                pendingBoundaryNodeIndex = -1; // Reset
            }
        }
    }
    
    /**
     * Predict categories for multiple patterns.
     * @param patterns array of input patterns
     * @param parameters DualVigilanceParameters
     * @return array of category indices
     */
    public int[] predict(Pattern[] patterns, Object parameters) {
        var predictions = new int[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            var result = stepFit(patterns[i], parameters);
            handlePendingBoundaryNode(result);
            if (result instanceof ActivationResult.Success success) {
                predictions[i] = success.categoryIndex();
            } else {
                predictions[i] = -1; // No match
            }
        }
        return predictions;
    }
    
    /**
     * Predict category for a single pattern.
     * @param pattern input pattern
     * @param parameters DualVigilanceParameters
     * @return category index or -1 if no match
     */
    public int predict(Pattern pattern, Object parameters) {
        var result = stepFit(pattern, parameters);
        handlePendingBoundaryNode(result);
        if (result instanceof ActivationResult.Success success) {
            return success.categoryIndex();
        }
        return -1;
    }
    
    /**
     * Handle pending boundary node marking after category creation.
     */
    private void handlePendingBoundaryNode(ActivationResult result) {
        if (pendingBoundaryNodeIndex >= 0 && result instanceof ActivationResult.Success success) {
            if (success.categoryIndex() == pendingBoundaryNodeIndex) {
                boundaryNodes.add(pendingBoundaryNodeIndex);
                initializeCategoryStatistics(pendingBoundaryNodeIndex);
            }
            pendingBoundaryNodeIndex = -1; // Reset
        }
    }
    
    // ==================== DUAL VIGILANCE SPECIFIC METHODS ====================
    
    /**
     * Check if a category is marked as a boundary node.
     */
    public boolean isBoundaryNode(int categoryIndex) {
        return boundaryNodes.contains(categoryIndex);
    }
    
    /**
     * Get the count of boundary nodes.
     */
    public int getBoundaryNodeCount() {
        return boundaryNodes.size();
    }
    
    /**
     * Check if there are any boundary nodes.
     */
    public boolean hasBoundaryNodes() {
        return !boundaryNodes.isEmpty();
    }
    
    /**
     * Get the set of boundary node indices.
     */
    public Set<Integer> getBoundaryNodes() {
        return new HashSet<>(boundaryNodes);
    }
    
    /**
     * Get category statistics for analysis.
     */
    public Map<String, Object> getCategoryStatistics(int categoryIndex) {
        return categoryStats.getOrDefault(categoryIndex, Collections.emptyMap());
    }
    
    /**
     * Get clustering metrics including boundary node information.
     */
    public Map<String, Object> getClusteringMetrics(Pattern[] data, int[] labels) {
        var metrics = new HashMap<String, Object>();
        
        // Calculate boundary node ratio
        double boundaryRatio = (double) getBoundaryNodeCount() / getCategoryCount();
        metrics.put("boundary_node_ratio", boundaryRatio);
        
        // Count core clusters (non-boundary nodes)
        int coreClusterCount = getCategoryCount() - getBoundaryNodeCount();
        metrics.put("core_cluster_count", coreClusterCount);
        
        // Calculate noise isolation score
        double noiseIsolation = calculateNoiseIsolationScore(labels);
        metrics.put("noise_isolation_score", noiseIsolation);
        
        return metrics;
    }
    
    /**
     * Serialize the DualVigilanceART state.
     */
    public byte[] serialize() {
        try {
            var baos = new java.io.ByteArrayOutputStream();
            var oos = new java.io.ObjectOutputStream(baos);
            
            // Write categories
            oos.writeInt(getCategoryCount());
            for (int i = 0; i < getCategoryCount(); i++) {
                var weight = (FuzzyWeight) getCategory(i);
                oos.writeInt(weight.originalDimension());
                oos.writeObject(weight.data());
            }
            
            // Write boundary nodes
            oos.writeInt(boundaryNodes.size());
            for (int node : boundaryNodes) {
                oos.writeInt(node);
            }
            
            // Write category map
            oos.writeInt(categoryMap.size());
            for (var entry : categoryMap.entrySet()) {
                oos.writeInt(entry.getKey());
                oos.writeInt(entry.getValue());
            }
            
            oos.close();
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize", e);
        }
    }
    
    /**
     * Deserialize a DualVigilanceART instance.
     */
    public static DualVigilanceART deserialize(byte[] data) {
        try {
            var bais = new java.io.ByteArrayInputStream(data);
            var ois = new java.io.ObjectInputStream(bais);
            
            // Read categories
            int categoryCount = ois.readInt();
            var categories = new ArrayList<FuzzyWeight>();
            for (int i = 0; i < categoryCount; i++) {
                int originalDim = ois.readInt();
                double[] weightData = (double[]) ois.readObject();
                categories.add(new FuzzyWeight(weightData, originalDim));
            }
            
            var art = new DualVigilanceART(categories);
            
            // Read boundary nodes
            int boundaryCount = ois.readInt();
            for (int i = 0; i < boundaryCount; i++) {
                art.boundaryNodes.add(ois.readInt());
            }
            
            // Read category map
            int mapSize = ois.readInt();
            for (int i = 0; i < mapSize; i++) {
                int key = ois.readInt();
                int value = ois.readInt();
                art.categoryMap.put(key, value);
            }
            
            ois.close();
            return art;
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize", e);
        }
    }
    
    // ==================== HELPER METHODS ====================
    
    private DualVigilanceParameters validateAndCastParameters(Object parameters) {
        if (!(parameters instanceof DualVigilanceParameters)) {
            throw new IllegalArgumentException(
                "Parameters must be DualVigilanceParameters, got: " + 
                (parameters == null ? "null" : parameters.getClass().getSimpleName()));
        }
        return (DualVigilanceParameters) parameters;
    }
    
    private Pattern ensureComplementCoding(Pattern input, WeightVector weight) {
        // Always apply complement coding
        int originalDim = input.dimension();
        int expectedDim = originalDim * 2;
        
        // Apply complement coding
        var data = new double[expectedDim];
        for (int i = 0; i < originalDim; i++) {
            data[i] = input.get(i);
            data[i + originalDim] = 1.0 - input.get(i);
        }
        
        return Pattern.of(data);
    }
    
    private void initializeCategoryStatistics(int categoryIndex) {
        var stats = new HashMap<String, Object>();
        stats.put("sample_count", 0);
        stats.put("avg_match_value", 0.0);
        stats.put("is_boundary", false);
        categoryStats.put(categoryIndex, stats);
    }
    
    private void updateCategoryStatistics(int categoryIndex, double matchValue) {
        var stats = categoryStats.computeIfAbsent(categoryIndex, k -> {
            var newStats = new HashMap<String, Object>();
            newStats.put("sample_count", 0);
            newStats.put("avg_match_value", 0.0);
            newStats.put("is_boundary", false);
            return newStats;
        });
        
        int count = (Integer) stats.get("sample_count");
        double avgMatch = (Double) stats.get("avg_match_value");
        
        // Update running average
        avgMatch = (avgMatch * count + matchValue) / (count + 1);
        count++;
        
        stats.put("sample_count", count);
        stats.put("avg_match_value", avgMatch);
        stats.put("is_boundary", boundaryNodes.contains(categoryIndex));
    }
    
    private double calculateNoiseIsolationScore(int[] labels) {
        if (labels.length == 0 || boundaryNodes.isEmpty()) {
            return 0.0;
        }
        
        // Count how many samples are assigned to boundary nodes
        int boundaryAssignments = 0;
        for (int label : labels) {
            if (boundaryNodes.contains(label)) {
                boundaryAssignments++;
            }
        }
        
        // Score is ratio of boundary assignments to total boundary nodes
        // Higher score means boundary nodes are effectively isolating noise
        return (double) boundaryAssignments / (boundaryNodes.size() * labels.length);
    }
}