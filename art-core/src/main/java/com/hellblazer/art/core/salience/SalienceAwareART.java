/*
 * Copyright (c) 2024 Hal Hildebrand. All rights reserved.
 * 
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core.salience;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.weights.FuzzyWeight;
import com.hellblazer.art.core.parameters.FuzzyParameters;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Salience-Aware Adaptive Resonance Theory implementation.
 * Extends Fuzzy ART with cluster-wise salient feature modeling for 
 * improved clustering of large-scale sparse data.
 */
public class SalienceAwareART extends BaseART {
    
    // Core parameters
    private double vigilance = 0.75;
    private double learningRate = 1.0;
    private double alpha = 0.001;
    
    // Salience-specific parameters
    private final Map<Integer, double[]> clusterSalience;
    private final List<SalienceCalculator> salienceCalculators;
    private double salienceUpdateRate = 0.01;
    
    // Cluster-specific parameters (self-adaptive)
    private final Map<Integer, Double> clusterVigilance;
    private final Map<Integer, Double> clusterLearningRate;
    
    // Sparse data handling
    private boolean useSparseMode = true;
    private double sparsityThreshold = 0.01;
    
    // Statistical measures for each cluster
    private final Map<Integer, ClusterStatistics> clusterStats;
    
    public SalienceAwareART() {
        super();
        this.clusterSalience = new ConcurrentHashMap<>();
        this.clusterVigilance = new ConcurrentHashMap<>();
        this.clusterLearningRate = new ConcurrentHashMap<>();
        this.clusterStats = new ConcurrentHashMap<>();
        this.salienceCalculators = initializeSalienceCalculators();
    }
    
    private List<SalienceCalculator> initializeSalienceCalculators() {
        List<SalienceCalculator> calculators = new ArrayList<>();
        calculators.add(new FrequencySalienceCalculator());
        calculators.add(new MeanSalienceCalculator());
        calculators.add(new StatisticalSalienceCalculator());
        return calculators;
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        // Get category index from weight
        int categoryIndex = categories.indexOf(weight);
        if (categoryIndex == -1) {
            // New category, use standard calculation
            return standardCalculateActivation(input, weight);
        }
        
        double[] salience = clusterSalience.getOrDefault(categoryIndex, 
                                                        getDefaultSalience(input.dimension()));
        
        // Convert to complement-coded form if needed
        Pattern complementInput = input;
        if (weight instanceof FuzzyWeight) {
            var fuzzyWeight = FuzzyWeight.fromInput(input);
            complementInput = Pattern.of(fuzzyWeight.data());
        }
        
        // Salience-weighted choice function
        double numerator = 0.0;
        double denominator = alpha;
        
        for (int i = 0; i < complementInput.dimension(); i++) {
            double fuzzyAnd = Math.min(complementInput.get(i), weight.get(i));
            numerator += salience[i] * fuzzyAnd;
            denominator += salience[i] * weight.get(i);
        }
        
        return denominator > 0 ? numerator / denominator : 0;
    }
    
    private double standardCalculateActivation(Pattern input, WeightVector weight) {
        // Standard Fuzzy ART activation without salience
        Pattern complementInput = input;
        if (weight instanceof FuzzyWeight) {
            var fuzzyWeight = FuzzyWeight.fromInput(input);
            complementInput = Pattern.of(fuzzyWeight.data());
        }
        
        // Calculate intersection manually
        double intersectionNorm = 0.0;
        double weightNorm = 0.0;
        for (int i = 0; i < complementInput.dimension(); i++) {
            intersectionNorm += Math.min(complementInput.get(i), weight.get(i));
            weightNorm += weight.get(i);
        }
        var denominator = alpha + weightNorm;
        
        return denominator > 0 ? intersectionNorm / denominator : 0;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        int categoryIndex = categories.indexOf(weight);
        
        // Get cluster-specific vigilance or use default
        double currentVigilance = (categoryIndex >= 0) 
            ? clusterVigilance.getOrDefault(categoryIndex, vigilance)
            : vigilance;
        
        if (categoryIndex == -1) {
            // New category, use standard vigilance test
            return standardCheckVigilance(input, weight, currentVigilance);
        }
        
        double[] salience = clusterSalience.getOrDefault(categoryIndex,
                                                        getDefaultSalience(input.dimension()));
        
        // Convert to complement-coded form if needed
        Pattern complementInput = input;
        if (weight instanceof FuzzyWeight) {
            var fuzzyWeight = FuzzyWeight.fromInput(input);
            complementInput = Pattern.of(fuzzyWeight.data());
        }
        
        // Salience-weighted match function
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (int i = 0; i < complementInput.dimension(); i++) {
            double fuzzyAnd = Math.min(complementInput.get(i), weight.get(i));
            numerator += salience[i] * fuzzyAnd;
            denominator += salience[i] * complementInput.get(i);
        }
        
        double matchValue = denominator > 0 ? numerator / denominator : 0;
        
        return matchValue >= currentVigilance 
            ? new MatchResult.Accepted(matchValue, currentVigilance)
            : new MatchResult.Rejected(matchValue, currentVigilance);
    }
    
    private MatchResult standardCheckVigilance(Pattern input, WeightVector weight, double vigilanceParam) {
        Pattern complementInput = input;
        if (weight instanceof FuzzyWeight) {
            var fuzzyWeight = FuzzyWeight.fromInput(input);
            complementInput = Pattern.of(fuzzyWeight.data());
        }
        
        // Calculate intersection and norms manually
        double intersectionNorm = 0.0;
        double inputNorm = 0.0;
        for (int i = 0; i < complementInput.dimension(); i++) {
            intersectionNorm += Math.min(complementInput.get(i), weight.get(i));
            inputNorm += complementInput.get(i);
        }
        
        if (inputNorm == 0.0) {
            return new MatchResult.Rejected(0.0, vigilanceParam);
        }
        
        double matchValue = intersectionNorm / inputNorm;
        return matchValue >= vigilanceParam 
            ? new MatchResult.Accepted(matchValue, vigilanceParam)
            : new MatchResult.Rejected(matchValue, vigilanceParam);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        int categoryIndex = categories.indexOf(currentWeight);
        
        // Get cluster-specific learning rate
        double beta = (categoryIndex >= 0)
            ? clusterLearningRate.getOrDefault(categoryIndex, learningRate)
            : learningRate;
        
        // Convert to complement-coded form if needed
        Pattern complementInput = input;
        if (currentWeight instanceof FuzzyWeight) {
            var fuzzyWeight = FuzzyWeight.fromInput(input);
            complementInput = Pattern.of(fuzzyWeight.data());
        }
        
        double[] salience = (categoryIndex >= 0)
            ? clusterSalience.getOrDefault(categoryIndex, getDefaultSalience(complementInput.dimension()))
            : getDefaultSalience(complementInput.dimension());
        
        // Update with salience-weighted learning
        double[] newData = new double[currentWeight.dimension()];
        for (int i = 0; i < currentWeight.dimension(); i++) {
            double fuzzyAnd = Math.min(complementInput.get(i), currentWeight.get(i));
            newData[i] = beta * salience[i] * fuzzyAnd + 
                        (1 - beta * salience[i]) * currentWeight.get(i);
            
            // Apply statistical bounds if available
            if (categoryIndex >= 0) {
                ClusterStatistics stats = clusterStats.get(categoryIndex);
                if (stats != null && stats.getSampleCount() > 1 && i < stats.getDimension()) {
                    double mean = stats.getFeatureMean(i);
                    double stdDev = stats.getFeatureStandardDeviation(i);
                    newData[i] = Math.max(mean - 2 * stdDev, Math.min(mean + 2 * stdDev, newData[i]));
                }
            }
        }
        
        // Update cluster statistics
        if (categoryIndex >= 0) {
            updateClusterStatistics(complementInput, categoryIndex);
            updateSalienceWeights(complementInput, categoryIndex);
            adaptClusterParameters(categoryIndex);
        }
        
        // Create new weight vector
        if (currentWeight instanceof FuzzyWeight fuzzy) {
            return new FuzzyWeight(newData, fuzzy.originalDimension());
        } else {
            // Fallback: create FuzzyWeight with half the dimension as original
            return new FuzzyWeight(newData, newData.length / 2);
        }
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        // Create complement-coded initial weight
        var fuzzyWeight = FuzzyWeight.fromInput(input);
        
        // Initialize statistics and salience for new category
        int newIndex = categories.size(); // Will be the next index
        
        // Initialize salience weights
        double[] initialSalience = getDefaultSalience(fuzzyWeight.dimension());
        clusterSalience.put(newIndex, initialSalience);
        
        // Initialize cluster statistics
        ClusterStatistics stats = new ClusterStatistics(fuzzyWeight.dimension());
        stats.updateStatistics(fuzzyWeight.data());
        clusterStats.put(newIndex, stats);
        
        // Initialize cluster-specific parameters
        clusterVigilance.put(newIndex, vigilance);
        clusterLearningRate.put(newIndex, learningRate);
        
        return fuzzyWeight;
    }
    
    private void updateClusterStatistics(Pattern input, int categoryIndex) {
        ClusterStatistics stats = clusterStats.get(categoryIndex);
        if (stats == null) {
            stats = new ClusterStatistics(input.dimension());
            clusterStats.put(categoryIndex, stats);
        }
        
        double[] inputData = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputData[i] = input.get(i);
        }
        stats.updateStatistics(inputData);
    }
    
    private void updateSalienceWeights(Pattern input, int categoryIndex) {
        ClusterStatistics stats = clusterStats.get(categoryIndex);
        if (stats == null || stats.getSampleCount() < 2) {
            return;
        }
        
        // Convert pattern to sparse vector for salience calculation
        double[] inputData = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputData[i] = input.get(i);
        }
        var sparseInput = new SparseVector(inputData, sparsityThreshold);
        
        double[] newSalience = new double[input.dimension()];
        
        // Combine multiple salience measures
        for (SalienceCalculator calculator : salienceCalculators) {
            double[] measure = calculator.calculate(stats, sparseInput);
            for (int i = 0; i < newSalience.length; i++) {
                newSalience[i] += measure[i] / salienceCalculators.size();
            }
        }
        
        // Smooth update with existing salience
        double[] currentSalience = clusterSalience.getOrDefault(categoryIndex,
                                                              getDefaultSalience(input.dimension()));
        for (int i = 0; i < newSalience.length; i++) {
            newSalience[i] = (1 - salienceUpdateRate) * currentSalience[i] + 
                           salienceUpdateRate * newSalience[i];
            
            // Normalize to [0, 1]
            newSalience[i] = Math.max(0.0, Math.min(1.0, newSalience[i]));
        }
        
        clusterSalience.put(categoryIndex, newSalience);
    }
    
    private void adaptClusterParameters(int categoryIndex) {
        ClusterStatistics stats = clusterStats.get(categoryIndex);
        if (stats == null || stats.getSampleCount() < 10) {
            return;  // Need sufficient samples for adaptation
        }
        
        // Calculate average variance
        double avgVariance = 0.0;
        for (int i = 0; i < stats.getDimension(); i++) {
            avgVariance += stats.getFeatureVariance(i);
        }
        avgVariance /= stats.getDimension();
        
        // Adapt vigilance based on cluster coherence
        double adaptedVigilance = vigilance * (1.0 + Math.exp(-avgVariance));
        clusterVigilance.put(categoryIndex, Math.min(0.95, adaptedVigilance));
        
        // Adapt learning rate based on cluster stability
        double stability = 1.0 / (1.0 + Math.log(stats.getSampleCount()));
        double adaptedLearningRate = learningRate * stability;
        clusterLearningRate.put(categoryIndex, Math.max(0.01, adaptedLearningRate));
    }
    
    private double[] getDefaultSalience(int dimension) {
        double[] salience = new double[dimension];
        Arrays.fill(salience, 1.0 / dimension);  // Equal initial weights
        return salience;
    }
    
    @Override
    protected String getAlgorithmName() {
        return "SalienceAwareART";
    }
    
    // Getters and setters for testing
    
    public double getVigilance() { return vigilance; }
    public double getLearningRate() { return learningRate; }
    public double getAlpha() { return alpha; }
    public double getSalienceUpdateRate() { return salienceUpdateRate; }
    public boolean isUsingSparseMode() { return useSparseMode; }
    public double getSparsityThreshold() { return sparsityThreshold; }
    public int getNumberOfCategories() { return categories.size(); }
    
    public Map<Integer, double[]> getClusterSalience() { 
        return new HashMap<>(clusterSalience); 
    }
    
    public Map<Integer, Double> getClusterVigilance() { 
        return new HashMap<>(clusterVigilance); 
    }
    
    public Map<Integer, Double> getClusterLearningRate() { 
        return new HashMap<>(clusterLearningRate); 
    }
    
    public ClusterStatistics getClusterStatistics(int index) {
        return clusterStats.get(index);
    }
    
    public WeightVector getPrototype(int index) {
        if (index >= 0 && index < categories.size()) {
            return (WeightVector) categories.get(index);
        }
        return null;
    }
    
    public void setClusterSalience(int index, double[] salience) {
        clusterSalience.put(index, salience);
    }
    
    public long estimateMemoryUsage() {
        long total = 0;
        // Estimate memory for salience arrays
        for (double[] salience : clusterSalience.values()) {
            total += salience.length * 8L;
        }
        // Estimate memory for statistics
        total += clusterStats.size() * 1000L; // Rough estimate per stats object
        return total;
    }
    
    // Simplified stepFit for testing
    public ActivationResult stepFit(Pattern input) {
        var params = new FuzzyParameters(vigilance, alpha, learningRate);
        return super.stepFit(input, params);
    }
    
    /**
     * Builder pattern for configuration
     */
    public static class Builder {
        private double vigilance = 0.75;
        private double learningRate = 1.0;
        private double alpha = 0.001;
        private double salienceUpdateRate = 0.01;
        private boolean useSparseMode = true;
        private double sparsityThreshold = 0.01;
        
        public Builder vigilance(double vigilance) {
            if (vigilance < 0 || vigilance > 1) {
                throw new IllegalArgumentException("Vigilance must be in [0, 1]");
            }
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate < 0 || learningRate > 1) {
                throw new IllegalArgumentException("Learning rate must be in [0, 1]");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder alpha(double alpha) {
            if (alpha < 0) {
                throw new IllegalArgumentException("Alpha must be non-negative");
            }
            this.alpha = alpha;
            return this;
        }
        
        public Builder salienceUpdateRate(double rate) {
            if (rate < 0 || rate > 1) {
                throw new IllegalArgumentException("Salience update rate must be in [0, 1]");
            }
            this.salienceUpdateRate = rate;
            return this;
        }
        
        public Builder useSparseMode(boolean sparse) {
            this.useSparseMode = sparse;
            return this;
        }
        
        public Builder sparsityThreshold(double threshold) {
            if (threshold < 0) {
                throw new IllegalArgumentException("Sparsity threshold must be non-negative");
            }
            this.sparsityThreshold = threshold;
            return this;
        }
        
        public SalienceAwareART build() {
            var art = new SalienceAwareART();
            art.vigilance = this.vigilance;
            art.learningRate = this.learningRate;
            art.alpha = this.alpha;
            art.salienceUpdateRate = this.salienceUpdateRate;
            art.useSparseMode = this.useSparseMode;
            art.sparsityThreshold = this.sparsityThreshold;
            return art;
        }
    }

    @Override
    public void close() throws Exception {
        // No-op for vanilla implementation
    }
}