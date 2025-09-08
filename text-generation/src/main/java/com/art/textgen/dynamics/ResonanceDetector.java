package com.art.textgen.dynamics;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ART-style Resonance Detector
 * Implements resonance between bottom-up input and top-down expectations
 * Critical for stability-plasticity balance
 */
public class ResonanceDetector {
    
    private static final double DEFAULT_VIGILANCE = 0.7;
    private static final double RESONANCE_THRESHOLD = 0.8;
    private static final int MAX_ITERATIONS = 100;
    
    private double vigilance;
    private final Map<Integer, CategoryNode> categories;
    private int nextCategoryId;
    
    public static class CategoryNode {
        public final int id;
        public double[] prototype;
        public double[] bottomUpWeights;
        public double[] topDownWeights;
        public double activation;
        public double matchScore;
        public int accessCount;
        public long lastAccess;
        public double commitment;  // How committed this node is to its pattern
        
        public CategoryNode(int id, int dimensionality) {
            this.id = id;
            this.prototype = new double[dimensionality];
            this.bottomUpWeights = new double[dimensionality];
            this.topDownWeights = new double[dimensionality];
            this.activation = 0.0;
            this.matchScore = 0.0;
            this.accessCount = 0;
            this.lastAccess = System.currentTimeMillis();
            this.commitment = 0.0;
            
            // Initialize with small random weights
            Random rand = new Random();
            for (int i = 0; i < dimensionality; i++) {
                bottomUpWeights[i] = rand.nextDouble() * 0.1;
                topDownWeights[i] = rand.nextDouble() * 0.1;
                prototype[i] = rand.nextDouble() * 0.1;
            }
        }
        
        public void updatePrototype(double[] input, double learningRate) {
            for (int i = 0; i < prototype.length; i++) {
                prototype[i] += learningRate * (input[i] - prototype[i]);
            }
            commitment = Math.min(1.0, commitment + 0.1);
        }
    }
    
    public static class ResonanceState {
        public final boolean isResonant;
        public final CategoryNode resonantCategory;
        public final double resonanceStrength;
        public final int iterations;
        public final Map<String, Double> diagnostics;
        
        public ResonanceState(boolean isResonant, CategoryNode category, 
                             double strength, int iterations) {
            this.isResonant = isResonant;
            this.resonantCategory = category;
            this.resonanceStrength = strength;
            this.iterations = iterations;
            this.diagnostics = new HashMap<>();
        }
        
        public void addDiagnostic(String key, double value) {
            diagnostics.put(key, value);
        }
    }
    
    public ResonanceDetector(double vigilance) {
        this.vigilance = vigilance;
        this.categories = new ConcurrentHashMap<>();
        this.nextCategoryId = 0;
    }
    
    public ResonanceDetector() {
        this(DEFAULT_VIGILANCE);
    }
    
    /**
     * Main resonance search algorithm
     */
    public ResonanceState searchResonance(double[] input) {
        // Normalize input
        double[] normalizedInput = normalize(input);
        
        // Reset all category activations
        categories.values().forEach(cat -> cat.activation = 0.0);
        
        // Bottom-up activation: compute match scores
        for (CategoryNode category : categories.values()) {
            category.matchScore = computeMatch(normalizedInput, category.bottomUpWeights);
            category.activation = category.matchScore;
        }
        
        // Sort categories by activation (match score)
        List<CategoryNode> sortedCategories = new ArrayList<>(categories.values());
        sortedCategories.sort((a, b) -> Double.compare(b.activation, a.activation));
        
        // Search for resonance
        for (CategoryNode category : sortedCategories) {
            if (category.activation < 0.01) continue; // Skip very low activations
            
            // Apply top-down expectation
            double[] expectation = computeExpectation(category);
            
            // Check vigilance criterion
            double matchDegree = computeVigilanceMatch(normalizedInput, expectation);
            
            if (matchDegree >= vigilance) {
                // Resonance achieved!
                ResonanceState state = achieveResonance(category, normalizedInput, matchDegree);
                
                // Learn if resonant
                if (state.isResonant) {
                    learnResonantPattern(category, normalizedInput);
                }
                
                return state;
            }
        }
        
        // No resonance found - create new category
        CategoryNode newCategory = createNewCategory(normalizedInput);
        return new ResonanceState(true, newCategory, 1.0, 1);
    }
    
    /**
     * Compute match between input and weights
     */
    private double computeMatch(double[] input, double[] weights) {
        double dotProduct = 0.0;
        double inputNorm = 0.0;
        double weightNorm = 0.0;
        
        for (int i = 0; i < input.length; i++) {
            dotProduct += input[i] * weights[i];
            inputNorm += input[i] * input[i];
            weightNorm += weights[i] * weights[i];
        }
        
        if (inputNorm == 0 || weightNorm == 0) return 0.0;
        
        // Cosine similarity
        return dotProduct / (Math.sqrt(inputNorm) * Math.sqrt(weightNorm));
    }
    
    /**
     * Compute top-down expectation from category
     */
    private double[] computeExpectation(CategoryNode category) {
        double[] expectation = new double[category.topDownWeights.length];
        
        for (int i = 0; i < expectation.length; i++) {
            expectation[i] = category.topDownWeights[i] * category.activation;
        }
        
        return normalize(expectation);
    }
    
    /**
     * Check vigilance match between input and expectation
     */
    private double computeVigilanceMatch(double[] input, double[] expectation) {
        double intersection = 0.0;
        double inputSum = 0.0;
        
        for (int i = 0; i < input.length; i++) {
            intersection += Math.min(input[i], expectation[i]);
            inputSum += input[i];
        }
        
        if (inputSum == 0) return 0.0;
        
        return intersection / inputSum;
    }
    
    /**
     * Achieve resonance through iterative matching
     */
    private ResonanceState achieveResonance(CategoryNode category, double[] input, 
                                           double initialMatch) {
        double[] F1 = input.clone(); // Feature layer
        double[] F2 = new double[F1.length]; // Category layer
        
        double resonanceStrength = initialMatch;
        int iterations = 0;
        double previousEnergy = Double.MAX_VALUE;
        
        while (iterations < MAX_ITERATIONS) {
            // Bottom-up pass
            for (int i = 0; i < F2.length; i++) {
                F2[i] = 0;
                for (int j = 0; j < F1.length; j++) {
                    F2[i] += F1[j] * category.bottomUpWeights[j];
                }
            }
            F2 = normalize(F2);
            
            // Top-down pass
            double[] newF1 = new double[F1.length];
            for (int i = 0; i < F1.length; i++) {
                newF1[i] = input[i] * category.topDownWeights[i];
            }
            newF1 = normalize(newF1);
            
            // Check convergence
            double energy = computeResonanceEnergy(F1, newF1, F2);
            
            if (Math.abs(energy - previousEnergy) < 0.001) {
                // Converged to resonance
                resonanceStrength = 1.0 - energy;
                break;
            }
            
            F1 = newF1;
            previousEnergy = energy;
            iterations++;
        }
        
        // If we reached this method, vigilance criterion was already satisfied
        // Therefore, resonance is achieved (ART theory compliance)
        boolean isResonant = true;
        
        ResonanceState state = new ResonanceState(isResonant, category, 
                                                  resonanceStrength, iterations);
        state.addDiagnostic("convergence_energy", previousEnergy);
        state.addDiagnostic("initial_match", initialMatch);
        
        return state;
    }
    
    /**
     * Compute resonance energy (lower is better)
     */
    private double computeResonanceEnergy(double[] F1, double[] F1_new, double[] F2) {
        double energy = 0.0;
        
        // Difference between consecutive F1 states
        for (int i = 0; i < F1.length; i++) {
            energy += Math.pow(F1[i] - F1_new[i], 2);
        }
        
        // Mismatch with F2
        for (int i = 0; i < F2.length; i++) {
            energy += Math.pow(F2[i] - F1_new[i], 2) * 0.5;
        }
        
        return energy;
    }
    
    /**
     * Learn the resonant pattern
     */
    private void learnResonantPattern(CategoryNode category, double[] input) {
        double learningRate = 0.1 * (1.0 - category.commitment * 0.5);
        
        // Update bottom-up weights
        for (int i = 0; i < category.bottomUpWeights.length; i++) {
            category.bottomUpWeights[i] += learningRate * 
                (input[i] - category.bottomUpWeights[i]);
        }
        
        // Update top-down weights
        for (int i = 0; i < category.topDownWeights.length; i++) {
            category.topDownWeights[i] += learningRate * 
                (input[i] - category.topDownWeights[i]);
        }
        
        // Update prototype
        category.updatePrototype(input, learningRate);
        
        // Update access statistics
        category.accessCount++;
        category.lastAccess = System.currentTimeMillis();
        
        // Normalize weights
        category.bottomUpWeights = normalize(category.bottomUpWeights);
        category.topDownWeights = normalize(category.topDownWeights);
    }
    
    /**
     * Create new category for unmatched input
     */
    private CategoryNode createNewCategory(double[] input) {
        CategoryNode newCategory = new CategoryNode(nextCategoryId++, input.length);
        
        // Initialize with input pattern
        System.arraycopy(input, 0, newCategory.bottomUpWeights, 0, input.length);
        System.arraycopy(input, 0, newCategory.topDownWeights, 0, input.length);
        System.arraycopy(input, 0, newCategory.prototype, 0, input.length);
        
        newCategory.activation = 1.0;
        newCategory.matchScore = 1.0;
        newCategory.commitment = 0.1;
        
        categories.put(newCategory.id, newCategory);
        
        return newCategory;
    }
    
    /**
     * Normalize vector to unit length
     */
    private double[] normalize(double[] vector) {
        double sum = 0.0;
        for (double v : vector) {
            sum += v * v;
        }
        
        if (sum == 0) return vector;
        
        double norm = Math.sqrt(sum);
        double[] normalized = new double[vector.length];
        
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        return normalized;
    }
    
    /**
     * Adjust vigilance parameter
     */
    public void setVigilance(double vigilance) {
        this.vigilance = Math.max(0.0, Math.min(1.0, vigilance));
    }
    
    /**
     * Get all categories
     */
    public Collection<CategoryNode> getCategories() {
        return categories.values();
    }
    
    /**
     * Prune unused categories
     */
    public void pruneCategories(long maxAge, int minAccess) {
        long currentTime = System.currentTimeMillis();
        
        categories.entrySet().removeIf(entry -> {
            CategoryNode cat = entry.getValue();
            long age = currentTime - cat.lastAccess;
            return age > maxAge && cat.accessCount < minAccess;
        });
    }
    
    /**
     * Reset all categories
     */
    public void reset() {
        categories.clear();
        nextCategoryId = 0;
    }
}
