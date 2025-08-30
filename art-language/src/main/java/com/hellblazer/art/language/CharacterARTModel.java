/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks - Language Module.
 */
package com.hellblazer.art.language;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.artmap.DeepARTMAP;
import com.hellblazer.art.core.artmap.DeepARTMAPParameters;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.parameters.TopoARTParameters;

import java.util.*;
import java.nio.charset.StandardCharsets;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Steel Thread MVP: Character-level ART Language Model
 * 
 * This is a simple character-level language model using DeepARTMAP
 * to demonstrate the feasibility of ART-based language processing.
 * 
 * Architecture:
 * 1. Character encoding (one-hot + context window)
 * 2. TopoART Layer 1: Character patterns (preserves sequential topology)
 * 3. FuzzyART Layer 2: Character combinations  
 * 4. FuzzyART Layer 3: Higher-level patterns
 * 5. Output mapping to next character prediction
 * 
 * @author Hal Hildebrand
 */
public class CharacterARTModel {
    
    // Model configuration
    private static final int CONTEXT_WINDOW = 10;  // Characters of context
    private static final int CHAR_VOCAB_SIZE = 128; // ASCII characters
    private static final int EMBEDDING_DIM = 16;    // Character embedding size
    
    // ART modules
    private final TopoART characterLayer;      // Layer 1: Character patterns with topology
    private final FuzzyART combinationLayer;   // Layer 2: Character combinations
    private final FuzzyART abstractLayer;      // Layer 3: Abstract patterns
    private final DeepARTMAP deepArtMap;       // Hierarchical processor
    
    // Character to category mappings
    private final Map<String, Integer> contextToCategoryMap = new HashMap<>();
    private final Map<Integer, Character> categoryToCharMap = new HashMap<>();
    private final List<Character> vocabulary = new ArrayList<>();
    
    // Sliding context window
    private final Deque<Character> contextWindow = new ArrayDeque<>(CONTEXT_WINDOW);
    
    public CharacterARTModel() {
        // Initialize vocabulary (ASCII printable characters)
        for (int i = 32; i < 127; i++) {
            vocabulary.add((char) i);
        }
        
        // Layer 1: TopoART for character patterns (preserves sequence topology)
        TopoARTParameters topoParams = new TopoARTParameters();
        topoParams.setVigilance(0.9);           // High vigilance for precise character matching
        topoParams.setLearningRate(0.1);        
        topoParams.setPhi(3);                    // Permanence threshold
        topoParams.setTau(100);                  // Cleanup cycle
        topoParams.setSbmLearningRate(0.05);    // Second-best match learning
        this.characterLayer = new TopoART(topoParams);
        
        // Layer 2: FuzzyART for character combinations
        FuzzyParameters comboParams = new FuzzyParameters();
        comboParams.setRho(0.7);                // Medium vigilance for combinations
        comboParams.setBeta(0.2);               // Moderate learning rate
        comboParams.setAlpha(0.001);            // Choice parameter
        this.combinationLayer = new FuzzyART(comboParams);
        
        // Layer 3: FuzzyART for abstract patterns
        FuzzyParameters abstractParams = new FuzzyParameters();
        abstractParams.setRho(0.5);             // Lower vigilance for abstraction
        abstractParams.setBeta(0.3);            // Higher learning for generalization
        abstractParams.setAlpha(0.001);
        this.abstractLayer = new FuzzyART(abstractParams);
        
        // Build DeepARTMAP hierarchy
        List<BaseART> modules = Arrays.asList(
            characterLayer,
            combinationLayer,
            abstractLayer
        );
        
        DeepARTMAPParameters deepParams = new DeepARTMAPParameters();
        deepParams.setSupervisedMode(false);    // Start with unsupervised learning
        deepParams.setMapFieldLearningRate(0.1);
        
        this.deepArtMap = new DeepARTMAP(modules, deepParams);
    }
    
    /**
     * Train the model on text data
     */
    public void train(String text) {
        System.out.println("Training on " + text.length() + " characters...");
        
        // Process text character by character
        for (int i = 0; i < text.length(); i++) {
            char currentChar = text.charAt(i);
            
            // Update context window
            if (contextWindow.size() >= CONTEXT_WINDOW) {
                contextWindow.removeFirst();
            }
            contextWindow.addLast(currentChar);
            
            // Skip if we don't have enough context yet
            if (contextWindow.size() < CONTEXT_WINDOW) {
                continue;
            }
            
            // Create input pattern from context window
            double[] inputPattern = encodeContext(contextWindow);
            
            // Process through DeepARTMAP
            Pattern pattern = new Pattern(new DenseVector(inputPattern));
            DeepARTMAPResult result = deepArtMap.unsupervisedLearn(
                Arrays.asList(pattern)  // Single pattern batch
            );
            
            // Map the resulting category to the current character
            String contextKey = contextToString(contextWindow);
            int topLevelCategory = result.getCategoryActivations()[2]; // Layer 3 category
            
            contextToCategoryMap.put(contextKey, topLevelCategory);
            categoryToCharMap.put(topLevelCategory, currentChar);
            
            // Progress indicator
            if ((i + 1) % 1000 == 0) {
                System.out.println("Processed " + (i + 1) + " characters. Categories: " +
                    "L1=" + characterLayer.getCategoryCount() + 
                    ", L2=" + combinationLayer.getCategoryCount() + 
                    ", L3=" + abstractLayer.getCategoryCount());
            }
        }
        
        System.out.println("Training complete. Final categories: " +
            "L1=" + characterLayer.getCategoryCount() + 
            ", L2=" + combinationLayer.getCategoryCount() + 
            ", L3=" + abstractLayer.getCategoryCount());
    }
    
    /**
     * Generate text starting from a seed
     */
    public String generate(String seed, int length) {
        StringBuilder generated = new StringBuilder(seed);
        
        // Initialize context with seed
        contextWindow.clear();
        for (char c : seed.toCharArray()) {
            if (contextWindow.size() >= CONTEXT_WINDOW) {
                contextWindow.removeFirst();
            }
            contextWindow.addLast(c);
        }
        
        // Generate new characters
        for (int i = 0; i < length; i++) {
            // Encode current context
            double[] inputPattern = encodeContext(contextWindow);
            Pattern pattern = new Pattern(new DenseVector(inputPattern));
            
            // Get prediction from DeepARTMAP
            DeepARTMAPResult result = deepArtMap.unsupervisedLearn(
                Arrays.asList(pattern)
            );
            
            // Find the best matching character for this category
            int topCategory = result.getCategoryActivations()[2];
            Character nextChar = categoryToCharMap.get(topCategory);
            
            // If no direct mapping, find closest category
            if (nextChar == null) {
                nextChar = findClosestCharacter(pattern);
            }
            
            // Add to generated text
            generated.append(nextChar);
            
            // Update context window
            contextWindow.removeFirst();
            contextWindow.addLast(nextChar);
        }
        
        return generated.toString();
    }
    
    /**
     * Encode context window into input pattern
     */
    private double[] encodeContext(Deque<Character> context) {
        // Simple one-hot encoding with positional information
        double[] pattern = new double[CONTEXT_WINDOW * EMBEDDING_DIM];
        
        int position = 0;
        for (Character c : context) {
            int charIndex = Math.max(0, Math.min(c - 32, 94)); // ASCII offset
            
            // One-hot encode with position weighting
            int baseIndex = position * EMBEDDING_DIM;
            pattern[baseIndex + (charIndex % EMBEDDING_DIM)] = 1.0;
            
            // Add positional encoding
            pattern[baseIndex + EMBEDDING_DIM - 1] = (double) position / CONTEXT_WINDOW;
            
            position++;
        }
        
        return pattern;
    }
    
    /**
     * Convert context to string key
     */
    private String contextToString(Deque<Character> context) {
        StringBuilder sb = new StringBuilder();
        for (Character c : context) {
            sb.append(c);
        }
        return sb.toString();
    }
    
    /**
     * Find closest matching character when no direct category mapping exists
     */
    private Character findClosestCharacter(Pattern pattern) {
        // Default to space if no matches
        return ' ';
    }
    
    /**
     * Load training data from file
     */
    public static String loadTrainingData(String filepath) throws IOException {
        return Files.readString(Path.of(filepath), StandardCharsets.UTF_8);
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        CharacterARTModel model = new CharacterARTModel();
        
        // Simple test with repetitive pattern
        String trainingText = "The quick brown fox jumps over the lazy dog. " +
                             "The quick brown fox jumps over the lazy dog. " +
                             "The quick brown fox jumps over the lazy dog. ";
        
        // Train the model
        model.train(trainingText);
        
        // Generate some text
        String generated = model.generate("The quick ", 20);
        System.out.println("\nGenerated text: " + generated);
        
        // Show model statistics
        System.out.println("\nModel Statistics:");
        System.out.println("Context mappings: " + model.contextToCategoryMap.size());
        System.out.println("Character categories: " + model.categoryToCharMap.size());
    }
}
