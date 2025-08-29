/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 */
package com.hellblazer.art.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Demonstration of polymorphic usage of vectorized ART algorithms through the common interface.
 * 
 * This example shows how the VectorizedARTAlgorithm interface enables consistent usage
 * patterns across different vectorized implementations, making it easy to:
 * - Switch between different algorithms
 * - Write generic processing functions
 * - Compare performance across implementations
 * - Use algorithms in collections and frameworks
 */
public class VectorizedARTPolymorphismDemo {
    
    /**
     * Generic function that works with any vectorized ART algorithm.
     * This demonstrates the power of the interface - the same code can work
     * with VectorizedFuzzyART, VectorizedHypersphereART, VectorizedTopoART, etc.
     */
    public static <T, P> void demonstrateAlgorithm(VectorizedARTAlgorithm<T, P> algorithm, 
                                                   Pattern[] trainingData, 
                                                   Pattern[] testData,
                                                   P parameters) {
        System.out.println("=== Testing " + algorithm.getAlgorithmType() + " ===");
        System.out.println("Vectorized: " + algorithm.isVectorized());
        System.out.println("Vector Species Length: " + algorithm.getVectorSpeciesLength());
        
        // Training phase
        System.out.println("\nTraining with " + trainingData.length + " patterns...");
        for (Pattern pattern : trainingData) {
            Object result = algorithm.learn(pattern, parameters);
            System.out.println("  Learned pattern -> Category: " + result);
        }
        
        System.out.println("Categories after training: " + algorithm.getCategoryCount());
        System.out.println("Is trained: " + algorithm.isTrained());
        
        // Testing phase
        System.out.println("\nTesting with " + testData.length + " patterns...");
        for (int i = 0; i < testData.length; i++) {
            Object prediction = algorithm.predict(testData[i], parameters);
            System.out.println("  Test pattern " + i + " -> Prediction: " + prediction);
        }
        
        // Performance monitoring
        System.out.println("\nPerformance Stats:");
        T stats = algorithm.getPerformanceStats();
        System.out.println("  " + stats);
        
        System.out.println("Algorithm completed successfully!\n");
    }
    
    /**
     * Example of managing multiple algorithms polymorphically.
     */
    public static void demonstrateMultipleAlgorithms() {
        System.out.println("=== Polymorphic Algorithm Management ===");
        
        // Create a collection of different vectorized algorithms
        List<VectorizedARTAlgorithm<?, ?>> algorithms = new ArrayList<>();
        
        // Add VectorizedHypersphereART
        var hypersphereParams = VectorizedHypersphereParameters.builder()
            .inputDimensions(2)
            .vigilance(0.8)
            .learningRate(0.1)
            .maxCategories(10)
            .enableSIMD(true)
            .build();
        algorithms.add(new VectorizedHypersphereART(hypersphereParams));
        
        // Future algorithms can be easily added:
        // algorithms.add(new VectorizedFuzzyART(fuzzyParams));
        // algorithms.add(new VectorizedTopoART(topoParams));
        
        // Generic operations on all algorithms
        System.out.println("Algorithm Summary:");
        for (VectorizedARTAlgorithm<?, ?> algorithm : algorithms) {
            System.out.println("  " + algorithm.getAlgorithmType() + 
                             " - Categories: " + algorithm.getCategoryCount() +
                             " - Vectorized: " + algorithm.isVectorized() +
                             " - Vector Length: " + algorithm.getVectorSpeciesLength());
        }
        
        // Cleanup all algorithms
        System.out.println("\nCleaning up algorithms...");
        algorithms.forEach(alg -> {
            try {
                alg.close();
                System.out.println("  Closed " + alg.getAlgorithmType());
            } catch (Exception e) {
                System.err.println("  Error closing " + alg.getAlgorithmType() + ": " + e.getMessage());
            }
        });
    }
    
    public static void main(String[] args) {
        System.out.println("Vectorized ART Algorithm Interface Demonstration");
        System.out.println("================================================\n");
        
        // Create sample data
        Pattern[] trainingData = {
            Pattern.of(new double[]{0.1, 0.2}),
            Pattern.of(new double[]{0.8, 0.9}),
            Pattern.of(new double[]{0.15, 0.25}),
            Pattern.of(new double[]{0.75, 0.85})
        };
        
        Pattern[] testData = {
            Pattern.of(new double[]{0.12, 0.22}),
            Pattern.of(new double[]{0.77, 0.87}),
            Pattern.of(new double[]{0.5, 0.5})
        };
        
        // Demonstrate VectorizedHypersphereART through the interface
        var params = VectorizedHypersphereParameters.builder()
            .inputDimensions(2)
            .vigilance(0.7)
            .learningRate(0.1)
            .maxCategories(5)
            .enableSIMD(true)
            .simdThreshold(4)
            .build();
        
        var hypersphereART = new VectorizedHypersphereART(params);
        
        // Use generic function - same code works for any vectorized ART algorithm
        demonstrateAlgorithm(hypersphereART, trainingData, testData, params);
        
        // Demonstrate multiple algorithms management
        demonstrateMultipleAlgorithms();
        
        // Cleanup
        try {
            hypersphereART.close();
        } catch (Exception e) {
            System.err.println("Error during cleanup: " + e.getMessage());
        }
        
        System.out.println("Demo completed!");
    }
}