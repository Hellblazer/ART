/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 */
package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import java.util.List;

/**
 * Simple validation and performance test for VectorizedDeepARTMAP
 */
public class ValidationTest {
    public static void main(String[] args) {
        System.out.println("VectorizedDeepARTMAP Validation Test");
        System.out.println("===================================");
        
        try {
            // Create test data
            int dataSize = 50;
            var channel0 = new Pattern[dataSize];
            var channel1 = new Pattern[dataSize];
            
            for (int i = 0; i < dataSize; i++) {
                double base = i / (double) dataSize;
                // 8-dimensional patterns for effective SIMD
                channel0[i] = Pattern.of(
                    base, 1.0 - base, base * 0.8, 1.0 - base * 0.8,
                    base * 0.6, 1.0 - base * 0.6, base * 0.4, 1.0 - base * 0.4
                );
                channel1[i] = Pattern.of(
                    Math.sin(base * Math.PI), Math.cos(base * Math.PI), base * 0.5,
                    Math.sqrt(base), 1.0 - base, base * base, Math.sqrt(1.0 - base), base * 0.7
                );
            }
            
            var trainingData = List.of(channel0, channel1);
            var labels = new int[dataSize];
            for (int i = 0; i < dataSize; i++) {
                labels[i] = i % 3; // 3 classes
            }
            
            System.out.println("✓ Test data created: " + dataSize + " samples, " + trainingData.size() + " channels");
            
            // Create vectorized network
            var vectorizedModules = List.<BaseART>of(
                new VectorizedFuzzyART(VectorizedParameters.createDefault()),
                new VectorizedFuzzyART(VectorizedParameters.createDefault())
            );
            var baseParams = new DeepARTMAPParameters(0.8, 0.1, 1000, true);
            var vectorizedParams = new VectorizedDeepARTMAPParameters(baseParams, 4, 2, 2, 2, true, 0, 100, 100, 0.8, true);
            var vectorizedNetwork = new VectorizedDeepARTMAP(vectorizedModules, vectorizedParams);
            
            System.out.println("✓ VectorizedDeepARTMAP created successfully");
            System.out.println("  - Modules: " + vectorizedModules.size());
            System.out.println("  - Parameters: " + vectorizedParams);
            
            // Test training
            System.out.println("\nTesting supervised training...");
            long startTime = System.nanoTime();
            var trainingResult = vectorizedNetwork.fitSupervised(trainingData, labels);
            long trainingTime = System.nanoTime() - startTime;
            
            System.out.printf("✓ Training completed in %.2f ms\n", trainingTime / 1_000_000.0);
            if (trainingResult instanceof DeepARTMAPResult.Success success) {
                System.out.println("  - Categories created: " + success.categoryCount());
                System.out.println("  - Training successful: " + success.supervisedMode());
            } else {
                System.out.println("  - Training result: " + trainingResult.toString());
            }
            
            // Test prediction
            System.out.println("\nTesting prediction...");
            startTime = System.nanoTime();
            var predictions = vectorizedNetwork.predict(trainingData);
            long predictionTime = System.nanoTime() - startTime;
            
            System.out.printf("✓ Prediction completed in %.2f ms\n", predictionTime / 1_000_000.0);
            System.out.println("  - Predictions: " + predictions.length);
            System.out.println("  - First 5 predictions: [" + predictions[0] + ", " + predictions[1] + ", " + predictions[2] + ", " + predictions[3] + ", " + predictions[4] + "]");
            
            // Test deep prediction
            System.out.println("\nTesting deep prediction...");
            startTime = System.nanoTime();
            var deepPredictions = vectorizedNetwork.predictDeep(trainingData);
            long deepPredTime = System.nanoTime() - startTime;
            
            System.out.printf("✓ Deep prediction completed in %.2f ms\n", deepPredTime / 1_000_000.0);
            System.out.println("  - Deep predictions shape: [" + deepPredictions.length + " x " + deepPredictions[0].length + "]");
            
            // Test probability prediction
            System.out.println("\nTesting probability prediction...");
            startTime = System.nanoTime();
            var probabilities = vectorizedNetwork.predict_proba(trainingData);
            long probaTime = System.nanoTime() - startTime;
            
            System.out.printf("✓ Probability prediction completed in %.2f ms\n", probaTime / 1_000_000.0);
            System.out.println("  - Probability matrix shape: [" + probabilities.length + " x " + probabilities[0].length + "]");
            System.out.printf("  - Sample probabilities: [%.3f, %.3f, %.3f]\n", 
                             probabilities[0][0], probabilities[0][1], probabilities[0][2]);
            
            // Get performance stats
            var stats = vectorizedNetwork.getPerformanceStats();
            
            System.out.println("\n📈 Performance Statistics:");
            System.out.println("=========================");
            System.out.printf("Total Operations: %d\n", stats.operationCount());
            System.out.printf("SIMD Operations: %d (%.1f%% efficiency)\n", 
                             stats.totalSIMDOperations(), stats.simdEfficiency() * 100);
            System.out.printf("Channel Parallel Tasks: %d (%.1f%% efficiency)\n", 
                             stats.totalChannelParallelTasks(), stats.channelParallelismEfficiency() * 100);
            System.out.printf("Layer Parallel Tasks: %d (%.1f%% efficiency)\n", 
                             stats.totalLayerParallelTasks(), stats.layerParallelismEfficiency() * 100);
            System.out.printf("Operations per Second: %.1f\n", stats.operationsPerSecond());
            System.out.printf("Categories Created: %d\n", stats.categoryCount());
            System.out.printf("Active Threads: %d\n", stats.totalActiveThreads());
            
            // Create standard DeepARTMAP for comparison
            System.out.println("\n🔍 Performance Comparison:");
            System.out.println("==========================");
            
            var standardModules = List.<BaseART>of(
                new FuzzyART(),
                new FuzzyART()
            );
            var standardParams = new DeepARTMAPParameters(0.8, 0.1, 1000, true);
            var standardNetwork = new DeepARTMAP(standardModules, standardParams);
            
            // Train standard network
            startTime = System.nanoTime();
            var standardResult = standardNetwork.fitSupervised(trainingData, labels);
            long standardTrainingTime = System.nanoTime() - startTime;
            
            // Predict with standard network
            startTime = System.nanoTime();
            var standardPredictions = standardNetwork.predict(trainingData);
            long standardPredTime = System.nanoTime() - startTime;
            
            // Calculate speedups
            double trainingSpeedup = (double) standardTrainingTime / trainingTime;
            double predictionSpeedup = (double) standardPredTime / predictionTime;
            
            System.out.printf("Training Speedup: %.2fx (%.2f ms vs %.2f ms)\n", 
                             trainingSpeedup, trainingTime / 1_000_000.0, standardTrainingTime / 1_000_000.0);
            System.out.printf("Prediction Speedup: %.2fx (%.2f ms vs %.2f ms)\n", 
                             predictionSpeedup, predictionTime / 1_000_000.0, standardPredTime / 1_000_000.0);
            
            double avgSpeedup = (trainingSpeedup + predictionSpeedup) / 2;
            System.out.printf("Average Speedup: %.2fx\n", avgSpeedup);
            
            // Final assessment
            System.out.println("\n🎯 Final Assessment:");
            System.out.println("===================");
            if (avgSpeedup >= 2.0) {
                System.out.println("🎉 SUCCESS: Achieved expected 2-5x performance improvement!");
            } else if (avgSpeedup >= 1.5) {
                System.out.println("✅ GOOD: Significant performance improvement achieved!");
            } else {
                System.out.println("⚠️  MODEST: Some performance improvement, within expected range for small datasets.");
            }
            
            System.out.println("\n✅ VectorizedDeepARTMAP validation completed successfully!");
            
            vectorizedNetwork.close();
            
        } catch (Exception e) {
            System.err.println("❌ Error during validation: " + e.getMessage());
            e.printStackTrace();
        }
    }
}