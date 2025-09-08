package com.art.textgen.tuning;

import com.art.textgen.generation.*;
import com.art.textgen.evaluation.TextGenerationMetrics;
import com.art.textgen.core.Vocabulary;
import java.util.*;
import java.io.*;

/**
 * Parameter tuning system for optimizing generation quality
 */
public class ParameterTuner {
    private final EnhancedPatternGenerator generator;
    private final TextGenerationMetrics metrics;
    private final List<String> testPrompts;
    private final Map<String, Double> bestParameters;
    
    public ParameterTuner() {
        Vocabulary vocab = new Vocabulary(10000);
        this.generator = new EnhancedPatternGenerator(vocab);
        this.metrics = new TextGenerationMetrics();
        this.testPrompts = loadTestPrompts();
        this.bestParameters = new HashMap<>();
    }
    
    /**
     * Grid search for optimal parameters
     */
    public void tuneParameters() {
        System.out.println("Starting parameter tuning...\n");
        
        // Parameter ranges to test
        double[] temperatures = {0.5, 0.7, 0.9, 1.0, 1.2};
        int[] topKValues = {20, 30, 40, 50, 60};
        double[] topPValues = {0.8, 0.85, 0.9, 0.95};
        double[] repetitionPenalties = {1.0, 1.1, 1.2, 1.3, 1.5};
        
        double bestScore = 0;
        Map<String, Object> bestConfig = new HashMap<>();
        
        int totalTests = temperatures.length * topKValues.length * 
                        topPValues.length * repetitionPenalties.length;
        int currentTest = 0;
        
        for (double temp : temperatures) {
            for (int topK : topKValues) {
                for (double topP : topPValues) {
                    for (double penalty : repetitionPenalties) {
                        currentTest++;
                        System.out.printf("Testing %d/%d: temp=%.1f, topK=%d, topP=%.2f, penalty=%.1f\n",
                            currentTest, totalTests, temp, topK, topP, penalty);
                        
                        // Configure generator
                        generator.setTemperature(temp);
                        generator.setTopK(topK);
                        generator.setTopP(topP);
                        generator.setRepetitionPenalty(penalty);
                        
                        // Test generation quality
                        double score = evaluateConfiguration();
                        
                        if (score > bestScore) {
                            bestScore = score;
                            bestConfig.put("temperature", temp);
                            bestConfig.put("topK", topK);
                            bestConfig.put("topP", topP);
                            bestConfig.put("repetitionPenalty", penalty);
                            bestConfig.put("score", score);
                            
                            System.out.printf("  ✅ New best score: %.3f\n", score);
                        }
                    }
                }
            }
        }
        
        // Save best parameters
        saveBestParameters(bestConfig);
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("BEST CONFIGURATION FOUND:");
        System.out.println("=".repeat(60));
        System.out.printf("Temperature: %.1f\n", bestConfig.get("temperature"));
        System.out.printf("Top-K: %d\n", bestConfig.get("topK"));
        System.out.printf("Top-P: %.2f\n", bestConfig.get("topP"));
        System.out.printf("Repetition Penalty: %.1f\n", bestConfig.get("repetitionPenalty"));
        System.out.printf("Quality Score: %.3f\n", bestConfig.get("score"));
        System.out.println("=".repeat(60));
    }
    
    /**
     * Evaluate current configuration
     */
    private double evaluateConfiguration() {
        double totalScore = 0;
        int count = 0;
        
        for (String prompt : testPrompts) {
            String generated = generator.generate(prompt, 100);
            
            // Calculate metrics
            double diversity = metrics.calculateDiversity(Arrays.asList(generated), 2);
            double coherence = metrics.calculateCoherence(generated, 3);
            double fluency = metrics.calculateFluency(generated);
            double readability = metrics.calculateReadability(generated);
            
            // Weighted score
            double score = (diversity * 0.2) + (coherence * 0.3) + 
                          (fluency * 0.3) + (readability * 0.2);
            totalScore += score;
            count++;
        }
        
        return count > 0 ? totalScore / count : 0;
    }
    
    /**
     * Load test prompts for evaluation
     */
    private List<String> loadTestPrompts() {
        List<String> prompts = new ArrayList<>();
        prompts.add("Once upon a time");
        prompts.add("The scientist discovered");
        prompts.add("In the beginning");
        prompts.add("She walked into the room");
        prompts.add("The future of technology");
        prompts.add("It was a dark and stormy");
        prompts.add("The most important thing");
        prompts.add("Yesterday I learned");
        prompts.add("The key to success");
        prompts.add("In conclusion");
        return prompts;
    }
    
    /**
     * Save best parameters to file
     */
    private void saveBestParameters(Map<String, Object> params) {
        try (PrintWriter writer = new PrintWriter("best_parameters.txt")) {
            writer.println("# ART Text Generation - Optimal Parameters");
            writer.println("# Generated: " + new Date());
            writer.println();
            writer.printf("temperature=%.1f\n", params.get("temperature"));
            writer.printf("topK=%d\n", params.get("topK"));
            writer.printf("topP=%.2f\n", params.get("topP"));
            writer.printf("repetitionPenalty=%.1f\n", params.get("repetitionPenalty"));
            writer.printf("qualityScore=%.3f\n", params.get("score"));
            
            System.out.println("Best parameters saved to: best_parameters.txt");
        } catch (IOException e) {
            System.err.println("Failed to save parameters: " + e.getMessage());
        }
    }
    
    /**
     * Quick tune - test fewer combinations for faster results
     */
    public void quickTune() {
        System.out.println("Starting quick parameter tuning (subset)...\n");
        
        // Reduced parameter ranges
        double[] temperatures = {0.7, 0.9, 1.1};
        int[] topKValues = {30, 40, 50};
        double[] topPValues = {0.85, 0.9, 0.95};
        double[] repetitionPenalties = {1.1, 1.2, 1.3};
        
        // Run tuning with reduced sets
        performTuning(temperatures, topKValues, topPValues, repetitionPenalties);
    }
    
    private void performTuning(double[] temps, int[] topKs, double[] topPs, double[] penalties) {
        Map<String, Object> results = new HashMap<>();
        double bestScore = 0;
        
        for (double temp : temps) {
            for (int topK : topKs) {
                for (double topP : topPs) {
                    for (double penalty : penalties) {
                        generator.setTemperature(temp);
                        generator.setTopK(topK);
                        generator.setTopP(topP);
                        generator.setRepetitionPenalty(penalty);
                        
                        double score = evaluateConfiguration();
                        String key = String.format("%.1f_%d_%.2f_%.1f", temp, topK, topP, penalty);
                        results.put(key, score);
                        
                        if (score > bestScore) {
                            bestScore = score;
                            bestParameters.put("temperature", temp);
                            bestParameters.put("topK", (double)topK);
                            bestParameters.put("topP", topP);
                            bestParameters.put("repetitionPenalty", penalty);
                        }
                    }
                }
            }
        }
        
        printResults(results);
    }
    
    private void printResults(Map<String, Object> results) {
        System.out.println("\nTop 5 Configurations:");
        results.entrySet().stream()
            .sorted((e1, e2) -> Double.compare((Double)e2.getValue(), (Double)e1.getValue()))
            .limit(5)
            .forEach(e -> System.out.printf("%s: %.3f\n", e.getKey(), e.getValue()));
    }
    
    public static void main(String[] args) {
        ParameterTuner tuner = new ParameterTuner();
        
        if (args.length > 0 && args[0].equals("--quick")) {
            tuner.quickTune();
        } else {
            tuner.tuneParameters();
        }
    }
}
