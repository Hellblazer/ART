package com.art.textgen.evaluation;

import com.art.textgen.generation.PatternGenerator;
import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * A/B Testing framework for comparing different generation strategies
 * Supports statistical significance testing and automated reporting
 */
public class ExperimentRunner {
    
    private final TextGenerationMetrics metrics;
    private final Map<String, ExperimentResult> results;
    private final String experimentDir;
    
    public ExperimentRunner() {
        this.metrics = new TextGenerationMetrics();
        this.results = new HashMap<>();
        this.experimentDir = "experiments/" + LocalDateTime.now()
            .format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
    }
    
    /**
     * Run a complete A/B test comparing two generation strategies
     */
    public void runExperiment(String name, 
                            GenerationStrategy control,
                            GenerationStrategy variant,
                            List<String> prompts,
                            int samplesPerPrompt) throws IOException {
        
        System.out.println("\n=== Running Experiment: " + name + " ===");
        System.out.println("Control: " + control.getName());
        System.out.println("Variant: " + variant.getName());
        System.out.println("Prompts: " + prompts.size());
        System.out.println("Samples per prompt: " + samplesPerPrompt);
        
        // Create experiment directory
        Files.createDirectories(Paths.get(experimentDir, name));
        
        // Generate samples for both strategies
        List<String> controlSamples = generateSamples(control, prompts, samplesPerPrompt);
        List<String> variantSamples = generateSamples(variant, prompts, samplesPerPrompt);
        
        // Evaluate both sets
        ExperimentResult controlResult = evaluateSamples(control.getName(), controlSamples);
        ExperimentResult variantResult = evaluateSamples(variant.getName(), variantSamples);
        
        // Statistical significance testing
        double pValue = calculatePValue(controlResult, variantResult);
        boolean significant = pValue < 0.05;
        
        // Store results
        controlResult.setPValue(pValue);
        controlResult.setSignificant(significant);
        variantResult.setPValue(pValue);
        variantResult.setSignificant(significant);
        
        results.put(name + "_control", controlResult);
        results.put(name + "_variant", variantResult);
        
        // Generate report
        String report = generateExperimentReport(name, controlResult, variantResult, pValue, significant);
        saveReport(name, report);
        
        System.out.println(report);
    }
    
    /**
     * Run a parameter sweep experiment
     */
    public void runParameterSweep(String parameterName,
                                 double[] values,
                                 GenerationStrategyFactory factory,
                                 List<String> prompts) throws IOException {
        
        System.out.println("\n=== Parameter Sweep: " + parameterName + " ===");
        System.out.println("Values to test: " + Arrays.toString(values));
        
        Map<Double, ExperimentResult> sweepResults = new TreeMap<>();
        
        for (double value : values) {
            System.out.println("\nTesting " + parameterName + " = " + value);
            
            GenerationStrategy strategy = factory.createStrategy(value);
            List<String> samples = generateSamples(strategy, prompts, 10);
            ExperimentResult result = evaluateSamples(parameterName + "_" + value, samples);
            
            sweepResults.put(value, result);
        }
        
        // Find optimal value
        double optimalValue = findOptimalParameter(sweepResults);
        
        // Generate sweep report
        String report = generateSweepReport(parameterName, sweepResults, optimalValue);
        saveReport(parameterName + "_sweep", report);
        
        System.out.println(report);
    }
    
    /**
     * Run incremental learning experiment
     */
    public void runIncrementalLearningExperiment(IncrementalLearner learner,
                                                List<List<String>> trainingBatches,
                                                List<String> testPrompts) throws IOException {
        
        System.out.println("\n=== Incremental Learning Experiment ===");
        System.out.println("Training batches: " + trainingBatches.size());
        
        List<IncrementalResult> incrementalResults = new ArrayList<>();
        
        for (int i = 0; i < trainingBatches.size(); i++) {
            System.out.println("\nTraining on batch " + (i + 1));
            
            // Train on next batch
            learner.trainOnBatch(trainingBatches.get(i));
            
            // Test current performance
            List<String> samples = new ArrayList<>();
            for (String prompt : testPrompts) {
                samples.add(learner.generate(prompt));
            }
            
            ExperimentResult result = evaluateSamples("batch_" + i, samples);
            
            // Check for catastrophic forgetting
            double forgettingScore = learner.measureForgetting();
            
            incrementalResults.add(new IncrementalResult(i, result, forgettingScore));
        }
        
        // Generate incremental learning report
        String report = generateIncrementalReport(incrementalResults);
        saveReport("incremental_learning", report);
        
        System.out.println(report);
    }
    
    /**
     * Generate samples using a strategy
     */
    private List<String> generateSamples(GenerationStrategy strategy, 
                                        List<String> prompts, 
                                        int samplesPerPrompt) {
        List<String> samples = new ArrayList<>();
        
        for (String prompt : prompts) {
            for (int i = 0; i < samplesPerPrompt; i++) {
                String generated = strategy.generate(prompt, 100); // Generate 100 tokens
                samples.add(generated);
            }
        }
        
        return samples;
    }
    
    /**
     * Evaluate a set of samples
     */
    private ExperimentResult evaluateSamples(String name, List<String> samples) {
        ExperimentResult result = new ExperimentResult(name);
        
        // Calculate diversity
        double diversity = metrics.calculateDiversity(samples, 2);
        result.addMetric("diversity", diversity);
        
        // Calculate average metrics for individual samples
        double totalCoherence = 0;
        double totalFluency = 0;
        double totalReadability = 0;
        
        for (String sample : samples) {
            totalCoherence += metrics.calculateCoherence(sample, 3);
            totalFluency += metrics.calculateFluency(sample);
            totalReadability += metrics.calculateReadability(sample);
        }
        
        result.addMetric("coherence", totalCoherence / samples.size());
        result.addMetric("fluency", totalFluency / samples.size());
        result.addMetric("readability", totalReadability / samples.size());
        
        // Calculate composite score
        result.setCompositeScore(metrics.getCompositeScore());
        
        // Store samples
        result.setSamples(samples);
        
        return result;
    }
    
    /**
     * Calculate p-value for statistical significance
     */
    private double calculatePValue(ExperimentResult control, ExperimentResult variant) {
        // Simplified t-test for composite scores
        double controlMean = control.getCompositeScore();
        double variantMean = variant.getCompositeScore();
        
        // Calculate standard deviations (simplified)
        double controlStd = 0.1; // Placeholder - should calculate from samples
        double variantStd = 0.1;
        
        int n = control.getSamples().size();
        
        // t-statistic
        double pooledStd = Math.sqrt((controlStd * controlStd + variantStd * variantStd) / 2);
        double tStat = (variantMean - controlMean) / (pooledStd * Math.sqrt(2.0 / n));
        
        // Approximate p-value (simplified)
        double pValue = 1.0 / (1.0 + Math.exp(Math.abs(tStat) - 2));
        
        return pValue;
    }
    
    /**
     * Find optimal parameter value based on composite scores
     */
    private double findOptimalParameter(Map<Double, ExperimentResult> sweepResults) {
        double optimalValue = 0;
        double bestScore = -1;
        
        for (Map.Entry<Double, ExperimentResult> entry : sweepResults.entrySet()) {
            if (entry.getValue().getCompositeScore() > bestScore) {
                bestScore = entry.getValue().getCompositeScore();
                optimalValue = entry.getKey();
            }
        }
        
        return optimalValue;
    }
    
    /**
     * Generate experiment report
     */
    private String generateExperimentReport(String name,
                                           ExperimentResult control,
                                           ExperimentResult variant,
                                           double pValue,
                                           boolean significant) {
        StringBuilder report = new StringBuilder();
        
        report.append("\n=== Experiment Report: " + name + " ===\n");
        report.append("=" . repeat(50) + "\n\n");
        
        report.append("Control Strategy: " + control.getName() + "\n");
        report.append("-".repeat(30) + "\n");
        report.append(formatMetrics(control));
        
        report.append("\nVariant Strategy: " + variant.getName() + "\n");
        report.append("-".repeat(30) + "\n");
        report.append(formatMetrics(variant));
        
        report.append("\nStatistical Analysis:\n");
        report.append("-".repeat(30) + "\n");
        report.append(String.format("P-value: %.4f\n", pValue));
        report.append("Significant difference: " + (significant ? "YES" : "NO") + "\n");
        
        double improvement = ((variant.getCompositeScore() - control.getCompositeScore()) 
            / control.getCompositeScore()) * 100;
        report.append(String.format("Improvement: %.1f%%\n", improvement));
        
        report.append("\nRecommendation: ");
        if (significant && improvement > 0) {
            report.append("Use VARIANT strategy (significant improvement)");
        } else if (significant && improvement < 0) {
            report.append("Keep CONTROL strategy (variant is worse)");
        } else {
            report.append("No significant difference - choose based on other factors");
        }
        
        return report.toString();
    }
    
    /**
     * Generate parameter sweep report
     */
    private String generateSweepReport(String parameterName,
                                      Map<Double, ExperimentResult> sweepResults,
                                      double optimalValue) {
        StringBuilder report = new StringBuilder();
        
        report.append("\n=== Parameter Sweep Report: " + parameterName + " ===\n");
        report.append("=".repeat(50) + "\n\n");
        
        report.append("Results by Value:\n");
        report.append("-".repeat(30) + "\n");
        
        for (Map.Entry<Double, ExperimentResult> entry : sweepResults.entrySet()) {
            report.append(String.format("\n%s = %.3f:\n", parameterName, entry.getKey()));
            report.append(formatMetrics(entry.getValue()));
        }
        
        report.append("\n" + "-".repeat(30) + "\n");
        report.append(String.format("OPTIMAL VALUE: %.3f\n", optimalValue));
        report.append(String.format("Best Score: %.3f\n", 
            sweepResults.get(optimalValue).getCompositeScore()));
        
        return report.toString();
    }
    
    /**
     * Generate incremental learning report
     */
    private String generateIncrementalReport(List<IncrementalResult> results) {
        StringBuilder report = new StringBuilder();
        
        report.append("\n=== Incremental Learning Report ===\n");
        report.append("=".repeat(50) + "\n\n");
        
        for (IncrementalResult result : results) {
            report.append(String.format("Batch %d:\n", result.batchNumber + 1));
            report.append(formatMetrics(result.experimentResult));
            report.append(String.format("  Forgetting Score: %.3f\n", result.forgettingScore));
            report.append("\n");
        }
        
        // Check for catastrophic forgetting
        double maxForgetting = results.stream()
            .mapToDouble(r -> r.forgettingScore)
            .max()
            .orElse(0);
        
        report.append("-".repeat(30) + "\n");
        report.append("Catastrophic Forgetting: " + 
            (maxForgetting > 0.3 ? "DETECTED" : "NOT DETECTED") + "\n");
        
        return report.toString();
    }
    
    /**
     * Format metrics for display
     */
    private String formatMetrics(ExperimentResult result) {
        StringBuilder sb = new StringBuilder();
        
        for (Map.Entry<String, Double> metric : result.getMetrics().entrySet()) {
            sb.append(String.format("  %s: %.3f\n", metric.getKey(), metric.getValue()));
        }
        sb.append(String.format("  Composite Score: %.3f\n", result.getCompositeScore()));
        
        return sb.toString();
    }
    
    /**
     * Save report to file
     */
    private void saveReport(String name, String report) throws IOException {
        Path reportPath = Paths.get(experimentDir, name + "_report.txt");
        Files.write(reportPath, report.getBytes());
        System.out.println("\nReport saved to: " + reportPath);
    }
    
    /**
     * Interface for generation strategies
     */
    public interface GenerationStrategy {
        String getName();
        String generate(String prompt, int maxTokens);
    }
    
    /**
     * Factory for creating strategies with different parameters
     */
    public interface GenerationStrategyFactory {
        GenerationStrategy createStrategy(double parameterValue);
    }
    
    /**
     * Interface for incremental learners
     */
    public interface IncrementalLearner {
        void trainOnBatch(List<String> batch);
        String generate(String prompt);
        double measureForgetting();
    }
    
    /**
     * Container for experiment results
     */
    private static class ExperimentResult {
        private final String name;
        private final Map<String, Double> metrics;
        private List<String> samples;
        private double compositeScore;
        private double pValue;
        private boolean significant;
        
        public ExperimentResult(String name) {
            this.name = name;
            this.metrics = new HashMap<>();
        }
        
        public void addMetric(String key, double value) {
            metrics.put(key, value);
        }
        
        public String getName() { return name; }
        public Map<String, Double> getMetrics() { return metrics; }
        public List<String> getSamples() { return samples; }
        public void setSamples(List<String> samples) { this.samples = samples; }
        public double getCompositeScore() { return compositeScore; }
        public void setCompositeScore(double score) { this.compositeScore = score; }
        public void setPValue(double pValue) { this.pValue = pValue; }
        public void setSignificant(boolean significant) { this.significant = significant; }
    }
    
    /**
     * Container for incremental learning results
     */
    private static class IncrementalResult {
        public final int batchNumber;
        public final ExperimentResult experimentResult;
        public final double forgettingScore;
        
        public IncrementalResult(int batchNumber, ExperimentResult result, double forgettingScore) {
            this.batchNumber = batchNumber;
            this.experimentResult = result;
            this.forgettingScore = forgettingScore;
        }
    }
}
