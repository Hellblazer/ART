package com.art.textgen.integration;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.evaluation.*;
import com.art.textgen.generation.*;
import com.art.textgen.monitoring.TrainingDashboard;
import com.art.textgen.training.*;

import javax.swing.SwingUtilities;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Integrated Training and Generation Pipeline
 * Connects all components: IncrementalTrainer, ContextAwareGenerator,
 * TextGenerationMetrics, TrainingDashboard, and ExperimentRunner
 * 
 * This is the main integration layer for the ART Text Generation system
 */
public class IntegratedPipeline {
    
    // Core components
    private final Vocabulary vocabulary;
    private final IncrementalTrainer incrementalTrainer;
    private final ContextAwareGenerator contextGenerator;
    private final AdvancedSamplingMethods sampler;
    private final TextGenerationMetrics metrics;
    private final TrainingDashboard dashboard;
    private final ExperimentRunner experimentRunner;
    
    // Existing components
    private final PatternGenerator basePatternGenerator;
    private final EnhancedPatternGenerator enhancedPatternGenerator;
    
    // Pipeline configuration
    private final Map<String, Object> configuration;
    private volatile boolean isTraining = false;
    private volatile boolean dashboardEnabled = true;
    
    // Statistics tracking
    private final Map<String, Double> pipelineStats;
    private final List<String> generatedSamples;
    private int epochCount = 0;
    
    public IntegratedPipeline() {
        this(new Vocabulary(64)); // Default 64-dimensional embeddings
    }
    
    public IntegratedPipeline(Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        
        // Initialize generation components
        this.basePatternGenerator = new PatternGenerator(vocabulary, 0.7); // Default vigilance
        this.enhancedPatternGenerator = new EnhancedPatternGenerator(vocabulary, 0.7);
        this.incrementalTrainer = new IncrementalTrainer(vocabulary, enhancedPatternGenerator);
        
        // Initialize context and sampling
        this.contextGenerator = new ContextAwareGenerator();
        this.sampler = new AdvancedSamplingMethods(1.0, 40, 0.9);
        
        // Initialize evaluation and monitoring
        this.metrics = new TextGenerationMetrics();
        this.experimentRunner = new ExperimentRunner();
        
        // Initialize dashboard (can be disabled)
        this.dashboard = dashboardEnabled ? new TrainingDashboard() : null;
        
        // Initialize tracking
        this.configuration = new HashMap<>();
        this.pipelineStats = new ConcurrentHashMap<>();
        this.generatedSamples = Collections.synchronizedList(new ArrayList<>());
        
        configureDefaults();
    }
    
    /**
     * Configure default pipeline parameters
     */
    private void configureDefaults() {
        configuration.put("batch_size", 32);
        configuration.put("max_epochs", 10);
        configuration.put("learning_rate", 0.001);
        configuration.put("vigilance", 0.7);
        configuration.put("temperature", 1.0);
        configuration.put("top_k", 40);
        configuration.put("top_p", 0.9);
        configuration.put("max_generation_length", 100);
        configuration.put("evaluation_interval", 100);
        configuration.put("checkpoint_interval", 500);
    }
    
    /**
     * Main training method with full integration
     */
    public void train(String corpusPath) throws IOException {
        System.out.println("\n=== Integrated Training Pipeline Started ===");
        System.out.println("Corpus: " + corpusPath);
        System.out.println("Configuration: " + configuration);
        
        isTraining = true;
        
        // Show dashboard if enabled
        if (dashboard != null) {
            SwingUtilities.invokeLater(() -> dashboard.setVisible(true));
            dashboard.updateStatus("Loading corpus...");
        }
        
        // Load corpus
        List<String> documents = loadCorpus(corpusPath);
        int totalDocuments = documents.size();
        
        updateDashboard("Loaded " + totalDocuments + " documents");
        
        // Training loop with incremental learning
        int maxEpochs = (int) configuration.get("max_epochs");
        int batchSize = (int) configuration.get("batch_size");
        
        for (int epoch = 0; epoch < maxEpochs && isTraining; epoch++) {
            epochCount = epoch + 1;
            System.out.println("\n--- Epoch " + epochCount + " / " + maxEpochs + " ---");
            updateDashboard("Starting epoch " + epochCount);
            
            // Shuffle documents for this epoch
            Collections.shuffle(documents);
            
            // Process in batches
            for (int i = 0; i < documents.size() && isTraining; i += batchSize) {
                int end = Math.min(i + batchSize, documents.size());
                List<String> batch = documents.subList(i, end);
                
                // Incremental training on batch
                trainBatch(batch, i / batchSize);
                
                // Periodic evaluation
                if ((i / batchSize) % 10 == 0) { // Simplified interval
                    evaluateGeneration();
                }
                
                // Update progress
                int progress = (int) ((i + batchSize) * 100.0 / documents.size());
                updateProgress(progress);
            }
            
            // End of epoch evaluation
            System.out.println("\nEpoch " + epochCount + " complete");
            performEpochEvaluation();
        }
        
        // Final evaluation
        performFinalEvaluation();
        
        isTraining = false;
        updateDashboard("Training complete");
        
        System.out.println("\n=== Training Pipeline Complete ===");
        printFinalStatistics();
    }
    
    /**
     * Train a single batch with incremental learning
     */
    private void trainBatch(List<String> batch, int batchNumber) {
        // Use incremental trainer
        incrementalTrainer.trainBatch(batch);
        
        // Get training statistics
        Map<String, Double> trainerStats = incrementalTrainer.getStatistics();
        
        // Update pipeline statistics
        for (Map.Entry<String, Double> stat : trainerStats.entrySet()) {
            pipelineStats.put("trainer_" + stat.getKey(), stat.getValue());
            updateMetric("trainer_" + stat.getKey(), stat.getValue());
        }
        
        log("Batch " + batchNumber + " trained (" + batch.size() + " documents)");
    }
    
    /**
     * Generate text using integrated components
     */
    public String generate(String prompt, int maxLength) {
        // Simplified generation for now
        // In full implementation, would integrate ContextAwareGenerator
        String generated = enhancedPatternGenerator.generate(prompt, maxLength);
        
        // Add to samples for evaluation
        generatedSamples.add(generated);
        if (dashboard != null) {
            dashboard.addSample(generated);
        }
        
        return generated;
    }
    
    /**
     * Evaluate current generation quality
     */
    private void evaluateGeneration() {
        System.out.println("\nEvaluating generation quality...");
        
        // Generate test samples
        String[] testPrompts = {
            "Once upon a time",
            "The future of technology",
            "In the beginning",
            "Science has shown that"
        };
        
        List<String> samples = new ArrayList<>();
        for (String prompt : testPrompts) {
            String generated = generate(prompt, 50);
            samples.add(generated);
        }
        
        // Calculate metrics
        double diversity = metrics.calculateDiversity(samples, 2);
        double avgCoherence = 0;
        double avgFluency = 0;
        double avgReadability = 0;
        
        for (String sample : samples) {
            avgCoherence += metrics.calculateCoherence(sample, 3);
            avgFluency += metrics.calculateFluency(sample);
            avgReadability += metrics.calculateReadability(sample);
        }
        
        avgCoherence /= samples.size();
        avgFluency /= samples.size();
        avgReadability /= samples.size();
        
        // Update statistics
        pipelineStats.put("diversity", diversity);
        pipelineStats.put("coherence", avgCoherence);
        pipelineStats.put("fluency", avgFluency);
        pipelineStats.put("readability", avgReadability);
        
        // Update dashboard
        updateMetric("diversity", diversity);
        updateMetric("coherence", avgCoherence);
        updateMetric("fluency", avgFluency);
        updateMetric("readability", avgReadability);
        
        System.out.printf("Diversity: %.3f, Coherence: %.3f, Fluency: %.3f, Readability: %.1f\n",
            diversity, avgCoherence, avgFluency, avgReadability);
    }
    
    /**
     * Perform comprehensive epoch evaluation
     */
    private void performEpochEvaluation() {
        System.out.println("\n=== Epoch " + epochCount + " Evaluation ===");
        
        evaluateGeneration();
        
        // Generate detailed report
        String report = metrics.generateReport();
        System.out.println(report);
    }
    
    /**
     * Perform final evaluation after training
     */
    private void performFinalEvaluation() {
        System.out.println("\n=== Final Evaluation ===");
        
        // Generate comprehensive test set
        List<String> finalSamples = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            String sample = generate("Test prompt " + i, 100);
            finalSamples.add(sample);
        }
        
        // Calculate all metrics
        double diversity = metrics.calculateDiversity(finalSamples, 2);
        double compositeScore = metrics.getCompositeScore();
        
        System.out.printf("Final Diversity: %.3f\n", diversity);
        System.out.printf("Final Composite Score: %.3f\n", compositeScore);
    }
    
    /**
     * Load corpus from directory
     */
    private List<String> loadCorpus(String path) throws IOException {
        List<String> documents = new ArrayList<>();
        
        Path corpusPath = Paths.get(path);
        if (Files.isDirectory(corpusPath)) {
            Files.walk(corpusPath)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".txt"))
                .forEach(file -> {
                    try {
                        String content = Files.readString(file);
                        documents.add(content);
                    } catch (IOException e) {
                        System.err.println("Failed to read: " + file);
                    }
                });
        } else if (Files.isRegularFile(corpusPath)) {
            documents.add(Files.readString(corpusPath));
        }
        
        return documents;
    }
    
    /**
     * Print final statistics
     */
    private void printFinalStatistics() {
        System.out.println("\n=== Final Pipeline Statistics ===");
        System.out.println("Epochs completed: " + epochCount);
        System.out.println("Samples generated: " + generatedSamples.size());
        
        for (Map.Entry<String, Double> stat : pipelineStats.entrySet()) {
            System.out.printf("%s: %.4f\n", stat.getKey(), stat.getValue());
        }
    }
    
    // Dashboard update methods
    
    private void updateDashboard(String status) {
        if (dashboard != null) {
            dashboard.updateStatus(status);
        }
        System.out.println("[STATUS] " + status);
    }
    
    private void updateMetric(String name, double value) {
        if (dashboard != null) {
            dashboard.updateMetric(name, value);
        }
    }
    
    private void updateProgress(int percent) {
        if (dashboard != null) {
            dashboard.updateProgress(percent);
        }
    }
    
    private void log(String message) {
        if (dashboard != null) {
            dashboard.log(message);
        }
    }
    
    // Configuration methods
    
    public void setConfiguration(String key, Object value) {
        configuration.put(key, value);
    }
    
    public Object getConfiguration(String key) {
        return configuration.get(key);
    }
    
    public void enableDashboard(boolean enable) {
        this.dashboardEnabled = enable;
    }
    
    public void stopTraining() {
        isTraining = false;
        updateDashboard("Training stopped by user");
    }
    
    public boolean isTraining() {
        return isTraining;
    }
    
    public void shutdown() {
        stopTraining();
        if (dashboard != null) {
            dashboard.shutdown();
        }
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        try {
            IntegratedPipeline pipeline = new IntegratedPipeline();
            
            // Configure pipeline
            pipeline.setConfiguration("max_epochs", 2);
            pipeline.setConfiguration("batch_size", 10);
            
            // Train on corpus
            pipeline.train("training-corpus");
            
            // Generate some samples
            System.out.println("\n=== Sample Generation ===");
            String sample1 = pipeline.generate("Once upon a time", 100);
            System.out.println("Sample 1: " + sample1);
            
            String sample2 = pipeline.generate("The future of technology", 100);
            System.out.println("Sample 2: " + sample2);
            
            // Shutdown
            pipeline.shutdown();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
