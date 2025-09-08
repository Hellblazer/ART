package com.art.textgen.training;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import com.art.textgen.dynamics.ResonanceDetector;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

/**
 * Training pipeline orchestrating corpus loading, pattern extraction, and model training
 */
public class TrainingPipeline {
    
    private final Vocabulary vocabulary;
    private final PatternGenerator patternGenerator;
    private final CorpusLoader corpusLoader;
    private final PatternExtractor patternExtractor;
    private final Map<String, Double> trainingMetrics;
    
    public TrainingPipeline(Vocabulary vocabulary, PatternGenerator patternGenerator) {
        this.vocabulary = vocabulary;
        this.patternGenerator = patternGenerator;
        this.corpusLoader = new CorpusLoader(vocabulary);
        this.patternExtractor = new PatternExtractor(vocabulary);
        this.trainingMetrics = new HashMap<>();
    }
    
    /**
     * Train from directory of text files
     */
    public void trainFromDirectory(String directoryPath) throws IOException {
        System.out.println("=== Starting Training Pipeline ===");
        
        // Step 1: Load corpus
        System.out.println("\n1. Loading corpus from: " + directoryPath);
        long startTime = System.currentTimeMillis();
        
        try {
            corpusLoader.loadFromDirectory(directoryPath);
        } catch (IOException e) {
            System.out.println("Directory not found, loading sample corpus instead");
            corpusLoader.loadSampleCorpus();
        }
        
        long loadTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("load_time_ms", (double) loadTime);
        
        // Step 2: Extract patterns
        System.out.println("\n2. Extracting patterns...");
        startTime = System.currentTimeMillis();
        
        List<List<String>> sentences = corpusLoader.getAllSentences();
        patternExtractor.extractPatterns(sentences);
        
        long extractTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("extract_time_ms", (double) extractTime);
        
        // Step 3: Train pattern generator
        System.out.println("\n3. Training pattern generator...");
        startTime = System.currentTimeMillis();
        
        patternExtractor.exportToPatternGenerator(patternGenerator);
        
        // Add semantic associations
        for (String word : vocabulary.getAllTokens()) {
            Set<String> associations = patternExtractor.getSemanticAssociations(word);
            // These would be added to vocabulary's semantic associations
            // (would need to add a method to Vocabulary to accept external associations)
        }
        
        long trainTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("train_time_ms", (double) trainTime);
        
        // Step 4: Calculate metrics
        calculateTrainingMetrics();
        
        System.out.println("\n=== Training Complete ===");
        printTrainingReport();
    }
    
    /**
     * Train from a single file
     */
    public void trainFromFile(String filePath) throws IOException {
        System.out.println("=== Training from File: " + filePath + " ===");
        
        long startTime = System.currentTimeMillis();
        
        // Load the single file
        corpusLoader.loadFile(Paths.get(filePath));
        
        long loadTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("load_time_ms", (double) loadTime);
        
        // Extract patterns
        startTime = System.currentTimeMillis();
        List<List<String>> sentences = corpusLoader.getAllSentences();
        patternExtractor.extractPatterns(sentences);
        
        long extractTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("extract_time_ms", (double) extractTime);
        
        // Train pattern generator
        startTime = System.currentTimeMillis();
        patternExtractor.exportToPatternGenerator(patternGenerator);
        
        long trainTime = System.currentTimeMillis() - startTime;
        trainingMetrics.put("train_time_ms", (double) trainTime);
        
        calculateTrainingMetrics();
    }
    
    /**
     * Train from sample corpus
     */
    public void trainFromSamples() {
        System.out.println("=== Training from Sample Corpus ===");
        
        corpusLoader.loadSampleCorpus();
        
        List<List<String>> sentences = corpusLoader.getAllSentences();
        patternExtractor.extractPatterns(sentences);
        patternExtractor.exportToPatternGenerator(patternGenerator);
        
        calculateTrainingMetrics();
        printTrainingReport();
    }
    
    /**
     * Calculate training metrics
     */
    private void calculateTrainingMetrics() {
        Map<String, Object> corpusStats = corpusLoader.getStatistics();
        Map<String, Object> patternStats = patternExtractor.getStatistics();
        
        trainingMetrics.put("total_documents", 
            ((Number) corpusStats.get("total_documents")).doubleValue());
        trainingMetrics.put("total_tokens", 
            ((Number) corpusStats.get("total_tokens")).doubleValue());
        trainingMetrics.put("unique_tokens", 
            ((Number) corpusStats.get("unique_tokens")).doubleValue());
        trainingMetrics.put("vocabulary_size", 
            ((Number) corpusStats.get("vocabulary_size")).doubleValue());
        
        trainingMetrics.put("ngram_patterns", 
            ((Number) patternStats.get("total_ngram_patterns")).doubleValue());
        trainingMetrics.put("syntactic_patterns", 
            ((Number) patternStats.get("unique_syntactic_patterns")).doubleValue());
        trainingMetrics.put("semantic_clusters", 
            ((Number) patternStats.get("semantic_clusters")).doubleValue());
        
        // Calculate vocabulary coverage
        double coverage = trainingMetrics.get("unique_tokens") / 
                         trainingMetrics.get("vocabulary_size");
        trainingMetrics.put("vocabulary_coverage", coverage);
    }
    
    /**
     * Print training report
     */
    private void printTrainingReport() {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║         TRAINING REPORT                ║");
        System.out.println("╠════════════════════════════════════════╣");
        
        System.out.printf("║ Documents:        %6.0f               ║\n", 
            trainingMetrics.get("total_documents"));
        System.out.printf("║ Total Tokens:     %6.0f               ║\n", 
            trainingMetrics.get("total_tokens"));
        System.out.printf("║ Unique Tokens:    %6.0f               ║\n", 
            trainingMetrics.get("unique_tokens"));
        System.out.printf("║ Vocabulary Size:  %6.0f               ║\n", 
            trainingMetrics.get("vocabulary_size"));
        
        System.out.println("╠════════════════════════════════════════╣");
        
        System.out.printf("║ N-gram Patterns:  %6.0f               ║\n", 
            trainingMetrics.get("ngram_patterns"));
        System.out.printf("║ Syntactic Types:  %6.0f               ║\n", 
            trainingMetrics.get("syntactic_patterns"));
        System.out.printf("║ Semantic Clusters:%6.0f               ║\n", 
            trainingMetrics.get("semantic_clusters"));
        
        System.out.println("╠════════════════════════════════════════╣");
        
        if (trainingMetrics.containsKey("load_time_ms")) {
            System.out.printf("║ Load Time:        %6.0f ms            ║\n", 
                trainingMetrics.get("load_time_ms"));
            System.out.printf("║ Extract Time:     %6.0f ms            ║\n", 
                trainingMetrics.get("extract_time_ms"));
            System.out.printf("║ Train Time:       %6.0f ms            ║\n", 
                trainingMetrics.get("train_time_ms"));
        }
        
        System.out.println("╚════════════════════════════════════════╝");
        
        // Print top patterns
        System.out.println("\nTop Patterns Learned:");
        List<PatternExtractor.PatternInfo> topPatterns = patternExtractor.getTopPatterns(5);
        for (int i = 0; i < topPatterns.size(); i++) {
            PatternExtractor.PatternInfo pattern = topPatterns.get(i);
            System.out.printf("%d. \"%s\" (freq: %d, importance: %.2f)\n", 
                i + 1, pattern.pattern, pattern.frequency, pattern.importance);
        }
    }
    
    /**
     * Get training metrics
     */
    public Map<String, Double> getMetrics() {
        return new HashMap<>(trainingMetrics);
    }
}
