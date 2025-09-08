package com.art.textgen.training;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Incremental training system for ART text generation
 * Implements curriculum learning, active learning, and memory management
 * Based on Phase 3.2 of EXECUTION_PLAN.md
 */
public class IncrementalTrainer {
    
    private final Vocabulary vocabulary;
    private final PatternGenerator patternGenerator;
    private final PatternExtractor patternExtractor;
    
    // Memory management
    private final Map<String, PatternCategory> patternMemory;
    private final PriorityQueue<PatternCategory> patternPriority;
    private final int maxMemorySize = 100000; // Maximum patterns to keep
    
    // Curriculum learning
    private int currentComplexityLevel = 1;
    private final int maxComplexityLevel = 5;
    
    // Active learning
    private final Set<String> uncertainPatterns;
    private final double uncertaintyThreshold = 0.3;
    
    // Statistics
    private final Map<String, Double> trainingStats;
    private int totalDocumentsProcessed = 0;
    private int totalPatternsExtracted = 0;
    
    public IncrementalTrainer(Vocabulary vocabulary, PatternGenerator patternGenerator) {
        this.vocabulary = vocabulary;
        this.patternGenerator = patternGenerator;
        this.patternExtractor = new PatternExtractor(vocabulary);
        
        this.patternMemory = new ConcurrentHashMap<>();
        this.patternPriority = new PriorityQueue<>(
            Comparator.comparingDouble(PatternCategory::getRelevanceScore).reversed()
        );
        
        this.uncertainPatterns = new HashSet<>();
        this.trainingStats = new HashMap<>();
    }
    
    /**
     * Train on a batch of documents without forgetting previous knowledge
     * Implements the core incremental learning algorithm
     */
    public void trainBatch(List<String> documents) {
        System.out.println("\n=== Incremental Training Batch ===");
        System.out.println("Documents in batch: " + documents.size());
        System.out.println("Current complexity level: " + currentComplexityLevel);
        System.out.println("Current memory size: " + patternMemory.size());
        
        // Filter documents by complexity for curriculum learning
        List<String> filteredDocs = filterByComplexity(documents, currentComplexityLevel);
        
        // Process each document
        for (String document : filteredDocs) {
            processDocument(document);
            totalDocumentsProcessed++;
            
            // Periodic memory management
            if (totalDocumentsProcessed % 10 == 0) {
                performMemoryManagement();
            }
        }
        
        // Update pattern weights and resonance
        updatePatternResonance();
        
        // Identify uncertain patterns for active learning
        identifyUncertainPatterns();
        
        // Progress curriculum if ready
        if (shouldProgressCurriculum()) {
            currentComplexityLevel = Math.min(currentComplexityLevel + 1, maxComplexityLevel);
            System.out.println("Progressed to complexity level: " + currentComplexityLevel);
        }
        
        // Update statistics
        updateStatistics();
        
        System.out.println("Batch training complete. Total patterns: " + patternMemory.size());
    }
    
    /**
     * Process a single document and extract patterns
     */
    private void processDocument(String document) {
        // Convert document to sentences for pattern extraction
        List<String> tokens = vocabulary.tokenize(document);
        List<List<String>> sentences = splitIntoSentences(tokens);
        
        // Extract patterns from sentences
        patternExtractor.extractPatterns(sentences);
        
        // Note: Since PatternExtractor doesn't return patterns directly,
        // we'll need to retrieve them from the extractor's internal state
        // For now, we'll count the processed sentences
        totalPatternsExtracted += sentences.size();
        
        // Process each sentence for pattern memory
        for (List<String> sentence : sentences) {
            String patternKey = String.join(" ", sentence);
            
            if (patternMemory.containsKey(patternKey)) {
                // Update existing pattern
                PatternCategory existing = patternMemory.get(patternKey);
                existing.reinforce();
                Pattern updatePattern = new Pattern(patternKey);
                existing.updateContext(updatePattern);
            } else {
                // Add new pattern
                Pattern pattern = new Pattern(patternKey);
                PatternCategory category = new PatternCategory(pattern);
                patternMemory.put(patternKey, category);
                patternPriority.offer(category);
            }
        }
    }
    
    /**
     * Split tokens into sentences
     */
    private List<List<String>> splitIntoSentences(List<String> tokens) {
        List<List<String>> sentences = new ArrayList<>();
        List<String> current = new ArrayList<>();
        
        for (String token : tokens) {
            current.add(token);
            if (token.matches("[.!?]")) {
                sentences.add(new ArrayList<>(current));
                current.clear();
            }
        }
        
        if (!current.isEmpty()) {
            sentences.add(current);
        }
        
        return sentences;
    }
    
    /**
     * Filter documents by complexity level for curriculum learning
     */
    private List<String> filterByComplexity(List<String> documents, int level) {
        return documents.stream()
            .filter(doc -> calculateComplexity(doc) <= level)
            .collect(Collectors.toList());
    }
    
    /**
     * Calculate document complexity (1-5 scale)
     */
    private int calculateComplexity(String document) {
        // Simple heuristic based on sentence length and vocabulary
        String[] sentences = document.split("[.!?]+");
        if (sentences.length == 0) return 1;
        
        double avgWordsPerSentence = Arrays.stream(sentences)
            .mapToInt(s -> s.split("\\s+").length)
            .average()
            .orElse(0);
        
        // Map to complexity level
        if (avgWordsPerSentence < 5) return 1;
        if (avgWordsPerSentence < 10) return 2;
        if (avgWordsPerSentence < 15) return 3;
        if (avgWordsPerSentence < 20) return 4;
        return 5;
    }
    
    /**
     * Perform memory management - prune rare patterns, merge similar ones
     */
    private void performMemoryManagement() {
        if (patternMemory.size() < maxMemorySize) {
            return; // No need to prune yet
        }
        
        System.out.println("Performing memory management...");
        
        // Prune rare patterns (those with low relevance scores)
        List<String> toPrune = new ArrayList<>();
        for (Map.Entry<String, PatternCategory> entry : patternMemory.entrySet()) {
            if (entry.getValue().getRelevanceScore() < 0.1) {
                toPrune.add(entry.getKey());
            }
        }
        
        for (String key : toPrune) {
            patternMemory.remove(key);
        }
        
        // Merge similar patterns
        mergeSimilarPatterns();
        
        System.out.println("Pruned " + toPrune.size() + " patterns");
    }
    
    /**
     * Merge patterns that are highly similar
     */
    private void mergeSimilarPatterns() {
        List<String> keys = new ArrayList<>(patternMemory.keySet());
        Set<String> toRemove = new HashSet<>();
        
        for (int i = 0; i < keys.size() - 1; i++) {
            if (toRemove.contains(keys.get(i))) continue;
            
            PatternCategory cat1 = patternMemory.get(keys.get(i));
            
            for (int j = i + 1; j < keys.size(); j++) {
                if (toRemove.contains(keys.get(j))) continue;
                
                PatternCategory cat2 = patternMemory.get(keys.get(j));
                
                if (cat1.similarity(cat2) > 0.9) {
                    // Merge cat2 into cat1
                    cat1.merge(cat2);
                    toRemove.add(keys.get(j));
                }
            }
        }
        
        // Remove merged patterns
        for (String key : toRemove) {
            patternMemory.remove(key);
        }
    }
    
    /**
     * Update pattern resonance based on recent usage
     */
    private void updatePatternResonance() {
        for (PatternCategory category : patternMemory.values()) {
            category.updateResonance();
        }
    }
    
    /**
     * Identify patterns with high uncertainty for active learning
     */
    private void identifyUncertainPatterns() {
        uncertainPatterns.clear();
        
        for (Map.Entry<String, PatternCategory> entry : patternMemory.entrySet()) {
            if (entry.getValue().getUncertainty() > uncertaintyThreshold) {
                uncertainPatterns.add(entry.getKey());
            }
        }
        
        if (!uncertainPatterns.isEmpty()) {
            System.out.println("Identified " + uncertainPatterns.size() + 
                " uncertain patterns for active learning");
        }
    }
    
    /**
     * Request specific examples for uncertain patterns (active learning)
     */
    public List<String> requestExamples() {
        List<String> requests = new ArrayList<>();
        
        for (String pattern : uncertainPatterns) {
            requests.add("Please provide examples of: " + pattern);
        }
        
        return requests;
    }
    
    /**
     * Learn from provided examples (active learning feedback)
     */
    public void learnFromExamples(Map<String, List<String>> examples) {
        for (Map.Entry<String, List<String>> entry : examples.entrySet()) {
            String patternKey = entry.getKey();
            
            if (patternMemory.containsKey(patternKey)) {
                PatternCategory category = patternMemory.get(patternKey);
                
                for (String example : entry.getValue()) {
                    category.addExample(example);
                }
                
                category.reduceUncertainty();
            }
        }
    }
    
    /**
     * Check if curriculum should progress to next level
     */
    private boolean shouldProgressCurriculum() {
        // Progress when we've seen enough documents and performance is good
        if (totalDocumentsProcessed < currentComplexityLevel * 20) {
            return false;
        }
        
        // Check pattern stability
        double avgResonance = patternMemory.values().stream()
            .mapToDouble(PatternCategory::getResonance)
            .average()
            .orElse(0);
        
        return avgResonance > 0.7;
    }
    
    /**
     * Update training statistics
     */
    private void updateStatistics() {
        trainingStats.put("total_documents", (double) totalDocumentsProcessed);
        trainingStats.put("total_patterns", (double) patternMemory.size());
        trainingStats.put("complexity_level", (double) currentComplexityLevel);
        trainingStats.put("uncertain_patterns", (double) uncertainPatterns.size());
        
        double avgResonance = patternMemory.values().stream()
            .mapToDouble(PatternCategory::getResonance)
            .average()
            .orElse(0);
        trainingStats.put("avg_resonance", avgResonance);
    }
    
    /**
     * Get training statistics
     */
    public Map<String, Double> getStatistics() {
        return new HashMap<>(trainingStats);
    }
    
    /**
     * Save the trained model
     */
    public void saveModel(String path) throws IOException {
        System.out.println("Saving incremental model to: " + path);
        
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(path))) {
            
            oos.writeObject(patternMemory);
            oos.writeInt(currentComplexityLevel);
            oos.writeInt(totalDocumentsProcessed);
            oos.writeObject(trainingStats);
        }
        
        System.out.println("Model saved successfully");
    }
    
    /**
     * Load a trained model
     */
    @SuppressWarnings("unchecked")
    public void loadModel(String path) throws IOException, ClassNotFoundException {
        System.out.println("Loading incremental model from: " + path);
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(path))) {
            
            patternMemory.clear();
            patternMemory.putAll((Map<String, PatternCategory>) ois.readObject());
            currentComplexityLevel = ois.readInt();
            totalDocumentsProcessed = ois.readInt();
            trainingStats.putAll((Map<String, Double>) ois.readObject());
        }
        
        System.out.println("Model loaded successfully");
    }
    
    /**
     * Pattern category with resonance and uncertainty tracking
     */
    private static class PatternCategory implements Serializable {
        private final Pattern pattern;
        private int frequency = 1;
        private double resonance = 0.5;
        private double uncertainty = 0.5;
        private final List<String> examples = new ArrayList<>();
        private long lastSeen = System.currentTimeMillis();
        
        public PatternCategory(Pattern pattern) {
            this.pattern = pattern;
        }
        
        public void reinforce() {
            frequency++;
            resonance = Math.min(1.0, resonance + 0.1);
            lastSeen = System.currentTimeMillis();
        }
        
        public void updateContext(Pattern newPattern) {
            // Update pattern context with new information
            // This would merge context from the new pattern
            // For now, just reinforce the pattern
            reinforce();
        }
        
        public void updateResonance() {
            // Decay resonance over time
            long timeSinceLastSeen = System.currentTimeMillis() - lastSeen;
            double decayFactor = Math.exp(-timeSinceLastSeen / 3600000.0); // 1 hour half-life
            resonance *= decayFactor;
        }
        
        public double getRelevanceScore() {
            return resonance * Math.log(1 + frequency) / (1 + uncertainty);
        }
        
        public double similarity(PatternCategory other) {
            // Calculate similarity between patterns
            // Simplified - would use actual pattern comparison
            return 0.5;
        }
        
        public void merge(PatternCategory other) {
            frequency += other.frequency;
            resonance = Math.max(resonance, other.resonance);
            uncertainty = Math.min(uncertainty, other.uncertainty);
            examples.addAll(other.examples);
        }
        
        public void addExample(String example) {
            examples.add(example);
            if (examples.size() > 10) {
                examples.remove(0); // Keep only recent examples
            }
        }
        
        public void reduceUncertainty() {
            uncertainty = Math.max(0, uncertainty - 0.2);
        }
        
        public double getResonance() { return resonance; }
        public double getUncertainty() { return uncertainty; }
    }
    
    /**
     * Simple Pattern class for demonstration
     */
    private static class Pattern implements Serializable {
        private final String content;
        
        public Pattern(String content) {
            this.content = content;
        }
        
        @Override
        public String toString() {
            return content;
        }
    }
    
    /**
     * Get total training steps
     */
    public int getTotalSteps() {
        return totalDocumentsProcessed;
    }
}
