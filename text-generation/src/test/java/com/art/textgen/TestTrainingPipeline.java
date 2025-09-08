package com.art.textgen;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import com.art.textgen.training.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * JUnit test class for the training pipeline
 */
public class TestTrainingPipeline {
    
    private Vocabulary vocabulary;
    private PatternGenerator patternGenerator;
    private TrainingPipeline pipeline;
    
    @BeforeEach
    public void setUp() {
        vocabulary = new Vocabulary(64);
        patternGenerator = new PatternGenerator(vocabulary, 0.8);
        pipeline = new TrainingPipeline(vocabulary, patternGenerator);
    }
    
    @Test
    @DisplayName("ART Training Pipeline Integration Test")
    public void testTrainingPipelineIntegration() {
        System.out.println("=== ART Text Generation Training Pipeline Test ===\n");
        
        // Test different training scenarios
        testSampleTraining(pipeline, vocabulary, patternGenerator);
        testDirectoryTraining(pipeline, vocabulary, patternGenerator);
        testAdaptiveVigilance(vocabulary, patternGenerator);
        testGenerationQuality(vocabulary, patternGenerator);
    }
    
    /**
     * Test training with sample corpus
     */
    private static void testSampleTraining(TrainingPipeline pipeline, 
                                          Vocabulary vocabulary, 
                                          PatternGenerator generator) {
        System.out.println("\n=== Testing Sample Corpus Training ===");
        
        // Train from samples
        pipeline.trainFromSamples();
        
        // Test generation with different prompts
        testGeneration(generator, vocabulary, "The future of", 30);
        testGeneration(generator, vocabulary, "Artificial intelligence", 30);
        testGeneration(generator, vocabulary, "Once upon a", 30);
    }
    
    /**
     * Test training from directory
     */
    private static void testDirectoryTraining(TrainingPipeline pipeline,
                                             Vocabulary vocabulary,
                                             PatternGenerator generator) {
        System.out.println("\n=== Testing Directory Training ===");
        
        // Create test corpus directory
        String corpusPath = createTestCorpus();
        
        try {
            // Train from directory
            pipeline.trainFromDirectory(corpusPath);
            
            // Test generation with corpus-specific prompts
            testGeneration(generator, vocabulary, "Neural networks", 30);
            testGeneration(generator, vocabulary, "The human brain", 30);
            testGeneration(generator, vocabulary, "Consciousness is", 30);
            
        } catch (IOException e) {
            System.err.println("Error training from directory: " + e.getMessage());
        }
    }
    
    /**
     * Test adaptive vigilance tuning
     */
    private static void testAdaptiveVigilance(Vocabulary vocabulary, 
                                             PatternGenerator generator) {
        System.out.println("\n=== Testing Adaptive Vigilance ===");
        
        // Test different vigilance levels
        double[] vigilanceLevels = {0.4, 0.6, 0.8};
        
        for (double vigilance : vigilanceLevels) {
            System.out.println("\nVigilance = " + vigilance);
            
            // Create new generator with specific vigilance
            PatternGenerator testGen = new PatternGenerator(vocabulary, 1.0 - vigilance);
            
            // Train with sample patterns
            trainWithPatterns(testGen);
            
            // Measure generation quality
            measureGenerationQuality(testGen, vocabulary, vigilance);
        }
    }
    
    /**
     * Test generation quality metrics
     */
    private static void testGenerationQuality(Vocabulary vocabulary,
                                             PatternGenerator generator) {
        System.out.println("\n=== Testing Generation Quality Metrics ===");
        
        // Generate multiple samples
        String[] prompts = {
            "The", "In the", "Machine learning", "The brain", "Science"
        };
        
        List<Double> perplexities = new ArrayList<>();
        List<Double> diversities = new ArrayList<>();
        List<Double> coherences = new ArrayList<>();
        
        for (String prompt : prompts) {
            GenerationMetrics metrics = evaluateGeneration(generator, vocabulary, prompt, 50);
            perplexities.add(metrics.perplexity);
            diversities.add(metrics.diversity);
            coherences.add(metrics.coherence);
            
            System.out.printf("Prompt: \"%s\"\n", prompt);
            System.out.printf("  Perplexity: %.2f\n", metrics.perplexity);
            System.out.printf("  Diversity: %.2f\n", metrics.diversity);
            System.out.printf("  Coherence: %.2f\n", metrics.coherence);
            System.out.printf("  Grammar Score: %.2f\n", metrics.grammarScore);
        }
        
        // Print averages
        System.out.println("\nAverage Metrics:");
        System.out.printf("  Perplexity: %.2f\n", average(perplexities));
        System.out.printf("  Diversity: %.2f\n", average(diversities));
        System.out.printf("  Coherence: %.2f\n", average(coherences));
    }
    
    /**
     * Create test corpus directory with sample files
     */
    private static String createTestCorpus() {
        String corpusDir = "test-corpus";
        
        try {
            Path dir = Paths.get(corpusDir);
            if (!Files.exists(dir)) {
                Files.createDirectory(dir);
            }
            
            // Create sample text files
            createCorpusFile(dir, "neuroscience.txt",
                "The human brain is a complex organ composed of billions of neurons. " +
                "These neurons communicate through electrical and chemical signals. " +
                "Neural networks in the brain process information in parallel. " +
                "Synaptic plasticity allows the brain to learn and adapt. " +
                "Memory formation involves changes in synaptic strength. " +
                "The cortex is organized into specialized regions for different functions. " +
                "Visual processing occurs in the occipital lobe. " +
                "Language processing involves Broca's and Wernicke's areas. " +
                "The hippocampus is crucial for memory consolidation. " +
                "Neurotransmitters like dopamine and serotonin regulate mood and behavior.");
            
            createCorpusFile(dir, "ai_basics.txt",
                "Artificial intelligence aims to create intelligent machines. " +
                "Machine learning enables computers to learn from data. " +
                "Deep learning uses neural networks with multiple layers. " +
                "Natural language processing helps computers understand human language. " +
                "Computer vision allows machines to interpret visual information. " +
                "Reinforcement learning teaches agents through rewards and penalties. " +
                "Supervised learning requires labeled training data. " +
                "Unsupervised learning discovers patterns in unlabeled data. " +
                "Transfer learning leverages knowledge from one task to another. " +
                "Neural networks are inspired by biological brain structures.");
            
            createCorpusFile(dir, "consciousness.txt",
                "Consciousness is the state of being aware of one's surroundings. " +
                "The hard problem of consciousness questions subjective experience. " +
                "Qualia refer to the subjective qualities of conscious experience. " +
                "Self-awareness is a key aspect of human consciousness. " +
                "The binding problem asks how separate neural processes create unified experience. " +
                "Attention and consciousness are closely related but distinct. " +
                "Altered states of consciousness include sleep, dreams, and meditation. " +
                "The global workspace theory proposes consciousness as information integration. " +
                "Phenomenal consciousness involves subjective experience. " +
                "Access consciousness refers to information available for use in reasoning.");
            
            return corpusDir;
            
        } catch (IOException e) {
            System.err.println("Error creating test corpus: " + e.getMessage());
            return "";
        }
    }
    
    /**
     * Create corpus file
     */
    private static void createCorpusFile(Path dir, String filename, String content) 
            throws IOException {
        Path file = dir.resolve(filename);
        Files.writeString(file, content);
    }
    
    /**
     * Train generator with patterns
     */
    private static void trainWithPatterns(PatternGenerator generator) {
        // Add common patterns
        String[][] patterns = {
            {"The", "quick", "brown", "fox"},
            {"The", "human", "brain", "is"},
            {"Neural", "networks", "can", "learn"},
            {"Artificial", "intelligence", "is", "the"},
            {"Machine", "learning", "algorithms", "are"},
            {"Deep", "learning", "models", "have"},
            {"Natural", "language", "processing", "enables"},
            {"Computer", "vision", "systems", "can"}
        };
        
        for (String[] pattern : patterns) {
            generator.learnPattern(Arrays.asList(pattern));
        }
    }
    
    /**
     * Test generation with a prompt
     */
    private static void testGeneration(PatternGenerator generator, 
                                      Vocabulary vocabulary,
                                      String prompt, 
                                      int length) {
        System.out.println("\nPrompt: \"" + prompt + "\"");
        System.out.print("Generated: " + prompt);
        
        List<String> context = vocabulary.tokenize(prompt);
        Set<String> usedTokens = new HashSet<>();
        
        for (int i = 0; i < length; i++) {
            String next = generator.generateNext(context);
            
            if (next.equals(Vocabulary.END_TOKEN)) {
                break;
            }
            
            System.out.print(" " + next);
            context.add(next);
            usedTokens.add(next);
            
            // Keep context window reasonable
            if (context.size() > 10) {
                context.remove(0);
            }
        }
        
        System.out.println();
        System.out.println("Vocabulary diversity: " + usedTokens.size() + " unique tokens");
    }
    
    /**
     * Measure generation quality
     */
    private static void measureGenerationQuality(PatternGenerator generator,
                                                Vocabulary vocabulary,
                                                double vigilance) {
        String prompt = "The";
        List<String> context = vocabulary.tokenize(prompt);
        List<String> generated = new ArrayList<>();
        
        // Generate 100 tokens
        for (int i = 0; i < 100; i++) {
            String next = generator.generateNext(context);
            if (next.equals(Vocabulary.END_TOKEN)) break;
            
            generated.add(next);
            context.add(next);
            if (context.size() > 10) context.remove(0);
        }
        
        // Calculate metrics
        Set<String> unique = new HashSet<>(generated);
        double diversity = unique.size() / (double) generated.size();
        
        System.out.printf("  Generated %d tokens, %d unique (diversity: %.2f)\n",
            generated.size(), unique.size(), diversity);
    }
    
    /**
     * Generation metrics class
     */
    static class GenerationMetrics {
        double perplexity;
        double diversity;
        double coherence;
        double grammarScore;
        
        GenerationMetrics(double perplexity, double diversity, 
                         double coherence, double grammarScore) {
            this.perplexity = perplexity;
            this.diversity = diversity;
            this.coherence = coherence;
            this.grammarScore = grammarScore;
        }
    }
    
    /**
     * Evaluate generation quality
     */
    private static GenerationMetrics evaluateGeneration(PatternGenerator generator,
                                                       Vocabulary vocabulary,
                                                       String prompt,
                                                       int length) {
        List<String> context = vocabulary.tokenize(prompt);
        List<String> generated = new ArrayList<>();
        Map<String, Integer> tokenFreq = new HashMap<>();
        
        // Generate tokens
        for (int i = 0; i < length; i++) {
            String next = generator.generateNext(context);
            if (next.equals(Vocabulary.END_TOKEN)) break;
            
            generated.add(next);
            tokenFreq.merge(next, 1, Integer::sum);
            context.add(next);
            if (context.size() > 10) context.remove(0);
        }
        
        // Calculate perplexity (simplified)
        double avgFreq = tokenFreq.values().stream()
            .mapToInt(Integer::intValue)
            .average()
            .orElse(1.0);
        double perplexity = Math.exp(-Math.log(1.0 / avgFreq));
        
        // Calculate diversity
        double diversity = tokenFreq.size() / (double) generated.size();
        
        // Calculate coherence (simplified - based on pattern consistency)
        double coherence = calculateCoherence(generated);
        
        // Calculate grammar score (simplified)
        double grammarScore = calculateGrammarScore(generated);
        
        return new GenerationMetrics(perplexity, diversity, coherence, grammarScore);
    }
    
    /**
     * Calculate coherence score
     */
    private static double calculateCoherence(List<String> tokens) {
        if (tokens.size() < 3) return 0.0;
        
        int coherentPairs = 0;
        
        for (int i = 0; i < tokens.size() - 2; i++) {
            // Check if trigrams make sense (simplified)
            String trigram = tokens.get(i) + " " + tokens.get(i+1) + " " + tokens.get(i+2);
            if (isCoherentTrigram(trigram)) {
                coherentPairs++;
            }
        }
        
        return coherentPairs / (double) (tokens.size() - 2);
    }
    
    /**
     * Check if trigram is coherent
     */
    private static boolean isCoherentTrigram(String trigram) {
        // Simplified coherence check
        String lower = trigram.toLowerCase();
        
        // Check for common patterns
        return !lower.contains("the the") &&
               !lower.contains("a a") &&
               !lower.contains("is is") &&
               !lower.contains("and and");
    }
    
    /**
     * Calculate grammar score
     */
    private static double calculateGrammarScore(List<String> tokens) {
        int correctGrammar = 0;
        
        for (int i = 0; i < tokens.size() - 1; i++) {
            String current = tokens.get(i).toLowerCase();
            String next = tokens.get(i + 1).toLowerCase();
            
            // Simple grammar rules
            if (isArticle(current) && !isVerb(next) && !isPreposition(next)) {
                correctGrammar++;
            } else if (isNoun(current) && (isVerb(next) || isPreposition(next))) {
                correctGrammar++;
            } else if (isVerb(current) && !isVerb(next)) {
                correctGrammar++;
            }
        }
        
        return correctGrammar / (double) Math.max(1, tokens.size() - 1);
    }
    
    // Helper methods for grammar checking
    private static boolean isArticle(String token) {
        return Arrays.asList("a", "an", "the").contains(token);
    }
    
    private static boolean isVerb(String token) {
        return token.endsWith("ing") || token.endsWith("ed") || 
               Arrays.asList("is", "are", "was", "were", "be", "have", "has", "had")
                   .contains(token);
    }
    
    private static boolean isPreposition(String token) {
        return Arrays.asList("in", "on", "at", "to", "for", "with", "by", "from")
            .contains(token);
    }
    
    private static boolean isNoun(String token) {
        // Simplified - words that don't fit other categories
        return !isArticle(token) && !isVerb(token) && !isPreposition(token);
    }
    
    /**
     * Calculate average
     */
    private static double average(List<Double> values) {
        return values.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
    }
}
