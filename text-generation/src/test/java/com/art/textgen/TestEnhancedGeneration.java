package com.art.textgen;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.*;
import com.art.textgen.training.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * JUnit test class for enhanced generation with repetition penalty and advanced sampling
 */
public class TestEnhancedGeneration {
    
    private Vocabulary vocabulary;
    private EnhancedPatternGenerator generator;
    private TrainingPipeline pipeline;
    
    @BeforeEach
    public void setUp() {
        vocabulary = new Vocabulary(64);
        generator = new EnhancedPatternGenerator(vocabulary, 0.8);
        pipeline = new TrainingPipeline(vocabulary, generator);
    }
    
    @Test
    @DisplayName("Enhanced ART Text Generation Test")
    public void testEnhancedGeneration() {
        System.out.println("=== Enhanced ART Text Generation Test ===\n");
        
        // Train
        System.out.println("1. Training from corpus...");
        try {
            // Try to train from downloaded corpus first
            pipeline.trainFromDirectory("training-corpus");
        } catch (Exception e) {
            // Fall back to sample corpus
            System.out.println("   Using sample corpus (run expand-corpus.sh for full corpus)");
            pipeline.trainFromSamples();
        }
        System.out.println();
        
        // Test different generation modes
        testGenerationModes(generator);
        
        // Compare with/without repetition penalty
        compareRepetitionPenalty(generator);
        
        // Test beam search
        testBeamSearch(generator);
        
        // Performance metrics
        testPerformanceMetrics(generator);
    }
    
    /**
     * Test different generation modes
     */
    private static void testGenerationModes(EnhancedPatternGenerator generator) {
        System.out.println("2. Testing Generation Modes\n");
        
        String prompt = "The future of artificial intelligence";
        EnhancedPatternGenerator.GenerationMode[] modes = {
            EnhancedPatternGenerator.GenerationMode.CONSERVATIVE,
            EnhancedPatternGenerator.GenerationMode.BALANCED,
            EnhancedPatternGenerator.GenerationMode.CREATIVE
        };
        
        for (var mode : modes) {
            generator.configureMode(mode);
            System.out.println("Mode: " + mode.name());
            System.out.println("  Temperature: " + mode.temperature);
            System.out.println("  Top-K: " + mode.topK);
            System.out.println("  Top-P: " + mode.topP);
            
            var result = generator.generateWithMetrics(prompt, 30);
            System.out.println("  Generated: " + result.getFullText());
            System.out.println("  Diversity: " + 
                String.format("%.2f", result.metrics.get("vocabulary_diversity")));
            System.out.println();
        }
    }
    
    /**
     * Compare generation with and without repetition penalty
     */
    private static void compareRepetitionPenalty(EnhancedPatternGenerator generator) {
        System.out.println("3. Repetition Penalty Comparison\n");
        
        String prompt = "Machine learning";
        
        // Without penalty (using base PatternGenerator)
        System.out.println("WITHOUT Repetition Penalty:");
        Vocabulary vocab = new Vocabulary(64);
        PatternGenerator baseGen = new PatternGenerator(vocab, 0.8);
        
        List<String> context = new ArrayList<>(Arrays.asList(prompt.split(" ")));
        System.out.print("  " + prompt);
        for (int i = 0; i < 40; i++) {
            String next = baseGen.generateNext(context);
            if (next == null || next.equals("<END>")) break;
            System.out.print(" " + next);
            context.add(next);
        }
        System.out.println("\n");
        
        // With penalty
        System.out.println("WITH Repetition Penalty:");
        generator.configureMode(EnhancedPatternGenerator.GenerationMode.BALANCED);
        var result = generator.generateWithMetrics(prompt, 40);
        System.out.println("  " + result.getFullText());
        System.out.println("  Repetition Rate: " + 
            String.format("%.2f", result.metrics.get("repetition_rate")));
        System.out.println();
    }
    
    /**
     * Test beam search generation
     */
    private static void testBeamSearch(EnhancedPatternGenerator generator) {
        System.out.println("4. Beam Search Generation\n");
        
        String prompt = "Once upon a time";
        List<String> context = Arrays.asList(prompt.split(" "));
        
        System.out.println("Prompt: " + prompt);
        
        // Beam width 1 (greedy)
        System.out.println("\nBeam Width 1 (Greedy):");
        List<String> beam1 = generator.generateWithBeamSearch(context, 20, 1);
        System.out.println("  " + prompt + " " + String.join(" ", beam1));
        
        // Beam width 3
        System.out.println("\nBeam Width 3:");
        List<String> beam3 = generator.generateWithBeamSearch(context, 20, 3);
        System.out.println("  " + prompt + " " + String.join(" ", beam3));
        
        // Beam width 5
        System.out.println("\nBeam Width 5:");
        List<String> beam5 = generator.generateWithBeamSearch(context, 20, 5);
        System.out.println("  " + prompt + " " + String.join(" ", beam5));
        System.out.println();
    }
    
    /**
     * Test performance metrics
     */
    private static void testPerformanceMetrics(EnhancedPatternGenerator generator) {
        System.out.println("5. Performance Metrics\n");
        
        String[] prompts = {
            "The human brain",
            "Quantum computing",
            "Climate change",
            "In the beginning",
            "Scientific research"
        };
        
        double totalPerplexity = 0;
        double totalDiversity = 0;
        double totalSpeed = 0;
        
        for (String prompt : prompts) {
            var result = generator.generateWithMetrics(prompt, 50);
            
            double perplexity = (Double) result.metrics.getOrDefault("estimated_perplexity", 0.0);
            double diversity = (Double) result.metrics.getOrDefault("diversity", 0.0);
            double speed = (Double) result.metrics.getOrDefault("tokens_per_second", 0.0);
            
            totalPerplexity += perplexity;
            totalDiversity += diversity;
            totalSpeed += speed;
            
            System.out.printf("Prompt: \"%s\"\n", prompt);
            System.out.printf("  Perplexity: %.2f\n", perplexity);
            System.out.printf("  Diversity: %.2f\n", diversity);
            System.out.printf("  Speed: %.1f tokens/sec\n", speed);
        }
        
        System.out.println("\nAverages:");
        System.out.printf("  Perplexity: %.2f\n", totalPerplexity / prompts.length);
        System.out.printf("  Diversity: %.2f\n", totalDiversity / prompts.length);
        System.out.printf("  Speed: %.1f tokens/sec\n", totalSpeed / prompts.length);
        
        // Success criteria
        System.out.println("\n✓ Success Criteria:");
        System.out.println("  Target Perplexity < 50: " + 
            (totalPerplexity / prompts.length < 50 ? "✅ ACHIEVED" : "❌ Not yet"));
        System.out.println("  Target Diversity > 0.7: " + 
            (totalDiversity / prompts.length > 0.7 ? "✅ ACHIEVED" : "❌ Not yet"));
        System.out.println("  Target Speed > 100 tokens/sec: " + 
            (totalSpeed / prompts.length > 100 ? "✅ ACHIEVED" : "❌ Not yet"));
    }
}
