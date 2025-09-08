package com.art.textgen;

import com.art.textgen.GrossbergTextGenerator;
import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.core.Vocabulary;
import com.art.textgen.training.TrainingPipeline;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Basic transformer replacement functionality test - lightweight version for main test suite
 * 
 * For comprehensive transformer replacement testing, run:
 * mvn test -Pintegration-tests -Dtest=TransformerReplacementTest
 */
@DisplayName("Basic Transformer Replacement")
public class BasicTransformerReplacementTest {
    
    private GrossbergTextGenerator grossbergGenerator;
    private EnhancedPatternGenerator patternGenerator;
    private TrainingPipeline pipeline;
    private Vocabulary vocabulary;
    
    @BeforeEach
    void setUp() {
        vocabulary = new Vocabulary(64); // Smaller vocab for faster tests
        patternGenerator = new EnhancedPatternGenerator(vocabulary);
        grossbergGenerator = new GrossbergTextGenerator();
        pipeline = new TrainingPipeline(vocabulary, patternGenerator);
        
        // Quick training for basic functionality
        pipeline.trainFromSamples();
    }
    
    @Test
    @DisplayName("Basic API Compatibility")
    void testBasicAPICompatibility() {
        String prompt = "The system works";
        int maxTokens = 20; // Small for speed
        
        // Test basic generation
        String generated = grossbergGenerator.generate(prompt, maxTokens)
            .limit(maxTokens)
            .collect(Collectors.joining(" "));
        
        assertNotNull(generated, "Generation should not return null");
        assertFalse(generated.trim().isEmpty(), "Generated text should not be empty");
        
        // Test token streaming
        List<String> tokens = grossbergGenerator.generate(prompt, maxTokens)
            .limit(maxTokens)
            .collect(Collectors.toList());
        
        assertFalse(tokens.isEmpty(), "Token stream should not be empty");
        assertTrue(tokens.size() <= maxTokens, "Token count should not exceed limit");
    }
    
    @Test
    @DisplayName("Basic Generation Quality")
    void testBasicGenerationQuality() {
        String prompt = "Artificial intelligence";
        String generated = grossbergGenerator.generate(prompt, 30)
            .limit(30)
            .collect(Collectors.joining(" "));
        
        assertNotNull(generated);
        assertFalse(generated.trim().isEmpty());
        
        // Basic quality checks
        String[] words = generated.trim().split("\\s+");
        assertTrue(words.length > 0, "Should generate at least one word");
        assertTrue(words.length <= 30, "Should respect token limit");
        
        // Check for reasonable text (no excessive repetition of single character)
        boolean hasVariety = Arrays.stream(words).distinct().count() > 1;
        assertTrue(hasVariety, "Generated text should have some variety");
    }
    
    @Test  
    @DisplayName("Basic Performance")
    void testBasicPerformance() {
        String prompt = "Quick test";
        
        long startTime = System.nanoTime();
        String generated = grossbergGenerator.generate(prompt, 20)
            .limit(20)
            .collect(Collectors.joining(" "));
        long endTime = System.nanoTime();
        
        double durationSeconds = (endTime - startTime) / 1_000_000_000.0;
        assertNotNull(generated);
        
        // Should generate reasonably quickly (less than 1 second for 20 tokens)
        assertTrue(durationSeconds < 1.0, 
            String.format("Basic generation took %.3f seconds, should be faster", durationSeconds));
    }
}