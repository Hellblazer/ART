package com.art.textgen.comprehensive;

import com.art.textgen.GrossbergTextGenerator;
import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.training.TrainingPipeline;
import com.art.textgen.core.Vocabulary;
import com.art.textgen.memory.RecursiveHierarchicalMemory;
import com.art.textgen.memory.MultiTimescaleMemoryBank;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Simplified comprehensive validation suite that works with existing codebase
 * 
 * VALIDATES:
 * - Core system integration and functionality
 * - Basic performance characteristics
 * - Memory system behavior
 * - Generation quality metrics
 */
@DisplayName("Simplified Comprehensive Validation Suite")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class SimplifiedValidationSuite {
    
    private GrossbergTextGenerator grossbergGenerator;
    private EnhancedPatternGenerator patternGenerator;
    private TrainingPipeline trainingPipeline;
    private Vocabulary vocabulary;
    private RecursiveHierarchicalMemory hierarchicalMemory;
    private MultiTimescaleMemoryBank timescaleBank;
    
    @BeforeEach
    void setUp() {
        vocabulary = new Vocabulary(64);
        patternGenerator = new EnhancedPatternGenerator(vocabulary);
        grossbergGenerator = new GrossbergTextGenerator();
        trainingPipeline = new TrainingPipeline(vocabulary, patternGenerator);
        hierarchicalMemory = new RecursiveHierarchicalMemory();
        timescaleBank = new MultiTimescaleMemoryBank();
    }
    
    @Test
    @Order(1)
    @DisplayName("CORE: End-to-End Training and Generation")
    void validateEndToEndWorkflow() {
        // Training phase
        assertDoesNotThrow(() -> {
            trainingPipeline.trainFromSamples();
        }, "Training should complete without errors");
        
        // Generation phase
        String prompt = "The neural network processes";
        List<String> generated = grossbergGenerator.generate(prompt, 50)
            .limit(50)
            .collect(Collectors.toList());
        
        assertFalse(generated.isEmpty(), "Should generate tokens");
        assertTrue(generated.size() >= 10, "Should generate reasonable number of tokens");
        
        String fullText = String.join(" ", generated);
        assertTrue(fullText.length() > prompt.length(), "Generated text should extend prompt");
        
        System.out.printf("✓ End-to-end workflow: %d tokens generated%n", generated.size());
    }
    
    @Test
    @Order(2)
    @DisplayName("MEMORY: Hierarchical Memory System")
    void validateHierarchicalMemory() {
        // Test adding tokens to hierarchical memory
        List<String> tokens = Arrays.asList("neural", "networks", "learn", "patterns", "from", "data");
        
        for (String token : tokens) {
            assertDoesNotThrow(() -> {
                hierarchicalMemory.addToken(token);
            }, "Should be able to add tokens to hierarchical memory");
        }
        
        // Test context retrieval
        List<Object> context = hierarchicalMemory.getActiveContext(10);
        assertNotNull(context, "Context should not be null");
        
        // Test capacity calculation
        double capacity = hierarchicalMemory.getEffectiveCapacity();
        assertTrue(capacity > 1000, "Effective capacity should be substantial");
        assertTrue(capacity < 50000, "Effective capacity should be reasonable");
        
        System.out.printf("✓ Hierarchical memory: %.0f effective capacity, %d context items%n", 
            capacity, context.size());
    }
    
    @Test
    @Order(3)
    @DisplayName("MEMORY: Multi-timescale Processing")
    void validateTimescaleProcessing() {
        // Test timescale memory updates
        List<String> sequence = Arrays.asList(
            "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
        );
        
        for (String token : sequence) {
            assertDoesNotThrow(() -> {
                timescaleBank.update(token);
            }, "Should be able to update timescale memory");
        }
        
        // Test active scales
        Set<MultiTimescaleMemoryBank.TimeScale> activeScales = timescaleBank.getActiveScales();
        assertNotNull(activeScales, "Active scales should not be null");
        assertFalse(activeScales.isEmpty(), "Should have active timescales");
        
        // Test predictions
        Map<MultiTimescaleMemoryBank.TimeScale, MultiTimescaleMemoryBank.Prediction> predictions = 
            timescaleBank.generatePredictions();
        assertNotNull(predictions, "Predictions should not be null");
        
        System.out.printf("✓ Timescale processing: %d active scales, %d predictions%n", 
            activeScales.size(), predictions.size());
    }
    
    @Test
    @Order(4)
    @DisplayName("GENERATION: Pattern Generation Quality")
    void validateGenerationQuality() {
        // Train first
        trainingPipeline.trainFromSamples();
        
        List<String> prompts = Arrays.asList(
            "The future of artificial intelligence",
            "Machine learning algorithms",
            "Deep neural networks"
        );
        
        List<String> generations = new ArrayList<>();
        
        for (String prompt : prompts) {
            String generated = patternGenerator.generate(prompt, 40);
            assertNotNull(generated, "Generated text should not be null");
            assertFalse(generated.trim().isEmpty(), "Generated text should not be empty");
            assertTrue(generated.length() > prompt.length(), "Generated text should extend prompt");
            
            generations.add(generated);
        }
        
        // Basic quality checks
        for (String generation : generations) {
            // Check for reasonable diversity (not all the same)
            String[] words = generation.split("\\s+");
            Set<String> uniqueWords = new HashSet<>(Arrays.asList(words));
            double diversity = (double) uniqueWords.size() / words.length;
            
            assertTrue(diversity > 0.3, 
                String.format("Diversity %.2f too low in generation", diversity));
        }
        
        System.out.printf("✓ Generation quality: %d prompts tested, avg length %.1f%n",
            generations.size(), 
            generations.stream().mapToInt(String::length).average().orElse(0));
    }
    
    @Test
    @Order(5)
    @DisplayName("PERFORMANCE: Generation Speed")
    void validateGenerationSpeed() {
        // Train first
        trainingPipeline.trainFromSamples();
        
        String prompt = "Performance test";
        int targetTokens = 100;
        
        // Measure generation speed
        long startTime = System.nanoTime();
        String generated = patternGenerator.generate(prompt, targetTokens);
        long endTime = System.nanoTime();
        
        double durationSeconds = (endTime - startTime) / 1_000_000_000.0;
        int actualTokens = generated.split("\\s+").length;
        double tokensPerSecond = actualTokens / durationSeconds;
        
        assertTrue(tokensPerSecond >= 1.0, 
            String.format("Generation speed %.1f tokens/s too slow", tokensPerSecond));
        assertTrue(durationSeconds <= 10.0, 
            String.format("Generation took %.1f seconds, too slow", durationSeconds));
        
        System.out.printf("✓ Performance: %.1f tokens/s, %.2f seconds%n", 
            tokensPerSecond, durationSeconds);
    }
    
    @Test
    @Order(6)
    @DisplayName("STABILITY: Continuous Generation")
    void validateContinuousGeneration() {
        // Train first
        trainingPipeline.trainFromSamples();
        
        String prompt = "The system generates";
        List<String> generations = new ArrayList<>();
        
        // Test multiple consecutive generations
        for (int i = 0; i < 10; i++) {
            String generated = grossbergGenerator.generate(prompt, 20)
                .limit(20)
                .collect(Collectors.joining(" "));
            
            assertNotNull(generated, "Continuous generation should not fail");
            assertFalse(generated.trim().isEmpty(), "Continuous generation should produce output");
            generations.add(generated);
        }
        
        // Check that generations are not identical (some variation)
        Set<String> uniqueGenerations = new HashSet<>(generations);
        double variationRatio = (double) uniqueGenerations.size() / generations.size();
        
        assertTrue(variationRatio >= 0.5, 
            String.format("Insufficient variation in continuous generation: %.2f", variationRatio));
        
        System.out.printf("✓ Continuous generation: %d iterations, %.2f variation ratio%n", 
            generations.size(), variationRatio);
    }
    
    @Test
    @Order(7)
    @DisplayName("INTEGRATION: System Component Interaction")
    void validateSystemIntegration() {
        // Test that all components work together
        String testSequence = "integration test sequence with multiple tokens for processing";
        String[] tokens = testSequence.split("\\s+");
        
        // Process through memory systems
        for (String token : tokens) {
            hierarchicalMemory.addToken(token);
            timescaleBank.update(token);
        }
        
        // Test generation after memory processing
        trainingPipeline.trainFromSamples();
        String generated = patternGenerator.generate("integration test", 30);
        
        // Validate integration
        assertNotNull(generated, "Integrated generation should work");
        assertFalse(generated.trim().isEmpty(), "Integrated generation should produce output");
        
        // Test memory retrieval still works
        List<Object> hierarchicalContext = hierarchicalMemory.getActiveContext(5);
        assertNotNull(hierarchicalContext, "Memory should remain functional");
        
        Set<MultiTimescaleMemoryBank.TimeScale> activeScales = timescaleBank.getActiveScales();
        assertFalse(activeScales.isEmpty(), "Timescale memory should remain active");
        
        System.out.printf("✓ System integration: %d memory items, %d timescales, generation functional%n",
            hierarchicalContext.size(), activeScales.size());
    }
    
    @Test
    @Order(8)
    @DisplayName("ROBUSTNESS: Error Handling and Recovery")
    void validateRobustness() {
        // Test various edge cases
        
        // Empty prompt
        assertDoesNotThrow(() -> {
            String result = patternGenerator.generate("", 10);
            // Should either work or fail gracefully, not crash
        }, "Empty prompt should be handled gracefully");
        
        // Very short generation
        String shortGenerated = patternGenerator.generate("test", 1);
        assertNotNull(shortGenerated, "Short generation should work");
        
        // After training
        trainingPipeline.trainFromSamples();
        
        // Long generation request
        String longGenerated = patternGenerator.generate("long test", 200);
        assertNotNull(longGenerated, "Long generation should work");
        
        // Multiple rapid generations
        for (int i = 0; i < 20; i++) {
            String rapidGenerated = patternGenerator.generate("rapid " + i, 10);
            assertNotNull(rapidGenerated, "Rapid generation " + i + " should work");
        }
        
        System.out.printf("✓ Robustness: handled edge cases and rapid generation%n");
    }
    
    @Test
    @Order(9)
    @DisplayName("MEMORY: Resource Usage")
    void validateMemoryUsage() {
        long startMemory = getUsedMemoryMB();
        
        // Perform memory-intensive operations
        trainingPipeline.trainFromSamples();
        
        // Add many tokens to memory systems
        for (int i = 0; i < 1000; i++) {
            hierarchicalMemory.addToken("token_" + (i % 100));
            timescaleBank.update("token_" + (i % 50));
        }
        
        // Generate multiple texts
        for (int i = 0; i < 50; i++) {
            patternGenerator.generate("memory test " + i, 20);
        }
        
        long endMemory = getUsedMemoryMB();
        long memoryUsed = endMemory - startMemory;
        
        // Memory usage should be reasonable (less than 1GB for this test)
        assertTrue(memoryUsed <= 1000, 
            String.format("Memory usage %d MB excessive", memoryUsed));
        
        System.out.printf("✓ Memory usage: %d MB for intensive operations%n", memoryUsed);
    }
    
    @Test
    @Order(10)
    @DisplayName("FINAL: Complete System Validation")
    void validateCompleteSystem() {
        // Final comprehensive test
        long startTime = System.currentTimeMillis();
        
        // Complete workflow
        trainingPipeline.trainFromSamples();
        
        String prompt = "Final comprehensive validation test";
        String generated = grossbergGenerator.generate(prompt, 75)
            .limit(75)
            .collect(Collectors.joining(" "));
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        
        // Final validations
        assertNotNull(generated, "Final generation should succeed");
        assertTrue(generated.length() > prompt.length(), "Final generation should extend prompt");
        assertTrue(totalTime <= 30000, "Complete workflow should complete in reasonable time");
        
        // Check system is still responsive
        String followUp = patternGenerator.generate("follow up", 20);
        assertNotNull(followUp, "System should remain responsive");
        
        System.out.printf("✓ Complete system validation: %d ms total, system responsive%n", totalTime);
    }
    
    // Helper method
    private long getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
    }
}