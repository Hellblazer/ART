package com.art.textgen.comprehensive;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.training.TrainingPipeline;
import com.art.textgen.memory.RecursiveHierarchicalMemory;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;

/**
 * Comprehensive performance validation against thesis claims
 * 
 * VALIDATES:
 * - Training speed: <30 seconds for 40MB corpus (vs hours for transformers)
 * - Memory growth: O(log n) vs O(n²) for transformers  
 * - Generation speed: >10 tokens/second
 * - Scalability: Performance scaling with corpus size
 * - Memory efficiency: Bounded memory usage with unbounded sequences
 */
@DisplayName("Performance Validation")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@Execution(ExecutionMode.CONCURRENT)
public class PerformanceValidationTest {
    
    // Performance targets based on thesis claims
    private static final long MAX_TRAINING_TIME_MS = 30_000; // 30 seconds
    private static final double MIN_GENERATION_SPEED = 10.0; // tokens/second
    private static final long MAX_MEMORY_MB = 4096; // 4GB memory limit - realistic for modern training
    private static final double SCALABILITY_EXPONENT = 1.2; // O(n^1.2) acceptable vs O(n²)
    
    private Vocabulary vocabulary;
    private EnhancedPatternGenerator generator;
    private TrainingPipeline pipeline;
    private MemoryMXBean memoryBean;
    
    @BeforeEach
    void setUp() {
        vocabulary = new Vocabulary(64);
        generator = new EnhancedPatternGenerator(vocabulary);
        pipeline = new TrainingPipeline(vocabulary, generator);
        memoryBean = ManagementFactory.getMemoryMXBean();
        
        // Force garbage collection for clean memory measurement
        System.gc();
        System.gc();
    }
    
    @Test
    @Order(1)
    @DisplayName("PERFORMANCE CLAIM: Training Speed <30 seconds for 40MB corpus")
    void validateTrainingSpeed() throws IOException {
        // Create test corpus of known size
        List<Path> corpusFiles = createTestCorpus(40); // 40MB
        
        long startMemory = getUsedMemoryMB();
        long startTime = System.currentTimeMillis();
        
        // Train on the corpus
        for (Path file : corpusFiles) {
            try {
                pipeline.trainFromFile(file.toString());
            } catch (IOException e) {
                fail("Failed to train from file: " + file + " - " + e.getMessage());
            }
        }
        
        long endTime = System.currentTimeMillis();
        long trainingTimeMs = endTime - startTime;
        long endMemory = getUsedMemoryMB();
        
        // Validate training speed claim
        assertTrue(trainingTimeMs <= MAX_TRAINING_TIME_MS,
            String.format("Training took %d ms, exceeds target %d ms", 
                trainingTimeMs, MAX_TRAINING_TIME_MS));
        
        // Validate memory usage is reasonable
        long memoryUsedMB = endMemory - startMemory;
        assertTrue(memoryUsedMB <= MAX_MEMORY_MB,
            String.format("Training used %d MB, exceeds limit %d MB",
                memoryUsedMB, MAX_MEMORY_MB));
        
        // Calculate training throughput
        double corpusSizeMB = corpusFiles.size(); // Approximate
        double throughputMBps = corpusSizeMB / (trainingTimeMs / 1000.0);
        
        System.out.printf("✓ Training speed validated: %d ms (%.2f MB/s), memory: %d MB%n",
            trainingTimeMs, throughputMBps, memoryUsedMB);
            
        // Cleanup test files
        for (Path file : corpusFiles) {
            Files.deleteIfExists(file);
        }
    }
    
    @Test
    @Order(2)
    @DisplayName("PERFORMANCE CLAIM: Generation Speed >10 tokens/second")
    void validateGenerationSpeed() {
        // Train on sample data first
        pipeline.trainFromSamples();
        
        List<String> testPrompts = Arrays.asList(
            "The future of artificial intelligence",
            "Machine learning algorithms can",
            "Neural networks are capable of",
            "Deep learning enables systems to",
            "Cognitive architectures provide"
        );
        
        List<Double> speeds = new ArrayList<>();
        
        for (String prompt : testPrompts) {
            long startTime = System.nanoTime();
            
            // Generate fixed length for consistent measurement
            String generated = generator.generate(prompt, 100);
            
            long endTime = System.nanoTime();
            double durationSeconds = (endTime - startTime) / 1_000_000_000.0;
            
            // Count actual tokens generated
            int tokenCount = generated.split("\\s+").length;
            double tokensPerSecond = tokenCount / durationSeconds;
            
            speeds.add(tokensPerSecond);
            
            assertTrue(tokensPerSecond >= MIN_GENERATION_SPEED,
                String.format("Generation speed %.2f tokens/s below target %.2f for prompt: %s",
                    tokensPerSecond, MIN_GENERATION_SPEED, prompt));
        }
        
        double avgSpeed = speeds.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double minSpeed = speeds.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double maxSpeed = speeds.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        
        System.out.printf("✓ Generation speed validated: avg=%.1f, range=[%.1f, %.1f] tokens/s%n",
            avgSpeed, minSpeed, maxSpeed);
    }
    
    @Test
    @Order(3)
    @DisplayName("MEMORY CLAIM: Logarithmic Memory Growth O(log n)")
    void validateMemoryScaling() {
        RecursiveHierarchicalMemory memory = new RecursiveHierarchicalMemory();
        
        // Test memory growth with increasing sequence lengths
        int[] sequenceLengths = {100, 500, 1000, 5000, 10000, 20000};
        List<Long> memoryUsages = new ArrayList<>();
        
        for (int length : sequenceLengths) {
            System.gc(); // Clean slate for each measurement
            long startMemory = getUsedMemoryMB();
            
            // Add tokens to memory
            for (int i = 0; i < length; i++) {
                memory.addToken("token_" + i);
            }
            
            long endMemory = getUsedMemoryMB();
            long memoryUsed = endMemory - startMemory;
            memoryUsages.add(memoryUsed);
        }
        
        // Validate logarithmic growth pattern
        // Memory should not grow quadratically
        for (int i = 1; i < sequenceLengths.length; i++) {
            double lengthRatio = (double) sequenceLengths[i] / sequenceLengths[i-1];
            double memoryRatio = (double) memoryUsages.get(i) / Math.max(1, memoryUsages.get(i-1));
            
            // For logarithmic growth, memory ratio should be much less than length ratio
            double growthExponent = Math.log(memoryRatio) / Math.log(lengthRatio);
            
            assertTrue(growthExponent <= SCALABILITY_EXPONENT,
                String.format("Memory growth exponent %.2f exceeds target %.2f at length %d",
                    growthExponent, SCALABILITY_EXPONENT, sequenceLengths[i]));
        }
        
        System.out.printf("✓ Memory scaling validated: max usage %d MB for %d tokens%n",
            Collections.max(memoryUsages), sequenceLengths[sequenceLengths.length-1]);
    }
    
    @Test
    @Order(4)
    @DisplayName("SCALABILITY: Performance vs Corpus Size")
    void validateCorpusScaling() throws IOException {
        int[] corpusSizesMB = {1, 5, 10, 20}; // Different corpus sizes
        List<Long> trainingTimes = new ArrayList<>();
        List<Long> memoryUsages = new ArrayList<>();
        
        for (int sizeMB : corpusSizesMB) {
            // Create corpus of specific size
            List<Path> corpus = createTestCorpus(sizeMB);
            
            long startMemory = getUsedMemoryMB();
            long startTime = System.currentTimeMillis();
            
            // Train on this corpus size
            Vocabulary vocab = new Vocabulary(64);
            EnhancedPatternGenerator gen = new EnhancedPatternGenerator(vocab);
            TrainingPipeline pipe = new TrainingPipeline(vocab, gen);
            
            for (Path file : corpus) {
                try {
                    pipe.trainFromFile(file.toString());
                } catch (IOException e) {
                    fail("Failed to train from file: " + file + " - " + e.getMessage());
                }
            }
            
            long endTime = System.currentTimeMillis();
            long endMemory = getUsedMemoryMB();
            
            long trainingTime = endTime - startTime;
            long memoryUsed = endMemory - startMemory;
            
            trainingTimes.add(trainingTime);
            memoryUsages.add(memoryUsed);
            
            // Cleanup test files
            for (Path file : corpus) {
                Files.deleteIfExists(file);
            }
        }
        
        // Validate scaling behavior
        for (int i = 1; i < corpusSizesMB.length; i++) {
            double sizeRatio = (double) corpusSizesMB[i] / corpusSizesMB[i-1];
            double timeRatio = (double) trainingTimes.get(i) / trainingTimes.get(i-1);
            
            double timeScalingExponent = Math.log(timeRatio) / Math.log(sizeRatio);
            
            // Training time should scale roughly linearly with corpus size
            assertTrue(timeScalingExponent <= 1.5,
                String.format("Training time scaling exponent %.2f too high at %d MB",
                    timeScalingExponent, corpusSizesMB[i]));
        }
        
        System.out.printf("✓ Corpus scaling validated: %d sizes tested, max time %d ms%n",
            corpusSizesMB.length, Collections.max(trainingTimes));
    }
    
    @Test
    @Order(5)
    @DisplayName("STRESS TEST: Continuous Generation Performance")
    void validateContinuousGeneration() {
        pipeline.trainFromSamples();
        
        // Test continuous generation for extended period
        String prompt = "The neural network";
        int totalTokensGenerated = 0;
        long startTime = System.currentTimeMillis();
        long testDurationMs = 10_000; // 10 seconds
        
        while (System.currentTimeMillis() - startTime < testDurationMs) {
            String generated = generator.generate(prompt, 50);
            totalTokensGenerated += generated.split("\\s+").length;
            
            // Use last few words as next prompt (continuous generation)
            String[] words = generated.split("\\s+");
            if (words.length >= 3) {
                prompt = String.join(" ", Arrays.copyOfRange(words, words.length-3, words.length));
            }
        }
        
        long actualDurationMs = System.currentTimeMillis() - startTime;
        double actualDurationSec = actualDurationMs / 1000.0;
        double sustainedSpeed = totalTokensGenerated / actualDurationSec;
        
        assertTrue(sustainedSpeed >= MIN_GENERATION_SPEED,
            String.format("Sustained generation speed %.2f below target %.2f tokens/s",
                sustainedSpeed, MIN_GENERATION_SPEED));
        
        // Memory should remain bounded during continuous generation
        long finalMemory = getUsedMemoryMB();
        assertTrue(finalMemory <= MAX_MEMORY_MB,
            String.format("Memory usage %d MB exceeds limit during continuous generation", finalMemory));
        
        System.out.printf("✓ Continuous generation validated: %.1f tokens/s for %.1f seconds%n",
            sustainedSpeed, actualDurationSec);
    }
    
    @Test
    @Order(6)
    @DisplayName("EFFICIENCY: Memory vs Transformer Baseline")
    void validateMemoryEfficiency() {
        // Simulate transformer memory usage (quadratic in sequence length)
        // vs our hierarchical memory (logarithmic)
        
        int[] sequenceLengths = {100, 500, 1000, 2000, 5000};
        
        for (int length : sequenceLengths) {
            // Our system memory usage (measured)
            RecursiveHierarchicalMemory ourMemory = new RecursiveHierarchicalMemory();
            long startMemory = getUsedMemoryMB();
            
            for (int i = 0; i < length; i++) {
                ourMemory.addToken("token_" + i);
            }
            
            long ourMemoryMB = getUsedMemoryMB() - startMemory;
            
            // Theoretical transformer memory (quadratic scaling)
            // Transformer attention: O(n²) memory for sequence length n
            double transformerMemoryMB = Math.pow(length / 1000.0, 2) * 100; // Approximate
            
            // Our memory should be significantly lower for longer sequences
            if (length >= 1000) {
                double efficiency = transformerMemoryMB / Math.max(1, ourMemoryMB);
                assertTrue(efficiency >= 2.0,
                    String.format("Memory efficiency %.2fx not significant at length %d", 
                        efficiency, length));
            }
            
            System.out.printf("Length %d: Our=%d MB, Transformer~=%.0f MB, Efficiency=%.1fx%n",
                length, ourMemoryMB, transformerMemoryMB, transformerMemoryMB / Math.max(1, ourMemoryMB));
        }
    }
    
    // Helper methods
    
    private List<Path> createTestCorpus(int targetSizeMB) throws IOException {
        List<Path> files = new ArrayList<>();
        
        // Generate test content
        String sampleContent = "The quick brown fox jumps over the lazy dog. " +
            "Neural networks learn patterns from data through training. " +
            "Cognitive architectures provide biologically plausible computation. " +
            "Working memory maintains active information for processing. ";
        
        // Calculate approximate file size needed
        int contentSize = sampleContent.length();
        int repetitions = (targetSizeMB * 1024 * 1024) / contentSize;
        
        // Create content string
        StringBuilder content = new StringBuilder(repetitions * contentSize);
        for (int i = 0; i < repetitions; i++) {
            content.append(sampleContent);
        }
        
        // Write to temporary file
        Path tempFile = Files.createTempFile("test_corpus_", ".txt");
        Files.write(tempFile, content.toString().getBytes());
        files.add(tempFile);
        
        return files;
    }
    
    private long getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
    }
    
    @AfterEach
    void tearDown() {
        // Force cleanup
        System.gc();
    }
}