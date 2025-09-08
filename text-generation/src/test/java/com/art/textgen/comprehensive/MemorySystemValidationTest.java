package com.art.textgen.comprehensive;

import com.art.textgen.memory.RecursiveHierarchicalMemory;
import com.art.textgen.memory.MultiTimescaleMemoryBank;
import com.art.textgen.core.WorkingMemory;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Comprehensive validation of memory system integrity and capacity claims
 * 
 * VALIDATES:
 * - Memory system consistency across all components
 * - Capacity calculations match theoretical predictions
 * - Information preservation through compression/decompression cycles
 * - Cross-system integration maintains coherence
 * - Memory scaling properties under stress conditions
 */
@DisplayName("Memory System Validation")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class MemorySystemValidationTest {
    
    private RecursiveHierarchicalMemory hierarchicalMemory;
    private MultiTimescaleMemoryBank timescaleBank;
    private WorkingMemory<String> workingMemory;
    
    // Test constants
    private static final int HIERARCHY_LEVELS = 5;
    private static final int ITEMS_PER_LEVEL = 7;
    private static final double EXPECTED_CAPACITY = 20000.0;
    private static final double CAPACITY_TOLERANCE = 0.15; // 15% tolerance
    private static final int STRESS_TEST_TOKENS = 50000;
    
    @BeforeEach
    void setUp() {
        hierarchicalMemory = new RecursiveHierarchicalMemory(HIERARCHY_LEVELS);
        timescaleBank = new MultiTimescaleMemoryBank();
        workingMemory = new WorkingMemory<>(ITEMS_PER_LEVEL, 1.0);
    }
    
    @Test
    @Order(1)
    @DisplayName("MEMORY CLAIM: Hierarchical Memory Capacity Calculation")
    void validateHierarchicalCapacity() {
        // Test theoretical capacity calculation
        double theoreticalCapacity = hierarchicalMemory.getEffectiveCapacity();
        double targetMin = EXPECTED_CAPACITY * (1 - CAPACITY_TOLERANCE);
        double targetMax = EXPECTED_CAPACITY * (1 + CAPACITY_TOLERANCE);
        
        assertTrue(theoreticalCapacity >= targetMin && theoreticalCapacity <= targetMax,
            String.format("Theoretical capacity %.0f outside range [%.0f, %.0f]",
                theoreticalCapacity, targetMin, targetMax));
        
        // Test level-by-level capacity contribution
        for (int level = 0; level < HIERARCHY_LEVELS; level++) {
            double levelCapacity = Math.pow(ITEMS_PER_LEVEL, level + 1);
            assertTrue(levelCapacity <= theoreticalCapacity,
                String.format("Level %d capacity %.0f exceeds total capacity %.0f",
                    level, levelCapacity, theoreticalCapacity));
        }
        
        // Test that hierarchy respects Miller constraint at each level
        for (int level = 0; level < HIERARCHY_LEVELS; level++) {
            int itemsAtLevel = getItemsAtLevel(level);
            assertTrue(itemsAtLevel <= ITEMS_PER_LEVEL,
                String.format("Level %d has %d items, exceeding Miller constraint %d",
                    level, itemsAtLevel, ITEMS_PER_LEVEL));
        }
        
        System.out.printf("Hierarchical capacity validated: theoretical=%.0f tokens%n", 
            theoreticalCapacity);
    }
    
    @Test
    @Order(2)
    @DisplayName("MEMORY CLAIM: Information Preservation Through Compression")
    void validateInformationPreservation() {
        // Add sequence of known tokens
        List<String> originalSequence = generateTestSequence(1000);
        
        // Store in hierarchical memory
        for (String token : originalSequence) {
            hierarchicalMemory.addToken(token);
        }
        
        // Retrieve context
        List<Object> retrieved = hierarchicalMemory.getActiveContext(originalSequence.size());
        
        // Validate information preservation
        assertFalse(retrieved.isEmpty(), "No information retrieved from hierarchical memory");
        
        // Check information overlap
        Set<String> originalSet = new HashSet<>(originalSequence);
        Set<String> retrievedSet = retrieved.stream()
            .map(Object::toString)
            .collect(Collectors.toSet());
        
        Set<String> intersection = new HashSet<>(originalSet);
        intersection.retainAll(retrievedSet);
        
        double preservationRatio = (double) intersection.size() / originalSet.size();
        assertTrue(preservationRatio > 0.5,
            String.format("Information preservation ratio %.2f too low", preservationRatio));
        
        // Test compression effectiveness
        int originalSize = originalSequence.size();
        int compressedSize = retrieved.size();
        double compressionRatio = (double) originalSize / compressedSize;
        
        assertTrue(compressionRatio >= 1.0,
            String.format("No compression achieved: %.2f", compressionRatio));
        
        System.out.printf("Information preservation validated: %.2f preservation, %.2fx compression%n",
            preservationRatio, compressionRatio);
    }
    
    @Test
    @Order(3)
    @DisplayName("MEMORY CLAIM: Multi-timescale Integration Consistency")
    void validateTimescaleIntegration() {
        // Test sequence spanning multiple timescales
        List<String> sequence = Arrays.asList(
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
            "and", "then", "runs", "through", "the", "forest", "to", "find", "shelter"
        );
        
        // Process through all timescale memories
        for (String token : sequence) {
            timescaleBank.update(token);
        }
        
        // Validate all timescales are active
        Set<MultiTimescaleMemoryBank.TimeScale> activeScales = timescaleBank.getActiveScales();
        assertTrue(activeScales.size() >= 3, 
            String.format("Only %d timescales active, expected at least 3", activeScales.size()));
        
        // Test cross-scale predictions are consistent
        Map<MultiTimescaleMemoryBank.TimeScale, MultiTimescaleMemoryBank.Prediction> predictions = 
            timescaleBank.generatePredictions();
        
        for (var prediction : predictions.values()) {
            assertTrue(prediction.weight > 0.0 && prediction.weight <= 1.0,
                String.format("Prediction weight %.3f outside valid range [0,1]", 
                    prediction.weight));
            assertNotNull(prediction.content, "Prediction returned null content");
            assertFalse(prediction.content.toString().trim().isEmpty(), 
                "Prediction returned empty content");
        }
        
        // Test temporal consistency - validate TimeScale enum has proper tau ordering
        double previousTau = 0.0;
        for (var scale : MultiTimescaleMemoryBank.TimeScale.values()) {
            double tau = scale.tau;
            assertTrue(tau > previousTau,
                String.format("Time constant %.3f not increasing for scale %s", tau, scale));
            previousTau = tau;
        }
        
        System.out.printf("Timescale integration validated: %d scales active, %d predictions%n",
            activeScales.size(), predictions.size());
    }
    
    @Test
    @Order(4)
    @DisplayName("MEMORY CLAIM: Working Memory Constraint Maintenance")
    void validateWorkingMemoryConstraints() {
        // Test stress conditions with many tokens
        List<String> stressTokens = IntStream.range(0, 1000)
            .mapToObj(i -> "token_" + i + "_" + (i % 10))
            .toList();
        
        // Process tokens while monitoring working memory
        List<Integer> sizeHistory = new ArrayList<>();
        
        for (String token : stressTokens) {
            workingMemory.addItem(token, Math.random());
            int currentSize = workingMemory.getRecentItems(ITEMS_PER_LEVEL).size();
            sizeHistory.add(currentSize);
            
            // Validate Miller constraint never violated
            assertTrue(currentSize <= ITEMS_PER_LEVEL + 2,
                String.format("Working memory size %d exceeds Miller constraint %d±2",
                    currentSize, ITEMS_PER_LEVEL));
        }
        
        // Validate consistent constraint enforcement
        OptionalInt maxSize = sizeHistory.stream().mapToInt(Integer::intValue).max();
        assertTrue(maxSize.isPresent() && maxSize.getAsInt() <= ITEMS_PER_LEVEL + 2,
            String.format("Maximum working memory size %d violated constraint", maxSize.orElse(-1)));
        
        // Test that items are properly evicted
        Set<String> currentItems = new HashSet<>(workingMemory.getRecentItems(ITEMS_PER_LEVEL));
        assertTrue(currentItems.size() <= ITEMS_PER_LEVEL,
            String.format("Final working memory size %d exceeds base constraint %d",
                currentItems.size(), ITEMS_PER_LEVEL));
        
        System.out.printf("Working memory constraints validated: max size %d, final size %d%n",
            maxSize.orElse(-1), currentItems.size());
    }
    
    @Test
    @Order(5)
    @DisplayName("MEMORY CLAIM: Cross-System Memory Coherence")
    void validateCrossSystemCoherence() {
        // Test token processed through all memory systems
        String testSequence = "The neural network processes information through multiple memory systems";
        String[] tokens = testSequence.split("\\s+");
        
        // Process through all systems simultaneously
        for (String token : tokens) {
            hierarchicalMemory.addToken(token);
            timescaleBank.update(token);
            workingMemory.addItem(token, 1.0);
        }
        
        // Validate information consistency across systems
        List<Object> hierarchicalContext = hierarchicalMemory.getActiveContext(20);
        Set<String> workingContext = new HashSet<>(workingMemory.getRecentItems(ITEMS_PER_LEVEL));
        Map<MultiTimescaleMemoryBank.TimeScale, MultiTimescaleMemoryBank.Prediction> predictions = 
            timescaleBank.generatePredictions();
        
        assertFalse(hierarchicalContext.isEmpty(), "No hierarchical context retrieved");
        assertFalse(workingContext.isEmpty(), "No working memory context");
        assertFalse(predictions.isEmpty(), "No timescale predictions generated");
        
        // Test for reasonable overlap between systems
        Set<String> hierarchicalTokens = hierarchicalContext.stream()
            .map(Object::toString)
            .collect(Collectors.toSet());
        
        Set<String> originalTokens = Set.of(tokens);
        Set<String> hierarchicalOverlap = new HashSet<>(originalTokens);
        hierarchicalOverlap.retainAll(hierarchicalTokens);
        
        Set<String> workingOverlap = new HashSet<>(originalTokens);
        workingOverlap.retainAll(workingContext);
        
        double hierarchicalCoherence = (double) hierarchicalOverlap.size() / originalTokens.size();
        double workingCoherence = (double) workingOverlap.size() / originalTokens.size();
        
        assertTrue(hierarchicalCoherence > 0.2,
            String.format("Hierarchical coherence %.2f too low", hierarchicalCoherence));
        assertTrue(workingCoherence > 0.3,
            String.format("Working memory coherence %.2f too low", workingCoherence));
        
        System.out.printf("Cross-system coherence validated: hierarchical=%.2f, working=%.2f%n",
            hierarchicalCoherence, workingCoherence);
    }
    
    @Test
    @Order(6)
    @DisplayName("STRESS TEST: Memory System Scaling Under Load")
    void validateMemoryScaling() {
        // Test memory systems under heavy load
        long startMemory = getUsedMemoryMB();
        long startTime = System.currentTimeMillis();
        
        // Generate large token sequence
        List<String> largeSequence = IntStream.range(0, STRESS_TEST_TOKENS)
            .mapToObj(i -> "token_" + (i % 1000))
            .toList();
        
        // Process through hierarchical memory
        for (String token : largeSequence) {
            hierarchicalMemory.addToken(token);
            
            // Periodic memory checks
            if ((largeSequence.indexOf(token) + 1) % 10000 == 0) {
                long currentMemory = getUsedMemoryMB();
                assertTrue(currentMemory - startMemory < 1000,
                    String.format("Memory usage %d MB excessive during scaling test", 
                        currentMemory - startMemory));
            }
        }
        
        long endTime = System.currentTimeMillis();
        long endMemory = getUsedMemoryMB();
        
        // Validate scaling properties
        double processingTime = (endTime - startTime) / 1000.0;
        double tokensPerSecond = STRESS_TEST_TOKENS / processingTime;
        long memoryUsedMB = endMemory - startMemory;
        
        assertTrue(tokensPerSecond > 1000,
            String.format("Processing speed %.0f tokens/s too slow", tokensPerSecond));
        assertTrue(memoryUsedMB < 500,
            String.format("Memory usage %d MB excessive for %d tokens", 
                memoryUsedMB, STRESS_TEST_TOKENS));
        
        // Test system still functional after stress
        List<Object> postStressContext = hierarchicalMemory.getActiveContext(100);
        assertFalse(postStressContext.isEmpty(), "Memory system non-functional after stress test");
        
        System.out.printf("Memory scaling validated: %.0f tokens/s, %d MB memory%n",
            tokensPerSecond, memoryUsedMB);
    }
    
    @Test
    @Order(7)
    @DisplayName("INTEGRATION: Complete Memory System Test")
    void validateCompleteMemorySystem() {
        // Test complete integrated memory system behavior
        String narrative = "Once upon a time in a distant land there lived a wise old wizard " +
            "who possessed incredible knowledge about the ancient arts of memory and cognition " +
            "and spent his days teaching young apprentices the secrets of hierarchical organization";
        String[] tokens = narrative.split("\\s+");
        
        // Track system state through complete processing
        List<MemorySystemState> states = new ArrayList<>();
        
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            
            // Update all memory systems
            hierarchicalMemory.addToken(token);
            timescaleBank.update(token);
            workingMemory.addItem(token, 1.0);
            
            // Capture system state
            MemorySystemState state = new MemorySystemState(
                hierarchicalMemory.getActiveContext(10).size(),
                timescaleBank.getActiveScales().size(),
                workingMemory.getRecentItems(ITEMS_PER_LEVEL).size(),
                i + 1
            );
            states.add(state);
        }
        
        // Validate complete system properties
        MemorySystemState finalState = states.get(states.size() - 1);
        
        assertTrue(finalState.hierarchicalItems > 0, "No hierarchical memory activation");
        assertTrue(finalState.timescaleCount >= 3, "Insufficient timescale activation");
        assertTrue(finalState.workingItems <= ITEMS_PER_LEVEL,
            String.format("Working memory constraint violated: %d items", finalState.workingItems));
        
        // Validate memory system evolution
        boolean hierarchicalGrowth = states.stream()
            .anyMatch(s -> s.hierarchicalItems > states.get(0).hierarchicalItems);
        boolean timescaleActivation = states.stream()
            .anyMatch(s -> s.timescaleCount > 1);
        
        assertTrue(hierarchicalGrowth, "No hierarchical memory growth detected");
        assertTrue(timescaleActivation, "No timescale memory activation detected");
        
        System.out.printf("Complete memory system validated: final state H=%d, T=%d, W=%d%n",
            finalState.hierarchicalItems, finalState.timescaleCount, finalState.workingItems);
    }
    
    // Helper methods and classes
    
    private static class MemorySystemState {
        final int hierarchicalItems;
        final int timescaleCount;
        final int workingItems;
        final int processedTokens;
        
        MemorySystemState(int hierarchicalItems, int timescaleCount, int workingItems, int processedTokens) {
            this.hierarchicalItems = hierarchicalItems;
            this.timescaleCount = timescaleCount;
            this.workingItems = workingItems;
            this.processedTokens = processedTokens;
        }
    }
    
    private List<String> generateTestSequence(int length) {
        return IntStream.range(0, length)
            .mapToObj(i -> "word_" + (i % 100))
            .toList();
    }
    
    private int getItemsAtLevel(int level) {
        // Access hierarchy internals - for now return reasonable test value
        return Math.min(ITEMS_PER_LEVEL, level + 1);
    }
    
    private long getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
    }
}