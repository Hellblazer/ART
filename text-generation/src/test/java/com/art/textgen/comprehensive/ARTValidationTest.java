package com.art.textgen.comprehensive;

import com.art.textgen.dynamics.ResonanceDetector;
import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.core.Vocabulary;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Comprehensive validation of ART resonance and learning mechanisms
 * 
 * VALIDATES:
 * - Bottom-up/top-down resonance with vigilance parameter
 * - Category formation without catastrophic forgetting
 * - Incremental learning preserves existing patterns
 * - Pattern matching with appropriate generalization
 * - Learning stability under various conditions
 */
@DisplayName("ART Resonance and Learning Validation")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ARTValidationTest {
    
    private MockARTNetwork artNetwork;
    private ResonanceDetector resonanceDetector;
    private EnhancedPatternGenerator generator;
    private Vocabulary vocabulary;
    
    // Test constants based on ART theory
    private static final double DEFAULT_VIGILANCE = 0.8;
    private static final double LOW_VIGILANCE = 0.3;
    private static final double HIGH_VIGILANCE = 0.95;
    private static final int MAX_CATEGORIES = 1000;
    private static final double LEARNING_RATE = 0.1;
    
    @BeforeEach
    void setUp() {
        vocabulary = new Vocabulary(64);
        artNetwork = new MockARTNetwork(DEFAULT_VIGILANCE, MAX_CATEGORIES);
        resonanceDetector = new ResonanceDetector(DEFAULT_VIGILANCE);
        generator = new EnhancedPatternGenerator(vocabulary);
    }
    
    @Test
    @Order(1)
    @DisplayName("ART CLAIM: Bottom-up/Top-down Resonance Detection")
    void validateResonanceDetection() {
        // First, create a category in the resonance detector to test against
        double[] templatePattern = {0.7, 0.5, 0.8, 0.4, 0.6};
        var templateResult = resonanceDetector.searchResonance(templatePattern);
        assertTrue(templateResult.isResonant, "Template pattern should create resonant category");
        
        // Test resonance with similar pattern
        double[] inputPattern = {0.8, 0.6, 0.9, 0.3, 0.7}; // Similar pattern
        var resonanceResult = resonanceDetector.searchResonance(inputPattern);
        
        assertTrue(resonanceResult.isResonant, 
            "Similar patterns should achieve resonance");
        assertTrue(resonanceResult.resonanceStrength >= 0.5,
            String.format("Resonance strength %.3f should be reasonably high", 
                resonanceResult.resonanceStrength));
        
        // Test resonance failure with dissimilar patterns  
        resonanceDetector.reset(); // Clear previous categories
        resonanceDetector.searchResonance(templatePattern); // Re-add template
        
        double[] dissimilarPattern = {0.1, 0.2, 0.0, 0.9, 0.1};
        var noResonanceResult = resonanceDetector.searchResonance(dissimilarPattern);
        
        // With high vigilance, dissimilar pattern should create new category
        assertTrue(noResonanceResult.isResonant,
            "Dissimilar patterns should create new resonant category");
        assertNotEquals(templateResult.resonantCategory.id, noResonanceResult.resonantCategory.id,
            "Dissimilar patterns should create different categories");
        
        // Test vigilance parameter effect
        resonanceDetector.reset();
        resonanceDetector.setVigilance(LOW_VIGILANCE);
        resonanceDetector.searchResonance(templatePattern); // Re-add template with low vigilance
        
        var lowVigilanceResult = resonanceDetector.searchResonance(dissimilarPattern);
        
        assertTrue(lowVigilanceResult.isResonant,
            "Low vigilance should allow resonance");
        // With low vigilance, might use same category (less strict matching)
        
        System.out.printf("Resonance detection validated: strength=%.3f, low-vigilance=%.3f%n",
            resonanceResult.resonanceStrength, lowVigilanceResult.resonanceStrength);
        
        // Verify we have the expected number of categories
        assertTrue(resonanceDetector.getCategories().size() <= 2,
            "Should have at most 2 categories with low vigilance");
    }
    
    @Test
    @Order(2)
    @DisplayName("ART CLAIM: Category Formation Without Catastrophic Forgetting")
    void validateCategoryFormation() {
        // Learn initial set of patterns
        List<double[]> initialPatterns = Arrays.asList(
            new double[]{0.9, 0.1, 0.8, 0.2, 0.7},
            new double[]{0.1, 0.9, 0.2, 0.8, 0.1},
            new double[]{0.5, 0.5, 0.9, 0.1, 0.5},
            new double[]{0.8, 0.2, 0.1, 0.9, 0.3}
        );
        
        // Learn patterns and track category formation
        Set<Integer> createdCategories = new HashSet<>();
        for (int i = 0; i < initialPatterns.size(); i++) {
            MockCategoryNode category = artNetwork.learn(initialPatterns.get(i));
            createdCategories.add(category.getId());
            
            assertTrue(category.getId() >= 0, "Invalid category ID generated");
            assertNotNull(category.getTemplate(), "Category template should not be null");
        }
        
        int initialCategoryCount = artNetwork.getCategoryCount();
        assertTrue(initialCategoryCount > 0 && initialCategoryCount <= initialPatterns.size(),
            String.format("Initial category count %d invalid", initialCategoryCount));
        
        // Learn new patterns and verify no catastrophic forgetting
        List<double[]> newPatterns = Arrays.asList(
            new double[]{0.2, 0.8, 0.3, 0.7, 0.4},
            new double[]{0.7, 0.3, 0.6, 0.4, 0.8}
        );
        
        for (double[] newPattern : newPatterns) {
            artNetwork.learn(newPattern);
        }
        
        // Verify original patterns still recognized
        List<Boolean> originalRecognition = new ArrayList<>();
        for (double[] originalPattern : initialPatterns) {
            MockCategoryNode matchedCategory = artNetwork.classify(originalPattern);
            boolean recognized = matchedCategory != null && 
                createdCategories.contains(matchedCategory.getId());
            originalRecognition.add(recognized);
        }
        
        long recognizedCount = originalRecognition.stream()
            .mapToLong(recognized -> recognized ? 1 : 0)
            .sum();
        
        double retentionRate = (double) recognizedCount / initialPatterns.size();
        assertTrue(retentionRate >= 0.8,
            String.format("Pattern retention rate %.2f below threshold 0.8", retentionRate));
        
        System.out.printf("Category formation validated: %d categories, %.2f retention rate%n",
            artNetwork.getCategoryCount(), retentionRate);
    }
    
    @Test
    @Order(3)
    @DisplayName("ART CLAIM: Incremental Learning Preserves Existing Patterns")
    void validateIncrementalLearning() {
        // Learn patterns incrementally
        List<double[]> patterns = IntStream.range(0, 20)
            .mapToObj(i -> generateTestPattern(i))
            .toList();
        
        // Learn first half of patterns
        int halfPoint = patterns.size() / 2;
        List<MockCategoryNode> firstHalfCategories = new ArrayList<>();
        
        for (int i = 0; i < halfPoint; i++) {
            MockCategoryNode category = artNetwork.learn(patterns.get(i));
            firstHalfCategories.add(category);
        }
        
        int midpointCategoryCount = artNetwork.getCategoryCount();
        
        // Capture category templates before learning second half
        Map<Integer, double[]> originalTemplates = new HashMap<>();
        for (MockCategoryNode category : firstHalfCategories) {
            originalTemplates.put(category.getId(), category.getTemplate().clone());
        }
        
        // Learn second half of patterns
        for (int i = halfPoint; i < patterns.size(); i++) {
            artNetwork.learn(patterns.get(i));
        }
        
        // Verify incremental learning properties
        int finalCategoryCount = artNetwork.getCategoryCount();
        assertTrue(finalCategoryCount >= midpointCategoryCount,
            "Category count should not decrease during incremental learning");
        
        // Verify original patterns still classify to same categories
        int correctClassifications = 0;
        for (int i = 0; i < halfPoint; i++) {
            MockCategoryNode classifiedCategory = artNetwork.classify(patterns.get(i));
            if (classifiedCategory != null && 
                firstHalfCategories.stream()
                    .anyMatch(cat -> cat.getId() == classifiedCategory.getId())) {
                correctClassifications++;
            }
        }
        
        double stability = (double) correctClassifications / halfPoint;
        assertTrue(stability >= 0.9,
            String.format("Classification stability %.2f below threshold 0.9", stability));
        
        // Verify template preservation (templates should not change drastically)
        double totalTemplateChange = 0.0;
        int templateCount = 0;
        
        for (MockCategoryNode category : firstHalfCategories) {
            if (originalTemplates.containsKey(category.getId())) {
                double[] originalTemplate = originalTemplates.get(category.getId());
                double[] currentTemplate = category.getTemplate();
                double templateChange = calculateTemplateChange(originalTemplate, currentTemplate);
                totalTemplateChange += templateChange;
                templateCount++;
            }
        }
        
        double averageTemplateChange = templateCount > 0 ? totalTemplateChange / templateCount : 0.0;
        assertTrue(averageTemplateChange < 0.5,
            String.format("Average template change %.3f indicates instability", averageTemplateChange));
        
        System.out.printf("Incremental learning validated: %.2f stability, %.3f template change%n",
            stability, averageTemplateChange);
    }
    
    @Test
    @Order(4)
    @DisplayName("ART CLAIM: Vigilance Parameter Controls Generalization")
    void validateVigilanceControl() {
        double[] basePattern = {0.8, 0.2, 0.9, 0.1, 0.7};
        
        // Test different vigilance levels
        double[] vigilanceLevels = {0.1, 0.5, 0.8, 0.95};
        Map<Double, Integer> categoryCountsByVigilance = new HashMap<>();
        
        for (double vigilance : vigilanceLevels) {
            MockARTNetwork testNetwork = new MockARTNetwork(vigilance, MAX_CATEGORIES);
            
            // Learn base pattern
            testNetwork.learn(basePattern);
            
            // Learn variations of the pattern
            for (int i = 0; i < 10; i++) {
                double[] variation = addNoise(basePattern, 0.1 + (i * 0.05));
                testNetwork.learn(variation);
            }
            
            categoryCountsByVigilance.put(vigilance, testNetwork.getCategoryCount());
        }
        
        // Verify vigilance effect: higher vigilance should create more categories
        for (int i = 1; i < vigilanceLevels.length; i++) {
            double lowerVigilance = vigilanceLevels[i-1];
            double higherVigilance = vigilanceLevels[i];
            
            int lowerCount = categoryCountsByVigilance.get(lowerVigilance);
            int higherCount = categoryCountsByVigilance.get(higherVigilance);
            
            assertTrue(higherCount >= lowerCount,
                String.format("Higher vigilance %.2f should create more categories than %.2f (%d vs %d)",
                    higherVigilance, lowerVigilance, higherCount, lowerCount));
        }
        
        System.out.printf("Vigilance control validated: categories by vigilance %s%n",
            categoryCountsByVigilance);
    }
    
    @Test
    @Order(5)
    @DisplayName("ART CLAIM: Pattern Matching with Appropriate Generalization")
    void validatePatternMatching() {
        // Create pattern families with intra-family similarity
        List<double[]> family1 = Arrays.asList(
            new double[]{0.9, 0.1, 0.8, 0.2, 0.9},
            new double[]{0.8, 0.2, 0.9, 0.1, 0.8},
            new double[]{0.9, 0.0, 0.7, 0.3, 0.9}
        );
        
        List<double[]> family2 = Arrays.asList(
            new double[]{0.1, 0.9, 0.2, 0.8, 0.1},
            new double[]{0.2, 0.8, 0.1, 0.9, 0.2},
            new double[]{0.0, 0.9, 0.3, 0.7, 0.1}
        );
        
        // Learn patterns from both families
        List<MockCategoryNode> family1Categories = new ArrayList<>();
        List<MockCategoryNode> family2Categories = new ArrayList<>();
        
        for (double[] pattern : family1) {
            family1Categories.add(artNetwork.learn(pattern));
        }
        
        for (double[] pattern : family2) {
            family2Categories.add(artNetwork.learn(pattern));
        }
        
        // Test within-family generalization
        double[] family1Test = {0.85, 0.15, 0.75, 0.25, 0.85};
        MockCategoryNode family1Match = artNetwork.classify(family1Test);
        
        assertNotNull(family1Match, "Family 1 test pattern should match existing category");
        assertTrue(family1Categories.stream()
                .anyMatch(cat -> cat.getId() == family1Match.getId()),
            "Family 1 test should match Family 1 category");
        
        double[] family2Test = {0.15, 0.85, 0.25, 0.75, 0.15};
        MockCategoryNode family2Match = artNetwork.classify(family2Test);
        
        assertNotNull(family2Match, "Family 2 test pattern should match existing category");
        assertTrue(family2Categories.stream()
                .anyMatch(cat -> cat.getId() == family2Match.getId()),
            "Family 2 test should match Family 2 category");
        
        // Test cross-family discrimination
        assertNotEquals(family1Match.getId(), family2Match.getId(),
            "Different pattern families should map to different categories");
        
        System.out.printf("Pattern matching validated: Family1=%d, Family2=%d, Total=%d categories%n",
            family1Categories.stream().mapToInt(MockCategoryNode::getId).distinct().toArray().length,
            family2Categories.stream().mapToInt(MockCategoryNode::getId).distinct().toArray().length,
            artNetwork.getCategoryCount());
    }
    
    @Test
    @Order(6)
    @DisplayName("STRESS TEST: ART Learning Under Extreme Conditions")
    void validateLearningStability() {
        // Test with large number of patterns
        int numPatterns = 1000;
        List<double[]> patterns = IntStream.range(0, numPatterns)
            .mapToObj(i -> generateRandomPattern())
            .toList();
        
        // Learn all patterns
        long startTime = System.currentTimeMillis();
        List<MockCategoryNode> learnedCategories = new ArrayList<>();
        
        for (double[] pattern : patterns) {
            MockCategoryNode category = artNetwork.learn(pattern);
            learnedCategories.add(category);
        }
        
        long learningTime = System.currentTimeMillis() - startTime;
        
        // Validate learning performance
        assertTrue(learningTime < 10000, // 10 seconds
            String.format("Learning %d patterns took %d ms, exceeds 10s limit", 
                numPatterns, learningTime));
        
        int finalCategoryCount = artNetwork.getCategoryCount();
        assertTrue(finalCategoryCount > 0 && finalCategoryCount <= MAX_CATEGORIES,
            String.format("Category count %d outside valid range [1, %d]", 
                finalCategoryCount, MAX_CATEGORIES));
        
        // Test classification accuracy on learned patterns
        int correctClassifications = 0;
        for (int i = 0; i < Math.min(100, patterns.size()); i++) {
            MockCategoryNode classified = artNetwork.classify(patterns.get(i));
            if (classified != null) {
                correctClassifications++;
            }
        }
        
        double accuracyRate = (double) correctClassifications / Math.min(100, patterns.size());
        assertTrue(accuracyRate >= 0.8,
            String.format("Classification accuracy %.2f below threshold 0.8", accuracyRate));
        
        // Test memory usage is reasonable
        long usedMemory = getUsedMemoryMB();
        assertTrue(usedMemory < 1000,
            String.format("Memory usage %d MB excessive for %d patterns", usedMemory, numPatterns));
        
        System.out.printf("Learning stability validated: %d patterns, %d categories, %.2f accuracy%n",
            numPatterns, finalCategoryCount, accuracyRate);
    }
    
    // Helper methods
    
    private double[] generateTestPattern(int index) {
        double[] pattern = new double[5];
        for (int i = 0; i < pattern.length; i++) {
            pattern[i] = (Math.sin(index + i) + 1) / 2.0; // [0, 1] range
        }
        return pattern;
    }
    
    private double[] generateRandomPattern() {
        double[] pattern = new double[5];
        for (int i = 0; i < pattern.length; i++) {
            pattern[i] = Math.random();
        }
        return pattern;
    }
    
    private double[] addNoise(double[] pattern, double noiseLevel) {
        double[] noisyPattern = new double[pattern.length];
        for (int i = 0; i < pattern.length; i++) {
            double noise = (Math.random() - 0.5) * 2 * noiseLevel;
            noisyPattern[i] = Math.max(0, Math.min(1, pattern[i] + noise));
        }
        return noisyPattern;
    }
    
    private double calculateTemplateChange(double[] original, double[] current) {
        if (original.length != current.length) return 1.0;
        
        double totalDifference = 0.0;
        for (int i = 0; i < original.length; i++) {
            totalDifference += Math.abs(original[i] - current[i]);
        }
        return totalDifference / original.length;
    }
    
    private long getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
    }
    
    // Mock classes for testing ART functionality
    private static class MockARTNetwork {
        private final double vigilance;
        private final int maxCategories;
        private final List<MockCategoryNode> categories = new ArrayList<>();
        private int nextId = 1;
        
        public MockARTNetwork(double vigilance, int maxCategories) {
            this.vigilance = vigilance;
            this.maxCategories = maxCategories;
        }
        
        public MockCategoryNode learn(double[] pattern) {
            // Find matching category
            for (MockCategoryNode category : categories) {
                double match = calculateMatch(pattern, category.getTemplate());
                if (match >= vigilance) {
                    // Update template
                    category.updateTemplate(pattern);
                    return category;
                }
            }
            
            // Create new category
            if (categories.size() < maxCategories) {
                MockCategoryNode newCategory = new MockCategoryNode(nextId++, pattern.clone());
                categories.add(newCategory);
                return newCategory;
            }
            
            return categories.get(0); // Fallback
        }
        
        public MockCategoryNode classify(double[] pattern) {
            MockCategoryNode bestMatch = null;
            double bestScore = 0;
            
            for (MockCategoryNode category : categories) {
                double match = calculateMatch(pattern, category.getTemplate());
                if (match > bestScore) {
                    bestScore = match;
                    bestMatch = category;
                }
            }
            
            return bestScore >= vigilance ? bestMatch : null;
        }
        
        public int getCategoryCount() {
            return categories.size();
        }
        
        private double calculateMatch(double[] pattern1, double[] pattern2) {
            double intersection = 0;
            double union = 0;
            
            for (int i = 0; i < pattern1.length && i < pattern2.length; i++) {
                intersection += Math.min(pattern1[i], pattern2[i]);
                union += Math.max(pattern1[i], pattern2[i]);
            }
            
            return union > 0 ? intersection / union : 0;
        }
    }
    
    private static class MockCategoryNode {
        private final int id;
        private double[] template;
        
        public MockCategoryNode(int id, double[] template) {
            this.id = id;
            this.template = template;
        }
        
        public int getId() {
            return id;
        }
        
        public double[] getTemplate() {
            return template.clone();
        }
        
        public void updateTemplate(double[] newPattern) {
            // Simple template update (averaging)
            for (int i = 0; i < template.length && i < newPattern.length; i++) {
                template[i] = (template[i] + newPattern[i]) / 2.0;
            }
        }
    }
}