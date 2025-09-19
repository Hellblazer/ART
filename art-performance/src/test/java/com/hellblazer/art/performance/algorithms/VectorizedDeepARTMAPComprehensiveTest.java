package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedDeepARTMAP;
import com.hellblazer.art.performance.algorithms.VectorizedDeepARTMAPParameters;
import com.hellblazer.art.core.artmap.DeepARTMAPParameters;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for VectorizedDeepARTMAP based on reference implementation patterns.
 * Tests cover multi-module initialization, hierarchical learning, deep label mapping,
 * supervised/unsupervised modes, and deep prediction capabilities.
 */
class VectorizedDeepARTMAPComprehensiveTest {

    private VectorizedDeepARTMAP deepArtmap;
    private VectorizedDeepARTMAPParameters parameters;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42);
        
        // Initialize with typical parameters for multi-module architecture  
        var baseParams = new DeepARTMAPParameters(0.75, 0.1, 1000, true);
        parameters = VectorizedDeepARTMAPParameters.defaults(baseParams);
        
        // Create simple modules for testing using FuzzyART
        var modules = List.<BaseART>of(
            new FuzzyART(),
            new FuzzyART()
        );
        
        deepArtmap = new VectorizedDeepARTMAP(modules, parameters);
    }

    @Test
    @DisplayName("Test DeepARTMAP initialization with multiple modules")
    void testMultiModuleInitialization() {
        assertNotNull(deepArtmap);
        
        // Test parameter retrieval
        var retrievedParams = deepArtmap.getParameters();
        assertNotNull(retrievedParams);
        assertEquals(VectorizedDeepARTMAPParameters.class, retrievedParams.getClass());
        
        // Test performance stats initialization
        var stats = deepArtmap.getPerformanceStats();
        assertNotNull(stats);
    }

    @Test
    @DisplayName("Test parameter validation and access")
    void testParameterValidation() {
        var params = deepArtmap.getParameters();
        assertNotNull(params);
        
        // Test that we can access parameter components
        assertNotNull(params.toString());
        
        // Test vectorization features
        // Note: VectorizedDeepARTMAP doesn't implement VectorizedARTAlgorithm interface
        // assertTrue(deepArtmap.isVectorized());
        int vectorLength = deepArtmap.getVectorSpeciesLength();
        assertTrue(vectorLength != 0);
    }

    @Test
    @DisplayName("Test supervised learning with hierarchical data")
    void testSupervisedHierarchicalLearning() {
        // Create hierarchical input data (multiple layers)
        var layer1Data = generateRandomPatterns(15, 4);
        var layer2Data = generateRandomPatterns(15, 3);
        var targetLabels = generateRandomPatterns(15, 2);
        
        // Train with hierarchical supervised learning
        for (int i = 0; i < layer1Data.size(); i++) {
            // In a real deep architecture, we'd combine layers
            var combinedPattern = combinePatterns(layer1Data.get(i), layer2Data.get(i));
            var result = deepArtmap.learn(combinedPattern, parameters);
            assertNotNull(result);
        }
        
        // Verify learning occurred
        assertTrue(deepArtmap.getCategoryCount() > 0);
    }

    @Test
    @DisplayName("Test unsupervised learning mode")
    void testUnsupervisedLearning() {
        var inputData = generateRandomPatterns(20, 6);
        
        // Train in unsupervised mode (no target labels)
        for (var pattern : inputData) {
            var result = deepArtmap.learn(pattern, parameters);
            assertNotNull(result);
        }
        
        // Should still create categories through unsupervised clustering
        assertTrue(deepArtmap.getCategoryCount() > 0);
    }

    @Test
    @DisplayName("Test deep prediction capabilities")
    void testDeepPrediction() {
        var trainingData = generateRandomPatterns(12, 5);
        
        // Train the deep model
        for (var pattern : trainingData) {
            deepArtmap.learn(pattern, parameters);
        }
        
        // Test prediction on training data
        for (var pattern : trainingData) {
            var prediction = deepArtmap.predict(pattern, parameters);
            assertNotNull(prediction);
        }
        
        // Test prediction on new hierarchical data
        var newData = generateRandomPatterns(5, 5);
        for (var pattern : newData) {
            var prediction = deepArtmap.predict(pattern, parameters);
            assertNotNull(prediction);
        }
    }

    @Test
    @DisplayName("Test hierarchical label mapping")
    void testHierarchicalLabelMapping() {
        // Create structured data that should form hierarchical clusters
        var structuredData = createStructuredHierarchicalData();
        
        // Train the model
        for (var pattern : structuredData) {
            deepArtmap.learn(pattern, parameters);
        }
        
        // Test that hierarchical structure is learned
        assertTrue(deepArtmap.getCategoryCount() > 1, 
            "Should learn multiple categories from hierarchical data");
        
        // Test prediction consistency
        for (var pattern : structuredData) {
            var prediction1 = deepArtmap.predict(pattern, parameters);
            var prediction2 = deepArtmap.predict(pattern, parameters);
            // Predictions should be consistent
            assertEquals(prediction1, prediction2, 
                "Predictions should be consistent for the same pattern");
        }
    }

    @Test
    @DisplayName("Test multi-level vigilance effects")
    void testMultiLevelVigilanceEffects() {
        var inputData = generateRandomPatterns(10, 4);
        
        // Test with low vigilance at all levels
        var lowVigilanceBaseParams = new DeepARTMAPParameters(0.05, 0.1, 10, true);
        var lowVigilanceParams = VectorizedDeepARTMAPParameters.defaults(lowVigilanceBaseParams);
        
        var lowVigilanceModules = List.<BaseART>of(
            new FuzzyART(),
            new FuzzyART()
        );
        var deepArtmapLow = new VectorizedDeepARTMAP(lowVigilanceModules, lowVigilanceParams);
        for (var pattern : inputData) {
            deepArtmapLow.learn(pattern, lowVigilanceParams);
        }
        int lowVigilanceCategories = deepArtmapLow.getCategoryCount();
        
        // Test with high vigilance at all levels
        var highVigilanceBaseParams = new DeepARTMAPParameters(0.95, 0.01, 10, true);
        var highVigilanceParams = VectorizedDeepARTMAPParameters.defaults(highVigilanceBaseParams);
        
        var highVigilanceModules = List.<BaseART>of(
            new FuzzyART(),
            new FuzzyART()
        );
        var deepArtmapHigh = new VectorizedDeepARTMAP(highVigilanceModules, highVigilanceParams);
        for (var pattern : inputData) {
            deepArtmapHigh.learn(pattern, highVigilanceParams);
        }
        int highVigilanceCategories = deepArtmapHigh.getCategoryCount();
        
        // Higher vigilance should generally create more or equal categories
        assertTrue(highVigilanceCategories >= lowVigilanceCategories,
            "High vigilance should create at least as many categories as low vigilance");
    }

    @Test
    @DisplayName("Test incremental deep learning")
    void testIncrementalDeepLearning() {
        var batch1 = generateRandomPatterns(8, 5);
        var batch2 = generateRandomPatterns(8, 5);
        var batch3 = generateRandomPatterns(8, 5);
        
        // Learn first batch
        for (var pattern : batch1) {
            deepArtmap.learn(pattern, parameters);
        }
        int categoriesAfterBatch1 = deepArtmap.getCategoryCount();
        
        // Learn second batch
        for (var pattern : batch2) {
            deepArtmap.learn(pattern, parameters);
        }
        int categoriesAfterBatch2 = deepArtmap.getCategoryCount();
        
        // Learn third batch
        for (var pattern : batch3) {
            deepArtmap.learn(pattern, parameters);
        }
        int categoriesAfterBatch3 = deepArtmap.getCategoryCount();
        
        // Categories should not decrease with incremental learning
        assertTrue(categoriesAfterBatch1 > 0);
        assertTrue(categoriesAfterBatch2 >= categoriesAfterBatch1);
        assertTrue(categoriesAfterBatch3 >= categoriesAfterBatch2);
    }

    @Test
    @DisplayName("Test performance tracking in deep architecture")
    void testDeepPerformanceTracking() {
        var inputData = generateRandomPatterns(20, 6);
        
        // Reset performance tracking
        deepArtmap.resetPerformanceTracking();
        
        // Perform deep learning operations
        for (var pattern : inputData) {
            deepArtmap.learn(pattern, parameters);
        }
        
        // Perform prediction operations
        for (var pattern : inputData.subList(0, 5)) {
            deepArtmap.predict(pattern, parameters);
        }
        
        // Check performance stats
        var stats = deepArtmap.getPerformanceStats();
        assertNotNull(stats);
        assertNotNull(stats.toString());
    }

    @Test
    @DisplayName("Test resource management in deep architecture")
    void testDeepResourceManagement() {
        assertDoesNotThrow(() -> {
            var testModules = List.<BaseART>of(new FuzzyART());
            var testParams = VectorizedDeepARTMAPParameters.defaults(new DeepARTMAPParameters(0.75, 0.1, 10, true));
            try (var testDeepArtmap = new VectorizedDeepARTMAP(testModules, testParams)) {
                var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4);
                testDeepArtmap.learn(pattern, parameters);
                testDeepArtmap.predict(pattern, parameters);
            }
        });
    }

    @Test
    @DisplayName("Test deep architecture with complex patterns")
    void testComplexPatternLearning() {
        // Create patterns with complex hierarchical structure
        var complexPatterns = createComplexHierarchicalPatterns();
        
        // Train on complex patterns
        for (var pattern : complexPatterns) {
            deepArtmap.learn(pattern, parameters);
        }
        
        // Test that complex structure is captured
        assertTrue(deepArtmap.getCategoryCount() > 0);
        
        // Test prediction on complex patterns
        for (var pattern : complexPatterns) {
            var prediction = deepArtmap.predict(pattern, parameters);
            assertNotNull(prediction);
        }
    }

    @Test
    @DisplayName("Test data validation for deep architecture")
    void testDeepDataValidation() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            deepArtmap.learn(null, parameters);
        });
        
        // Test null parameters
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        assertThrows(NullPointerException.class, () -> {
            deepArtmap.learn(pattern, (VectorizedDeepARTMAPParameters) null);
        });
        
        // Test valid inputs
        assertDoesNotThrow(() -> {
            deepArtmap.learn(pattern, parameters);
        });
    }

    /**
     * Generate random patterns for testing
     */
    private List<Pattern> generateRandomPatterns(int count, int dimensions) {
        return IntStream.range(0, count)
            .mapToObj(i -> {
                double[] values = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    values[j] = random.nextDouble();
                }
                return Pattern.of(values);
            })
            .toList();
    }

    /**
     * Create structured hierarchical data for testing deep learning
     */
    private List<Pattern> createStructuredHierarchicalData() {
        return List.of(
            // Group 1: Similar patterns at level 1
            Pattern.of(0.1, 0.1, 0.8, 0.8, 0.2),
            Pattern.of(0.12, 0.08, 0.82, 0.78, 0.18),
            Pattern.of(0.08, 0.12, 0.78, 0.82, 0.22),
            
            // Group 2: Similar patterns at level 2
            Pattern.of(0.7, 0.3, 0.1, 0.2, 0.9),
            Pattern.of(0.72, 0.28, 0.12, 0.18, 0.88),
            Pattern.of(0.68, 0.32, 0.08, 0.22, 0.92),
            
            // Group 3: Different pattern structure
            Pattern.of(0.5, 0.5, 0.5, 0.5, 0.5),
            Pattern.of(0.48, 0.52, 0.48, 0.52, 0.48)
        );
    }

    /**
     * Create complex hierarchical patterns
     */
    private List<Pattern> createComplexHierarchicalPatterns() {
        return List.of(
            // Complex pattern 1: Multiple levels of structure
            Pattern.of(0.1, 0.9, 0.1, 0.9, 0.1, 0.9),
            Pattern.of(0.11, 0.89, 0.12, 0.88, 0.09, 0.91),
            
            // Complex pattern 2: Different hierarchical structure  
            Pattern.of(0.3, 0.3, 0.7, 0.7, 0.4, 0.6),
            Pattern.of(0.32, 0.28, 0.68, 0.72, 0.38, 0.62),
            
            // Complex pattern 3: Mixed structure
            Pattern.of(0.0, 1.0, 0.5, 0.2, 0.8, 0.3),
            Pattern.of(0.02, 0.98, 0.48, 0.22, 0.78, 0.32)
        );
    }

    /**
     * Combine two patterns for hierarchical input simulation
     */
    private Pattern combinePatterns(Pattern pattern1, Pattern pattern2) {
        int dim1 = pattern1.dimension();
        int dim2 = pattern2.dimension();
        
        double[] combined = new double[dim1 + dim2];
        
        // Copy values from first pattern
        for (int i = 0; i < dim1; i++) {
            combined[i] = pattern1.get(i);
        }
        
        // Copy values from second pattern
        for (int i = 0; i < dim2; i++) {
            combined[dim1 + i] = pattern2.get(i);
        }
        
        return Pattern.of(combined);
    }
}