package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.FusionART;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test FusionART based on the Python reference implementation.
 * 
 * FusionART combines multiple ART modules (one per channel) where:
 * - Each module processes its own channel of data
 * - Activation is a weighted combination of channel activations
 * - All channels must pass vigilance for resonance
 * - Each channel updates independently during learning
 */
class FusionARTReferenceTest {
    
    private FusionART fusionART;
    private FuzzyART moduleA;
    private FuzzyART moduleB;
    private double[] gammaValues;
    private int[] channelDims;
    
    @BeforeEach
    void setUp() {
        // Initialize FusionART with two FuzzyART modules
        // Matching Python: module_a = FuzzyART(0.5, 0.01, 1.0)
        moduleA = new FuzzyART();
        var paramsA = FuzzyParameters.builder()
            .vigilance(0.5)
            .learningRate(0.01)
            .build();
        
        // Matching Python: module_b = FuzzyART(0.7, 0.01, 1.0)
        moduleB = new FuzzyART();
        var paramsB = FuzzyParameters.builder()
            .vigilance(0.7)
            .learningRate(0.01)
            .build();
        
        // gamma_values = np.array([0.5, 0.5])
        gammaValues = new double[] {0.5, 0.5};
        
        // channel_dims = [4, 4]
        channelDims = new int[] {4, 4};
        
        fusionART = new FusionART(
            List.of(moduleA, moduleB),
            gammaValues,
            channelDims
        );
    }
    
    @Test
    void testInitialization() {
        // Test that the model initializes correctly
        assertEquals(2, fusionART.getNumChannels());
        assertArrayEquals(gammaValues, fusionART.getGammaValues());
        assertArrayEquals(channelDims, fusionART.getChannelDims());
        assertEquals(8, fusionART.getTotalDimension()); // 4 + 4
    }
    
    @Test
    void testValidateParams() {
        // Test parameter validation
        
        // Valid params: gamma values sum to 1.0
        var validGamma = new double[] {0.3, 0.7};
        assertDoesNotThrow(() -> FusionART.validateGammaValues(validGamma));
        
        // Invalid: gamma values don't sum to 1.0
        var invalidGamma = new double[] {0.6, 0.6};
        assertThrows(IllegalArgumentException.class, 
            () -> FusionART.validateGammaValues(invalidGamma));
        
        // Invalid: gamma values outside [0,1]
        var invalidGamma2 = new double[] {1.5, -0.5};
        assertThrows(IllegalArgumentException.class,
            () -> FusionART.validateGammaValues(invalidGamma2));
    }
    
    @Test
    void testPrepareAndRestoreData() {
        // Test prepare_data and restore_data methods
        var random = new Random(42);
        
        // Create two channels of data (10 samples, 2 features each)
        var channel1Data = generateRandomData(10, 2, random);
        var channel2Data = generateRandomData(10, 2, random);
        
        // Prepare data - converting Pattern[][] to List<Pattern[]>
        var channelList = new ArrayList<Pattern[]>();
        
        // Convert the 2D arrays to 1D arrays for each channel
        var channel1Array = new Pattern[10];
        var channel2Array = new Pattern[10];
        for (int i = 0; i < 10; i++) {
            channel1Array[i] = channel1Data[i][0];
            channel2Array[i] = channel2Data[i][0];
        }
        
        channelList.add(channel1Array);
        channelList.add(channel2Array);
        
        var preparedData = fusionART.prepareData(channelList);
        
        // After complement coding: 2 features -> 4 features per channel
        // Total: 8 features
        assertEquals(10, preparedData.length); // 10 samples
        assertEquals(8, preparedData[0].dimension()); // 8 total features
        
        // Restore data
        var restoredChannels = fusionART.restoreData(preparedData);
        assertEquals(2, restoredChannels.size());
        
        // Check dimensions match prepared data (4 features per channel after complement coding)
        assertEquals(10, restoredChannels.get(0).length);
        assertEquals(4, restoredChannels.get(0)[0].dimension());
    }
    
    @Test
    void testStepFit() {
        // Test the step_fit method
        var random = new Random(42);
        
        // Create sample data for both channels
        var sample = new double[8]; // Total dimension after complement coding
        for (int i = 0; i < 4; i++) {
            sample[i] = random.nextDouble(); // Channel 1
            sample[i + 4] = random.nextDouble(); // Channel 2
        }
        
        var pattern = Pattern.of(sample);
        var params = fusionART.createDefaultParameters();
        
        // First step_fit should create a category
        var result = fusionART.stepFit(pattern, params);
        assertNotNull(result);
        assertTrue(result instanceof com.hellblazer.art.core.results.ActivationResult.Success);
        assertEquals(0, ((com.hellblazer.art.core.results.ActivationResult.Success) result).categoryIndex());
        assertEquals(1, fusionART.getCategoryCount());
    }
    
    @Test
    void testFitMultipleSamples() {
        // Test fitting multiple samples
        var random = new Random(42);
        var patterns = new ArrayList<Pattern>();
        
        // Generate 10 random patterns
        for (int i = 0; i < 10; i++) {
            var sample = new double[8];
            for (int j = 0; j < 8; j++) {
                sample[j] = random.nextDouble();
            }
            patterns.add(Pattern.of(sample));
        }
        
        var params = fusionART.createDefaultParameters();
        
        // Fit all patterns
        for (var pattern : patterns) {
            fusionART.stepFit(pattern, params);
        }
        
        // Should have created some categories (exact number depends on vigilance)
        assertTrue(fusionART.getCategoryCount() > 0);
        assertTrue(fusionART.getCategoryCount() <= 10);
    }
    
    @Test
    void testPredict() {
        // Test prediction after training
        var random = new Random(42);
        var params = fusionART.createDefaultParameters();
        
        // Train with a pattern
        var trainPattern = generateRandomPattern(8, random);
        fusionART.stepFit(trainPattern, params);
        
        // Predict with the same pattern
        var prediction = fusionART.stepPredict(trainPattern, params);
        assertEquals(0, prediction); // Should predict the first category
        
        // Predict with a slightly different pattern
        var testPattern = generateSimilarPattern(trainPattern, 0.05, random);
        var testPrediction = fusionART.stepPredict(testPattern, params);
        assertNotNull(testPrediction);
    }
    
    @Test
    void testChannelSkipping() {
        // Test that we can skip channels during processing
        var random = new Random(42);
        var pattern = generateRandomPattern(8, random);
        var params = fusionART.createDefaultParameters();
        
        // Train normally
        fusionART.stepFit(pattern, params);
        
        // Test with skipping channel 1 (index 1)
        var skipChannels = List.of(1);
        var resultWithSkip = fusionART.stepFitWithSkip(pattern, params, skipChannels);
        
        assertNotNull(resultWithSkip);
        // Processing should still work with one channel
    }
    
    @Test
    void testGetClusterCenters() {
        // Test getting cluster centers after training
        var random = new Random(42);
        var params = fusionART.createDefaultParameters();
        
        // Train with several patterns
        for (int i = 0; i < 5; i++) {
            var pattern = generateRandomPattern(8, random);
            fusionART.stepFit(pattern, params);
        }
        
        // Get cluster centers
        var centers = fusionART.getClusterCenters();
        assertNotNull(centers);
        assertEquals(fusionART.getCategoryCount(), centers.size());
        
        // Each center should have the full dimension
        for (var center : centers) {
            assertEquals(8, center.length);
        }
    }
    
    @Test
    void testJoinAndSplitChannelData() {
        // Test joining and splitting channel data
        var channel1 = new double[] {0.1, 0.2, 0.3, 0.4};
        var channel2 = new double[] {0.5, 0.6, 0.7, 0.8};
        
        // Join channels
        var joined = fusionART.joinChannelData(List.of(
            Pattern.of(channel1),
            Pattern.of(channel2)
        ));
        
        assertEquals(8, joined.dimension());
        
        // Check values are in correct order
        for (int i = 0; i < 4; i++) {
            assertEquals(channel1[i], joined.get(i), 1e-6);
            assertEquals(channel2[i], joined.get(i + 4), 1e-6);
        }
        
        // Split channels
        var split = fusionART.splitChannelData(joined);
        assertEquals(2, split.size());
        
        for (int i = 0; i < 4; i++) {
            assertEquals(channel1[i], split.get(0).get(i), 1e-6);
            assertEquals(channel2[i], split.get(1).get(i), 1e-6);
        }
    }
    
    // Helper methods
    
    private Pattern[][] generateRandomData(int samples, int features, Random random) {
        var data = new Pattern[samples][1];
        for (int i = 0; i < samples; i++) {
            var values = new double[features];
            for (int j = 0; j < features; j++) {
                values[j] = random.nextDouble();
            }
            data[i][0] = Pattern.of(values);
        }
        return data;
    }
    
    private Pattern generateRandomPattern(int dim, Random random) {
        var values = new double[dim];
        for (int i = 0; i < dim; i++) {
            values[i] = random.nextDouble();
        }
        return Pattern.of(values);
    }
    
    private Pattern generateSimilarPattern(Pattern original, double noise, Random random) {
        var values = new double[original.dimension()];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.max(0, Math.min(1, 
                original.get(i) + (random.nextDouble() - 0.5) * noise));
        }
        return Pattern.of(values);
    }
}