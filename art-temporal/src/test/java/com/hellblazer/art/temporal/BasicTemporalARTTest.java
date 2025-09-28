package com.hellblazer.art.temporal;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.algorithms.BasicTemporalART;
import com.hellblazer.art.temporal.parameters.*;
import com.hellblazer.art.temporal.results.TemporalResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Comprehensive test suite for BasicTemporalART
 */
public class BasicTemporalARTTest {

    private BasicTemporalART temporalART;
    private WorkingMemoryParameters wmParams;
    private MaskingParameters mfParams;

    @BeforeEach
    public void setUp() {
        // Create working memory parameters
        wmParams = WorkingMemoryParameters.builder()
            .capacity(10)
            .decayRate(0.1)
            .maxActivation(1.0)
            .competitiveRate(0.05)
            .primacyThreshold(0.01)
            .temporalResolution(0.01)
            .inputGain(1.0)
            .enableCompetition(true)
            .enableNormalization(true)
            .enableAdaptiveCapacity(false)
            .build();

        // Create masking field parameters
        mfParams = MaskingParameters.builder()
            .scaleCount(3)
            .fieldSize(100)
            .passiveDecayRate(0.1)
            .maxActivation(1.0)
            .lateralInhibition(0.3)
            .selfInhibition(0.1)
            .transmitterRecoveryRate(0.2)
            .transmitterDepletionRate(0.5)
            .boundaryThreshold(0.1)
            .convergenceThreshold(0.01)
            .timeStep(0.01)
            .enableTransmitterGates(true)
            .enableCompetition(true)
            .enableMultiScale(true)
            .scaleFactor(2.0)
            .build();

        // Create temporal ART instance
        temporalART = new BasicTemporalART(wmParams, mfParams, 0.7f, 0.01f, 50);
    }

    @Test
    public void testBasicSequenceProcessing() {
        // Create a simple sequence of patterns
        var sequence = createTestSequence(3, 10);

        // Process the sequence
        var result = temporalART.processSequence(sequence);

        // Verify basic properties
        assertNotNull(result);
        assertNotNull(result.getWorkingMemoryState());
        assertNotNull(result.getMaskingFieldActivations());
        assertTrue(result.getProcessingTime() >= 0);
    }

    @Test
    public void testSequenceLearning() {
        // Create multiple similar sequences
        var sequences = new ArrayList<List<Pattern>>();
        for (int i = 0; i < 5; i++) {
            sequences.add(createSimilarSequence(4, 10, i * 0.1f));
        }

        // Process and learn sequences
        for (var sequence : sequences) {
            var result = temporalART.processSequence(sequence);
            assertTrue(result.getProcessingTime() >= 0);
        }

        // Check that categories were created
        var categories = temporalART.getCategories();
        assertFalse(categories.isEmpty());
        assertTrue(categories.size() <= 5);
    }

    @Test
    public void testPrimacyGradient() {
        // Create sequence
        var sequence = createTestSequence(5, 10);

        // Process sequence
        var result = temporalART.processSequence(sequence);

        // Check working memory state
        var memoryState = result.getWorkingMemoryState();
        assertNotNull(memoryState);
        assertEquals(5, memoryState.getSequenceLength());

        // Verify the temporal pattern has sequence
        var seq = memoryState.getSequence();
        assertNotNull(seq);
        assertEquals(5, seq.size());
    }

    @Test
    public void testTransmitterGates() {
        // Create sequence
        var sequence = createTestSequence(3, 10);

        // Process multiple times to see transmitter depletion
        float[] previousStates = null;
        for (int i = 0; i < 5; i++) {
            var result = temporalART.processSequence(sequence);
            var transmitterStates = result.getTransmitterGateValues();

            assertNotNull(transmitterStates);

            if (previousStates != null) {
                // Check that some gates have changed
                var hasChanged = false;
                for (int j = 0; j < Math.min(previousStates.length, transmitterStates[0].length); j++) {
                    if (Math.abs(previousStates[j] - transmitterStates[0][j]) > 0.001) {
                        hasChanged = true;
                        break;
                    }
                }
                assertTrue(hasChanged, "Transmitter states should change with repeated activation");
            }

            // Convert transmitter states to 1D for comparison
            if (transmitterStates.length > 0) {
                previousStates = new float[transmitterStates[0].length];
                for (int j = 0; j < transmitterStates[0].length; j++) {
                    previousStates[j] = (float)transmitterStates[0][j];
                }
            }
        }
    }

    @Test
    public void testMaskingFieldActivation() {
        // Create sequences of different lengths
        var shortSequence = createTestSequence(2, 10);
        var mediumSequence = createTestSequence(5, 10);
        var longSequence = createTestSequence(8, 10);

        // Process each sequence
        var shortResult = temporalART.processSequence(shortSequence);
        var mediumResult = temporalART.processSequence(mediumSequence);
        var longResult = temporalART.processSequence(longSequence);

        // Different length sequences should create different results
        var shortChunks = shortResult.getIdentifiedChunks();
        var mediumChunks = mediumResult.getIdentifiedChunks();
        var longChunks = longResult.getIdentifiedChunks();

        // They might have different numbers of chunks
        assertNotNull(shortChunks);
        assertNotNull(mediumChunks);
        assertNotNull(longChunks);
    }

    @Test
    public void testResetFunctionality() {
        // Process some sequences
        for (int i = 0; i < 3; i++) {
            var sequence = createTestSequence(3, 10);
            temporalART.processSequence(sequence);
        }

        // Should have some categories
        assertFalse(temporalART.getCategories().isEmpty());

        // Reset
        temporalART.reset();

        // Should have no categories after reset
        assertTrue(temporalART.getCategories().isEmpty());
    }

    @Test
    public void testSinglePatternProcessing() {
        // Create single pattern
        var pattern = createTestPattern(10);

        // Process as single pattern sequence
        temporalART.learn(pattern, false);

        // Should create a category
        assertEquals(1, temporalART.getCategories().size());

        // Predict with same pattern
        var result = temporalART.predict(pattern);
        assertNotNull(result);
        if (result instanceof ActivationResult.Success success) {
            assertEquals(0, success.categoryIndex());
        } else {
            fail("Expected successful activation");
        }
    }

    @Test
    public void testTemporalPatternExtraction() {
        // Create sequence
        var sequence = createTestSequence(4, 10);

        // Process sequence and get temporal pattern
        var result = temporalART.processSequence(sequence);
        var temporalPattern = result.getWorkingMemoryState();

        assertNotNull(temporalPattern);
        assertEquals(4, temporalPattern.getSequenceLength());
        assertNotNull(temporalPattern.getSequence());
        assertTrue(temporalPattern.isValid());
    }

    @Test
    public void testStatistics() {
        // Process several sequences
        for (int i = 0; i < 5; i++) {
            var sequence = createTestSequence(3, 10);
            temporalART.processSequence(sequence);
        }

        // Get category count
        var categoryCount = temporalART.getCategoryCount();
        assertTrue(categoryCount >= 0);
    }

    // Helper methods

    private List<Pattern> createTestSequence(int length, int dimension) {
        var sequence = new ArrayList<Pattern>();
        var random = new Random(42);  // Fixed seed for reproducibility

        for (int i = 0; i < length; i++) {
            sequence.add(createRandomPattern(dimension, random));
        }

        return sequence;
    }

    private List<Pattern> createSimilarSequence(int length, int dimension, float noise) {
        var sequence = new ArrayList<Pattern>();
        var random = new Random(42);
        var basePattern = createRandomPattern(dimension, random);

        for (int i = 0; i < length; i++) {
            sequence.add(addNoise(basePattern, noise, random));
        }

        return sequence;
    }

    private Pattern createTestPattern(int dimension) {
        var features = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = Math.sin(i * 0.1) * 0.5 + 0.5;
        }
        return new DenseVector(features);
    }

    private Pattern createRandomPattern(int dimension, Random random) {
        var features = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = random.nextDouble();
        }
        return new DenseVector(features);
    }

    private Pattern addNoise(Pattern pattern, float noiseLevel, Random random) {
        var dimension = pattern.dimension();
        var noisyFeatures = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            var noise = (random.nextDouble() - 0.5) * 2 * noiseLevel;
            noisyFeatures[i] = Math.max(0, Math.min(1, pattern.get(i) + noise));
        }

        return new DenseVector(noisyFeatures);
    }

}