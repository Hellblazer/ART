package com.hellblazer.art.hybrid.pan.weight;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BPARTWeight implementation.
 */
class BPARTWeightTest {

    private PANParameters parameters;
    private Pattern testPattern;

    @BeforeEach
    void setUp() {
        parameters = PANParameters.defaultParameters();

        // Create test pattern
        double[] data = new double[10];
        for (int i = 0; i < data.length; i++) {
            data[i] = 0.5 + i * 0.05;
        }
        testPattern = new DenseVector(data);
    }

    @Test
    void testCreateFromPattern() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);

        assertNotNull(weight);
        assertEquals(testPattern.dimension() * parameters.hiddenUnits(), weight.forwardWeights().length);
        assertEquals(parameters.hiddenUnits(), weight.backwardWeights().length);
        assertEquals(parameters.hiddenUnits(), weight.hiddenBias().length);
        assertEquals(0, weight.updateCount());
    }

    @Test
    void testCreateFromPatternWithTarget() {
        double[] targetData = {0, 0, 1, 0, 0};  // One-hot encoded
        Pattern target = new DenseVector(targetData);

        BPARTWeight weight = BPARTWeight.createFromPatternWithTarget(testPattern, target, parameters);

        assertNotNull(weight);
        // Output bias should be initialized to target mean
        assertEquals(0.2, weight.outputBias(), 0.01);
    }

    @Test
    void testCalculateActivation() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);

        double activation = weight.calculateActivation(testPattern);

        assertTrue(activation >= 0.0);
        assertTrue(activation <= 1.0);  // Sigmoid bounded
    }

    @Test
    void testComputeGradients() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);

        // Test unsupervised gradients
        double[] gradients = weight.computeGradients(testPattern, null);
        assertNotNull(gradients);
        assertTrue(gradients.length > 0);

        // Test supervised gradients
        double[] targetData = {1.0};
        Pattern target = new DenseVector(targetData);
        double[] supervisedGradients = weight.computeGradients(testPattern, target);
        assertNotNull(supervisedGradients);
    }

    @Test
    void testWeightVectorInterface() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);

        // Test dimension
        int expectedDim = weight.forwardWeights().length + weight.backwardWeights().length;
        assertEquals(expectedDim, weight.dimension());

        // Test get
        double firstForward = weight.forwardWeights()[0];
        assertEquals(firstForward, weight.get(0));

        double firstBackward = weight.backwardWeights()[0];
        assertEquals(firstBackward, weight.get(weight.forwardWeights().length));

        // Test out of bounds
        assertThrows(IndexOutOfBoundsException.class, () -> weight.get(expectedDim + 1));

        // Test L1 norm
        double l1Norm = weight.l1Norm();
        assertTrue(l1Norm >= 0);
    }

    @Test
    void testUpdate() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);
        long initialCount = weight.updateCount();

        // Create gradient pattern
        double[] gradientData = new double[weight.dimension()];
        for (int i = 0; i < gradientData.length; i++) {
            gradientData[i] = 0.01;
        }
        Pattern gradientPattern = new DenseVector(gradientData);

        // Update weights
        var updatedWeight = (BPARTWeight) weight.update(gradientPattern, parameters);

        assertNotNull(updatedWeight);
        assertNotEquals(weight, updatedWeight);  // Immutable
        assertEquals(initialCount + 1, updatedWeight.updateCount());

        // Weights should have changed
        assertNotEquals(weight.forwardWeights()[0], updatedWeight.forwardWeights()[0]);
    }

    @Test
    void testNegativeWeightConstraint() {
        // Test with negative weights disabled
        var noNegParams = new PANParameters(
            0.7, 6,
            parameters.cnnConfig(),
            false,
            0.01, 0.9, 0.0001,
            false,  // No negative weights
            64,
            0.95, 0.8,
            100, 10, 0.1,
            0.1
        );

        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, noNegParams);

        // Create gradient that would make weights negative
        double[] gradientData = new double[weight.dimension()];
        for (int i = 0; i < gradientData.length; i++) {
            gradientData[i] = 100.0;  // Large positive gradient
        }
        Pattern gradientPattern = new DenseVector(gradientData);

        var updatedWeight = (BPARTWeight) weight.update(gradientPattern, noNegParams);

        // Check no negative weights
        for (double w : updatedWeight.forwardWeights()) {
            assertTrue(w >= 0, "Forward weight should be non-negative");
        }
        for (double w : updatedWeight.backwardWeights()) {
            assertTrue(w >= 0, "Backward weight should be non-negative");
        }
    }

    @Test
    void testSimilarity() {
        BPARTWeight weight1 = BPARTWeight.createFromPattern(testPattern, parameters);
        BPARTWeight weight2 = BPARTWeight.createFromPattern(testPattern, parameters);

        // Same pattern should give high similarity
        double similarity = weight1.similarity(weight2);
        assertTrue(similarity >= 0);
        assertTrue(similarity <= 1.0);

        // Different patterns should give lower similarity
        double[] differentData = new double[10];
        for (int i = 0; i < differentData.length; i++) {
            differentData[i] = 0.1;
        }
        Pattern differentPattern = new DenseVector(differentData);
        BPARTWeight weight3 = BPARTWeight.createFromPattern(differentPattern, parameters);

        double diffSimilarity = weight1.similarity(weight3);
        assertTrue(diffSimilarity < similarity);
    }

    @Test
    void testLTMConsolidation() {
        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, parameters);

        // Initially shouldn't consolidate
        assertFalse(weight.shouldConsolidateToLTM(0.1));

        // After many updates with low error, should consolidate
        var updated = weight;
        for (int i = 0; i < 101; i++) {
            double[] gradientData = new double[updated.dimension()];
            Pattern gradientPattern = new DenseVector(gradientData);
            updated = (BPARTWeight) updated.update(gradientPattern, parameters);
        }

        assertTrue(updated.updateCount() > 100);
        assertTrue(updated.shouldConsolidateToLTM(0.1));
    }

    @Test
    void testLightInductionBias() {
        // Parameters with bias factor
        var biasParams = new PANParameters(
            0.7, 6,
            parameters.cnnConfig(),
            false,
            0.01, 0.9, 0.0001, true, 64,
            0.95, 0.8,
            100, 10, 0.1,
            0.5  // Large bias factor
        );

        BPARTWeight weight = BPARTWeight.createFromPattern(testPattern, biasParams);
        double initialBias = weight.outputBias();

        // Update should increase bias due to light induction
        double[] gradientData = new double[weight.dimension()];
        Pattern gradientPattern = new DenseVector(gradientData);
        var updated = (BPARTWeight) weight.update(gradientPattern, biasParams);

        // Output bias should have increased due to light induction
        assertTrue(updated.outputBias() > initialBias);
    }
}