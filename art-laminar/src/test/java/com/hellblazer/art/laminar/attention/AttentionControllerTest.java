package com.hellblazer.art.laminar.attention;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.AttentionParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AttentionController - spatial, feature, and object-based attention mechanisms.
 *
 * Tests verify biological accuracy of attention mechanisms from Grossberg's canonical circuit:
 * - Gaussian spatial gain fields
 * - Similarity-based feature enhancement
 * - Template matching for object attention
 * - Combined multi-modal attention
 *
 * @author Hal Hildebrand
 */
public class AttentionControllerTest {

    private AttentionController controller;
    private AttentionParameters defaultParams;

    @BeforeEach
    public void setUp() {
        defaultParams = new AttentionParameters();
        controller = new AttentionController(32, 32, defaultParams);
    }

    /**
     * Test 1: Spatial Attention - Gaussian Gain Field
     *
     * Verifies that spatial attention creates a Gaussian gain field centered
     * on the attended location, with proper falloff according to sigma parameter.
     *
     * Biological basis: Spatial attention enhances processing at attended locations
     * through multiplicative gain modulation (Reynolds & Heeger, 2009).
     */
    @Test
    public void testSpatialAttention() {
        // Set attention to center of field
        int centerX = 16;
        int centerY = 16;
        controller.setAttentionLocation(centerX, centerY);

        // Test gain at center - should be maximum
        double centerGain = controller.computeSpatialGain(centerX, centerY);
        assertEquals(1.0, centerGain, 0.01, "Center should have maximum gain (1.0 before scaling)");

        // Test gain at nearby location - should be high but less than center
        double nearbyGain = controller.computeSpatialGain(centerX + 2, centerY + 2);
        assertTrue(nearbyGain > 0.7, "Nearby location should have high gain");
        assertTrue(nearbyGain < centerGain, "Nearby gain should be less than center");

        // Test gain at distant location - should be low
        double distantGain = controller.computeSpatialGain(0, 0);
        assertTrue(distantGain < 0.3, "Distant location should have low gain");

        // Verify Gaussian profile - should decrease monotonically with distance
        double gain1 = controller.computeSpatialGain(centerX + 1, centerY);
        double gain2 = controller.computeSpatialGain(centerX + 2, centerY);
        double gain3 = controller.computeSpatialGain(centerX + 3, centerY);
        assertTrue(gain1 > gain2, "Gain should decrease with distance (1 vs 2)");
        assertTrue(gain2 > gain3, "Gain should decrease with distance (2 vs 3)");
    }

    /**
     * Test 2: Feature Attention - Similarity-Based Enhancement
     *
     * Verifies that feature attention enhances processing of features similar
     * to the attended feature, using cosine similarity.
     *
     * Biological basis: Feature-based attention enhances processing of attended
     * features across the visual field (Maunsell & Treue, 2006).
     */
    @Test
    public void testFeatureAttention() {
        // Create an attended feature pattern (vertical orientation)
        var attendedFeature = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        controller.setAttendedFeature(attendedFeature);

        // Test identical feature - should get maximum enhancement
        var identicalFeature = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        double identicalGain = controller.computeFeatureGain(identicalFeature);
        assertTrue(identicalGain >= 1.0, "Identical feature should have gain >= 1.0");
        assertTrue(identicalGain <= defaultParams.maxFeatureGain(),
                   "Gain should not exceed max feature gain");

        // Test similar feature - should get moderate enhancement
        var similarFeature = new DenseVector(new double[]{0.7, 0.3, 0.0, 0.0});
        double similarGain = controller.computeFeatureGain(similarFeature);
        assertTrue(similarGain > 1.0, "Similar feature should have enhancement > 1.0");
        assertTrue(similarGain <= identicalGain, "Similar gain should be <= identical");

        // Test orthogonal feature - should get minimal enhancement
        var orthogonalFeature = new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0});
        double orthogonalGain = controller.computeFeatureGain(orthogonalFeature);
        assertTrue(orthogonalGain >= 1.0, "Orthogonal feature should still have base gain");
        assertTrue(orthogonalGain < similarGain, "Orthogonal gain should be less than similar");
    }

    /**
     * Test 3: Object Attention - Template Matching
     *
     * Verifies that object attention enhances processing based on match to
     * an attended object template.
     *
     * Biological basis: Object-based attention enhances all features of an
     * attended object (Scholl, 2001).
     */
    @Test
    public void testObjectAttention() {
        // Create an attended object template
        var objectTemplate = new DenseVector(new double[]{
            0.8, 0.6, 0.4, 0.2,  // Example object feature vector
            0.9, 0.7, 0.5, 0.3
        });
        controller.setAttendedObject(objectTemplate);

        // Test perfect match - should get strong enhancement
        var perfectMatch = new DenseVector(new double[]{
            0.8, 0.6, 0.4, 0.2,
            0.9, 0.7, 0.5, 0.3
        });
        double perfectGain = controller.computeObjectGain(perfectMatch);
        assertTrue(perfectGain >= 1.0, "Perfect match should have gain >= 1.0");
        assertTrue(perfectGain <= defaultParams.maxObjectGain(),
                   "Gain should not exceed max object gain");

        // Test partial match - should get moderate enhancement (more different pattern)
        var partialMatch = new DenseVector(new double[]{
            0.5, 0.4, 0.6, 0.3,
            0.6, 0.5, 0.7, 0.4
        });
        double partialGain = controller.computeObjectGain(partialMatch);
        assertTrue(partialGain > 1.0, "Partial match should have enhancement");
        assertTrue(partialGain <= perfectGain, "Partial gain should be <= perfect");

        // Test poor match - should get minimal enhancement
        var poorMatch = new DenseVector(new double[]{
            0.1, 0.1, 0.9, 0.9,
            0.1, 0.1, 0.9, 0.9
        });
        double poorGain = controller.computeObjectGain(poorMatch);
        assertTrue(poorGain >= 1.0, "Poor match should still have base gain");
        assertTrue(poorGain < partialGain, "Poor gain should be less than partial");
    }

    /**
     * Test 4: Combined Attention - All Mechanisms Together
     *
     * Verifies that spatial, feature, and object attention can be combined
     * to provide integrated attentional modulation.
     *
     * Biological basis: Multiple attention mechanisms operate simultaneously
     * and interact to determine final gain (Baldauf & Desimone, 2014).
     */
    @Test
    public void testCombinedAttention() {
        // Set up all attention types
        int centerX = 16;
        int centerY = 16;
        controller.setAttentionLocation(centerX, centerY);

        var attendedFeature = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        controller.setAttendedFeature(attendedFeature);

        // Compute combined gain at center with matching feature
        var matchingFeature = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        double combinedGain = controller.computeCombinedGain(
            centerX, centerY, matchingFeature
        );

        // Combined gain should reflect both spatial and feature enhancement
        assertTrue(combinedGain > 1.0, "Combined gain should show enhancement");

        // Test that location matters - same feature at distant location
        double distantGain = controller.computeCombinedGain(
            0, 0, matchingFeature
        );
        assertTrue(distantGain < combinedGain,
                   "Distant location should have lower gain despite matching feature");

        // Test that feature matters - center location with non-matching feature
        var nonMatchingFeature = new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0});
        double centerNonMatchGain = controller.computeCombinedGain(
            centerX, centerY, nonMatchingFeature
        );
        assertTrue(centerNonMatchGain < combinedGain,
                   "Non-matching feature at center should have lower gain");

        // Reset and verify gains return to baseline
        controller.reset();
        double resetGain = controller.computeCombinedGain(
            centerX, centerY, matchingFeature
        );
        assertTrue(Math.abs(resetGain - 1.0) < 0.1,
                   "Reset should bring gains close to baseline (1.0)");
    }
}