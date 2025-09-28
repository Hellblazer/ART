package com.hellblazer.art.temporal.masking;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ItemNode class.
 */
public class ItemNodeTest {

    @Test
    public void testItemNodeCreation() {
        double[] pattern = {0.1, 0.2, 0.3, 0.4, 0.5};
        double strength = 0.8;
        int position = 2;
        double creationTime = 100.0;

        var node = new ItemNode(pattern, strength, position, creationTime);

        assertEquals(strength, node.getStrength());
        assertEquals(position, node.getPosition());
        assertEquals(creationTime, node.getCreationTime());
        assertEquals(1, node.getAccessCount());
        assertArrayEquals(pattern, node.getPattern());
    }

    @Test
    public void testPatternMatching() {
        double[] original = {0.5, 0.5, 0.5};
        var node = new ItemNode(original, 1.0, 0, 0.0);

        // Exact match
        assertTrue(node.matches(original, 0.99));

        // Similar pattern
        double[] similar = {0.49, 0.51, 0.5};
        assertTrue(node.matches(similar, 0.95));

        // Different pattern
        double[] different = {1.0, 0.0, 0.0};
        assertFalse(node.matches(different, 0.9));

        // Wrong dimension
        double[] wrongDim = {0.5, 0.5};
        assertFalse(node.matches(wrongDim, 0.5));
    }

    @Test
    public void testCosineSimilarity() {
        // Orthogonal vectors
        double[] pattern1 = {1.0, 0.0, 0.0};
        double[] pattern2 = {0.0, 1.0, 0.0};
        var node = new ItemNode(pattern1, 1.0, 0, 0.0);
        assertFalse(node.matches(pattern2, 0.1));

        // Parallel vectors
        double[] pattern3 = {2.0, 0.0, 0.0};
        assertTrue(node.matches(pattern3, 0.99));

        // Anti-parallel vectors
        double[] pattern4 = {-1.0, 0.0, 0.0};
        assertFalse(node.matches(pattern4, 0.1));
    }

    @Test
    public void testStrengthening() {
        double[] pattern = {0.1, 0.2, 0.3};
        double initialStrength = 0.5;
        var node = new ItemNode(pattern, initialStrength, 0, 0.0);

        assertEquals(1, node.getAccessCount());

        // Strengthen node
        node.strengthen(0.3);
        assertEquals(0.8, node.getStrength(), 0.001);
        assertEquals(2, node.getAccessCount());

        // Strengthen again
        node.strengthen(0.2);
        assertEquals(1.0, node.getStrength(), 0.001);
        assertEquals(3, node.getAccessCount());
    }

    @Test
    public void testDecay() {
        double[] pattern = {0.5, 0.5, 0.5};
        var node = new ItemNode(pattern, 1.0, 0, 0.0);

        double initialTime = 0.0;
        double currentTime = 10.0;
        double decayRate = 0.1;

        // Apply decay
        node.decay(decayRate, currentTime);

        // Strength should decrease
        assertTrue(node.getStrength() < 1.0);
        assertTrue(node.getStrength() > 0.0);

        // Verify exponential decay formula
        double expectedStrength = 1.0 * Math.exp(-decayRate * (currentTime - node.getLastAccessTime()));
        assertEquals(expectedStrength, node.getStrength(), 0.001);
    }

    @Test
    public void testPatternImmutability() {
        double[] pattern = {0.1, 0.2, 0.3};
        var node = new ItemNode(pattern, 1.0, 0, 0.0);

        // Modify original pattern
        pattern[0] = 0.9;

        // Node pattern should be unchanged
        double[] nodePattern = node.getPattern();
        assertEquals(0.1, nodePattern[0], 0.001);

        // Modify returned pattern
        nodePattern[1] = 0.9;

        // Node pattern should still be unchanged
        double[] nodePattern2 = node.getPattern();
        assertEquals(0.2, nodePattern2[1], 0.001);
    }

    @Test
    public void testZeroPattern() {
        double[] zeroPattern = {0.0, 0.0, 0.0};
        var node = new ItemNode(zeroPattern, 0.5, 0, 0.0);

        // Should not match with non-zero pattern
        double[] nonZero = {0.1, 0.1, 0.1};
        assertFalse(node.matches(nonZero, 0.9));

        // Should match with another zero pattern
        double[] anotherZero = {0.0, 0.0, 0.0};
        assertFalse(node.matches(anotherZero, 0.9)); // Cosine similarity undefined for zero vectors
    }

    @Test
    public void testNormalizedPatterns() {
        // Test with normalized patterns (unit vectors)
        double norm = Math.sqrt(3.0);
        double[] pattern1 = {1.0/norm, 1.0/norm, 1.0/norm};
        var node = new ItemNode(pattern1, 1.0, 0, 0.0);

        // Same normalized pattern
        assertTrue(node.matches(pattern1, 0.999));

        // Scaled version (should match due to cosine similarity)
        double[] scaled = {2.0/norm, 2.0/norm, 2.0/norm};
        assertTrue(node.matches(scaled, 0.999));
    }
}