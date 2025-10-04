package com.hellblazer.art.cortical.temporal;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ItemNode class.
 * Part of LIST PARSE temporal chunking system.
 */
class ItemNodeTest {

    @Test
    void testItemNodeCreation() {
        var pattern = new double[]{0.1, 0.2, 0.3, 0.4, 0.5};
        var strength = 0.8;
        var position = 2;
        var creationTime = 100.0;

        var node = new ItemNode(pattern, strength, position, creationTime);

        assertEquals(strength, node.getStrength());
        assertEquals(position, node.getPosition());
        assertEquals(creationTime, node.getCreationTime());
        assertEquals(1, node.getAccessCount());
        assertArrayEquals(pattern, node.getPattern());
    }

    @Test
    void testPatternMatching() {
        var original = new double[]{0.5, 0.5, 0.5};
        var node = new ItemNode(original, 1.0, 0, 0.0);

        // Exact match
        assertTrue(node.matches(original, 0.99));

        // Similar pattern
        var similar = new double[]{0.49, 0.51, 0.5};
        assertTrue(node.matches(similar, 0.95));

        // Different pattern
        var different = new double[]{1.0, 0.0, 0.0};
        assertFalse(node.matches(different, 0.9));

        // Wrong dimension
        var wrongDim = new double[]{0.5, 0.5};
        assertFalse(node.matches(wrongDim, 0.5));
    }

    @Test
    void testCosineSimilarity() {
        // Orthogonal vectors
        var pattern1 = new double[]{1.0, 0.0, 0.0};
        var pattern2 = new double[]{0.0, 1.0, 0.0};
        var node = new ItemNode(pattern1, 1.0, 0, 0.0);
        assertFalse(node.matches(pattern2, 0.1));

        // Parallel vectors
        var pattern3 = new double[]{2.0, 0.0, 0.0};
        assertTrue(node.matches(pattern3, 0.99));

        // Anti-parallel vectors
        var pattern4 = new double[]{-1.0, 0.0, 0.0};
        assertFalse(node.matches(pattern4, 0.1));
    }

    @Test
    void testStrengthening() {
        var pattern = new double[]{0.1, 0.2, 0.3};
        var initialStrength = 0.5;
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
    void testDecay() {
        var pattern = new double[]{0.5, 0.5, 0.5};
        var node = new ItemNode(pattern, 1.0, 0, 0.0);

        var initialTime = 0.0;
        var currentTime = 10.0;
        var decayRate = 0.1;

        // Apply decay
        node.decay(decayRate, currentTime);

        // Strength should decrease
        assertTrue(node.getStrength() < 1.0);
        assertTrue(node.getStrength() > 0.0);

        // Verify exponential decay formula
        var expectedStrength = 1.0 * Math.exp(-decayRate * (currentTime - node.getLastAccessTime()));
        assertEquals(expectedStrength, node.getStrength(), 0.001);
    }

    @Test
    void testPatternImmutability() {
        var pattern = new double[]{0.1, 0.2, 0.3};
        var node = new ItemNode(pattern, 1.0, 0, 0.0);

        // Modify original pattern
        pattern[0] = 0.9;

        // Node pattern should be unchanged
        var nodePattern = node.getPattern();
        assertEquals(0.1, nodePattern[0], 0.001);

        // Modify returned pattern
        nodePattern[1] = 0.9;

        // Node pattern should still be unchanged
        var nodePattern2 = node.getPattern();
        assertEquals(0.2, nodePattern2[1], 0.001);
    }

    @Test
    void testZeroPattern() {
        var zeroPattern = new double[]{0.0, 0.0, 0.0};
        var node = new ItemNode(zeroPattern, 0.5, 0, 0.0);

        // Should not match with non-zero pattern
        var nonZero = new double[]{0.1, 0.1, 0.1};
        assertFalse(node.matches(nonZero, 0.9));

        // Should match with another zero pattern
        var anotherZero = new double[]{0.0, 0.0, 0.0};
        assertFalse(node.matches(anotherZero, 0.9)); // Cosine similarity undefined for zero vectors
    }

    @Test
    void testNormalizedPatterns() {
        // Test with normalized patterns (unit vectors)
        var norm = Math.sqrt(3.0);
        var pattern1 = new double[]{1.0/norm, 1.0/norm, 1.0/norm};
        var node = new ItemNode(pattern1, 1.0, 0, 0.0);

        // Same normalized pattern
        assertTrue(node.matches(pattern1, 0.999));

        // Scaled version (should match due to cosine similarity)
        var scaled = new double[]{2.0/norm, 2.0/norm, 2.0/norm};
        assertTrue(node.matches(scaled, 0.999));
    }
}
