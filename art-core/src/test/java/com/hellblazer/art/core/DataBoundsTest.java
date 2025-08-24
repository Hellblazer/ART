package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for DataBounds record.
 * Tests bounds checking and validation for vector normalization.
 */
class DataBoundsTest {
    
    @Test
    @DisplayName("Data bounds creation and basic operations")
    void testDataBoundsBasics() {
        var min = new double[]{0.0, -1.0, 5.0};
        var max = new double[]{10.0, 1.0, 15.0};
        var bounds = DataBounds.of(min, max);
        
        assertEquals(3, bounds.dimension());
        assertEquals(0.0, bounds.min(0), 1e-10);
        assertEquals(-1.0, bounds.min(1), 1e-10);
        assertEquals(5.0, bounds.min(2), 1e-10);
        assertEquals(10.0, bounds.max(0), 1e-10);
        assertEquals(1.0, bounds.max(1), 1e-10);
        assertEquals(15.0, bounds.max(2), 1e-10);
    }
    
    @Test
    @DisplayName("Range calculation")
    void testRange() {
        var bounds = DataBounds.of(
            new double[]{0.0, -5.0, 10.0},
            new double[]{10.0, 5.0, 10.0}  // Zero range in third dimension
        );
        
        assertEquals(10.0, bounds.range(0), 1e-10);
        assertEquals(10.0, bounds.range(1), 1e-10);
        assertEquals(0.0, bounds.range(2), 1e-10);
    }
    
    @Test
    @DisplayName("Value containment check")
    void testContains() {
        var bounds = DataBounds.of(
            new double[]{0.0, -1.0},
            new double[]{10.0, 1.0}
        );
        
        assertTrue(bounds.contains(Vector.of(5.0, 0.0)));
        assertTrue(bounds.contains(Vector.of(0.0, -1.0))); // Boundary values
        assertTrue(bounds.contains(Vector.of(10.0, 1.0)));
        assertFalse(bounds.contains(Vector.of(-1.0, 0.0))); // Below minimum
        assertFalse(bounds.contains(Vector.of(11.0, 0.0))); // Above maximum
        assertFalse(bounds.contains(Vector.of(5.0, 2.0))); // Above maximum
    }
    
    @Test
    @DisplayName("Bounds expansion to include new vectors")
    void testExpand() {
        var bounds = DataBounds.of(
            new double[]{0.0, 0.0},
            new double[]{5.0, 5.0}
        );
        
        var newPoint = Vector.of(-2.0, 7.0);
        var expanded = bounds.expand(newPoint);
        
        assertEquals(-2.0, expanded.min(0), 1e-10);
        assertEquals(0.0, expanded.min(1), 1e-10);
        assertEquals(5.0, expanded.max(0), 1e-10);
        assertEquals(7.0, expanded.max(1), 1e-10);
    }
    
    @Test
    @DisplayName("Bounds expansion with already contained vector")
    void testExpandWithContainedVector() {
        var bounds = DataBounds.of(
            new double[]{0.0, 0.0},
            new double[]{5.0, 5.0}
        );
        
        var containedPoint = Vector.of(2.0, 3.0);
        var expanded = bounds.expand(containedPoint);
        
        // Should be unchanged
        assertEquals(bounds, expanded);
    }
    
    @Test
    @DisplayName("Null input handling")
    void testNullInputs() {
        assertThrows(NullPointerException.class, 
            () -> DataBounds.of(null, new double[]{1.0}));
        assertThrows(NullPointerException.class, 
            () -> DataBounds.of(new double[]{0.0}, null));
        
        var bounds = DataBounds.of(new double[]{0.0}, new double[]{1.0});
        assertThrows(NullPointerException.class, 
            () -> bounds.contains(null));
        assertThrows(NullPointerException.class, 
            () -> bounds.expand(null));
    }
    
    @Test
    @DisplayName("Dimension mismatch handling")
    void testDimensionMismatch() {
        assertThrows(IllegalArgumentException.class,
            () -> DataBounds.of(new double[]{0.0}, new double[]{1.0, 2.0}));
            
        var bounds = DataBounds.of(new double[]{0.0, 0.0}, new double[]{1.0, 1.0});
        assertThrows(IllegalArgumentException.class,
            () -> bounds.contains(Vector.of(0.5))); // Wrong dimension
    }
    
    @Test
    @DisplayName("Empty bounds handling")
    void testEmptyBounds() {
        assertThrows(IllegalArgumentException.class,
            () -> DataBounds.of(new double[0], new double[0]));
    }
    
    @Test
    @DisplayName("Invalid bounds (min > max)")
    void testInvalidBounds() {
        assertThrows(IllegalArgumentException.class,
            () -> DataBounds.of(new double[]{5.0}, new double[]{0.0}));
        assertThrows(IllegalArgumentException.class,
            () -> DataBounds.of(new double[]{0.0, 5.0}, new double[]{1.0, 0.0}));
    }
    
    @Test
    @DisplayName("Index bounds checking")
    void testIndexBounds() {
        var bounds = DataBounds.of(new double[]{0.0, 1.0}, new double[]{2.0, 3.0});
        
        assertThrows(IndexOutOfBoundsException.class, () -> bounds.min(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> bounds.max(2));
        assertThrows(IndexOutOfBoundsException.class, () -> bounds.range(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> bounds.range(2));
    }
    
    @Test
    @DisplayName("Data bounds equality and hashcode")
    void testEqualityAndHashCode() {
        var bounds1 = DataBounds.of(new double[]{0.0, 1.0}, new double[]{2.0, 3.0});
        var bounds2 = DataBounds.of(new double[]{0.0, 1.0}, new double[]{2.0, 3.0});
        var bounds3 = DataBounds.of(new double[]{0.0, 1.0}, new double[]{2.0, 4.0});
        
        assertEquals(bounds1, bounds2);
        assertNotEquals(bounds1, bounds3);
        assertEquals(bounds1.hashCode(), bounds2.hashCode());
    }
    
    @Test
    @DisplayName("Bounds immutability")
    void testBoundsImmutability() {
        var originalMin = new double[]{0.0, 1.0};
        var originalMax = new double[]{2.0, 3.0};
        var bounds = DataBounds.of(originalMin, originalMax);
        
        // Modify original arrays
        originalMin[0] = 999.0;
        originalMax[0] = 999.0;
        
        // Bounds should be unaffected
        assertEquals(0.0, bounds.min(0), 1e-10);
        assertEquals(2.0, bounds.max(0), 1e-10);
    }
}