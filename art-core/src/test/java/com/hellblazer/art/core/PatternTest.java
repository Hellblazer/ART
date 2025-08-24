package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Pattern interface and implementations.
 * Tests all fundamental pattern operations required for ART algorithms.
 */
class PatternTest {
    
    @Test
    @DisplayName("Dense vector creation and basic operations")
    void testDenseVectorBasics() {
        var data = new double[]{3.0, -4.0, 0.0, 5.0};
        var vector = Pattern.of(data);
        
        assertEquals(4, vector.dimension());
        assertEquals(3.0, vector.get(0), 1e-10);
        assertEquals(-4.0, vector.get(1), 1e-10);
        assertEquals(0.0, vector.get(2), 1e-10);
        assertEquals(5.0, vector.get(3), 1e-10);
    }
    
    @Test
    @DisplayName("L1 norm calculation")
    void testL1Norm() {
        var vector = Pattern.of(3.0, -4.0, 0.0, 5.0);
        assertEquals(12.0, vector.l1Norm(), 1e-10);
    }
    
    @Test
    @DisplayName("L2 norm calculation")
    void testL2Norm() {
        var vector = Pattern.of(3.0, 4.0, 0.0);
        assertEquals(5.0, vector.l2Norm(), 1e-10);
    }
    
    @Test
    @DisplayName("Vector normalization with data bounds")
    void testNormalization() {
        var vector = Pattern.of(2.0, 4.0, 6.0);
        var bounds = DataBounds.of(
            new double[]{0.0, 2.0, 4.0}, 
            new double[]{4.0, 6.0, 8.0}
        );
        
        var normalized = vector.normalize(bounds);
        
        assertEquals(0.5, normalized.get(0), 1e-10);
        assertEquals(0.5, normalized.get(1), 1e-10);
        assertEquals(0.5, normalized.get(2), 1e-10);
    }
    
    @Test
    @DisplayName("Vector normalization with zero range")
    void testNormalizationWithZeroRange() {
        var vector = Pattern.of(3.0, 3.0);
        var bounds = DataBounds.of(
            new double[]{3.0, 2.0}, 
            new double[]{3.0, 6.0}  // First dimension has zero range
        );
        
        var normalized = vector.normalize(bounds);
        
        assertEquals(0.0, normalized.get(0), 1e-10);  // Zero range -> 0
        assertEquals(0.25, normalized.get(1), 1e-10); // (3-2)/(6-2) = 0.25
    }
    
    @Test
    @DisplayName("Vector element-wise minimum operation")
    void testMin() {
        var a = Pattern.of(0.8, 0.3, 0.9);
        var b = Pattern.of(0.6, 0.7, 0.2);
        
        var result = a.min(b);
        
        assertEquals(0.6, result.get(0), 1e-10);
        assertEquals(0.3, result.get(1), 1e-10);
        assertEquals(0.2, result.get(2), 1e-10);
    }
    
    @Test
    @DisplayName("Vector element-wise maximum operation")
    void testMax() {
        var a = Pattern.of(0.8, 0.3, 0.9);
        var b = Pattern.of(0.6, 0.7, 0.2);
        
        var result = a.max(b);
        
        assertEquals(0.8, result.get(0), 1e-10);
        assertEquals(0.7, result.get(1), 1e-10);
        assertEquals(0.9, result.get(2), 1e-10);
    }
    
    @Test
    @DisplayName("Vector scaling operation")
    void testScale() {
        var vector = Pattern.of(2.0, -3.0, 4.0);
        var scaled = vector.scale(0.5);
        
        assertEquals(1.0, scaled.get(0), 1e-10);
        assertEquals(-1.5, scaled.get(1), 1e-10);
        assertEquals(2.0, scaled.get(2), 1e-10);
    }
    
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 5, 10, 100})
    @DisplayName("Vector dimension property")
    void testVectorDimensions(int dimension) {
        var data = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            data[i] = Math.random();
        }
        
        var vector = Pattern.of(data);
        assertEquals(dimension, vector.dimension());
    }
    
    @Test
    @DisplayName("Index bounds checking")
    void testIndexBounds() {
        var vector = Pattern.of(1.0, 2.0, 3.0);
        
        assertThrows(IndexOutOfBoundsException.class, () -> vector.get(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> vector.get(3));
    }
    
    @Test
    @DisplayName("Null data handling")
    void testNullData() {
        assertThrows(NullPointerException.class, () -> Pattern.of((double[]) null));
    }
    
    @Test
    @DisplayName("Empty vector handling")
    void testEmptyVector() {
        assertThrows(IllegalArgumentException.class, () -> Pattern.of(new double[0]));
    }
    
    @Test
    @DisplayName("Vector immutability")
    void testVectorImmutability() {
        var originalData = new double[]{1.0, 2.0, 3.0};
        var vector = Pattern.of(originalData);
        
        // Modify original array
        originalData[0] = 999.0;
        
        // Pattern should be unaffected
        assertEquals(1.0, vector.get(0), 1e-10);
    }
    
    @Test
    @DisplayName("Dimension mismatch in binary operations")
    void testDimensionMismatch() {
        var a = Pattern.of(1.0, 2.0);
        var b = Pattern.of(1.0, 2.0, 3.0);
        
        assertThrows(IllegalArgumentException.class, () -> a.min(b));
        assertThrows(IllegalArgumentException.class, () -> a.max(b));
    }
    
    @Test
    @DisplayName("Vector equality and hashcode")
    void testEqualityAndHashCode() {
        var a = Pattern.of(1.0, 2.0, 3.0);
        var b = Pattern.of(1.0, 2.0, 3.0);
        var c = Pattern.of(1.0, 2.0, 4.0);
        
        assertEquals(a, b);
        assertNotEquals(a, c);
        assertEquals(a.hashCode(), b.hashCode());
    }
}