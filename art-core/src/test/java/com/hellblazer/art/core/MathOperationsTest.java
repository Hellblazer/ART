package com.hellblazer.art.core;

import com.hellblazer.art.core.utils.MathOperations;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test the mathematical operations required for TopoART.
 */
class MathOperationsTest {
    
    @Test
    @DisplayName("Complement coding transformation")
    void testComplementCoding() {
        var input = new double[]{0.3, 0.7, 0.5};
        var coded = MathOperations.complementCode(input);
        
        assertEquals(6, coded.length);
        assertEquals(0.3, coded[0], 1e-10);
        assertEquals(0.7, coded[1], 1e-10);
        assertEquals(0.5, coded[2], 1e-10);
        assertEquals(0.7, coded[3], 1e-10);  // 1 - 0.3
        assertEquals(0.3, coded[4], 1e-10);  // 1 - 0.7
        assertEquals(0.5, coded[5], 1e-10);  // 1 - 0.5
    }
    
    @Test
    @DisplayName("Component-wise minimum operation")
    void testComponentWiseMin() {
        var a = new double[]{0.3, 0.8, 0.2, 0.9};
        var b = new double[]{0.5, 0.6, 0.4, 0.7};
        var result = MathOperations.componentWiseMin(a, b);
        
        var expected = new double[]{0.3, 0.6, 0.2, 0.7};
        assertArrayEquals(expected, result, 1e-10);
    }
    
    @Test
    @DisplayName("City block norm (L1 norm)")
    void testCityBlockNorm() {
        var vector = new double[]{0.3, -0.2, 0.5, -0.1};
        var norm = MathOperations.cityBlockNorm(vector);
        
        assertEquals(1.1, norm, 1e-10);
    }
}