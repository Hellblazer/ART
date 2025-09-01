package com.hellblazer.art.core.preprocessing;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class DataPreprocessorTest {
    
    private DataPreprocessor preprocessor;
    
    @BeforeEach
    void setUp() {
        preprocessor = new DataPreprocessor();
    }
    
    @Test
    void testNormalizeMinMax() {
        var data = new double[][] {
            {0.0, 10.0, 100.0},
            {5.0, 20.0, 200.0},
            {10.0, 30.0, 300.0}
        };
        
        var result = preprocessor.normalize(data);
        
        assertNotNull(result);
        assertEquals(3, result.normalized().length);
        assertEquals(3, result.normalized()[0].length);
        
        // Check normalization to [0,1]
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, result.normalized()[0], 0.001);
        assertArrayEquals(new double[]{0.5, 0.5, 0.5}, result.normalized()[1], 0.001);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, result.normalized()[2], 0.001);
        
        // Check stored min/max values
        assertArrayEquals(new double[]{0.0, 10.0, 100.0}, result.min(), 0.001);
        assertArrayEquals(new double[]{10.0, 30.0, 300.0}, result.max(), 0.001);
    }
    
    @Test
    void testNormalizeWithProvidedBounds() {
        var data = new double[][] {
            {2.0, 15.0},
            {4.0, 25.0}
        };
        var min = new double[]{0.0, 10.0};
        var max = new double[]{10.0, 30.0};
        
        var result = preprocessor.normalize(data, min, max);
        
        assertArrayEquals(new double[]{0.2, 0.25}, result.normalized()[0], 0.001);
        assertArrayEquals(new double[]{0.4, 0.75}, result.normalized()[1], 0.001);
    }
    
    @Test
    void testDenormalize() {
        var normalized = new double[][] {
            {0.0, 0.5},
            {1.0, 1.0}
        };
        var min = new double[]{10.0, 20.0};
        var max = new double[]{20.0, 40.0};
        
        var result = preprocessor.denormalize(normalized, min, max);
        
        assertArrayEquals(new double[]{10.0, 30.0}, result[0], 0.001);
        assertArrayEquals(new double[]{20.0, 40.0}, result[1], 0.001);
    }
    
    @Test
    void testHandleConstantColumns() {
        var data = new double[][] {
            {5.0, 10.0, 7.0},
            {5.0, 20.0, 7.0},
            {5.0, 30.0, 7.0}
        };
        
        var result = preprocessor.normalize(data);
        
        // Constant columns should be set to 0
        assertEquals(0.0, result.normalized()[0][0], 0.001);
        assertEquals(0.0, result.normalized()[1][0], 0.001);
        assertEquals(0.0, result.normalized()[2][0], 0.001);
        
        // Constant last column
        assertEquals(0.0, result.normalized()[0][2], 0.001);
        assertEquals(0.0, result.normalized()[1][2], 0.001);
        assertEquals(0.0, result.normalized()[2][2], 0.001);
        
        // Variable middle column normalized properly
        assertEquals(0.0, result.normalized()[0][1], 0.001);
        assertEquals(0.5, result.normalized()[1][1], 0.001);
        assertEquals(1.0, result.normalized()[2][1], 0.001);
    }
    
    @Test
    void testComplementCoding() {
        var data = new double[][] {
            {0.2, 0.8},
            {0.5, 0.3}
        };
        
        var result = preprocessor.complementCode(data);
        
        assertEquals(2, result.length);
        assertEquals(4, result[0].length);
        
        // Original values followed by complements
        assertArrayEquals(new double[]{0.2, 0.8, 0.8, 0.2}, result[0], 0.001);
        assertArrayEquals(new double[]{0.5, 0.3, 0.5, 0.7}, result[1], 0.001);
    }
    
    @Test
    void testDeComplementCoding() {
        var complementCoded = new double[][] {
            {0.2, 0.8, 0.8, 0.2},
            {0.5, 0.3, 0.5, 0.7}
        };
        
        var result = preprocessor.deComplementCode(complementCoded);
        
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        
        // Should average the original and inverted complement
        assertArrayEquals(new double[]{0.2, 0.8}, result[0], 0.001);
        assertArrayEquals(new double[]{0.5, 0.3}, result[1], 0.001);
    }
    
    @Test
    void testL1Normalization() {
        var data = new double[][] {
            {3.0, 4.0},    // L1 norm = 7
            {1.0, 1.0}     // L1 norm = 2
        };
        
        var result = preprocessor.l1Normalize(data);
        
        assertArrayEquals(new double[]{3.0/7.0, 4.0/7.0}, result[0], 0.001);
        assertArrayEquals(new double[]{0.5, 0.5}, result[1], 0.001);
    }
    
    @Test
    void testL2Normalization() {
        var data = new double[][] {
            {3.0, 4.0},    // L2 norm = 5
            {0.0, 2.0}     // L2 norm = 2
        };
        
        var result = preprocessor.l2Normalize(data);
        
        assertArrayEquals(new double[]{0.6, 0.8}, result[0], 0.001);
        assertArrayEquals(new double[]{0.0, 1.0}, result[1], 0.001);
    }
    
    @Test
    void testHandleMissingValues() {
        var data = new double[][] {
            {1.0, Double.NaN, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, Double.NaN}
        };
        
        var result = preprocessor.handleMissingValues(data, MissingValueStrategy.MEAN);
        
        // NaN in column 1 should be replaced with mean of 5.0 and 8.0 = 6.5
        assertEquals(6.5, result[0][1], 0.001);
        
        // NaN in column 2 should be replaced with mean of 3.0 and 6.0 = 4.5
        assertEquals(4.5, result[2][2], 0.001);
    }
    
    @Test
    void testHandleMissingValuesWithZero() {
        var data = new double[][] {
            {1.0, Double.NaN, 3.0},
            {4.0, 5.0, Double.NaN}
        };
        
        var result = preprocessor.handleMissingValues(data, MissingValueStrategy.ZERO);
        
        assertEquals(0.0, result[0][1], 0.001);
        assertEquals(0.0, result[1][2], 0.001);
    }
    
    @Test
    void testBatchProcessing() {
        var batch1 = new double[][] {
            {0.0, 10.0},
            {5.0, 20.0}
        };
        var batch2 = new double[][] {
            {10.0, 30.0},
            {2.0, 15.0}
        };
        
        // First, find bounds across all batches
        var bounds = preprocessor.findBounds(batch1, batch2);
        assertEquals(0.0, bounds.min()[0], 0.001);
        assertEquals(10.0, bounds.max()[0], 0.001);
        assertEquals(10.0, bounds.min()[1], 0.001);
        assertEquals(30.0, bounds.max()[1], 0.001);
        
        // Then normalize each batch with the same bounds
        var result1 = preprocessor.normalize(batch1, bounds.min(), bounds.max());
        var result2 = preprocessor.normalize(batch2, bounds.min(), bounds.max());
        
        // Check batch1 normalization
        assertArrayEquals(new double[]{0.0, 0.0}, result1.normalized()[0], 0.001);
        assertArrayEquals(new double[]{0.5, 0.5}, result1.normalized()[1], 0.001);
        
        // Check batch2 normalization
        assertArrayEquals(new double[]{1.0, 1.0}, result2.normalized()[0], 0.001);
        assertArrayEquals(new double[]{0.2, 0.25}, result2.normalized()[1], 0.001);
    }
    
    @Test
    void testPipeline() {
        var data = new double[][] {
            {10.0, Double.NaN, 30.0},
            {20.0, 50.0, 60.0},
            {30.0, 70.0, 90.0}
        };
        
        var pipeline = preprocessor.createPipeline()
            .addStep(PreprocessingStep.HANDLE_MISSING, MissingValueStrategy.MEAN)
            .addStep(PreprocessingStep.NORMALIZE)
            .addStep(PreprocessingStep.COMPLEMENT_CODE)
            .build();
        
        var result = pipeline.process(data);
        
        // Should handle missing, normalize, then complement code
        assertEquals(3, result.length);
        assertEquals(6, result[0].length); // Doubled due to complement coding
        
        // All values should be in [0,1]
        for (var row : result) {
            for (var val : row) {
                assertTrue(val >= 0.0 && val <= 1.0);
            }
        }
    }
    
    @Test
    void testEmptyData() {
        var data = new double[0][0];
        
        assertThrows(IllegalArgumentException.class, () -> {
            preprocessor.normalize(data);
        });
    }
    
    @Test
    void testSingleRow() {
        var data = new double[][] {
            {5.0, 10.0, 15.0}
        };
        
        var result = preprocessor.normalize(data);
        
        // Single row should result in all zeros (no variation)
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, result.normalized()[0], 0.001);
    }
}