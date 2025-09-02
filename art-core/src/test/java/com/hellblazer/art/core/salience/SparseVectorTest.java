package com.hellblazer.art.core.salience;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SparseVector Tests")
class SparseVectorTest {

    private static final double EPSILON = 1e-10;
    private SparseVector sparseVector;

    @BeforeEach
    void setUp() {
        sparseVector = new SparseVector(10);
    }

    @Test
    @DisplayName("Should create sparse vector from dense array")
    void testCreateFromDenseArray() {
        double[] denseArray = {1.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3};
        var sparse = new SparseVector(denseArray, 0.01);
        
        assertEquals(10, sparse.getDimension());
        assertEquals(1.0, sparse.get(0), EPSILON);
        assertEquals(0.0, sparse.get(1), EPSILON);
        assertEquals(0.5, sparse.get(2), EPSILON);
        assertEquals(0.8, sparse.get(5), EPSILON);
        assertEquals(0.3, sparse.get(9), EPSILON);
        assertEquals(4, sparse.getNonZeroCount());
    }

    @Test
    @DisplayName("Should handle sparsity threshold correctly")
    void testSparsityThreshold() {
        double[] denseArray = {1.0, 0.001, 0.5, 0.0001, 0.8};
        var sparse = new SparseVector(denseArray, 0.01);
        
        assertEquals(1.0, sparse.get(0), EPSILON);
        assertEquals(0.0, sparse.get(1), EPSILON); // Below threshold
        assertEquals(0.5, sparse.get(2), EPSILON);
        assertEquals(0.0, sparse.get(3), EPSILON); // Below threshold
        assertEquals(0.8, sparse.get(4), EPSILON);
        assertEquals(3, sparse.getNonZeroCount());
    }

    @Test
    @DisplayName("Should get and set elements correctly")
    void testGetAndSet() {
        sparseVector.set(2, 0.5);
        sparseVector.set(7, 0.8);
        
        assertEquals(0.5, sparseVector.get(2), EPSILON);
        assertEquals(0.8, sparseVector.get(7), EPSILON);
        assertEquals(0.0, sparseVector.get(3), EPSILON);
        assertEquals(2, sparseVector.getNonZeroCount());
        
        // Setting to zero should remove element
        sparseVector.set(2, 0.0);
        assertEquals(0.0, sparseVector.get(2), EPSILON);
        assertEquals(1, sparseVector.getNonZeroCount());
    }

    @Test
    @DisplayName("Should perform complement coding correctly")
    void testComplementCoding() {
        double[] values = {0.8, 0.0, 0.3, 0.0, 1.0};
        var sparse = new SparseVector(values, 0.01);
        var complement = sparse.complement();
        
        assertEquals(10, complement.getDimension()); // Double dimension
        
        // Original values
        assertEquals(0.8, complement.get(0), EPSILON);
        assertEquals(0.0, complement.get(1), EPSILON);
        assertEquals(0.3, complement.get(2), EPSILON);
        assertEquals(0.0, complement.get(3), EPSILON);
        assertEquals(1.0, complement.get(4), EPSILON);
        
        // Complement values (1 - original)
        assertEquals(0.2, complement.get(5), EPSILON);
        assertEquals(1.0, complement.get(6), EPSILON);
        assertEquals(0.7, complement.get(7), EPSILON);
        assertEquals(1.0, complement.get(8), EPSILON);
        assertEquals(0.0, complement.get(9), EPSILON);
    }

    @Test
    @DisplayName("Should perform fuzzy AND operation between sparse vectors")
    void testFuzzyAndSparse() {
        double[] values1 = {0.8, 0.0, 0.6, 0.0, 0.9};
        double[] values2 = {0.5, 0.7, 0.8, 0.0, 0.4};
        
        var sparse1 = new SparseVector(values1, 0.01);
        var sparse2 = new SparseVector(values2, 0.01);
        
        var result = sparse1.fuzzyAnd(sparse2);
        
        assertEquals(0.5, result.get(0), EPSILON); // min(0.8, 0.5)
        assertEquals(0.0, result.get(1), EPSILON); // min(0.0, 0.7)
        assertEquals(0.6, result.get(2), EPSILON); // min(0.6, 0.8)
        assertEquals(0.0, result.get(3), EPSILON); // min(0.0, 0.0)
        assertEquals(0.4, result.get(4), EPSILON); // min(0.9, 0.4)
    }

    @Test
    @DisplayName("Should integrate with Pattern interface")
    void testPatternIntegration() {
        double[] values = {0.8, 0.0, 0.3, 0.0, 1.0};
        var sparse = new SparseVector(values, 0.01);
        
        Pattern pattern = sparse.asPattern();
        assertNotNull(pattern);
        assertEquals(5, pattern.dimension());
        
        // Verify pattern values match sparse vector
        for (int i = 0; i < 5; i++) {
            assertEquals(sparse.get(i), pattern.get(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should calculate L1 norm correctly")
    void testL1Norm() {
        double[] values = {0.8, 0.0, 0.3, 0.0, 0.4};
        var sparse = new SparseVector(values, 0.01);
        
        double expectedNorm = 0.8 + 0.3 + 0.4;
        assertEquals(expectedNorm, sparse.normL1(), EPSILON);
    }

    @Test
    @DisplayName("Should calculate L2 norm correctly")
    void testL2Norm() {
        double[] values = {3.0, 0.0, 4.0, 0.0, 0.0};
        var sparse = new SparseVector(values, 0.01);
        
        double expectedNorm = Math.sqrt(9.0 + 16.0);
        assertEquals(expectedNorm, sparse.normL2(), EPSILON);
    }

    @Test
    @DisplayName("Should normalize vector correctly")
    void testNormalization() {
        double[] values = {3.0, 0.0, 4.0, 0.0, 0.0};
        var sparse = new SparseVector(values, 0.01);
        
        var normalizedL1 = sparse.normalizeL1();
        assertEquals(1.0, normalizedL1.normL1(), EPSILON);
        assertEquals(3.0/7.0, normalizedL1.get(0), EPSILON);
        assertEquals(4.0/7.0, normalizedL1.get(2), EPSILON);
        
        var normalizedL2 = sparse.normalizeL2();
        assertEquals(1.0, normalizedL2.normL2(), EPSILON);
        assertEquals(0.6, normalizedL2.get(0), EPSILON);
        assertEquals(0.8, normalizedL2.get(2), EPSILON);
    }

    @Test
    @DisplayName("Should calculate dot product correctly")
    void testDotProduct() {
        double[] values1 = {2.0, 0.0, 3.0, 0.0, 4.0};
        double[] values2 = {1.0, 5.0, 2.0, 0.0, 3.0};
        
        var sparse1 = new SparseVector(values1, 0.01);
        var sparse2 = new SparseVector(values2, 0.01);
        
        double expected = 2.0*1.0 + 3.0*2.0 + 4.0*3.0; // 2 + 6 + 12 = 20
        assertEquals(expected, sparse1.dot(sparse2), EPSILON);
    }

    @Test
    @DisplayName("Should add vectors correctly")
    void testVectorAddition() {
        double[] values1 = {1.0, 0.0, 2.0, 0.0, 3.0};
        double[] values2 = {0.5, 1.0, 0.0, 2.0, 1.0};
        
        var sparse1 = new SparseVector(values1, 0.01);
        var sparse2 = new SparseVector(values2, 0.01);
        
        var result = sparse1.add(sparse2);
        assertEquals(1.5, result.get(0), EPSILON);
        assertEquals(1.0, result.get(1), EPSILON);
        assertEquals(2.0, result.get(2), EPSILON);
        assertEquals(2.0, result.get(3), EPSILON);
        assertEquals(4.0, result.get(4), EPSILON);
    }

    @Test
    @DisplayName("Should multiply by scalar correctly")
    void testScalarMultiplication() {
        double[] values = {2.0, 0.0, 3.0, 0.0, 4.0};
        var sparse = new SparseVector(values, 0.01);
        
        var result = sparse.multiply(2.5);
        assertEquals(5.0, result.get(0), EPSILON);
        assertEquals(0.0, result.get(1), EPSILON);
        assertEquals(7.5, result.get(2), EPSILON);
        assertEquals(0.0, result.get(3), EPSILON);
        assertEquals(10.0, result.get(4), EPSILON);
    }

    @Test
    @DisplayName("Should calculate mean and variance correctly")
    void testMeanAndVariance() {
        double[] values = {2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0};
        var sparse = new SparseVector(values, 0.01);
        
        double expectedMean = 20.0 / 7.0;
        assertEquals(expectedMean, sparse.mean(), EPSILON);
        
        // Variance calculation
        double expectedVariance = sparse.variance();
        assertTrue(expectedVariance > 0);
    }

    @Test
    @DisplayName("Should handle salience weighting")
    void testSalienceWeighting() {
        double[] values = {1.0, 0.0, 0.5, 0.0, 0.8};
        double[] salience = {0.5, 0.1, 0.2, 0.1, 0.1};
        
        var sparse = new SparseVector(values, 0.01);
        var weighted = sparse.applySalience(salience);
        
        assertEquals(0.5, weighted.get(0), EPSILON);  // 1.0 * 0.5
        assertEquals(0.0, weighted.get(1), EPSILON);  // 0.0 * 0.1
        assertEquals(0.1, weighted.get(2), EPSILON);  // 0.5 * 0.2
        assertEquals(0.0, weighted.get(3), EPSILON);  // 0.0 * 0.1
        assertEquals(0.08, weighted.get(4), EPSILON); // 0.8 * 0.1
    }

    @Test
    @DisplayName("Should return non-zero indices correctly")
    void testNonZeroIndices() {
        double[] values = {0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0};
        var sparse = new SparseVector(values, 0.01);
        
        Set<Integer> indices = sparse.getNonZeroIndices();
        assertEquals(3, indices.size());
        assertTrue(indices.contains(1));
        assertTrue(indices.contains(3));
        assertTrue(indices.contains(6));
        assertFalse(indices.contains(0));
    }

    @Test
    @DisplayName("Should calculate sparsity ratio correctly")
    void testSparsityRatio() {
        double[] values = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0};
        var sparse = new SparseVector(values, 0.01);
        
        double sparsityRatio = sparse.getSparsityRatio();
        assertEquals(0.3, sparsityRatio, EPSILON); // 3 non-zero out of 10
    }

    @Test
    @DisplayName("Should handle thread-safe read operations")
    void testThreadSafetyRead() throws InterruptedException {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var sparse = new SparseVector(values, 0.01);
        
        int threadCount = 10;
        CountDownLatch latch = new CountDownLatch(threadCount);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        
        for (int i = 0; i < threadCount; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 1000; j++) {
                        double sum = 0;
                        for (int k = 0; k < 5; k++) {
                            sum += sparse.get(k);
                        }
                        assertEquals(15.0, sum, EPSILON);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await();
        executor.shutdown();
    }

    @Test
    @DisplayName("Should handle large sparse vectors efficiently")
    void testLargeSparseVector() {
        int dimension = 100000;
        var sparse = new SparseVector(dimension);
        
        // Set only a few elements
        sparse.set(100, 1.0);
        sparse.set(5000, 2.0);
        sparse.set(50000, 3.0);
        sparse.set(99999, 4.0);
        
        assertEquals(4, sparse.getNonZeroCount());
        assertEquals(1.0, sparse.get(100), EPSILON);
        assertEquals(2.0, sparse.get(5000), EPSILON);
        assertEquals(3.0, sparse.get(50000), EPSILON);
        assertEquals(4.0, sparse.get(99999), EPSILON);
        assertEquals(0.0, sparse.get(25000), EPSILON);
        
        // Memory efficiency - should use much less than dense array
        assertTrue(sparse.getMemoryUsage() < dimension * 8);
    }

    @Test
    @DisplayName("Should convert to dense array correctly")
    void testToDenseArray() {
        double[] values = {1.0, 0.0, 2.0, 0.0, 3.0};
        var sparse = new SparseVector(values, 0.01);
        
        double[] dense = sparse.toDenseArray();
        assertArrayEquals(values, dense, EPSILON);
    }

    @Test
    @DisplayName("Should handle edge cases correctly")
    void testEdgeCases() {
        // Empty vector
        var empty = new SparseVector(5);
        assertEquals(0, empty.getNonZeroCount());
        assertEquals(0.0, empty.normL1(), EPSILON);
        assertEquals(0.0, empty.normL2(), EPSILON);
        
        // Single element
        var single = new SparseVector(1);
        single.set(0, 5.0);
        assertEquals(5.0, single.get(0), EPSILON);
        assertEquals(1, single.getNonZeroCount());
        
        // All zeros
        double[] zeros = new double[10];
        var allZeros = new SparseVector(zeros, 0.01);
        assertEquals(0, allZeros.getNonZeroCount());
        
        // All non-zeros
        double[] ones = new double[5];
        Arrays.fill(ones, 1.0);
        var allOnes = new SparseVector(ones, 0.01);
        assertEquals(5, allOnes.getNonZeroCount());
    }
}