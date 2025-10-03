package com.hellblazer.art.cortical.memory;

import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for WeightMatrixPool - Phase 4E.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Basic rent/return functionality</li>
 *   <li>Thread safety under concurrent access</li>
 *   <li>Pool size limits and behavior</li>
 *   <li>Memory usage estimation</li>
 * </ul>
 *
 * @author Phase 4E: Memory Optimization
 */
class WeightMatrixPoolTest {

    private static final int ROWS = 128;
    private static final int COLS = 256;

    private WeightMatrixPool pool;

    @BeforeEach
    void setup() {
        pool = new WeightMatrixPool(ROWS, COLS);
    }

    @Test
    void testRentFromEmptyPool() {
        var matrix = pool.rent();

        assertNotNull(matrix);
        assertEquals(ROWS, matrix.getRows());
        assertEquals(COLS, matrix.getCols());
    }

    @Test
    void testReturnAndReuse() {
        // Rent matrix
        var matrix1 = pool.rent();
        matrix1.set(0, 0, 42.0);

        // Return it
        pool.returnMatrix(matrix1);

        // Rent again - should get same instance
        var matrix2 = pool.rent();

        assertSame(matrix1, matrix2, "Should reuse pooled matrix");
        assertEquals(42.0, matrix2.get(0, 0), "Matrix should retain data (not cleared)");
    }

    @Test
    void testRentZeroed() {
        // Create matrix with data
        var matrix1 = pool.rent();
        matrix1.set(0, 0, 42.0);
        matrix1.set(10, 10, 99.0);

        // Return it
        pool.returnMatrix(matrix1);

        // Rent zeroed - should clear data
        var matrix2 = pool.rentZeroed();

        assertEquals(0.0, matrix2.get(0, 0), "Should be zeroed");
        assertEquals(0.0, matrix2.get(10, 10), "Should be zeroed");
    }

    @Test
    void testMultipleRentReturn() {
        var matrix1 = pool.rent();
        var matrix2 = pool.rent();
        var matrix3 = pool.rent();

        assertNotSame(matrix1, matrix2);
        assertNotSame(matrix2, matrix3);
        assertNotSame(matrix1, matrix3);

        // Return all
        pool.returnMatrix(matrix1);
        pool.returnMatrix(matrix2);
        pool.returnMatrix(matrix3);

        assertEquals(3, pool.getPoolSize());
    }

    @Test
    void testMaxPoolSize() {
        var maxSize = 4;
        var limitedPool = new WeightMatrixPool(ROWS, COLS, maxSize);

        // Create more matrices than pool size
        var matrices = new ArrayList<WeightMatrix>();
        for (int i = 0; i < maxSize + 2; i++) {
            matrices.add(limitedPool.rent());
        }

        // Return all
        for (var matrix : matrices) {
            limitedPool.returnMatrix(matrix);
        }

        // Pool should not exceed max size
        assertTrue(limitedPool.getPoolSize() <= maxSize,
            "Pool size should not exceed max: " + limitedPool.getPoolSize());
    }

    @Test
    void testReturnNullMatrixTolerated() {
        // Should not throw
        assertDoesNotThrow(() -> pool.returnMatrix(null));
    }

    @Test
    void testReturnWrongDimensionsThrows() {
        var wrongMatrix = new WeightMatrix(64, 64);

        assertThrows(IllegalArgumentException.class,
            () -> pool.returnMatrix(wrongMatrix),
            "Should reject matrix with wrong dimensions");
    }

    @Test
    void testPrewarm() {
        var targetSize = 5;
        pool.prewarm(targetSize);

        assertEquals(targetSize, pool.getPoolSize(),
            "Should have " + targetSize + " matrices after prewarm");

        // All should be immediately available
        for (int i = 0; i < targetSize; i++) {
            assertNotNull(pool.rent());
        }

        assertEquals(0, pool.getPoolSize(),
            "Pool should be empty after renting all prewarmed matrices");
    }

    @Test
    void testPrewarmRespectsMaxSize() {
        var maxSize = 3;
        var limitedPool = new WeightMatrixPool(ROWS, COLS, maxSize);

        limitedPool.prewarm(10);  // Request more than max

        assertEquals(maxSize, limitedPool.getPoolSize(),
            "Prewarm should respect max pool size");
    }

    @Test
    void testClear() {
        pool.prewarm(5);
        assertEquals(5, pool.getPoolSize());

        pool.clear();

        assertEquals(0, pool.getPoolSize(), "Pool should be empty after clear");
    }

    @Test
    void testThreadSafety() throws InterruptedException {
        var threadCount = 8;
        var operationsPerThread = 1000;
        var latch = new CountDownLatch(threadCount);
        var errorCount = new AtomicInteger(0);

        // Multiple threads rent/return concurrently
        var threads = new Thread[threadCount];
        for (int t = 0; t < threadCount; t++) {
            threads[t] = new Thread(() -> {
                try {
                    for (int i = 0; i < operationsPerThread; i++) {
                        var matrix = pool.rent();
                        // Simulate some work
                        matrix.set(0, 0, i);
                        pool.returnMatrix(matrix);
                    }
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                } finally {
                    latch.countDown();
                }
            });
            threads[t].start();
        }

        // Wait for all threads
        latch.await();

        assertEquals(0, errorCount.get(), "Should have no errors in concurrent access");
    }

    @Test
    void testMemoryEstimation() {
        pool.prewarm(10);

        var estimatedMemory = pool.estimateMemoryUsage();

        // Each matrix: 128 * 256 * 8 = 262,144 bytes + 24 bytes overhead
        var expectedPerMatrix = ROWS * COLS * 8L + 24L;
        var expectedTotal = expectedPerMatrix * 10;

        assertEquals(expectedTotal, estimatedMemory,
            "Memory estimation should be accurate");
    }

    @Test
    void testGetDimensions() {
        var dims = pool.getDimensions();

        assertEquals(2, dims.length);
        assertEquals(ROWS, dims[0]);
        assertEquals(COLS, dims[1]);
    }

    @Test
    void testGetMaxPoolSize() {
        var maxSize = 20;
        var customPool = new WeightMatrixPool(ROWS, COLS, maxSize);

        assertEquals(maxSize, customPool.getMaxPoolSize());
    }

    @Test
    void testToString() {
        pool.prewarm(5);

        var str = pool.toString();

        assertTrue(str.contains("128x256"));
        assertTrue(str.contains("poolSize=5"));
    }

    @Test
    void testInvalidConstructorParameters() {
        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(0, 10),
            "Zero rows should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(10, 0),
            "Zero cols should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(-1, 10),
            "Negative rows should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(10, -1),
            "Negative cols should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(10, 10, 0),
            "Zero maxPoolSize should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new WeightMatrixPool(10, 10, -1),
            "Negative maxPoolSize should throw");
    }

    @Test
    void testRentReturnCycle() {
        // Simulate typical usage pattern
        for (int cycle = 0; cycle < 100; cycle++) {
            var matrix = pool.rent();
            matrix.set(0, 0, cycle);
            pool.returnMatrix(matrix);
        }

        // Pool should have exactly 1 matrix (reused 100 times)
        assertEquals(1, pool.getPoolSize(),
            "Should reuse same matrix across cycles");
    }

    @Test
    void testConcurrentRentWithPrewarm() throws InterruptedException {
        pool.prewarm(4);

        var threadCount = 4;
        var latch = new CountDownLatch(threadCount);
        var rentedMatrices = new ArrayList<WeightMatrix>();

        // Each thread rents once
        for (int t = 0; t < threadCount; t++) {
            new Thread(() -> {
                var matrix = pool.rent();
                synchronized (rentedMatrices) {
                    rentedMatrices.add(matrix);
                }
                latch.countDown();
            }).start();
        }

        latch.await();

        // All rented matrices should be distinct
        assertEquals(threadCount, rentedMatrices.size());
        for (int i = 0; i < rentedMatrices.size(); i++) {
            for (int j = i + 1; j < rentedMatrices.size(); j++) {
                assertNotSame(rentedMatrices.get(i), rentedMatrices.get(j),
                    "Matrices should be distinct instances");
            }
        }
    }
}
