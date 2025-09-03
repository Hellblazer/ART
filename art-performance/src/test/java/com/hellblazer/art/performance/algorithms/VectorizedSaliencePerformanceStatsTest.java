package com.hellblazer.art.performance.algorithms;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSaliencePerformanceStats
 */
class VectorizedSaliencePerformanceStatsTest {
    
    @Test
    @DisplayName("Test empty stats creation")
    void testEmptyStats() {
        var stats = VectorizedSaliencePerformanceStats.empty();
        
        assertNotNull(stats);
        assertEquals(0L, stats.totalOperations());
        assertEquals(0L, stats.simdOperations());
        assertEquals(0.0, stats.averageProcessingTime());
        assertEquals(0.0, stats.averageSalienceComputationTime());
        assertEquals(0L, stats.statisticsUpdateCount());
        assertEquals(0.0, stats.averageCategoryUtilization());
        assertNotNull(stats.categorySalienceScores());
        assertTrue(stats.categorySalienceScores().isEmpty());
        assertEquals(0L, stats.sparseVectorOperations());
        assertEquals(0.0, stats.memoryEfficiencyRatio());
    }
    
    @Test
    @DisplayName("Test stats with data")
    void testStatsWithData() {
        Map<Integer, Double> salienceScores = new HashMap<>();
        salienceScores.put(0, 0.8);
        salienceScores.put(1, 0.6);
        salienceScores.put(2, 0.9);
        
        var stats = new VectorizedSaliencePerformanceStats(
            1000L,      // totalOperations
            800L,       // simdOperations
            1.5,        // averageProcessingTime (ms)
            0.3,        // averageSalienceComputationTime (ms)
            500L,       // statisticsUpdateCount
            0.75,       // averageCategoryUtilization
            salienceScores,
            200L,       // sparseVectorOperations
            0.85        // memoryEfficiencyRatio
        );
        
        assertEquals(1000L, stats.totalOperations());
        assertEquals(800L, stats.simdOperations());
        assertEquals(1.5, stats.averageProcessingTime());
        assertEquals(0.3, stats.averageSalienceComputationTime());
        assertEquals(500L, stats.statisticsUpdateCount());
        assertEquals(0.75, stats.averageCategoryUtilization());
        assertEquals(3, stats.categorySalienceScores().size());
        assertEquals(0.8, stats.categorySalienceScores().get(0));
        assertEquals(200L, stats.sparseVectorOperations());
        assertEquals(0.85, stats.memoryEfficiencyRatio());
    }
    
    @Test
    @DisplayName("Test stats merge operation")
    void testStatsMerge() {
        var stats1 = new VectorizedSaliencePerformanceStats(
            100L, 80L, 1.0, 0.2, 50L, 0.7,
            Map.of(0, 0.5, 1, 0.6),
            20L, 0.8
        );
        
        var stats2 = new VectorizedSaliencePerformanceStats(
            200L, 160L, 2.0, 0.4, 100L, 0.8,
            Map.of(1, 0.7, 2, 0.9),
            40L, 0.9
        );
        
        var merged = stats1.merge(stats2);
        
        assertNotNull(merged);
        assertEquals(300L, merged.totalOperations());
        assertEquals(240L, merged.simdOperations());
        // Average should be weighted
        assertTrue(merged.averageProcessingTime() > 1.0);
        assertTrue(merged.averageProcessingTime() < 2.0);
        assertEquals(150L, merged.statisticsUpdateCount());
        assertEquals(60L, merged.sparseVectorOperations());
        // Should have entries from both maps
        assertEquals(3, merged.categorySalienceScores().size());
    }
    
    @Test
    @DisplayName("Test SIMD utilization ratio calculation")
    void testSimdUtilizationRatio() {
        var stats = new VectorizedSaliencePerformanceStats(
            1000L, 750L, 1.0, 0.2, 100L, 0.5,
            Map.of(), 100L, 0.9
        );
        
        assertEquals(0.75, stats.getSimdUtilizationRatio());
    }
    
    @Test
    @DisplayName("Test salience overhead calculation")
    void testSalienceOverhead() {
        var stats = new VectorizedSaliencePerformanceStats(
            1000L, 500L, 2.0, 0.5, 100L, 0.5,
            Map.of(), 100L, 0.9
        );
        
        // Salience computation time as percentage of total processing time
        assertEquals(0.25, stats.getSalienceOverheadRatio());
    }
    
    @Test
    @DisplayName("Test sparse efficiency metrics")
    void testSparseEfficiency() {
        var stats = new VectorizedSaliencePerformanceStats(
            1000L, 500L, 2.0, 0.5, 100L, 0.5,
            Map.of(), 200L, 0.85
        );
        
        assertEquals(0.2, stats.getSparseOperationRatio());
        assertEquals(0.85, stats.memoryEfficiencyRatio());
    }
    
    @Test
    @DisplayName("Test toString provides useful information")
    void testToString() {
        var stats = VectorizedSaliencePerformanceStats.empty();
        var str = stats.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("totalOperations"));
        assertTrue(str.contains("simdOperations"));
        assertTrue(str.contains("memoryEfficiencyRatio"));
    }
    
    @Test
    @DisplayName("Test immutability of salience scores map")
    void testSalienceScoresImmutability() {
        Map<Integer, Double> mutableMap = new HashMap<>();
        mutableMap.put(0, 0.5);
        
        var stats = new VectorizedSaliencePerformanceStats(
            100L, 50L, 1.0, 0.2, 10L, 0.5,
            mutableMap, 10L, 0.9
        );
        
        // Modifying original map should not affect stats
        mutableMap.put(1, 0.7);
        assertEquals(1, stats.categorySalienceScores().size());
        
        // Returned map should be immutable
        assertThrows(UnsupportedOperationException.class, () -> {
            stats.categorySalienceScores().put(2, 0.8);
        });
    }
}