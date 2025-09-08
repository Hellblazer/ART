package com.hellblazer.art.nlp.processor.optimization;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;
import com.hellblazer.art.nlp.processor.optimization.PerformanceOptimizer.OptimizationConfig;
import com.hellblazer.art.nlp.processor.optimization.PerformanceOptimizer.OptimizationStrategy;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Map;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for PerformanceOptimizer.
 */
public class PerformanceOptimizerTest {
    
    private PerformanceOptimizer optimizer;
    private OptimizationConfig defaultConfig;
    
    @BeforeEach
    void setUp() {
        defaultConfig = OptimizationConfig.defaultConfig();
        optimizer = new PerformanceOptimizer(defaultConfig);
    }
    
    @Test
    void testBasicOptimization() {
        var inputText = "This is a test sentence for optimization";
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.8, 120),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80)
        );
        
        var optimized = optimizer.optimizeChannelProcessing(inputText, channelResults);
        
        assertThat(optimized).isNotNull();
        assertThat(optimized).hasSize(3);
        assertThat(optimized.keySet()).containsExactlyInAnyOrder("semantic", "entity", "syntactic");
        
        // Verify performance stats are updated
        var stats = optimizer.getPerformanceStats();
        assertThat(stats.channelMetrics()).hasSize(3);
        assertThat(stats.cacheStats().currentSize()).isEqualTo(1);
    }
    
    @Test
    void testCacheHitOptimization() {
        var inputText = "Same input text";
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        
        // First call - should cache
        var result1 = optimizer.optimizeChannelProcessing(inputText, channelResults);
        var stats1 = optimizer.getPerformanceStats();
        
        // Second call - should hit cache
        var result2 = optimizer.optimizeChannelProcessing(inputText, channelResults);
        var stats2 = optimizer.getPerformanceStats();
        
        assertThat(result1).isEqualTo(result2);
        assertThat(stats2.cacheStats().hits()).isEqualTo(stats1.cacheStats().hits() + 1);
        assertThat(stats2.cacheStats().getHitRate()).isGreaterThan(0.0);
    }
    
    @Test
    void testConsensusStrategyRecommendation() {
        // Many channels with high variance - should recommend Hierarchical
        var manyChannelsHighVariance = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.95, 100),
            "entity", ChannelResult.success("entity", 2, 0.6, 120),
            "syntactic", ChannelResult.success("syntactic", 1, 0.4, 80),
            "lexical", ChannelResult.success("lexical", 3, 0.8, 90),
            "phonetic", ChannelResult.success("phonetic", 2, 0.3, 110)
        );
        
        var recommendation1 = optimizer.recommendConsensusStrategy(manyChannelsHighVariance);
        assertThat(recommendation1).isEqualTo("HierarchicalConsensus");
        
        // High confidence channels - should recommend Attention
        var highConfidenceChannels = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.85, 120),
            "syntactic", ChannelResult.success("syntactic", 1, 0.88, 80)
        );
        
        var recommendation2 = optimizer.recommendConsensusStrategy(highConfidenceChannels);
        assertThat(recommendation2).isEqualTo("AttentionConsensus");
        
        // Few channels - should recommend Weighted
        var fewChannels = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.7, 100)
        );
        
        var recommendation3 = optimizer.recommendConsensusStrategy(fewChannels);
        assertThat(recommendation3).isEqualTo("WeightedVoting");
    }
    
    @Test
    void testFusionStrategyRecommendation() {
        // Many channels - should recommend PCA
        var manyChannels = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.8, 120),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80),
            "lexical", ChannelResult.success("lexical", 3, 0.6, 90)
        );
        
        var recommendation1 = optimizer.recommendFusionStrategy(manyChannels);
        assertThat(recommendation1).isEqualTo("PCA");
        
        // Medium channels - should recommend Attention
        var mediumChannels = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.8, 120)
        );
        
        var recommendation2 = optimizer.recommendFusionStrategy(mediumChannels);
        assertThat(recommendation2).isEqualTo("AttentionFusion");
        
        // Single channel - should recommend Simple
        var singleChannel = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        
        var recommendation3 = optimizer.recommendFusionStrategy(singleChannel);
        assertThat(recommendation3).isEqualTo("SimpleConcat");
    }
    
    @Test
    void testConsensusPerformanceRecording() {
        var result = ConsensusResult.create(1, 0.85, "TestStrategy", Map.of("test", 1.0));
        
        optimizer.recordConsensusPerformance("TestStrategy", result, 150);
        
        var stats = optimizer.getPerformanceStats();
        assertThat(stats.strategyMetrics()).containsKey("TestStrategy");
        
        var strategyStats = stats.strategyMetrics().get("TestStrategy");
        assertThat(strategyStats.getTotalExecutions()).isEqualTo(1);
        assertThat(strategyStats.getAverageConfidence()).isEqualTo(0.85);
        assertThat(strategyStats.getAverageProcessingTime()).isEqualTo(150.0);
    }
    
    @Test
    void testFusionPerformanceRecording() {
        var result = new double[]{0.1, 0.2, 0.3}; // Mock fusion result
        
        optimizer.recordFusionPerformance("TestFusion", result, 200);
        
        var stats = optimizer.getPerformanceStats();
        assertThat(stats.strategyMetrics()).containsKey("Fusion_TestFusion");
        
        var fusionStats = stats.strategyMetrics().get("Fusion_TestFusion");
        assertThat(fusionStats.getTotalExecutions()).isEqualTo(1);
        assertThat(fusionStats.getAverageConfidence()).isEqualTo(1.0); // Success
        assertThat(fusionStats.getAverageProcessingTime()).isEqualTo(200.0);
    }
    
    @Test
    void testChannelMetricsAccumulation() {
        var channelResults1 = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        var channelResults2 = Map.of(
            "semantic", ChannelResult.success("semantic", 2, 0.8, 120)
        );
        
        optimizer.optimizeChannelProcessing("text1", channelResults1);
        optimizer.optimizeChannelProcessing("text2", channelResults2);
        
        var stats = optimizer.getPerformanceStats();
        var semanticMetrics = stats.channelMetrics().get("semantic");
        
        assertThat(semanticMetrics.getTotalExecutions()).isEqualTo(2);
        assertThat(semanticMetrics.getSuccessfulExecutions()).isEqualTo(2);
        assertThat(semanticMetrics.getSuccessRate()).isEqualTo(1.0);
        assertThat(semanticMetrics.getAverageConfidence()).isCloseTo(0.85, within(1e-10)); // (0.9 + 0.8) / 2
        assertThat(semanticMetrics.getAverageProcessingTime()).isEqualTo(110.0); // (100 + 120) / 2
        assertThat(semanticMetrics.getMinProcessingTime()).isEqualTo(100.0);
        assertThat(semanticMetrics.getMaxProcessingTime()).isEqualTo(120.0);
    }
    
    @Test
    void testFailedChannelMetrics() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "failed", ChannelResult.failed("failed", "Error", 50)
        );
        
        optimizer.optimizeChannelProcessing("test", channelResults);
        
        var stats = optimizer.getPerformanceStats();
        var failedMetrics = stats.channelMetrics().get("failed");
        
        assertThat(failedMetrics.getTotalExecutions()).isEqualTo(1);
        assertThat(failedMetrics.getSuccessfulExecutions()).isEqualTo(0);
        assertThat(failedMetrics.getSuccessRate()).isEqualTo(0.0);
        assertThat(failedMetrics.getAverageConfidence()).isEqualTo(0.0);
    }
    
    @Test
    void testCacheLRUEviction() {
        var smallCacheConfig = new OptimizationConfig(true, 2, 300000, true, true);
        var smallOptimizer = new PerformanceOptimizer(smallCacheConfig);
        
        var result1 = Map.of("ch1", ChannelResult.success("ch1", 1, 0.9, 100));
        var result2 = Map.of("ch2", ChannelResult.success("ch2", 2, 0.8, 120));
        var result3 = Map.of("ch3", ChannelResult.success("ch3", 3, 0.7, 80));
        
        smallOptimizer.optimizeChannelProcessing("text1", result1);
        smallOptimizer.optimizeChannelProcessing("text2", result2);
        
        var stats1 = smallOptimizer.getPerformanceStats();
        assertThat(stats1.cacheStats().currentSize()).isEqualTo(2);
        
        // This should evict the oldest entry
        smallOptimizer.optimizeChannelProcessing("text3", result3);
        
        var stats2 = smallOptimizer.getPerformanceStats();
        assertThat(stats2.cacheStats().currentSize()).isEqualTo(2); // Still maxed out
    }
    
    @Test
    void testClearOperations() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        
        optimizer.optimizeChannelProcessing("test", channelResults);
        optimizer.recordConsensusPerformance("TestStrategy", 
            ConsensusResult.create(1, 0.8, "Test", Map.of("test", 1.0)), 100);
        
        var statsBefore = optimizer.getPerformanceStats();
        assertThat(statsBefore.cacheStats().currentSize()).isGreaterThan(0);
        assertThat(statsBefore.channelMetrics()).isNotEmpty();
        assertThat(statsBefore.strategyMetrics()).isNotEmpty();
        
        optimizer.clearCache();
        var statsAfterCacheClear = optimizer.getPerformanceStats();
        assertThat(statsAfterCacheClear.cacheStats().currentSize()).isEqualTo(0);
        assertThat(statsAfterCacheClear.cacheStats().hits()).isEqualTo(0);
        assertThat(statsAfterCacheClear.cacheStats().misses()).isEqualTo(0);
        
        optimizer.clearMetrics();
        var statsAfterMetricsClear = optimizer.getPerformanceStats();
        assertThat(statsAfterMetricsClear.channelMetrics()).isEmpty();
        assertThat(statsAfterMetricsClear.strategyMetrics()).isEmpty();
    }
    
    @Test
    void testOptimizationStrategyManagement() {
        assertThat(optimizer.getOptimizationStrategy()).isEqualTo(OptimizationStrategy.ADAPTIVE);
        
        optimizer.setOptimizationStrategy(OptimizationStrategy.CACHE_ONLY);
        assertThat(optimizer.getOptimizationStrategy()).isEqualTo(OptimizationStrategy.CACHE_ONLY);
        
        var stats = optimizer.getPerformanceStats();
        assertThat(stats.currentStrategy()).isEqualTo(OptimizationStrategy.CACHE_ONLY);
    }
    
    @Test
    void testDisabledOptimization() {
        var disabledConfig = OptimizationConfig.disabledConfig();
        var disabledOptimizer = new PerformanceOptimizer(disabledConfig);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        
        var result1 = disabledOptimizer.optimizeChannelProcessing("test1", channelResults);
        var result2 = disabledOptimizer.optimizeChannelProcessing("test1", channelResults); // Same input
        
        var stats = disabledOptimizer.getPerformanceStats();
        assertThat(stats.cacheStats().currentSize()).isEqualTo(0);
        assertThat(stats.cacheStats().hits()).isEqualTo(0);
        
        // Results should still be processed
        assertThat(result1).isEqualTo(result2);
    }
    
    @Test
    void testCacheOnlyConfig() {
        var cacheOnlyConfig = OptimizationConfig.cacheOnlyConfig(500, 60000);
        
        assertThat(cacheOnlyConfig.enableCaching()).isTrue();
        assertThat(cacheOnlyConfig.maxCacheSize()).isEqualTo(500);
        assertThat(cacheOnlyConfig.cacheExpirationMs()).isEqualTo(60000);
        assertThat(cacheOnlyConfig.enableMetrics()).isTrue();
        assertThat(cacheOnlyConfig.enableAdaptiveStrategies()).isFalse();
    }
    
    @Test
    void testConfigurationValidation() {
        assertThatThrownBy(() -> new OptimizationConfig(true, -1, 1000, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("maxCacheSize must be non-negative");
            
        assertThatThrownBy(() -> new OptimizationConfig(true, 100, -1, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("cacheExpirationMs must be non-negative");
    }
    
    @Test
    void testNullInputValidation() {
        assertThatThrownBy(() -> new PerformanceOptimizer(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("config cannot be null");
            
        assertThatThrownBy(() -> optimizer.optimizeChannelProcessing(null, Map.of()))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("inputText cannot be null");
            
        assertThatThrownBy(() -> optimizer.optimizeChannelProcessing("test", null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("channelResults cannot be null");
            
        assertThatThrownBy(() -> optimizer.recommendConsensusStrategy(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("channelResults cannot be null");
            
        assertThatThrownBy(() -> optimizer.recommendFusionStrategy(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("channelResults cannot be null");
            
        assertThatThrownBy(() -> optimizer.setOptimizationStrategy(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("strategy cannot be null");
    }
    
    @Test
    void testCacheStatsCalculations() {
        var stats = new PerformanceOptimizer.CacheStats(80, 20, 500, 1000);
        
        assertThat(stats.getHitRate()).isEqualTo(0.8); // 80 / (80 + 20)
        assertThat(stats.getUtilization()).isEqualTo(0.5); // 500 / 1000
        
        var emptyStats = new PerformanceOptimizer.CacheStats(0, 0, 0, 1000);
        assertThat(emptyStats.getHitRate()).isEqualTo(0.0);
        assertThat(emptyStats.getUtilization()).isEqualTo(0.0);
        
        var maxStats = new PerformanceOptimizer.CacheStats(100, 0, 0, 0);
        assertThat(maxStats.getHitRate()).isEqualTo(1.0);
        assertThat(maxStats.getUtilization()).isEqualTo(0.0);
    }
}