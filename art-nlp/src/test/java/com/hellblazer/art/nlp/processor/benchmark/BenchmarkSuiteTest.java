package com.hellblazer.art.nlp.processor.benchmark;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;
import com.hellblazer.art.nlp.processor.benchmark.BenchmarkSuite.BenchmarkConfig;
import com.hellblazer.art.nlp.processor.benchmark.BenchmarkSuite.BenchmarkType;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for BenchmarkSuite.
 */
public class BenchmarkSuiteTest {
    
    private BenchmarkSuite benchmarkSuite;
    private BenchmarkConfig fastConfig;
    
    @BeforeEach
    void setUp() {
        fastConfig = BenchmarkConfig.fastConfig();
        benchmarkSuite = new BenchmarkSuite(fastConfig);
    }
    
    @AfterEach
    void tearDown() {
        benchmarkSuite.shutdown();
    }
    
    @Test
    void testConsensusBenchmark() throws ExecutionException, InterruptedException {
        // Create a simple consensus function that always returns a result
        Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction = (Map<String, ChannelResult> channelResults) -> {
            var firstSuccess = channelResults.values().stream()
                .filter(ChannelResult::isSuccess)
                .findFirst();
            
            // Always return a result, even if no channels succeeded
            var category = firstSuccess.map(ChannelResult::category).orElse(0);
            var confidence = firstSuccess.map(ChannelResult::confidence).orElse(0.5);
            
            var contributions = new HashMap<String, Double>();
            var totalWeight = (double) channelResults.size();
            channelResults.forEach((id, result) -> 
                contributions.put(id, result.isSuccess() ? 1.0 / totalWeight : 0.0));
            
            return ConsensusResult.create(category, confidence, "TestConsensus", contributions);
        };
        
        // Generate test data with custom config to ensure some successes
        var noFailConfig = new BenchmarkConfig(2, List.of(5), 42L, 0.0, 3); // 0% failure rate
        var noFailSuite = new BenchmarkSuite(noFailConfig);
        var testData = noFailSuite.generateSyntheticData(20, "semantic", "entity", "syntactic");
        noFailSuite.shutdown();
        
        // Run benchmark
        var future = benchmarkSuite.benchmarkConsensus("TestStrategy", consensusFunction, testData);
        var result = future.get();
        
        assertThat(result).isNotNull();
        assertThat(result.strategyName()).isEqualTo("TestStrategy");
        assertThat(result.benchmarkType()).isEqualTo(BenchmarkType.CONSENSUS);
        assertThat(result.iterationCount()).isEqualTo(20);
        assertThat(result.totalTime()).isGreaterThan(Duration.ZERO);
        assertThat(result.performanceMetrics()).hasSize(20);
        
        // Should have some accuracy metrics for successful consensus results
        assertThat(result.accuracyMetrics()).isNotEmpty();
        
        // Verify statistics
        var stats = benchmarkSuite.getStatistics();
        assertThat(stats.strategyStatistics()).containsKey("consensus_TestStrategy");
    }
    
    @Test
    void testFusionBenchmark() throws ExecutionException, InterruptedException {
        // Create a simple fusion function that concatenates successful channel vectors
        Function<Map<String, ChannelResult>, Object> fusionFunction = (Map<String, ChannelResult> channelResults) -> {
            var successfulChannels = channelResults.values().stream()
                .filter(ChannelResult::isSuccess)
                .toList();
            
            if (successfulChannels.isEmpty()) {
                return null;
            }
            
            // Simple fusion: create a vector with one element per successful channel
            var result = new double[successfulChannels.size()];
            for (int i = 0; i < successfulChannels.size(); i++) {
                result[i] = successfulChannels.get(i).confidence();
            }
            return result;
        };
        
        // Generate test data
        var testData = benchmarkSuite.generateSyntheticData(15, "semantic", "entity");
        
        // Run benchmark
        var future = benchmarkSuite.benchmarkFusion("TestFusion", fusionFunction, testData);
        var result = future.get();
        
        assertThat(result).isNotNull();
        assertThat(result.strategyName()).isEqualTo("TestFusion");
        assertThat(result.benchmarkType()).isEqualTo(BenchmarkType.FUSION);
        assertThat(result.iterationCount()).isEqualTo(15);
        assertThat(result.performanceMetrics()).hasSize(15);
        assertThat(result.fusionMetrics()).isNotEmpty();
        
        // Verify fusion-specific metrics
        var fusionMetric = result.fusionMetrics().getFirst();
        assertThat(fusionMetric.resultDimensionality()).isGreaterThan(0);
        assertThat(fusionMetric.compressionRatio()).isGreaterThanOrEqualTo(0.0);
        assertThat(fusionMetric.channelUtilization()).isBetween(0.0, 1.0);
    }
    
    @Test
    void testScalabilityBenchmark() throws ExecutionException, InterruptedException {
        // Simple strategy function
        Function<Map<String, ChannelResult>, Object> strategyFunction = (Map<String, ChannelResult> channelResults) -> {
            return channelResults.values().stream()
                .filter(ChannelResult::isSuccess)
                .mapToDouble(ChannelResult::confidence)
                .average()
                .orElse(0.0);
        };
        
        // Data generator
        Supplier<Map<String, ChannelResult>> dataGenerator = () -> Map.of(
            "test", ChannelResult.success("test", 1, 0.8, 100)
        );
        
        // Run scalability benchmark
        var future = benchmarkSuite.benchmarkScalability("ScalableStrategy", strategyFunction, dataGenerator);
        var result = future.get();
        
        assertThat(result).isNotNull();
        assertThat(result.strategyName()).isEqualTo("ScalableStrategy");
        assertThat(result.scalabilityPoints()).isNotEmpty();
        
        // Verify scalability points are ordered by size
        var sizes = result.scalabilityPoints().stream()
            .mapToInt(BenchmarkSuite.ScalabilityPoint::dataSize)
            .toArray();
        
        for (int i = 1; i < sizes.length; i++) {
            assertThat(sizes[i]).isGreaterThan(sizes[i-1]);
        }
        
        // Verify each point has valid metrics
        for (var point : result.scalabilityPoints()) {
            assertThat(point.dataSize()).isGreaterThan(0);
            assertThat(point.totalTime()).isGreaterThan(Duration.ZERO);
            assertThat(point.throughput()).isGreaterThanOrEqualTo(0.0);
            assertThat(point.successRate()).isBetween(0.0, 1.0);
        }
    }
    
    @Test
    void testStrategyComparison() throws ExecutionException, InterruptedException {
        // Create multiple strategies to compare
        Map<String, Function<Map<String, ChannelResult>, ?>> strategies = Map.of(
            "Fast", (Function<Map<String, ChannelResult>, String>) channelResults -> "fast_result",
            "Slow", (Function<Map<String, ChannelResult>, String>) channelResults -> {
                try { Thread.sleep(1); } catch (InterruptedException e) { /* ignore */ }
                return "slow_result";
            },
            "Failing", (Function<Map<String, ChannelResult>, String>) channelResults -> { throw new RuntimeException("Simulated failure"); }
        );
        
        var testData = benchmarkSuite.generateSyntheticData(10, "semantic", "entity");
        
        // Run comparison
        var future = benchmarkSuite.compareStrategies(strategies, testData);
        var result = future.get();
        
        assertThat(result).isNotNull();
        assertThat(result.strategyMetrics()).hasSize(3);
        assertThat(result.strategyMetrics().keySet()).containsExactlyInAnyOrder("Fast", "Slow", "Failing");
        
        // Verify each strategy has metrics for all test cases
        for (var entry : result.strategyMetrics().entrySet()) {
            var strategyName = entry.getKey();
            var metrics = entry.getValue();
            
            if (!strategyName.equals("Failing")) {
                assertThat(metrics).hasSize(10);
            }
            
            // Verify metrics structure
            for (var metric : metrics) {
                assertThat(metric.strategyName()).isEqualTo(strategyName);
                assertThat(metric.processingTime()).isGreaterThanOrEqualTo(Duration.ZERO);
            }
        }
    }
    
    @Test
    void testSyntheticDataGeneration() {
        var data = benchmarkSuite.generateSyntheticData(100, "ch1", "ch2", "ch3");
        
        assertThat(data).hasSize(100);
        
        for (var channelResults : data) {
            assertThat(channelResults).hasSize(3);
            assertThat(channelResults.keySet()).containsExactlyInAnyOrder("ch1", "ch2", "ch3");
            
            for (var result : channelResults.values()) {
                assertThat(result.channelId()).isIn("ch1", "ch2", "ch3");
                assertThat(result.processingTimeMs()).isGreaterThan(0);
                
                if (result.isSuccess()) {
                    assertThat(result.confidence()).isBetween(0.5, 1.0);
                    assertThat(result.category()).isBetween(0, fastConfig.syntheticCategoryCount() - 1);
                }
            }
        }
        
        // Verify failure rate is approximately correct
        var totalResults = data.stream()
            .flatMap(map -> map.values().stream())
            .toList();
        
        var failureRate = (double) totalResults.stream()
            .mapToInt(result -> result.isFailure() ? 1 : 0)
            .sum() / totalResults.size();
        
        // Should be approximately the configured failure rate (with some tolerance)
        assertThat(failureRate).isBetween(0.0, 0.2); // Fast config has 0.05 failure rate
    }
    
    @Test
    void testBenchmarkStatistics() throws ExecutionException, InterruptedException {
        // Run a simple benchmark first
        var testData = benchmarkSuite.generateSyntheticData(5, "test");
        Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction = (Map<String, ChannelResult> cr) -> 
            ConsensusResult.create(1, 0.8, "Test", Map.of("test", 1.0));
        
        var future = benchmarkSuite.benchmarkConsensus("StatTest", consensusFunction, testData);
        future.get(); // Wait for completion
        
        // Get statistics
        var stats = benchmarkSuite.getStatistics();
        
        assertThat(stats).isNotNull();
        assertThat(stats.totalBenchmarks()).isEqualTo(1);
        assertThat(stats.strategyStatistics()).containsKey("consensus_StatTest");
        assertThat(stats.generatedAt()).isNotNull();
        
        var strategyStats = stats.strategyStatistics().get("consensus_StatTest");
        assertThat(strategyStats.strategyName()).isEqualTo("StatTest");
        assertThat(strategyStats.benchmarkType()).isEqualTo(BenchmarkType.CONSENSUS);
        assertThat(strategyStats.totalIterations()).isEqualTo(5);
        assertThat(strategyStats.averageProcessingTime()).isGreaterThan(Duration.ZERO);
        assertThat(strategyStats.successRate()).isBetween(0.0, 1.0);
        assertThat(strategyStats.throughput()).isGreaterThanOrEqualTo(0.0);
    }
    
    @Test
    void testResultExport() throws ExecutionException, InterruptedException {
        // Run a benchmark to have data to export
        var testData = benchmarkSuite.generateSyntheticData(3, "test");
        Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction = (Map<String, ChannelResult> cr) -> 
            ConsensusResult.create(1, 0.9, "Export", Map.of("test", 1.0));
        
        var future = benchmarkSuite.benchmarkConsensus("ExportTest", consensusFunction, testData);
        future.get();
        
        // Export results
        var exported = benchmarkSuite.exportResults();
        
        assertThat(exported).containsKey("consensus_ExportTest");
        assertThat(exported).containsKey("exportTime");
        assertThat(exported).containsKey("totalBenchmarks");
        assertThat(exported.get("totalBenchmarks")).isEqualTo(1);
        
        // Verify benchmark result structure
        @SuppressWarnings("unchecked")
        var benchmarkData = (Map<String, Object>) exported.get("consensus_ExportTest");
        assertThat(benchmarkData).containsKey("strategyName");
        assertThat(benchmarkData).containsKey("benchmarkType");
        assertThat(benchmarkData).containsKey("iterationCount");
        assertThat(benchmarkData.get("strategyName")).isEqualTo("ExportTest");
    }
    
    @Test
    void testClearResults() throws ExecutionException, InterruptedException {
        // Run a benchmark
        var testData = benchmarkSuite.generateSyntheticData(2, "test");
        Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction = (Map<String, ChannelResult> cr) -> 
            ConsensusResult.create(1, 0.7, "Clear", Map.of("test", 1.0));
        
        var future = benchmarkSuite.benchmarkConsensus("ClearTest", consensusFunction, testData);
        future.get();
        
        // Verify data exists
        var statsBefore = benchmarkSuite.getStatistics();
        assertThat(statsBefore.totalBenchmarks()).isEqualTo(1);
        
        // Clear results
        benchmarkSuite.clearResults();
        
        // Verify data is cleared
        var statsAfter = benchmarkSuite.getStatistics();
        assertThat(statsAfter.totalBenchmarks()).isEqualTo(0);
        assertThat(statsAfter.strategyStatistics()).isEmpty();
    }
    
    @Test
    void testDifferentBenchmarkConfigurations() {
        var defaultConfig = BenchmarkConfig.defaultConfig();
        var thoroughConfig = BenchmarkConfig.thoroughConfig();
        
        // Default config
        assertThat(defaultConfig.maxConcurrentBenchmarks()).isEqualTo(Runtime.getRuntime().availableProcessors());
        assertThat(defaultConfig.scalabilityTestSizes()).hasSize(5);
        assertThat(defaultConfig.syntheticFailureRate()).isEqualTo(0.1);
        
        // Thorough config
        assertThat(thoroughConfig.scalabilityTestSizes()).hasSize(8);
        assertThat(thoroughConfig.syntheticFailureRate()).isEqualTo(0.15);
        assertThat(thoroughConfig.syntheticCategoryCount()).isEqualTo(10);
        
        // Fast config (already used in tests)
        assertThat(fastConfig.scalabilityTestSizes()).hasSize(3);
        assertThat(fastConfig.syntheticFailureRate()).isEqualTo(0.05);
    }
    
    @Test
    void testConfigurationValidation() {
        // Test invalid max concurrent benchmarks
        assertThatThrownBy(() -> new BenchmarkConfig(0, List.of(10), 42L, 0.1, 5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("maxConcurrentBenchmarks must be positive");
            
        assertThatThrownBy(() -> new BenchmarkConfig(-1, List.of(10), 42L, 0.1, 5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("maxConcurrentBenchmarks must be positive");
        
        // Test null scalability test sizes
        assertThatThrownBy(() -> new BenchmarkConfig(2, null, 42L, 0.1, 5))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("scalabilityTestSizes cannot be null");
        
        // Test invalid failure rate
        assertThatThrownBy(() -> new BenchmarkConfig(2, List.of(10), 42L, -0.1, 5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("syntheticFailureRate must be in [0.0, 1.0]");
            
        assertThatThrownBy(() -> new BenchmarkConfig(2, List.of(10), 42L, 1.5, 5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("syntheticFailureRate must be in [0.0, 1.0]");
        
        // Test invalid category count
        assertThatThrownBy(() -> new BenchmarkConfig(2, List.of(10), 42L, 0.1, 0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("syntheticCategoryCount must be positive");
    }
    
    @Test
    void testNullInputValidation() {
        assertThatThrownBy(() -> new BenchmarkSuite(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("config cannot be null");
            
        var testData = List.of(Map.of("test", ChannelResult.success("test", 1, 0.8, 100)));
        Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction = (Map<String, ChannelResult> cr) -> 
            ConsensusResult.create(1, 0.8, "Test", Map.of("test", 1.0));
        
        assertThatThrownBy(() -> benchmarkSuite.benchmarkConsensus(null, consensusFunction, testData))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("strategyName cannot be null");
            
        assertThatThrownBy(() -> benchmarkSuite.benchmarkConsensus("Test", null, testData))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("consensusFunction cannot be null");
            
        assertThatThrownBy(() -> benchmarkSuite.benchmarkConsensus("Test", consensusFunction, null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("testData cannot be null");
    }
    
    @Test
    void testRecordDataClasses() {
        // Test PerformanceMetric
        var perfMetric = new BenchmarkSuite.PerformanceMetric(1, Duration.ofMillis(50), 3, true);
        assertThat(perfMetric.iteration()).isEqualTo(1);
        assertThat(perfMetric.processingTime()).isEqualTo(Duration.ofMillis(50));
        assertThat(perfMetric.channelCount()).isEqualTo(3);
        assertThat(perfMetric.success()).isTrue();
        
        // Test AccuracyMetric
        var accuracyMetric = new BenchmarkSuite.AccuracyMetric(2, 0.85, 1, 0.9);
        assertThat(accuracyMetric.iteration()).isEqualTo(2);
        assertThat(accuracyMetric.confidence()).isEqualTo(0.85);
        assertThat(accuracyMetric.predictedCategory()).isEqualTo(1);
        assertThat(accuracyMetric.channelAgreement()).isEqualTo(0.9);
        
        // Test FusionMetric
        var fusionMetric = new BenchmarkSuite.FusionMetric(3, 128, 0.25, 0.8);
        assertThat(fusionMetric.iteration()).isEqualTo(3);
        assertThat(fusionMetric.resultDimensionality()).isEqualTo(128);
        assertThat(fusionMetric.compressionRatio()).isEqualTo(0.25);
        assertThat(fusionMetric.channelUtilization()).isEqualTo(0.8);
        
        // Test ScalabilityPoint
        var scalabilityPoint = new BenchmarkSuite.ScalabilityPoint(100, Duration.ofSeconds(1), 100.0, 0.95);
        assertThat(scalabilityPoint.dataSize()).isEqualTo(100);
        assertThat(scalabilityPoint.totalTime()).isEqualTo(Duration.ofSeconds(1));
        assertThat(scalabilityPoint.throughput()).isEqualTo(100.0);
        assertThat(scalabilityPoint.successRate()).isEqualTo(0.95);
    }
    
    @Test
    void testBenchmarkResultToMap() {
        var perfMetrics = List.of(
            new BenchmarkSuite.PerformanceMetric(0, Duration.ofMillis(10), 2, true),
            new BenchmarkSuite.PerformanceMetric(1, Duration.ofMillis(20), 2, true)
        );
        var accuracyMetrics = List.of(
            new BenchmarkSuite.AccuracyMetric(0, 0.8, 1, 0.9),
            new BenchmarkSuite.AccuracyMetric(1, 0.9, 1, 0.95)
        );
        
        var result = new BenchmarkSuite.BenchmarkResult(
            "TestStrategy", BenchmarkType.CONSENSUS, 2, Duration.ofMillis(30),
            perfMetrics, accuracyMetrics, List.of("error1")
        );
        
        var map = result.toMap();
        
        assertThat(map).containsKey("strategyName");
        assertThat(map).containsKey("benchmarkType");
        assertThat(map).containsKey("iterationCount");
        assertThat(map).containsKey("totalTime");
        assertThat(map).containsKey("errorCount");
        assertThat(map).containsKey("avgProcessingTime");
        assertThat(map).containsKey("avgConfidence");
        
        assertThat(map.get("strategyName")).isEqualTo("TestStrategy");
        assertThat(map.get("benchmarkType")).isEqualTo("CONSENSUS");
        assertThat(map.get("iterationCount")).isEqualTo(2);
        assertThat(map.get("errorCount")).isEqualTo(1);
    }
}