package com.hellblazer.art.nlp.processor.benchmark;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Comprehensive benchmarking and metrics suite for ART-NLP processing.
 * Provides performance benchmarks, accuracy metrics, scalability tests,
 * and statistical analysis of consensus and fusion strategies.
 */
public class BenchmarkSuite {
    private static final Logger log = LoggerFactory.getLogger(BenchmarkSuite.class);
    
    private final Map<String, BenchmarkResult> results;
    private final BenchmarkConfig config;
    private final ExecutorService executor;
    
    public BenchmarkSuite() {
        this(BenchmarkConfig.defaultConfig());
    }
    
    public BenchmarkSuite(BenchmarkConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        this.config = config;
        this.results = new ConcurrentHashMap<>();
        this.executor = Executors.newFixedThreadPool(config.maxConcurrentBenchmarks());
    }
    
    /**
     * Run performance benchmark for a consensus strategy.
     */
    public CompletableFuture<BenchmarkResult> benchmarkConsensus(
            String strategyName,
            Function<Map<String, ChannelResult>, ConsensusResult> consensusFunction,
            List<Map<String, ChannelResult>> testData) {
        
        Objects.requireNonNull(strategyName, "strategyName cannot be null");
        Objects.requireNonNull(consensusFunction, "consensusFunction cannot be null");
        Objects.requireNonNull(testData, "testData cannot be null");
        
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Starting consensus benchmark: {}", strategyName);
            
            var performanceMetrics = new ArrayList<PerformanceMetric>();
            var accuracyMetrics = new ArrayList<AccuracyMetric>();
            var errors = new ArrayList<String>();
            
            var startTime = Instant.now();
            
            for (int i = 0; i < testData.size(); i++) {
                var channelResults = testData.get(i);
                var iterationStart = System.nanoTime();
                
                try {
                    var result = consensusFunction.apply(channelResults);
                    var processingTime = System.nanoTime() - iterationStart;
                    
                    // Record performance
                    performanceMetrics.add(new PerformanceMetric(
                        i, Duration.ofNanos(processingTime), channelResults.size(), result != null
                    ));
                    
                    // Record accuracy if result is valid
                    if (result != null) {
                        accuracyMetrics.add(new AccuracyMetric(
                            i, result.confidence(), result.category(), 
                            calculateChannelAgreement(channelResults, result)
                        ));
                    }
                    
                } catch (Exception e) {
                    errors.add(String.format("Iteration %d: %s", i, e.getMessage()));
                    log.warn("Error in consensus benchmark iteration {}: {}", i, e.getMessage());
                }
                
                // Progress logging
                if ((i + 1) % Math.max(1, testData.size() / 10) == 0) {
                    log.debug("Consensus benchmark progress: {}/{}", i + 1, testData.size());
                }
            }
            
            var totalTime = Duration.between(startTime, Instant.now());
            
            var benchmarkResult = new BenchmarkResult(
                strategyName,
                BenchmarkType.CONSENSUS,
                testData.size(),
                totalTime,
                performanceMetrics,
                accuracyMetrics,
                errors
            );
            
            results.put("consensus_" + strategyName, benchmarkResult);
            
            log.debug("Completed consensus benchmark: {} - {} iterations in {}", 
                     strategyName, testData.size(), totalTime);
            
            return benchmarkResult;
        }, executor);
    }
    
    /**
     * Run performance benchmark for a fusion strategy.
     */
    public CompletableFuture<BenchmarkResult> benchmarkFusion(
            String strategyName,
            Function<Map<String, ChannelResult>, Object> fusionFunction,
            List<Map<String, ChannelResult>> testData) {
        
        Objects.requireNonNull(strategyName, "strategyName cannot be null");
        Objects.requireNonNull(fusionFunction, "fusionFunction cannot be null");
        Objects.requireNonNull(testData, "testData cannot be null");
        
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Starting fusion benchmark: {}", strategyName);
            
            var performanceMetrics = new ArrayList<PerformanceMetric>();
            var fusionMetrics = new ArrayList<FusionMetric>();
            var errors = new ArrayList<String>();
            
            var startTime = Instant.now();
            
            for (int i = 0; i < testData.size(); i++) {
                var channelResults = testData.get(i);
                var iterationStart = System.nanoTime();
                
                try {
                    var result = fusionFunction.apply(channelResults);
                    var processingTime = System.nanoTime() - iterationStart;
                    
                    // Record performance
                    performanceMetrics.add(new PerformanceMetric(
                        i, Duration.ofNanos(processingTime), channelResults.size(), result != null
                    ));
                    
                    // Record fusion-specific metrics
                    if (result != null) {
                        var dimensionality = estimateResultDimensionality(result);
                        var compression = channelResults.size() > 0 ? 
                            (double) dimensionality / channelResults.size() : 0.0;
                        
                        fusionMetrics.add(new FusionMetric(
                            i, dimensionality, compression, 
                            calculateChannelUtilization(channelResults)
                        ));
                    }
                    
                } catch (Exception e) {
                    errors.add(String.format("Iteration %d: %s", i, e.getMessage()));
                    log.warn("Error in fusion benchmark iteration {}: {}", i, e.getMessage());
                }
                
                // Progress logging
                if ((i + 1) % Math.max(1, testData.size() / 10) == 0) {
                    log.debug("Fusion benchmark progress: {}/{}", i + 1, testData.size());
                }
            }
            
            var totalTime = Duration.between(startTime, Instant.now());
            
            var benchmarkResult = new BenchmarkResult(
                strategyName,
                BenchmarkType.FUSION,
                testData.size(),
                totalTime,
                performanceMetrics,
                Collections.emptyList(), // No accuracy metrics for fusion
                errors,
                fusionMetrics
            );
            
            results.put("fusion_" + strategyName, benchmarkResult);
            
            log.debug("Completed fusion benchmark: {} - {} iterations in {}", 
                     strategyName, testData.size(), totalTime);
            
            return benchmarkResult;
        }, executor);
    }
    
    /**
     * Run scalability benchmark with increasing data sizes.
     */
    public CompletableFuture<ScalabilityResult> benchmarkScalability(
            String strategyName,
            Function<Map<String, ChannelResult>, ?> strategyFunction,
            Supplier<Map<String, ChannelResult>> dataGenerator) {
        
        Objects.requireNonNull(strategyName, "strategyName cannot be null");
        Objects.requireNonNull(strategyFunction, "strategyFunction cannot be null");
        Objects.requireNonNull(dataGenerator, "dataGenerator cannot be null");
        
        return CompletableFuture.supplyAsync(() -> {
            log.debug("Starting scalability benchmark: {}", strategyName);
            
            var scalabilityPoints = new ArrayList<ScalabilityPoint>();
            
            for (var dataSize : config.scalabilityTestSizes()) {
                var testData = IntStream.range(0, dataSize)
                    .mapToObj(i -> dataGenerator.get())
                    .toList();
                
                var startTime = System.nanoTime();
                var successCount = 0;
                
                for (var data : testData) {
                    try {
                        var result = strategyFunction.apply(data);
                        if (result != null) {
                            successCount++;
                        }
                    } catch (Exception e) {
                        log.debug("Error in scalability test (size {}): {}", dataSize, e.getMessage());
                    }
                }
                
                var processingTime = Duration.ofNanos(System.nanoTime() - startTime);
                var millis = processingTime.toMillis();
                var throughput = (dataSize > 0 && millis > 0) ? (double) dataSize / millis * 1000.0 : 0.0;
                var successRate = (double) successCount / dataSize;
                
                scalabilityPoints.add(new ScalabilityPoint(
                    dataSize, processingTime, throughput, successRate
                ));
                
                log.debug("Scalability point: size={}, time={}, throughput={:.2f} ops/sec, success={:.2f}%", 
                         dataSize, processingTime, throughput, successRate * 100);
            }
            
            var scalabilityResult = new ScalabilityResult(strategyName, scalabilityPoints);
            
            log.debug("Completed scalability benchmark: {}", strategyName);
            
            return scalabilityResult;
        }, executor);
    }
    
    /**
     * Compare multiple strategies on the same test data.
     */
    public CompletableFuture<ComparisonResult> compareStrategies(
            Map<String, Function<Map<String, ChannelResult>, ?>> strategies,
            List<Map<String, ChannelResult>> testData) {
        
        Objects.requireNonNull(strategies, "strategies cannot be null");
        Objects.requireNonNull(testData, "testData cannot be null");
        
        var comparisons = strategies.entrySet().stream()
            .map(entry -> {
                var strategyName = entry.getKey();
                var strategyFunction = entry.getValue();
                
                return CompletableFuture.supplyAsync(() -> {
                    var metrics = new ArrayList<ComparisonMetric>();
                    
                    for (int i = 0; i < testData.size(); i++) {
                        var data = testData.get(i);
                        var startTime = System.nanoTime();
                        
                        try {
                            var result = strategyFunction.apply(data);
                            var processingTime = Duration.ofNanos(System.nanoTime() - startTime);
                            
                            metrics.add(new ComparisonMetric(
                                i, strategyName, processingTime, result != null
                            ));
                        } catch (Exception e) {
                            log.debug("Error in strategy comparison {}: {}", strategyName, e.getMessage());
                        }
                    }
                    
                    return Map.entry(strategyName, metrics);
                }, executor);
            })
            .toList();
        
        return CompletableFuture.allOf(comparisons.toArray(new CompletableFuture[0]))
            .thenApply(v -> {
                var allMetrics = new HashMap<String, List<ComparisonMetric>>();
                
                for (var future : comparisons) {
                    var entry = future.join();
                    allMetrics.put(entry.getKey(), entry.getValue());
                }
                
                return new ComparisonResult(allMetrics);
            });
    }
    
    /**
     * Generate synthetic test data for benchmarking.
     */
    public List<Map<String, ChannelResult>> generateSyntheticData(int size, String... channelIds) {
        var random = new Random(config.randomSeed());
        var testData = new ArrayList<Map<String, ChannelResult>>(size);
        
        for (int i = 0; i < size; i++) {
            var channelResults = new HashMap<String, ChannelResult>();
            
            for (var channelId : channelIds) {
                var success = random.nextDouble() > config.syntheticFailureRate();
                
                if (success) {
                    var category = random.nextInt(config.syntheticCategoryCount());
                    var confidence = 0.5 + random.nextDouble() * 0.5; // 0.5 to 1.0
                    var processingTime = 50 + random.nextInt(200); // 50-250ms
                    
                    channelResults.put(channelId, 
                        ChannelResult.success(channelId, category, confidence, processingTime));
                } else {
                    var processingTime = 10 + random.nextInt(100); // 10-110ms (failures are faster)
                    channelResults.put(channelId, 
                        ChannelResult.failed(channelId, "Synthetic failure", processingTime));
                }
            }
            
            testData.add(channelResults);
        }
        
        log.debug("Generated {} synthetic test cases with {} channels", size, channelIds.length);
        return testData;
    }
    
    /**
     * Get comprehensive benchmark statistics.
     */
    public BenchmarkStatistics getStatistics() {
        var strategyStats = new HashMap<String, StrategyStatistics>();
        
        for (var entry : results.entrySet()) {
            var result = entry.getValue();
            var stats = calculateStrategyStatistics(result);
            strategyStats.put(entry.getKey(), stats);
        }
        
        return new BenchmarkStatistics(strategyStats, results.size(), Instant.now());
    }
    
    /**
     * Clear all benchmark results.
     */
    public void clearResults() {
        results.clear();
        log.debug("Cleared all benchmark results");
    }
    
    /**
     * Export benchmark results for analysis.
     */
    public Map<String, Object> exportResults() {
        var export = new HashMap<String, Object>();
        
        for (var entry : results.entrySet()) {
            var result = entry.getValue();
            export.put(entry.getKey(), result.toMap());
        }
        
        export.put("exportTime", Instant.now().toString());
        export.put("totalBenchmarks", results.size());
        
        return export;
    }
    
    /**
     * Shutdown the benchmark suite and cleanup resources.
     */
    public void shutdown() {
        executor.shutdown();
        log.debug("Benchmark suite shut down");
    }
    
    // Private helper methods
    
    private double calculateChannelAgreement(Map<String, ChannelResult> channelResults, ConsensusResult consensusResult) {
        var successful = channelResults.values().stream()
            .filter(ChannelResult::isSuccess)
            .toList();
        
        if (successful.isEmpty()) return 0.0;
        
        var consensusCategory = consensusResult.category();
        var agreementCount = successful.stream()
            .mapToLong(result -> result.category() == consensusCategory ? 1 : 0)
            .sum();
        
        return (double) agreementCount / successful.size();
    }
    
    private int estimateResultDimensionality(Object result) {
        if (result instanceof double[]) {
            return ((double[]) result).length;
        } else if (result instanceof Collection) {
            return ((Collection<?>) result).size();
        } else if (result instanceof Map) {
            return ((Map<?, ?>) result).size();
        } else {
            return 1; // Scalar result
        }
    }
    
    private double calculateChannelUtilization(Map<String, ChannelResult> channelResults) {
        if (channelResults.isEmpty()) return 0.0;
        
        var successfulCount = channelResults.values().stream()
            .mapToLong(result -> result.isSuccess() ? 1 : 0)
            .sum();
        
        return (double) successfulCount / channelResults.size();
    }
    
    private StrategyStatistics calculateStrategyStatistics(BenchmarkResult result) {
        var avgProcessingTime = result.performanceMetrics().stream()
            .mapToDouble(metric -> metric.processingTime().toNanos())
            .average()
            .orElse(0.0);
        
        var successRate = result.performanceMetrics().stream()
            .mapToDouble(metric -> metric.success() ? 1.0 : 0.0)
            .average()
            .orElse(0.0);
        
        var avgConfidence = result.accuracyMetrics().stream()
            .mapToDouble(AccuracyMetric::confidence)
            .average()
            .orElse(0.0);
        
        var throughput = result.totalTime().toMillis() > 0 ? 
            (double) result.iterationCount() / result.totalTime().toMillis() * 1000.0 : 0.0;
        
        return new StrategyStatistics(
            result.strategyName(),
            result.benchmarkType(),
            result.iterationCount(),
            Duration.ofNanos((long) avgProcessingTime),
            successRate,
            avgConfidence,
            throughput,
            result.errors().size()
        );
    }
    
    // Data classes for benchmark results
    
    public record PerformanceMetric(
        int iteration,
        Duration processingTime,
        int channelCount,
        boolean success
    ) {}
    
    public record AccuracyMetric(
        int iteration,
        double confidence,
        int predictedCategory,
        double channelAgreement
    ) {}
    
    public record FusionMetric(
        int iteration,
        int resultDimensionality,
        double compressionRatio,
        double channelUtilization
    ) {}
    
    public record ComparisonMetric(
        int iteration,
        String strategyName,
        Duration processingTime,
        boolean success
    ) {}
    
    public record ScalabilityPoint(
        int dataSize,
        Duration totalTime,
        double throughput,
        double successRate
    ) {}
    
    public record BenchmarkResult(
        String strategyName,
        BenchmarkType benchmarkType,
        int iterationCount,
        Duration totalTime,
        List<PerformanceMetric> performanceMetrics,
        List<AccuracyMetric> accuracyMetrics,
        List<String> errors,
        List<FusionMetric> fusionMetrics
    ) {
        public BenchmarkResult(String strategyName, BenchmarkType benchmarkType, 
                             int iterationCount, Duration totalTime,
                             List<PerformanceMetric> performanceMetrics,
                             List<AccuracyMetric> accuracyMetrics,
                             List<String> errors) {
            this(strategyName, benchmarkType, iterationCount, totalTime,
                 performanceMetrics, accuracyMetrics, errors, Collections.emptyList());
        }
        
        public Map<String, Object> toMap() {
            var map = new HashMap<String, Object>();
            map.put("strategyName", strategyName);
            map.put("benchmarkType", benchmarkType.toString());
            map.put("iterationCount", iterationCount);
            map.put("totalTime", totalTime.toString());
            map.put("errorCount", errors.size());
            
            if (!performanceMetrics.isEmpty()) {
                map.put("avgProcessingTime", performanceMetrics.stream()
                    .mapToDouble(m -> m.processingTime().toNanos()).average().orElse(0.0));
            }
            
            if (!accuracyMetrics.isEmpty()) {
                map.put("avgConfidence", accuracyMetrics.stream()
                    .mapToDouble(AccuracyMetric::confidence).average().orElse(0.0));
            }
            
            return map;
        }
    }
    
    public record ScalabilityResult(
        String strategyName,
        List<ScalabilityPoint> scalabilityPoints
    ) {}
    
    public record ComparisonResult(
        Map<String, List<ComparisonMetric>> strategyMetrics
    ) {}
    
    public record StrategyStatistics(
        String strategyName,
        BenchmarkType benchmarkType,
        int totalIterations,
        Duration averageProcessingTime,
        double successRate,
        double averageConfidence,
        double throughput,
        int errorCount
    ) {}
    
    public record BenchmarkStatistics(
        Map<String, StrategyStatistics> strategyStatistics,
        int totalBenchmarks,
        Instant generatedAt
    ) {}
    
    public enum BenchmarkType {
        CONSENSUS, FUSION, SCALABILITY, COMPARISON
    }
    
    /**
     * Configuration for benchmark execution.
     */
    public record BenchmarkConfig(
        int maxConcurrentBenchmarks,
        List<Integer> scalabilityTestSizes,
        long randomSeed,
        double syntheticFailureRate,
        int syntheticCategoryCount
    ) {
        public BenchmarkConfig {
            if (maxConcurrentBenchmarks <= 0) {
                throw new IllegalArgumentException("maxConcurrentBenchmarks must be positive: " + maxConcurrentBenchmarks);
            }
            Objects.requireNonNull(scalabilityTestSizes, "scalabilityTestSizes cannot be null");
            if (syntheticFailureRate < 0.0 || syntheticFailureRate > 1.0) {
                throw new IllegalArgumentException("syntheticFailureRate must be in [0.0, 1.0]: " + syntheticFailureRate);
            }
            if (syntheticCategoryCount <= 0) {
                throw new IllegalArgumentException("syntheticCategoryCount must be positive: " + syntheticCategoryCount);
            }
        }
        
        public static BenchmarkConfig defaultConfig() {
            return new BenchmarkConfig(
                Runtime.getRuntime().availableProcessors(),
                List.of(10, 50, 100, 500, 1000),
                42L,
                0.1,
                5
            );
        }
        
        public static BenchmarkConfig fastConfig() {
            return new BenchmarkConfig(
                Runtime.getRuntime().availableProcessors(),
                List.of(5, 25, 50),
                42L,
                0.05,
                3
            );
        }
        
        public static BenchmarkConfig thoroughConfig() {
            return new BenchmarkConfig(
                Runtime.getRuntime().availableProcessors(),
                List.of(10, 25, 50, 100, 250, 500, 1000, 2000),
                42L,
                0.15,
                10
            );
        }
    }
}