package com.hellblazer.art.hartcq;

import com.hellblazer.art.hartcq.core.StreamProcessor;
import com.hellblazer.art.hartcq.core.Tokenizer;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Runner class for HART-CQ performance benchmarks.
 * 
 * This class executes the comprehensive performance benchmark suite and validates
 * that the system meets the required performance criteria:
 * - Throughput: >100 sentences/second
 * - Latency: Acceptable p95/p99 percentiles
 * - Memory: Reasonable heap allocation
 * - Determinism: Consistent outputs
 * 
 * Usage:
 * - Run all benchmarks: java PerformanceBenchmarkRunner
 * - Quick verification: java PerformanceBenchmarkRunner --quick
 * - Specific benchmark: java PerformanceBenchmarkRunner --benchmark singleSentenceProcessing
 */
public class PerformanceBenchmarkRunner {
    private static final Logger logger = LoggerFactory.getLogger(PerformanceBenchmarkRunner.class);
    
    // Performance requirements
    private static final double MIN_SENTENCES_PER_SECOND = 100.0;
    private static final double MAX_AVERAGE_LATENCY_MS = 10.0;
    private static final double MAX_P95_LATENCY_MS = 50.0;
    private static final double MAX_P99_LATENCY_MS = 100.0;
    
    // Test sentences for quick verification
    private static final String[] VERIFICATION_SENTENCES = {
        "The quick brown fox jumps over the lazy dog.",
        "Java performance testing with HART-CQ system.",
        "Neural network processing with adaptive resonance theory.",
        "Sliding window algorithm for text processing efficiency.",
        "Comprehensive performance benchmarks ensure system reliability."
    };

    public static void main(String[] args) throws RunnerException {
        var runner = new PerformanceBenchmarkRunner();
        
        if (args.length > 0) {
            switch (args[0]) {
                case "--quick" -> {
                    logger.info("Running quick performance verification...");
                    runner.runQuickVerification();
                }
                case "--benchmark" -> {
                    if (args.length > 1) {
                        runner.runSpecificBenchmark(args[1]);
                    } else {
                        logger.error("Please specify benchmark name after --benchmark");
                        showUsage();
                    }
                }
                case "--help" -> showUsage();
                default -> {
                    logger.info("Running full benchmark suite...");
                    runner.runFullBenchmarkSuite();
                }
            }
        } else {
            logger.info("Running full benchmark suite...");
            runner.runFullBenchmarkSuite();
        }
    }
    
    /**
     * Run the complete benchmark suite and analyze results.
     */
    public void runFullBenchmarkSuite() throws RunnerException {
        logger.info("=".repeat(80));
        logger.info("HART-CQ COMPREHENSIVE PERFORMANCE BENCHMARK SUITE");
        logger.info("=".repeat(80));
        
        var options = new OptionsBuilder()
                .include(PerformanceBenchmark.class.getSimpleName())
                .forks(2)
                .warmupIterations(5)
                .measurementIterations(10)
                .timeUnit(TimeUnit.MILLISECONDS)
                .build();
        
        var results = new Runner(options).run();
        
        analyzeResults(results);
        generatePerformanceReport(results);
    }
    
    /**
     * Run a specific benchmark by name.
     */
    public void runSpecificBenchmark(String benchmarkName) throws RunnerException {
        logger.info("Running specific benchmark: {}", benchmarkName);
        
        var options = new OptionsBuilder()
                .include(PerformanceBenchmark.class.getSimpleName() + "." + benchmarkName)
                .forks(1)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();
        
        var results = new Runner(options).run();
        analyzeSpecificResults(results, benchmarkName);
    }
    
    /**
     * Quick verification to check if system meets basic performance requirements.
     */
    public void runQuickVerification() {
        logger.info("=".repeat(60));
        logger.info("HART-CQ QUICK PERFORMANCE VERIFICATION");
        logger.info("=".repeat(60));
        
        var verificationResults = new VerificationResults();
        
        // Test 1: Throughput verification
        verificationResults.throughputTest = verifyThroughput();
        
        // Test 2: Latency verification
        verificationResults.latencyTest = verifyLatency();
        
        // Test 3: Determinism verification
        verificationResults.determinismTest = verifyDeterminism();
        
        // Test 4: Memory allocation verification
        verificationResults.memoryTest = verifyMemoryUsage();
        
        // Test 5: Concurrent processing verification
        verificationResults.concurrencyTest = verifyConcurrentProcessing();
        
        // Report results
        reportVerificationResults(verificationResults);
    }
    
    /**
     * Verify that throughput meets the >100 sentences/second requirement.
     */
    private boolean verifyThroughput() {
        logger.info("Testing throughput requirement: >100 sentences/second");
        
        try (var processor = new StreamProcessor()) {
            var startTime = System.nanoTime();
            var completedFutures = new ArrayList<CompletableFuture<StreamProcessor.ProcessingResult>>();
            
            // Process test sentences
            for (int i = 0; i < 500; i++) {
                var sentence = VERIFICATION_SENTENCES[i % VERIFICATION_SENTENCES.length] + " Test " + (i + 1);
                completedFutures.add(processor.processStream(sentence));
            }
            
            // Wait for all to complete
            var successCount = 0;
            for (var future : completedFutures) {
                var result = future.join();
                if (result.isSuccessful()) {
                    successCount++;
                }
            }
            
            var endTime = System.nanoTime();
            var durationSeconds = (endTime - startTime) / 1_000_000_000.0;
            var sentencesPerSecond = successCount / durationSeconds;
            
            logger.info("Processed {} sentences in {:.3f} seconds", successCount, durationSeconds);
            logger.info("Throughput: {:.1f} sentences/second", sentencesPerSecond);
            
            var passed = sentencesPerSecond >= MIN_SENTENCES_PER_SECOND;
            logger.info("Throughput test: {} (requirement: >{} sentences/second)", 
                       passed ? "PASSED" : "FAILED", MIN_SENTENCES_PER_SECOND);
            
            return passed;
            
        } catch (Exception e) {
            logger.error("Throughput test failed with exception", e);
            return false;
        }
    }
    
    /**
     * Verify latency characteristics.
     */
    private boolean verifyLatency() {
        logger.info("Testing latency characteristics");
        
        try (var processor = new StreamProcessor()) {
            var latencies = new ArrayList<Long>();
            
            // Warm up
            for (int i = 0; i < 50; i++) {
                var future = processor.processStream(VERIFICATION_SENTENCES[0]);
                future.join();
            }
            
            // Measure latencies
            for (int i = 0; i < 200; i++) {
                var sentence = VERIFICATION_SENTENCES[i % VERIFICATION_SENTENCES.length];
                var startTime = System.nanoTime();
                
                var future = processor.processStream(sentence);
                var result = future.join();
                
                if (result.isSuccessful()) {
                    var latency = System.nanoTime() - startTime;
                    latencies.add(latency);
                }
            }
            
            // Calculate percentiles
            latencies.sort(Long::compareTo);
            var p50 = getPercentile(latencies, 50) / 1_000_000.0; // Convert to ms
            var p95 = getPercentile(latencies, 95) / 1_000_000.0;
            var p99 = getPercentile(latencies, 99) / 1_000_000.0;
            var average = latencies.stream().mapToLong(Long::longValue).average().orElse(0) / 1_000_000.0;
            
            logger.info("Latency statistics:");
            logger.info("  Average: {:.2f} ms", average);
            logger.info("  P50: {:.2f} ms", p50);
            logger.info("  P95: {:.2f} ms", p95);
            logger.info("  P99: {:.2f} ms", p99);
            
            var passed = average <= MAX_AVERAGE_LATENCY_MS && 
                         p95 <= MAX_P95_LATENCY_MS && 
                         p99 <= MAX_P99_LATENCY_MS;
            
            logger.info("Latency test: {}", passed ? "PASSED" : "FAILED");
            return passed;
            
        } catch (Exception e) {
            logger.error("Latency test failed with exception", e);
            return false;
        }
    }
    
    /**
     * Verify deterministic processing.
     */
    private boolean verifyDeterminism() {
        logger.info("Testing deterministic processing");
        
        try (var processor = new StreamProcessor()) {
            var testSentence = "Determinism test sentence for consistent output verification.";
            var results = new ArrayList<StreamProcessor.ProcessingResult>();
            
            // Process same sentence multiple times
            for (int i = 0; i < 10; i++) {
                var future = processor.processStream(testSentence);
                results.add(future.join());
            }
            
            // Check that all results are consistent (same number of tokens processed)
            var firstResult = results.get(0);
            var allConsistent = results.stream()
                    .allMatch(result -> result.isSuccessful() && 
                             result.getTotalTokens() == firstResult.getTotalTokens() &&
                             result.getWindowsProcessed() == firstResult.getWindowsProcessed());
            
            logger.info("Processed same input {} times", results.size());
            logger.info("All results consistent: {}", allConsistent);
            logger.info("Determinism test: {}", allConsistent ? "PASSED" : "FAILED");
            
            return allConsistent;
            
        } catch (Exception e) {
            logger.error("Determinism test failed with exception", e);
            return false;
        }
    }
    
    /**
     * Verify reasonable memory usage.
     */
    private boolean verifyMemoryUsage() {
        logger.info("Testing memory allocation patterns");
        
        try {
            // Force GC before test
            System.gc();
            Thread.sleep(100);
            
            var memoryBefore = getUsedMemory();
            
            // Process a batch of sentences
            try (var processor = new StreamProcessor()) {
                var futures = new ArrayList<CompletableFuture<StreamProcessor.ProcessingResult>>();
                
                for (int i = 0; i < 100; i++) {
                    var sentence = VERIFICATION_SENTENCES[i % VERIFICATION_SENTENCES.length] + " Memory test " + i;
                    futures.add(processor.processStream(sentence));
                }
                
                // Wait for completion
                futures.forEach(CompletableFuture::join);
            }
            
            System.gc();
            Thread.sleep(100);
            
            var memoryAfter = getUsedMemory();
            var memoryUsedMB = (memoryAfter - memoryBefore) / (1024.0 * 1024.0);
            
            logger.info("Memory used for processing: {:.2f} MB", memoryUsedMB);
            
            // Reasonable limit: less than 50MB for this test
            var passed = memoryUsedMB < 50.0;
            logger.info("Memory test: {} (used {:.2f} MB)", passed ? "PASSED" : "FAILED", memoryUsedMB);
            
            return passed;
            
        } catch (Exception e) {
            logger.error("Memory test failed with exception", e);
            return false;
        }
    }
    
    /**
     * Verify concurrent processing capability.
     */
    private boolean verifyConcurrentProcessing() {
        logger.info("Testing concurrent processing capability");
        
        try (var processor = new StreamProcessor()) {
            var numThreads = 5;
            var sentencesPerThread = 20;
            var futures = new ArrayList<CompletableFuture<Integer>>();
            
            var startTime = System.nanoTime();
            
            // Launch concurrent processing
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                var future = CompletableFuture.supplyAsync(() -> {
                    var processed = 0;
                    for (int i = 0; i < sentencesPerThread; i++) {
                        var sentence = VERIFICATION_SENTENCES[i % VERIFICATION_SENTENCES.length] + 
                                      " Thread " + threadId + " Sentence " + i;
                        var result = processor.processStream(sentence).join();
                        if (result.isSuccessful()) {
                            processed++;
                        }
                    }
                    return processed;
                });
                futures.add(future);
            }
            
            // Wait for all threads to complete
            var totalProcessed = futures.stream()
                    .mapToInt(CompletableFuture::join)
                    .sum();
            
            var endTime = System.nanoTime();
            var durationSeconds = (endTime - startTime) / 1_000_000_000.0;
            var throughput = totalProcessed / durationSeconds;
            
            logger.info("Concurrent processing: {} sentences in {:.3f} seconds", totalProcessed, durationSeconds);
            logger.info("Concurrent throughput: {:.1f} sentences/second", throughput);
            
            var passed = totalProcessed == (numThreads * sentencesPerThread) && throughput > 50.0;
            logger.info("Concurrent processing test: {}", passed ? "PASSED" : "FAILED");
            
            return passed;
            
        } catch (Exception e) {
            logger.error("Concurrent processing test failed with exception", e);
            return false;
        }
    }
    
    /**
     * Analyze benchmark results and check performance requirements.
     */
    private void analyzeResults(Collection<RunResult> results) {
        logger.info("\n" + "=".repeat(80));
        logger.info("PERFORMANCE ANALYSIS RESULTS");
        logger.info("=".repeat(80));
        
        var requirementsMet = 0;
        var totalRequirements = 0;
        
        for (var result : results) {
            var benchmarkName = result.getParams().getBenchmark();
            var score = result.getPrimaryResult().getScore();
            var unit = result.getPrimaryResult().getScoreUnit();
            
            logger.info("\n{}: {:.3f} {}", benchmarkName, score, unit);
            
            // Check specific requirements
            if (benchmarkName.contains("singleSentenceProcessing")) {
                totalRequirements++;
                if (unit.contains("ops/ms")) {
                    var sentencesPerSecond = score * 1000; // Convert ops/ms to ops/s
                    if (sentencesPerSecond >= MIN_SENTENCES_PER_SECOND) {
                        requirementsMet++;
                        logger.info("  ✓ Throughput requirement met: {:.1f} sentences/second", sentencesPerSecond);
                    } else {
                        logger.warn("  ✗ Throughput requirement NOT met: {:.1f} sentences/second (required: >{})", 
                                   sentencesPerSecond, MIN_SENTENCES_PER_SECOND);
                    }
                }
            }
        }
        
        logger.info("\n" + "=".repeat(80));
        logger.info("FINAL ASSESSMENT: {}/{} requirements met", requirementsMet, totalRequirements);
        if (requirementsMet == totalRequirements && totalRequirements > 0) {
            logger.info("🎉 ALL PERFORMANCE REQUIREMENTS MET!");
        } else {
            logger.warn("⚠️  Some performance requirements not met. Review results above.");
        }
        logger.info("=".repeat(80));
    }
    
    /**
     * Analyze results for a specific benchmark.
     */
    private void analyzeSpecificResults(Collection<RunResult> results, String benchmarkName) {
        logger.info("\nResults for benchmark: {}", benchmarkName);
        
        for (var result : results) {
            var score = result.getPrimaryResult().getScore();
            var unit = result.getPrimaryResult().getScoreUnit();
            var error = result.getPrimaryResult().getScoreError();
            
            logger.info("Score: {:.3f} ± {:.3f} {}", score, error, unit);
            
            if (result.getSecondaryResults() != null) {
                result.getSecondaryResults().forEach((key, value) -> 
                    logger.info("{}: {:.3f} {}", key, value.getScore(), value.getScoreUnit()));
            }
        }
    }
    
    /**
     * Generate a comprehensive performance report.
     */
    private void generatePerformanceReport(Collection<RunResult> results) {
        var report = new StringBuilder();
        report.append("\n").append("=".repeat(100)).append("\n");
        report.append("HART-CQ PERFORMANCE BENCHMARK REPORT\n");
        report.append("Generated: ").append(java.time.LocalDateTime.now()).append("\n");
        report.append("=".repeat(100)).append("\n");
        
        for (var result : results) {
            var benchmark = result.getParams().getBenchmark();
            var mode = result.getParams().getMode();
            var score = result.getPrimaryResult().getScore();
            var error = result.getPrimaryResult().getScoreError();
            var unit = result.getPrimaryResult().getScoreUnit();
            
            report.append(String.format("\n%s (%s):\n", benchmark, mode));
            report.append(String.format("  Score: %.3f ± %.3f %s\n", score, error, unit));
            
            if (result.getSecondaryResults() != null && !result.getSecondaryResults().isEmpty()) {
                report.append("  Additional metrics:\n");
                result.getSecondaryResults().forEach((key, value) -> 
                    report.append(String.format("    %s: %.3f %s\n", key, value.getScore(), value.getScoreUnit())));
            }
        }
        
        report.append("\n").append("=".repeat(100)).append("\n");
        
        logger.info(report.toString());
    }
    
    /**
     * Report verification test results.
     */
    private void reportVerificationResults(VerificationResults results) {
        logger.info("\n" + "=".repeat(60));
        logger.info("VERIFICATION RESULTS SUMMARY");
        logger.info("=".repeat(60));
        
        logger.info("Throughput Test:           {}", results.throughputTest ? "✓ PASSED" : "✗ FAILED");
        logger.info("Latency Test:              {}", results.latencyTest ? "✓ PASSED" : "✗ FAILED");
        logger.info("Determinism Test:          {}", results.determinismTest ? "✓ PASSED" : "✗ FAILED");
        logger.info("Memory Usage Test:         {}", results.memoryTest ? "✓ PASSED" : "✗ FAILED");
        logger.info("Concurrent Processing:     {}", results.concurrencyTest ? "✓ PASSED" : "✗ FAILED");
        
        var passedTests = (results.throughputTest ? 1 : 0) +
                         (results.latencyTest ? 1 : 0) +
                         (results.determinismTest ? 1 : 0) +
                         (results.memoryTest ? 1 : 0) +
                         (results.concurrencyTest ? 1 : 0);
        
        logger.info("\nOverall: {}/5 tests passed", passedTests);
        
        if (passedTests == 5) {
            logger.info("🎉 HART-CQ meets all performance requirements!");
        } else {
            logger.warn("⚠️  Some performance requirements not met. Review failed tests.");
        }
        
        logger.info("=".repeat(60));
    }
    
    /**
     * Calculate percentile from sorted list.
     */
    private long getPercentile(ArrayList<Long> sortedList, int percentile) {
        if (sortedList.isEmpty()) return 0;
        var index = (int) Math.ceil((percentile / 100.0) * sortedList.size()) - 1;
        return sortedList.get(Math.max(0, Math.min(index, sortedList.size() - 1)));
    }
    
    /**
     * Get currently used memory in bytes.
     */
    private long getUsedMemory() {
        var runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
    
    /**
     * Show usage information.
     */
    private static void showUsage() {
        System.out.println("HART-CQ Performance Benchmark Runner");
        System.out.println("Usage:");
        System.out.println("  java PerformanceBenchmarkRunner              - Run full benchmark suite");
        System.out.println("  java PerformanceBenchmarkRunner --quick      - Quick verification only");
        System.out.println("  java PerformanceBenchmarkRunner --benchmark <name> - Run specific benchmark");
        System.out.println("  java PerformanceBenchmarkRunner --help       - Show this help");
        System.out.println();
        System.out.println("Available benchmarks:");
        System.out.println("  singleSentenceProcessing, smallBatchProcessing, mediumBatchProcessing,");
        System.out.println("  largeDocumentProcessing, concurrentProcessing, channelProcessingOverhead,");
        System.out.println("  templateMatchingPerformance, tokenizationPerformance, determinismTest,");
        System.out.println("  scalabilityTest, stressTest, memoryAllocationTest");
    }
    
    /**
     * Holds results of verification tests.
     */
    private static class VerificationResults {
        boolean throughputTest;
        boolean latencyTest;
        boolean determinismTest;
        boolean memoryTest;
        boolean concurrencyTest;
    }
}