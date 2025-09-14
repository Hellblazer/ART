package com.hellblazer.art.hartcq;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.core.MultiChannelProcessor;
import com.hellblazer.art.hartcq.core.StreamProcessor;
import com.hellblazer.art.hartcq.Tokenizer;
import com.hellblazer.art.hartcq.templates.TemplateManager;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive performance benchmark for HART-CQ system.
 * 
 * Tests multiple performance aspects:
 * - Throughput: >100 sentences/second requirement
 * - Latency: p50, p95, p99 percentiles
 * - Scalability: varying input sizes
 * - Determinism: consistent outputs
 * - Memory usage: heap allocation tracking
 * 
 * Benchmark scenarios:
 * - Single sentence processing
 * - Batch processing (100 sentences)
 * - Large document processing (1000 sentences)
 * - Concurrent processing (10 threads)
 * - Template matching performance
 * - Channel processing overhead
 */
@BenchmarkMode({Mode.Throughput, Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 3, timeUnit = TimeUnit.SECONDS)
@Fork(2)
@Threads(1)
public class PerformanceBenchmark {

    // Test data sizes
    private static final int SMALL_BATCH_SIZE = 10;
    private static final int MEDIUM_BATCH_SIZE = 100;
    private static final int LARGE_BATCH_SIZE = 1000;
    private static final int CONCURRENT_THREADS = 10;

    // Test sentences of varying complexity
    private static final String[] TEST_SENTENCES = {
        "The quick brown fox jumps over the lazy dog.",
        "Java 24 introduces powerful new features for modern application development.",
        "Neural networks can process complex patterns in data through adaptive resonance theory.",
        "The HART-CQ system implements a 20-token sliding window mechanism for deterministic text processing.",
        "Performance optimization requires careful consideration of memory allocation patterns, CPU utilization, and concurrent processing overhead.",
        "Artificial intelligence systems must balance accuracy, speed, and resource consumption while maintaining deterministic behavior for reliable production deployment.",
        "In the realm of computational linguistics, tokenization represents the foundational step in natural language processing pipelines, breaking down complex textual input into manageable discrete units.",
        "The implementation of sliding window algorithms in high-performance computing environments necessitates careful attention to memory management, cache coherency, and parallel processing coordination mechanisms."
    };

    // State objects
    private StreamProcessor streamProcessor;
    private MultiChannelProcessor channelProcessor;
    private TemplateManager templateManager;
    private Tokenizer tokenizer;
    
    // Test data
    private List<String> smallBatch;
    private List<String> mediumBatch;
    private List<String> largeBatch;
    private List<com.hellblazer.art.hartcq.Token> sampleTokens;
    private Map<String, String> templateVariables;

    @Setup(Level.Trial)
    public void setupBenchmark() {
        // Initialize processors
        streamProcessor = new StreamProcessor(Runtime.getRuntime().availableProcessors());
        channelProcessor = new MultiChannelProcessor();
        templateManager = new TemplateManager();
        tokenizer = new Tokenizer();

        // Initialize template variables
        templateVariables = new HashMap<>();
        templateVariables.put("user", "testUser");
        templateVariables.put("context", "benchmark");
        templateVariables.put("timestamp", String.valueOf(System.currentTimeMillis()));

        // Generate test data
        generateTestData();

        // Pre-tokenize sample data
        var sampleText = "Performance testing requires comprehensive evaluation of system capabilities.";
        sampleTokens = tokenizer.tokenize(sampleText);
    }

    @TearDown(Level.Trial)
    public void teardownBenchmark() {
        if (streamProcessor != null) {
            streamProcessor.close();
        }
        if (channelProcessor != null) {
            channelProcessor.shutdown();
        }
    }

    private void generateTestData() {
        smallBatch = generateSentences(SMALL_BATCH_SIZE);
        mediumBatch = generateSentences(MEDIUM_BATCH_SIZE);
        largeBatch = generateSentences(LARGE_BATCH_SIZE);
    }

    private List<String> generateSentences(int count) {
        var sentences = new ArrayList<String>();
        for (int i = 0; i < count; i++) {
            sentences.add(TEST_SENTENCES[i % TEST_SENTENCES.length] + " Sentence " + (i + 1) + ".");
        }
        return sentences;
    }

    // ========== THROUGHPUT BENCHMARKS ==========

    /**
     * Test single sentence processing throughput.
     * Target: >100 sentences/second
     */
    @Benchmark
    public void singleSentenceProcessing(Blackhole bh) {
        var sentence = TEST_SENTENCES[0];
        var future = streamProcessor.processStream(sentence);
        var result = future.join();
        bh.consume(result);
    }

    /**
     * Test small batch processing (10 sentences).
     */
    @Benchmark
    public void smallBatchProcessing(Blackhole bh) {
        for (var sentence : smallBatch) {
            var future = streamProcessor.processStream(sentence);
            bh.consume(future.join());
        }
    }

    /**
     * Test medium batch processing (100 sentences).
     */
    @Benchmark
    public void mediumBatchProcessing(Blackhole bh) {
        for (var sentence : mediumBatch) {
            var future = streamProcessor.processStream(sentence);
            bh.consume(future.join());
        }
    }

    /**
     * Test large document processing (1000 sentences).
     */
    @Benchmark
    public void largeDocumentProcessing(Blackhole bh) {
        var document = String.join(" ", largeBatch);
        var future = streamProcessor.processStream(document);
        bh.consume(future.join());
    }

    // ========== CONCURRENT PROCESSING BENCHMARKS ==========

    /**
     * Test concurrent processing with multiple threads.
     */
    @Benchmark
    @Threads(10)
    public void concurrentProcessing(Blackhole bh) {
        var sentence = TEST_SENTENCES[(int)(Thread.currentThread().getId() % TEST_SENTENCES.length)];
        var future = streamProcessor.processStream(sentence);
        bh.consume(future.join());
    }

    /**
     * Test concurrent batch processing.
     */
    @Benchmark
    @Threads(5)
    public void concurrentBatchProcessing(Blackhole bh) {
        var batch = mediumBatch.subList(0, 20); // Smaller batches for concurrency
        for (var sentence : batch) {
            var future = streamProcessor.processStream(sentence);
            bh.consume(future.join());
        }
    }

    // ========== CHANNEL PROCESSING BENCHMARKS ==========

    /**
     * Test multi-channel processing overhead.
     */
    @Benchmark
    public void channelProcessingOverhead(Blackhole bh) {
        var tokens = sampleTokens.toArray(new com.hellblazer.art.hartcq.Token[0]);
        var output = channelProcessor.processWindow(tokens);
        bh.consume(output);
    }

    /**
     * Test batch channel processing.
     */
    @Benchmark
    public void batchChannelProcessing(Blackhole bh) {
        var tokenWindows = new ArrayList<com.hellblazer.art.hartcq.Token[]>();
        for (int i = 0; i < 50; i++) {
            var tokens = tokenizer.tokenize(TEST_SENTENCES[i % TEST_SENTENCES.length]);
            if (tokens.size() >= 20) {
                tokenWindows.add(tokens.subList(0, 20).toArray(new com.hellblazer.art.hartcq.Token[0]));
            }
        }
        var outputs = channelProcessor.processBatch(tokenWindows);
        bh.consume(outputs);
    }

    // ========== TEMPLATE PROCESSING BENCHMARKS ==========

    /**
     * Test template matching performance.
     */
    @Benchmark
    public void templateMatchingPerformance(Blackhole bh) {
        var input = "Process this text with template matching";
        var result = templateManager.processInput(input, templateVariables);
        bh.consume(result);
    }

    /**
     * Test deterministic template processing.
     */
    @Benchmark
    public void deterministicTemplateProcessing(Blackhole bh) {
        var input = "Generate deterministic output for this input";
        var result = templateManager.processDeterministic(input, templateVariables);
        bh.consume(result);
    }

    // ========== TOKENIZATION BENCHMARKS ==========

    /**
     * Test tokenization performance.
     */
    @Benchmark
    public void tokenizationPerformance(Blackhole bh) {
        var text = String.join(" ", TEST_SENTENCES);
        var tokens = tokenizer.tokenize(text);
        bh.consume(tokens);
    }

    /**
     * Test word-only tokenization.
     */
    @Benchmark
    public void wordTokenizationPerformance(Blackhole bh) {
        var text = String.join(" ", TEST_SENTENCES);
        var tokens = tokenizer.tokenizeWords(text);
        bh.consume(tokens);
    }

    /**
     * Test chunked tokenization.
     */
    @Benchmark
    public void chunkedTokenizationPerformance(Blackhole bh) {
        var text = String.join(" ", mediumBatch);
        var windows = tokenizer.tokenizeForWindows(text, 50);
        bh.consume(windows);
    }

    // ========== MEMORY ALLOCATION BENCHMARKS ==========

    /**
     * Test memory allocation patterns during processing.
     */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void memoryAllocationTest(Blackhole bh) {
        // Force garbage collection before test
        System.gc();
        
        var processor = new StreamProcessor(2);
        try {
            var futures = new ArrayList<CompletableFuture<StreamProcessor.ProcessingResult>>();
            
            for (var sentence : smallBatch) {
                futures.add(processor.processStream(sentence));
            }
            
            for (var future : futures) {
                bh.consume(future.join());
            }
        } finally {
            processor.close();
        }
    }

    // ========== DETERMINISM BENCHMARKS ==========

    /**
     * Test processing determinism - same input should produce same output.
     */
    @Benchmark
    public void determinismTest(Blackhole bh) {
        var input = TEST_SENTENCES[0];
        
        // Process same input multiple times
        var results = new ArrayList<StreamProcessor.ProcessingResult>();
        for (int i = 0; i < 5; i++) {
            var future = streamProcessor.processStream(input);
            results.add(future.join());
        }
        
        bh.consume(results);
    }

    // ========== SCALABILITY BENCHMARKS ==========

    @Param({"10", "50", "100", "200", "500"})
    private int sentenceCount;

    /**
     * Test scalability with increasing input sizes.
     */
    @Benchmark
    public int scalabilityTest(Blackhole bh) {
        var sentences = generateSentences(this.sentenceCount);
        var processedCount = 0;
        
        for (var sentence : sentences) {
            var future = streamProcessor.processStream(sentence);
            var result = future.join();
            if (result.isSuccessful()) {
                processedCount++;
            }
            bh.consume(result);
        }
        
        return processedCount;
    }

    // ========== STRESS TESTING BENCHMARKS ==========

    /**
     * Stress test with maximum load.
     */
    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(20)
    public void stressTest(Blackhole bh) {
        var sentence = TEST_SENTENCES[(int)(System.nanoTime() % TEST_SENTENCES.length)];
        var future = streamProcessor.processStream(sentence);
        bh.consume(future.join());
    }

    /**
     * Test pipeline performance under continuous load.
     */
    @Benchmark
    public void pipelineStressTest(Blackhole bh) {
        var futures = new ArrayList<CompletableFuture<StreamProcessor.ProcessingResult>>();
        
        // Submit continuous stream of work
        for (int i = 0; i < 100; i++) {
            var sentence = TEST_SENTENCES[i % TEST_SENTENCES.length];
            futures.add(streamProcessor.processStream(sentence));
        }
        
        // Wait for all to complete
        var results = futures.stream()
                             .map(CompletableFuture::join)
                             .toList();
        
        bh.consume(results);
    }

    /**
     * Main method to run benchmarks standalone.
     */
    public static void main(String[] args) throws RunnerException {
        var opt = new OptionsBuilder()
                .include(PerformanceBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();

        new Runner(opt).run();
    }
}