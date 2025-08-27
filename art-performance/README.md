# ART Performance Module

The `art-performance` module provides high-performance, vectorized implementations of ART algorithms optimized for modern hardware. This module leverages Java 24's Vector API, virtual threads, and advanced concurrency features to deliver maximum throughput and efficiency.

## Overview

The performance module transforms the scientific algorithms from `art-core` into production-ready, high-throughput implementations suitable for:

- **Large-scale Data Processing**: Handle datasets with millions of patterns
- **Real-time Applications**: Sub-millisecond response times for prediction
- **Batch Processing**: Efficient handling of pattern batches
- **Resource-constrained Environments**: Optimized memory usage and CPU utilization
- **Production Systems**: Industrial-strength reliability and monitoring

## Key Performance Features

### SIMD Vectorization
- Java 24 Vector API: Leverages 256-bit SIMD instructions
- Auto-vectorization: Automatic detection of vectorizable operations
- Cross-platform: Works on x86-64, ARM64, and other SIMD-capable architectures
- Performance Gains: 4-8x speedup over scalar implementations

### Advanced Concurrency
- Virtual Threads: Lightweight threading for I/O-bound operations
- Work Stealing: Efficient load balancing across CPU cores
- Lock-free Algorithms: Minimize contention and maximize throughput
- Structured Concurrency: Proper resource management and cancellation

### Memory Optimization
- Zero-copy Operations: Direct memory access where possible
- Buffer Pooling: Reuse of allocated memory buffers
- Garbage Collection Friendly: Minimal allocation in hot paths
- Memory-mapped I/O: Efficient handling of large datasets

## Module Structure

```
art-performance/
├── src/main/java/com/hellblazer/art/performance/
│   ├── algorithms/              # Vectorized algorithm implementations
│   │   ├── ValidationTest.java              # Algorithm validation utilities
│   │   ├── VectorizedART.java              # Base vectorized ART class
│   │   ├── VectorizedDeepARTMAP.java       # Hierarchical supervised learning
│   │   ├── VectorizedDeepARTMAPParameters.java
│   │   ├── VectorizedDeepARTMAPPerformanceStats.java
│   │   ├── VectorizedFuzzyART.java         # High-performance fuzzy ART
│   │   ├── VectorizedFuzzyWeight.java
│   │   ├── VectorizedHypersphereART.java   # Geometric clustering optimization
│   │   ├── VectorizedHypersphereParameters.java
│   │   ├── VectorizedHypersphereWeight.java
│   │   ├── VectorizedParameters.java       # Performance parameter base
│   │   ├── VectorizedPerformanceStats.java
│   │   └── VectorizedWeight.java
│   └── supervised/              # Vectorized supervised learning
│       ├── VectorizedARTMAP.java          # High-performance ARTMAP
│       ├── VectorizedARTMAPParameters.java
│       └── VectorizedARTMAPResult.java
└── src/test/java/com/hellblazer/art/performance/
    ├── algorithms/              # Algorithm correctness tests
    │   ├── VectorizedARTTest.java
    │   ├── VectorizedDeepARTMAPTest.java
    │   ├── VectorizedFuzzyARTTest.java
    │   └── VectorizedHypersphereARTTest.java
    ├── benchmarks/              # JMH performance benchmarks
    │   ├── VectorizedARTMAPBenchmark.java
    │   ├── VectorizedDeepARTMAPBenchmark.java
    │   ├── VectorizedFuzzyARTBenchmark.java
    │   └── VectorizedHypersphereARTBenchmark.java
    └── supervised/              # Supervised learning tests
        ├── VectorizedARTMAPAdvancedTest.java
        ├── VectorizedARTMAPRegressionTest.java
        ├── VectorizedARTMAPRobustnessTest.java
        ├── VectorizedARTMAPSimpleTest.java
        └── VectorizedARTMAPTest.java
```

## Vectorized Implementations

### VectorizedFuzzyART - High-Performance Pattern Recognition

The flagship implementation with full SIMD optimization:

```java
public class VectorizedFuzzyART extends FuzzyART {
    public VectorizedFuzzyART(VectorizedParameters parameters);
    
    // High-performance learning
    public int learn(Pattern input);                    // Vectorized single pattern
    public List<Integer> learnBatch(List<Pattern> patterns);  // Batch processing
    
    // Performance monitoring
    public VectorizedPerformanceStats getPerformanceStats();
    public boolean isSIMDEnabled();
    public int getParallelThreads();
    
    // Resource management
    public void compactMemory();
    public long getMemoryUsage();
}
```

**Performance Features:**
- **SIMD Distance Calculations**: Vector operations for fuzzy intersection
- **Parallel Category Competition**: Multi-threaded winner selection
- **Optimized Memory Layout**: Cache-friendly data structures
- **Batch Processing**: Amortized overhead across multiple patterns

**Usage Example:**
```java
var parameters = VectorizedParameters.builder()
    .vigilance(0.85)
    .learningRate(0.1)
    .inputDimensions(128)
    .maxCategories(10000)
    .enableSIMD(true)
    .parallelThreads(8)
    .build();

try (var vectorizedART = new VectorizedFuzzyART(parameters)) {
    // Process large dataset efficiently
    var patterns = loadLargeDataset();  // e.g., 1M patterns
    var categories = vectorizedART.learnBatch(patterns);
    
    // Performance analysis
    var stats = vectorizedART.getPerformanceStats();
    System.out.printf("Throughput: %.1f patterns/sec%n", stats.getThroughput());
    System.out.printf("SIMD speedup: %.1fx%n", stats.getSIMDSpeedup());
}
```

### VectorizedHypersphereART - Geometric Optimization

Specialized for geometric distance calculations:

```java
public class VectorizedHypersphereART extends HypersphereART {
    public VectorizedHypersphereART(VectorizedHypersphereParameters parameters);
    
    // Optimized distance calculations
    public double[] calculateDistancesVectorized(Pattern input);
    public List<Double> calculateBatchDistances(List<Pattern> patterns, int categoryIndex);
    
    // Geometric analysis
    public Vector3f[] getCenterVectorsOptimized();
    public float[] getRadiiOptimized();
}
```

**Optimization Features:**
- **SIMD Euclidean Distance**: Vector operations for distance calculations
- **Parallel Radius Updates**: Concurrent radius adaptation
- **Geometric Batching**: Efficient processing of geometric operations
- **Memory-aligned Data**: SIMD-friendly memory layouts

### VectorizedARTMAP - Supervised Learning Performance

High-throughput supervised learning:

```java
public class VectorizedARTMAP extends ARTMAP {
    public VectorizedARTMAP(VectorizedARTMAPParameters parameters);
    
    // Batch supervised learning
    public List<ARTMAPResult> learnBatch(List<Pattern> inputs, List<Pattern> targets);
    public List<ARTMAPResult> predictBatch(List<Pattern> inputs);
    
    // Asynchronous processing
    public CompletableFuture<ARTMAPResult> learnAsync(Pattern input, Pattern target);
    public Stream<ARTMAPResult> learnStream(Stream<Pair<Pattern, Pattern>> data);
    
    // Performance optimization
    public void enableParallelSearch(boolean enabled);
    public void setSearchStrategy(SearchStrategy strategy);
}
```

## Performance Parameters

### VectorizedParameters - Advanced Configuration

```java
public class VectorizedParameters extends FuzzyParameters {
    public static Builder builder();
    
    public static class Builder extends FuzzyParameters.Builder {
        // SIMD Configuration
        public Builder enableSIMD(boolean enabled);           // Enable Vector API
        public Builder vectorSize(int size);                  // Vector register size
        public Builder preferredSpecies(VectorSpecies<Float> species);  // SIMD species
        
        // Concurrency Configuration  
        public Builder parallelThreads(int threads);          // Thread pool size
        public Builder workStealingPool(boolean enabled);     // Work stealing threads
        public Builder virtualThreads(boolean enabled);       // Virtual thread execution
        
        // Memory Configuration
        public Builder batchSize(int size);                   // Optimal batch size
        public Builder memoryStrategy(MemoryStrategy strategy); // Memory management
        public Builder bufferPoolSize(int size);              // Buffer pool capacity
        public Builder preAllocateMemory(boolean enabled);    // Pre-allocation strategy
        
        // Performance Tuning
        public Builder cacheOptimization(CacheStrategy strategy); // CPU cache optimization
        public Builder numaAware(boolean enabled);            // NUMA topology awareness
        public Builder cpuAffinity(int[] cores);              // CPU core binding
        
        public VectorizedParameters build();
    }
}
```

### Memory Strategies

```java
public enum MemoryStrategy {
    CONSERVATIVE {
        // Minimize memory usage, slower performance
        // Buffer size: 1KB, No pre-allocation
    },
    BALANCED {
        // Balance memory and performance
        // Buffer size: 64KB, Limited pre-allocation  
    },
    PERFORMANCE {
        // Maximum performance, higher memory usage
        // Buffer size: 1MB, Aggressive pre-allocation
    },
    POOL_REUSE {
        // Buffer pooling for repeated operations
        // Shared buffer pools, Memory recycling
    },
    STREAMING {
        // Optimized for continuous data streams
        // Ring buffers, Pipeline optimization
    }
}
```

### Cache Optimization Strategies

```java
public enum CacheStrategy {
    L1_OPTIMIZED,    // Optimize for L1 cache (32KB typical)
    L2_OPTIMIZED,    // Optimize for L2 cache (256KB typical)
    L3_OPTIMIZED,    // Optimize for L3 cache (8MB typical)
    AUTO_DETECT,     // Automatically detect cache hierarchy
    DISABLE          // Disable cache optimizations
}
```

## Performance Monitoring

### VectorizedPerformanceStats

Comprehensive performance metrics:

```java
public class VectorizedPerformanceStats {
    // Throughput Metrics
    public double getThroughput();                    // Patterns per second
    public double getLatencyP50();                    // 50th percentile latency
    public double getLatencyP95();                    // 95th percentile latency  
    public double getLatencyP99();                    // 99th percentile latency
    
    // SIMD Metrics
    public double getSIMDUtilization();              // SIMD instruction usage %
    public double getSIMDSpeedup();                  // Speedup vs scalar code
    public VectorSpecies<Float> getVectorSpecies();  // Active SIMD species
    
    // Memory Metrics
    public long getMemoryUsageMB();                  // Current memory usage
    public long getMemoryAllocated();                // Total allocated memory
    public double getMemoryEfficiency();             // Utilization percentage
    public int getBufferPoolHitRate();               // Buffer pool hit rate %
    
    // Concurrency Metrics
    public double getThreadUtilization();            // Thread pool utilization %
    public int getActiveThreads();                   // Currently active threads
    public double getWorkStealingEfficiency();       // Work stealing hit rate %
    
    // Algorithm Metrics
    public int getCategoryCount();                   // Number of categories
    public double getAverageActivationTime();        // Per-pattern activation time
    public double getCategoryUtilization();          // Category usage distribution
    
    // System Metrics
    public double getCpuUtilization();               // CPU usage percentage
    public double getCacheHitRate();                 // CPU cache hit rate
    public Map<String, Object> getDetailedMetrics(); // Extended metrics
}
```

### Performance Profiling

```java
// Enable detailed profiling
var parameters = VectorizedParameters.builder()
    .enableProfiling(true)
    .profilingInterval(Duration.ofSeconds(1))
    .detailedMemoryTracking(true)
    .build();

try (var art = new VectorizedFuzzyART(parameters)) {
    // Training with monitoring
    var patterns = loadDataset();
    var startTime = System.nanoTime();
    
    var categories = art.learnBatch(patterns);
    
    var endTime = System.nanoTime();
    var stats = art.getPerformanceStats();
    
    // Performance report
    System.out.printf("Processing time: %.3f seconds%n", 
        (endTime - startTime) / 1e9);
    System.out.printf("Throughput: %.1f patterns/sec%n", 
        stats.getThroughput());
    System.out.printf("SIMD speedup: %.2fx%n", 
        stats.getSIMDSpeedup());
    System.out.printf("Memory efficiency: %.1f%%%n", 
        stats.getMemoryEfficiency());
    System.out.printf("Thread utilization: %.1f%%%n", 
        stats.getThreadUtilization());
}
```

## Benchmarking Suite

### JMH Performance Benchmarks

The module includes comprehensive JMH benchmarks:

```java
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS) 
@State(Scope.Benchmark)
public class VectorizedFuzzyARTBenchmark {
    
    @Param({"32", "64", "128", "256"})
    private int dimensions;
    
    @Param({"100", "500", "1000", "5000"})
    private int categoryCount;
    
    @Param({"1", "10", "100", "1000"})
    private int batchSize;
    
    @Benchmark
    public List<Integer> benchmarkVectorizedBatch(BenchmarkState state) {
        return state.vectorizedART.learnBatch(state.patterns);
    }
    
    @Benchmark
    public List<Integer> benchmarkScalarBatch(BenchmarkState state) {
        return state.patterns.stream()
            .map(state.scalarART::stepFit)
            .collect(Collectors.toList());
    }
}
```

### Running Benchmarks

```bash
# Run all performance benchmarks
mvn test -pl art-performance -Dtest=**/*Benchmark

# Run specific algorithm benchmarks  
mvn test -pl art-performance -Dtest=VectorizedFuzzyARTBenchmark

# Run with JMH profiling
mvn exec:java -Dexec.mainClass="org.openjdk.jmh.Main" \
  -Dexec.args="-prof gc -prof stack com.hellblazer.art.performance.benchmarks.*"

# Generate benchmark report
mvn exec:java -Dexec.mainClass="org.openjdk.jmh.Main" \
  -Dexec.args="-rf json -rff benchmark-results.json"
```

## Advanced Usage Patterns

### Streaming Data Processing

```java
public class StreamingProcessor {
    private final VectorizedFuzzyART art;
    private final ExecutorService executor;
    
    public StreamingProcessor(VectorizedParameters parameters) {
        this.art = new VectorizedFuzzyART(parameters);
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
    }
    
    public CompletableFuture<Stream<Integer>> processAsync(Stream<Pattern> patterns) {
        return CompletableFuture.supplyAsync(() ->
            patterns.parallel()
                .map(art::learn)
                .collect(Collectors.toList())
                .stream(),
            executor
        );
    }
    
    public void processRealtimeStream(BlockingQueue<Pattern> inputQueue,
                                    BlockingQueue<Integer> outputQueue) {
        var patterns = new ArrayList<Pattern>();
        
        while (!Thread.currentThread().isInterrupted()) {
            // Collect batch
            inputQueue.drainTo(patterns, art.getOptimalBatchSize());
            
            if (!patterns.isEmpty()) {
                // Process batch
                var results = art.learnBatch(patterns);
                
                // Output results
                outputQueue.addAll(results);
                patterns.clear();
            }
        }
    }
}
```

### Memory-constrained Processing

```java
// Configure for memory-constrained environment
var parameters = VectorizedParameters.builder()
    .memoryStrategy(MemoryStrategy.CONSERVATIVE)
    .batchSize(64)                    // Small batches
    .bufferPoolSize(10)               // Limited buffer pool
    .preAllocateMemory(false)         // No pre-allocation
    .enableSIMD(true)                 // Still use SIMD for speed
    .parallelThreads(2)               // Fewer threads
    .build();

try (var art = new VectorizedFuzzyART(parameters)) {
    // Process in small batches to limit memory usage
    var patterns = loadLargeDataset();
    var batchSize = parameters.getBatchSize();
    
    for (int i = 0; i < patterns.size(); i += batchSize) {
        var batch = patterns.subList(i, Math.min(i + batchSize, patterns.size()));
        var results = art.learnBatch(batch);
        
        // Process results immediately to free memory
        processResults(results);
        
        // Periodic memory cleanup
        if (i % (batchSize * 10) == 0) {
            art.compactMemory();
            System.gc(); // Suggest garbage collection
        }
    }
}
```

### High-throughput Server Application

```java
@Component
public class ARTClassificationService {
    private final VectorizedFuzzyART classifier;
    private final ARTPerformanceMonitor monitor;
    
    public ARTClassificationService() {
        var parameters = VectorizedParameters.builder()
            .memoryStrategy(MemoryStrategy.PERFORMANCE)
            .enableSIMD(true)
            .parallelThreads(Runtime.getRuntime().availableProcessors())
            .batchSize(1000)
            .cacheOptimization(CacheStrategy.AUTO_DETECT)
            .numaAware(true)
            .build();
            
        this.classifier = new VectorizedFuzzyART(parameters);
        this.monitor = new ARTPerformanceMonitor(classifier);
    }
    
    @PostMapping("/classify/batch")
    public ResponseEntity<ClassificationResponse> classifyBatch(
            @RequestBody List<double[]> patterns) {
        
        var denseVectors = patterns.stream()
            .map(DenseVector::new)
            .collect(Collectors.toList());
            
        var results = classifier.learnBatch(denseVectors);
        
        var response = new ClassificationResponse(
            results,
            monitor.getCurrentMetrics()
        );
        
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/performance")
    public ResponseEntity<VectorizedPerformanceStats> getPerformanceStats() {
        return ResponseEntity.ok(classifier.getPerformanceStats());
    }
}
```

## Testing Strategy

### Correctness Validation

All vectorized implementations are validated against reference implementations:

```java
@Test
void vectorizedResultsShouldMatchScalarResults() {
    var patterns = generateRandomPatterns(1000, 64);
    
    // Scalar reference
    var scalarART = new FuzzyART(FuzzyParameters.of(0.8, 0.1, 0.001));
    var scalarResults = patterns.stream()
        .map(scalarART::stepFit)
        .collect(Collectors.toList());
    
    // Vectorized implementation  
    var vectorizedParams = VectorizedParameters.builder()
        .vigilance(0.8)
        .learningRate(0.1)
        .enableSIMD(true)
        .build();
    
    var vectorizedART = new VectorizedFuzzyART(vectorizedParams);
    var vectorizedResults = vectorizedART.learnBatch(patterns);
    
    // Results must be identical
    assertEquals(scalarResults, vectorizedResults);
}
```

### Performance Regression Testing

```java
@Test 
void vectorizedShouldBeFasterThanScalar() {
    var patterns = generateRandomPatterns(10000, 128);
    
    // Measure scalar performance
    var scalarStart = System.nanoTime();
    var scalarResults = patterns.stream()
        .map(scalarART::stepFit)
        .collect(Collectors.toList());
    var scalarTime = System.nanoTime() - scalarStart;
    
    // Measure vectorized performance
    var vectorizedStart = System.nanoTime();
    var vectorizedResults = vectorizedART.learnBatch(patterns);
    var vectorizedTime = System.nanoTime() - vectorizedStart;
    
    // Vectorized should be significantly faster
    var speedup = (double) scalarTime / vectorizedTime;
    assertTrue(speedup >= 2.0, 
        String.format("Expected >=2x speedup, got %.2fx", speedup));
}
```

## Best Practices

### Performance Optimization

1. **Choose Optimal Batch Sizes**:
```java
// Use built-in optimization
int optimalBatch = VectorizedART.getOptimalBatchSize();

// Or calculate based on your data
int optimalBatch = Math.max(64, 
    Math.min(1024, Runtime.getRuntime().availableProcessors() * 128));
```

2. **Configure Memory Strategy**:
```java
// For throughput-critical applications
.memoryStrategy(MemoryStrategy.PERFORMANCE)

// For memory-constrained environments  
.memoryStrategy(MemoryStrategy.CONSERVATIVE)

// For long-running services
.memoryStrategy(MemoryStrategy.POOL_REUSE)
```

3. **Enable SIMD Optimization**:
```java
// Always enable SIMD unless targeting legacy hardware
.enableSIMD(true)
.vectorSize(256)  // Use 256-bit SIMD if available
```

### Resource Management

1. **Use Try-with-Resources**:
```java
try (var art = new VectorizedFuzzyART(parameters)) {
    // Processing code
} // Automatic cleanup of threads and memory
```

2. **Monitor Performance**:
```java
// Regular performance monitoring
if (art.getPerformanceStats().getMemoryUsageMB() > MAX_MEMORY_MB) {
    art.compactMemory();
}

// Automatic alerting
if (art.getPerformanceStats().getThroughput() < MIN_THROUGHPUT) {
    alertingService.sendAlert("ART performance degraded");
}
```

3. **Graceful Shutdown**:
```java
// In Spring Boot applications
@PreDestroy
public void shutdown() {
    if (art != null) {
        art.close(); // Properly shutdown thread pools and release resources
    }
}
```

The art-performance module transforms ART algorithms into production-ready, high-performance implementations suitable for demanding real-world applications while maintaining full compatibility with the core algorithms.