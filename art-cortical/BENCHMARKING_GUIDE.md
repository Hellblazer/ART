# art-cortical Benchmarking Guide

**Phase 4A: Performance Measurement Infrastructure**
**Date**: October 3, 2025

---

## Overview

This guide covers the JMH (Java Microbenchmark Harness) benchmarking suite for art-cortical. The benchmark infrastructure establishes baseline performance metrics and identifies optimization opportunities.

---

## Quick Start

### Run All Benchmarks
```bash
./run-benchmarks.sh all
```

### Run Specific Benchmark Suites
```bash
./run-benchmarks.sh layer      # Layer-level benchmarks
./run-benchmarks.sh circuit    # Full circuit benchmarks
./run-benchmarks.sh learning   # Learning rule benchmarks
./run-benchmarks.sh dynamics   # Shunting dynamics benchmarks
./run-benchmarks.sh simd       # SIMD batch processing
```

### Run with Profiling
```bash
./run-benchmarks.sh learning gc    # With GC profiler
./run-benchmarks.sh all stack      # With stack profiler
```

---

## Benchmark Suites

### 1. LayerBenchmarks
**Location**: `src/test/java/com/hellblazer/art/cortical/benchmarks/LayerBenchmarks.java`

**Purpose**: Measure individual layer processing performance

**Benchmarks**:
- `benchmarkLayer4Forward` - Layer 4 bottom-up (fastest layer, 10-50ms)
- `benchmarkLayer23Forward` - Layer 2/3 grouping (30-150ms)
- `benchmarkLayer1Forward` - Layer 1 attention (200-1000ms, slowest)
- `benchmarkLayer6TopDown` - Layer 6 feedback (100-500ms)
- `benchmarkLayer4Reset` - Layer state reset cost

**Parameters**:
- `dimension`: [64, 128, 256, 512] neurons

**Expected Results**:
- Layer 4: ~10-50 µs/pattern
- Layer 2/3: ~20-100 µs/pattern
- Layer 1: ~50-200 µs/pattern
- Layer 6: ~30-150 µs/pattern

**Run Individually**:
```bash
mvn test -Dtest=LayerBenchmarks#benchmarkLayer4Forward
```

---

### 2. CircuitBenchmarks
**Location**: `src/test/java/com/hellblazer/art/cortical/benchmarks/CircuitBenchmarks.java`

**Purpose**: Measure full 6-layer cortical circuit end-to-end performance

**Benchmarks**:
- `benchmarkFullCircuit` - Complete circuit processing
- `benchmarkBottomUpOnly` - Layer 4 → 2/3 → 1 pathway
- `benchmarkTopDownOnly` - Layer 6 → 2/3 pathway
- `benchmarkDetailedOutput` - With detailed activation tracking

**Parameters**:
- `dimension`: [64, 128, 256] neurons

**Expected Results**:
- Full circuit: ~100-500 µs/pattern
- Bottom-up only: ~80-350 µs/pattern
- Top-down only: ~50-250 µs/pattern

**Run Individually**:
```bash
mvn test -Dtest=CircuitBenchmarks#benchmarkFullCircuit
```

---

### 3. LearningBenchmarks
**Location**: `src/test/java/com/hellblazer/art/cortical/benchmarks/LearningBenchmarks.java`

**Purpose**: Measure learning rule weight update performance

**Benchmarks**:
- `benchmarkHebbianLearning` - Classic Hebbian plasticity
- `benchmarkBCMLearning` - BCM sliding threshold
- `benchmarkInstarLearning` - ART instar (bottom-up)
- `benchmarkOutstarLearning` - ART outstar (top-down)
- `benchmarkBidirectionalLearning` - Combined instar/outstar
- `benchmarkWeightMatrixAllocation` - Allocation overhead
- `benchmarkWeightMatrixCopy` - Copy overhead

**Parameters**:
- `preSize`: [32, 64, 128, 256] input neurons
- `postSize`: [32, 64, 128] output neurons

**Expected Results**:
- Hebbian: ~100-500 ns/update
- BCM: ~200-1000 ns/update (includes threshold adaptation)
- Instar/Outstar: ~150-600 ns/update
- Bidirectional: ~300-1200 ns/update

**Run Individually**:
```bash
mvn test -Dtest=LearningBenchmarks#benchmarkHebbianLearning
```

---

### 4. DynamicsBenchmarks
**Location**: `src/test/java/com/hellblazer/art/cortical/benchmarks/DynamicsBenchmarks.java`

**Purpose**: Measure shunting dynamics convergence performance

**Benchmarks**:
- `benchmarkSingleIteration` - One dynamics update step
- `benchmarkFastConvergence` - Layer 4 convergence (10ms)
- `benchmarkMediumConvergence` - Layer 2/3 convergence (30ms)
- `benchmarkSlowConvergence` - Layer 1 convergence (200ms)
- `benchmarkLyapunovEnergy` - Energy computation
- `benchmarkDynamicsReset` - State reset cost

**Parameters**:
- `dimension`: [64, 128, 256, 512] neurons

**Expected Results**:
- Single iteration: ~10-50 µs
- Fast convergence: ~50-500 µs (5-10 iterations)
- Medium convergence: ~100-1000 µs
- Slow convergence: ~200-2000 µs

**Run Individually**:
```bash
mvn test -Dtest=DynamicsBenchmarks#benchmarkFastConvergence
```

---

### 5. SIMDBenchmark
**Location**: `src/test/java/com/hellblazer/art/cortical/batch/SIMDBenchmark.java`

**Purpose**: Measure SIMD batch processing speedup

**Benchmarks**:
- `baselineSequential` - Non-SIMD baseline
- `simdBatchProcessing` - Full SIMD pipeline
- `transposeOnly` - Transpose overhead
- `simdOperationsOnly` - Pure SIMD benefit
- `batchCreation` - Batch creation cost
- `drivingStrengthSIMD` - Driving strength application
- `saturationSIMD` - Saturation function

**Parameters**:
- `batchSize`: [32, 64, 128, 256] patterns
- `dimension`: [64, 128, 256] neurons

**Expected Results**:
- Batch 32: 1.30x speedup (art-laminar baseline)
- Batch 64: 1.40x-1.50x speedup (Phase 1B target)
- Batch 128+: 1.50x+ speedup (optimal)

**Run Individually**:
```bash
mvn test -Dtest=SIMDBenchmark#simdBatchProcessing
```

---

## Profiling

### GC Profiler (Allocation Tracking)
Measures garbage collection and memory allocation:
```bash
./run-benchmarks.sh learning gc
```

**Output Includes**:
- Allocation rate (MB/sec)
- GC count and time
- Normalized allocation per operation

### Stack Profiler (Hotspot Analysis)
Identifies CPU hotspots:
```bash
./run-benchmarks.sh all stack
```

**Output Includes**:
- Method call stacks
- Time spent in each method
- Hotspot identification

### Perf Profiler (Linux Only)
Hardware performance counters:
```bash
./run-benchmarks.sh dynamics perf
```

**Output Includes**:
- CPU cycles
- Cache misses
- Branch mispredictions

---

## Interpreting Results

### Throughput Mode (ops/sec)
Higher is better. Measures how many operations can be completed per second.

**Example**:
```
Benchmark                                  Mode  Cnt      Score     Error  Units
LayerBenchmarks.benchmarkLayer4Forward    thrpt    5  50000.123 ± 1234.567  ops/s
```

This means Layer 4 can process ~50,000 patterns/second.

### Average Time Mode (µs/op)
Lower is better. Measures time per operation.

**Example**:
```
Benchmark                                  Mode  Cnt   Score   Error  Units
LayerBenchmarks.benchmarkLayer4Forward     avgt    5  20.123 ± 0.456  us/op
```

This means Layer 4 takes ~20µs per pattern.

---

## Baseline Performance Report

### Phase 4A Baseline (October 3, 2025)

**Hardware**: [To be measured]
**Java**: Java 24
**JVM Args**: `--add-modules=jdk.incubator.vector -Xmx2G`

#### Layer Performance (dimension=128)
| Layer | Time (µs/op) | Throughput (ops/s) |
|-------|--------------|-------------------|
| Layer 4 | TBD | TBD |
| Layer 2/3 | TBD | TBD |
| Layer 1 | TBD | TBD |
| Layer 6 | TBD | TBD |

#### Circuit Performance (dimension=128)
| Benchmark | Time (µs/op) | Throughput (ops/s) |
|-----------|--------------|-------------------|
| Full Circuit | TBD | TBD |
| Bottom-Up Only | TBD | TBD |
| Top-Down Only | TBD | TBD |

#### Learning Performance (preSize=128, postSize=64)
| Learning Rule | Time (ns/op) | Allocations (bytes/op) |
|---------------|--------------|----------------------|
| Hebbian | TBD | TBD |
| BCM | TBD | TBD |
| Instar | TBD | TBD |
| Outstar | TBD | TBD |

#### SIMD Performance (batchSize=64, dimension=128)
| Benchmark | Time (µs/op) | Speedup vs Baseline |
|-----------|--------------|-------------------|
| Sequential | TBD | 1.00x |
| SIMD Batch | TBD | TBD |

---

## Next Steps

### Phase 4B: Layer 4 SIMD Optimization
**Target**: 10-100x speedup with vectorized batch processing

**Approach**:
1. Measure baseline with current SIMDBenchmark
2. Implement `Layer4SIMDBatch` optimizations
3. Benchmark improved version
4. Compare speedup

### Phase 4C: Shunting Dynamics Parallelization
**Target**: 3-5x speedup with parallel neuron updates

**Approach**:
1. Measure baseline with DynamicsBenchmarks
2. Implement parallel dynamics updates
3. Benchmark with different thread counts
4. Identify optimal parallelization strategy

### Phase 4D: Learning Vectorization
**Target**: 5-10x speedup with SIMD weight updates

**Approach**:
1. Measure baseline with LearningBenchmarks
2. Implement SIMD variants (HebbianLearningSIMD, etc.)
3. Benchmark vectorized versions
4. Validate precision preservation

---

## Best Practices

### 1. Warm-Up
JMH automatically warms up the JVM. Default: 3 iterations, 1 second each.

### 2. Measurement
After warm-up, JMH runs measurement iterations. Default: 5 iterations, 2 seconds each.

### 3. Forking
JMH forks a new JVM to avoid polluting results. Default: 1 fork.

### 4. Blackholes
Always consume benchmark results with `Blackhole.consume()` to prevent dead code elimination.

### 5. State Management
Use `@State(Scope.Benchmark)` for shared state across iterations.

### 6. Multiple Parameters
Use `@Param` for testing different configurations (dimensions, batch sizes).

---

## Troubleshooting

### Issue: Benchmark Fails to Compile
**Solution**: Run `mvn test-compile` to check for compilation errors.

### Issue: JVM Crashes During Benchmark
**Cause**: SIMD code may crash on unsupported hardware.
**Solution**: Check that Java Vector API is available (`--add-modules=jdk.incubator.vector`).

### Issue: Results Vary Significantly
**Cause**: System load, GC, or JIT compilation variance.
**Solution**: Increase warmup/measurement iterations or use `-wi 5 -i 10`.

### Issue: OutOfMemoryError
**Cause**: Large batch sizes or dimensions.
**Solution**: Increase heap size with `-Xmx4G` or reduce parameters.

---

## References

- **JMH Documentation**: https://github.com/openjdk/jmh
- **Java Vector API**: https://openjdk.org/jeps/338
- **Performance Tuning**: PHASE_4_PERFORMANCE_PLAN.md

---

**Status**: Phase 4A Complete ✅
**Next**: Run baseline benchmarks and proceed to Phase 4B
