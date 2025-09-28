# Temporal Performance

High-performance vectorized implementations of temporal ART components using Java Vector API.

## Overview

This module provides SIMD-optimized versions of temporal processing components, achieving significant speedups through vectorization. All implementations maintain full API compatibility with standard versions while leveraging modern CPU vector instructions.

## Performance Results

### Measured Speedups

| Component | Standard Time | Vectorized Time | Speedup |
|-----------|---------------|-----------------|---------|
| Working Memory Store | 34.20ms | 2.44ms | **14.00x** |
| Shunting Dynamics | 96.65ms | 63.20ms | **1.53x** |
| Multi-Scale Dynamics | 93.87ms | 92.76ms | 1.01x |

*Measurements: dimension=100, iterations=1000, Java 24, Apple M1*

## Vectorized Components

### VectorizedWorkingMemory

SIMD-optimized working memory with vectorized:
- Pattern storage operations
- Primacy gradient computation
- Temporal pattern formation
- Matrix operations

**Usage:**
```java
var params = WorkingMemoryParameters.paperDefaults();
var vectorizedMemory = new VectorizedWorkingMemory(params);

// Identical API to standard version
vectorizedMemory.storeItem(pattern, timestamp);
var temporalPattern = vectorizedMemory.getTemporalPattern();
```

### VectorizedShuntingDynamics

Vectorized shunting network with optimized:
- Activation updates
- Lateral inhibition computation
- Energy calculations
- Convergence checking

**Usage:**
```java
var params = ShuntingParameters.competitiveDefaults(dimension);
var vectorizedShunting = new VectorizedShuntingDynamics(params, dimension);

// Process with SIMD
vectorizedShunting.setExcitatoryInput(input);
var state = vectorizedShunting.evolve(currentState, dt);
```

### VectorizedMaskingField

Optimized masking field with vectorized:
- Distance computations
- Weight updates
- Multi-scale competition
- Winner selection

### VectorizedTemporalART

Complete temporal ART with all vectorized components:
```java
var params = TemporalARTParameters.speechDefaults();
var vectorizedTemporal = new VectorizedTemporalART(params);

// Full speedup across all operations
vectorizedTemporal.processSequence(sequence);
```

## Optimization Techniques

### SIMD Operations
- Vector lane processing for array operations
- Parallel reduction for summations
- Masked operations for conditional updates
- Fused multiply-add instructions

### Memory Optimization
- Cache-aligned data structures
- Minimized memory allocations
- Reused temporary buffers
- Optimized memory access patterns

### Algorithm Optimization
- Loop unrolling where beneficial
- Strength reduction for expensive operations
- Lazy evaluation of unused components
- Early termination for convergence

## Vector API Usage

The implementation uses Java's Vector API (incubator):

```java
// Example: Vectorized dot product
public double dotProductVectorized(double[] a, double[] b) {
    var species = DoubleVector.SPECIES_PREFERRED;
    double sum = 0.0;
    int i = 0;

    // Vector loop
    for (; i < species.loopBound(a.length); i += species.length()) {
        var va = DoubleVector.fromArray(species, a, i);
        var vb = DoubleVector.fromArray(species, b, i);
        sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < a.length; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}
```

## Benchmarking

### Running Benchmarks

```bash
# Quick performance test
mvn test -pl art-temporal/temporal-performance -Dtest=QuickPerformanceTest

# JMH benchmarks (when enabled)
mvn test -pl art-temporal/temporal-performance -Dtest=PerformanceBenchmark
```

### Performance Test

The `QuickPerformanceTest` provides rapid performance comparison:
- Configurable dimensions and iterations
- Direct timing measurements
- Speedup calculations
- Warmup iterations

### JMH Benchmarks

The `PerformanceBenchmark` provides detailed measurements:
- Multiple parameter configurations
- Statistical analysis
- Throughput and latency metrics
- Multi-threaded testing

## Compatibility

### Fallback Behavior
- Automatic fallback to scalar operations on unsupported hardware
- Graceful degradation for edge cases
- Full API compatibility maintained

### Hardware Requirements
- Best performance on CPUs with AVX2/AVX-512
- Apple Silicon (M1/M2) with NEON
- Automatic detection of vector width

## Integration

### Drop-in Replacement

Vectorized implementations are drop-in replacements:

```java
// Standard version
var standard = new WorkingMemory(params);

// Vectorized version - same API
var vectorized = new VectorizedWorkingMemory(params);
```

### Performance Monitoring

Track performance improvements:
```java
var vectorized = new VectorizedTemporalART(params);
var stats = vectorized.getPerformanceStats();

System.out.println("Vector operations: " + stats.vectorOpsCount());
System.out.println("Speedup factor: " + stats.averageSpeedup());
```

## Test Coverage

24 tests validate:
- Functional equivalence with standard versions
- Performance improvements
- Edge case handling
- Vector API usage
- Memory efficiency

## Future Optimizations

Potential improvements:
- GPU acceleration via OpenCL/CUDA
- Further vectorization of remaining scalar code
- Parallel processing across sequences
- Custom SIMD intrinsics for critical paths

## Dependencies

- temporal-core: Base interfaces
- temporal-dynamics: Standard implementations
- temporal-memory: Standard working memory
- temporal-masking: Standard masking field
- temporal-integration: Standard TemporalART
- Java Vector API: SIMD operations (incubator)