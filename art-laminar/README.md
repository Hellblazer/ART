# ART Laminar Circuits Module

Integration of Grossberg's canonical laminar circuit with SIMD batch processing for ART neural networks.

## Overview

This module implements Grossberg's canonical laminar circuit architecture from "A Canonical Laminar Neocortical Circuit Whose Bottom-Up, Horizontal, and Top-Down Pathways Control Attention, Learning, and Prediction" with high-performance SIMD batch processing optimizations.

## Current Status

**Phase 6C Complete**: Stateful batch processing with mini-batch SIMD optimization
- **Test Suite**: 402/402 tests passing (100%)
- **Performance**: 1.30x speedup over baseline sequential processing
- **Semantic Equivalence**: 0.00e+00 max difference (bit-exact)

## Key Features

### Laminar Circuit Architecture
- **6 cortical layers**: Layers 1, 2/3, 4, 5, 6 with biologically-inspired dynamics
- **Multiple pathways**: Bottom-up, top-down, and horizontal connections
- **ART integration**: Complete ART matching and learning dynamics
- **Temporal dynamics**: Multi-scale processing with proper time scale separation

### High-Performance SIMD Processing
- **Mini-batch optimization**: 32-pattern batches for optimal SIMD efficiency
- **Java Vector API**: Hardware-accelerated SIMD operations
- **Automatic fallback**: Graceful degradation to sequential when SIMD not beneficial
- **Bit-exact equivalence**: 0.00e+00 difference from sequential processing

### State Management
- **Stateful processing**: Proper state evolution across patterns
- **Sequential semantics**: Pattern N+1 sees effects of Pattern N
- **Layer-level SIMD**: SIMD optimization within each layer's processing
- **ART learning**: Sequential category learning with SIMD-optimized layers

## Performance

**Throughput**: 1049.7 patterns/sec (vs 807 baseline)
- **Single-pattern**: 1239.04 μs/pattern
- **Batch (Phase 6C)**: 952.64 μs/pattern
- **Speedup**: 1.30x ✅

**Optimized Layers**:
- Layer 4: SIMD shunting dynamics
- Layer 5: SIMD burst firing and amplification
- Layer 6: SIMD ART matching rule
- Layer 2/3: Sequential (bipole network complexity)
- Layer 1: Sequential (complex attention state)

## Structure

```
art-laminar/
├── src/main/java/com/hellblazer/art/laminar/
│   ├── batch/                  # SIMD batch processing
│   │   ├── BatchDataLayout.java           # Transpose infrastructure
│   │   ├── BatchShuntingDynamics.java     # Exact dynamics
│   │   ├── Layer4SIMDBatch.java           # Layer 4 SIMD
│   │   ├── Layer5SIMDBatch.java           # Layer 5 SIMD
│   │   ├── Layer6SIMDBatch.java           # Layer 6 SIMD
│   │   ├── Layer23SIMDBatch.java          # Layer 2/3 SIMD
│   │   ├── Layer1SIMDBatch.java           # Layer 1 SIMD
│   │   └── StatefulBatchProcessor.java    # Stateful interface
│   ├── layers/                 # Layer implementations
│   │   ├── Layer1Implementation.java      # Top-down priming
│   │   ├── Layer23Implementation.java     # Horizontal grouping
│   │   ├── Layer4Implementation.java      # Thalamic input
│   │   ├── Layer5Implementation.java      # Output amplification
│   │   └── Layer6Implementation.java      # ART matching
│   ├── integration/            # Circuit integration
│   │   └── ARTLaminarCircuit.java         # Full circuit
│   ├── canonical/              # Canonical circuit (temporal dynamics)
│   ├── temporal/               # Temporal chunking (LIST PARSE)
│   └── network/                # Bipole cell networks
└── src/test/java/              # Comprehensive test suite (402 tests)
```

## Phase Progression

### ✅ Phase 1-2: Temporal Dynamics & Chunking
- Temporal dynamics integration with RK4
- LIST PARSE temporal chunking
- Working memory with primacy gradient
- Time scale separation validation

### ✅ Phase 3: Layer SIMD Foundation
- Transpose-and-vectorize architecture
- Layer 4, 5, 6 SIMD batch processing
- BatchShuntingDynamics for exact equivalence
- Cost-benefit analysis for automatic SIMD selection

### ✅ Phase 5: Complete Layer SIMD
- Layer 1 SIMD (top-down processing)
- Layer 2/3 SIMD (without bipole network)
- Individual layer semantic equivalence (0.00e+00)
- All 5 layers with SIMD capability

### ✅ Phase 6A: Stateful Batch Processing
- StatefulBatchProcessor interface
- Sequential pattern processing with layer-level SIMD
- Proper state evolution maintained
- Circuit integration complete

### ✅ Phase 6C: Performance Optimization
- Mini-batch SIMD (32 patterns per batch)
- 1.30x speedup achieved
- Interface casting elimination
- Production-ready performance

## Building and Testing

```bash
# Compile
mvn compile -pl art-laminar

# Run all tests (402 tests)
mvn test -pl art-laminar

# Run specific test category
mvn test -pl art-laminar -Dtest=BatchProcessingTest
mvn test -pl art-laminar -Dtest=Layer4SIMDBatchTest
mvn test -pl art-laminar -Dtest=ARTLaminarCircuitTest
```

**Test Results**: 402/402 tests passing (100%)
- Batch Processing: 14 tests (Phase 6C validation)
- Layer SIMD: 73 tests (all layers)
- Integration: 6 tests (circuit-level)
- Canonical Circuit: 59 tests
- Temporal Dynamics: 50 tests
- Validation: 200+ tests

## Usage Example

```java
// Create circuit with default parameters
var params = ARTCircuitParameters.builder(256)
    .vigilance(0.85)
    .learningRate(0.8)
    .maxCategories(100)
    .build();

var circuit = new ARTLaminarCircuit(params);

// Process single pattern
var pattern = new DenseVector(inputData);
var result = circuit.process(pattern);

// Process batch with SIMD optimization (Phase 6C)
var patterns = new Pattern[100];
// ... fill patterns ...
var batchResult = circuit.processBatch(patterns);

// Access performance statistics
var stats = batchResult.statistics();
System.out.printf("Speedup: %.2fx%n", stats.getSpeedup(baselineTime));
System.out.printf("Throughput: %.1f patterns/sec%n",
    stats.getPatternsPerSecond());
```

## Dependencies

- **art-core**: Core ART interfaces and pattern types
- **art-temporal**: Temporal dynamics (shunting, transmitter gates)
- **art-performance**: Vectorization interfaces
- **Java 24** with preview features and Vector API

## Documentation

All phase documentation has been archived in `archive/` directory:
- Phase completion reports (Phase 3, 5, 6A, 6C)
- Design documents
- Performance analyses
- Validation reports

Current documentation:
- **README.md**: This overview (updated)
- **PHASE6C-COMPLETION.md**: Latest completion report
- **archive/**: Historical development documentation

## Theoretical Foundation

Based on Grossberg's canonical laminar circuit:
- Shunting dynamics for neural activation
- ART matching rule for category learning
- Multi-scale temporal processing
- Top-down/bottom-up integration

**Key Equations** (see PHASE6C-COMPLETION.md for details):
- Shunting dynamics: `dx/dt = -Ax + (B-x)E - xI`
- ART matching: `M = |X ∩ E| / |X|`
- Vigilance test: `if M ≥ ρ: learn; else: search`

## Performance Characteristics

**Optimal Configuration**:
- Batch size: ≥32 patterns (SIMD threshold)
- Dimension: ≥64 (SIMD beneficial)
- Pattern type: Independent or weakly dependent

**Expected Speedup**:
- Small batches (< 32): ~1.0x (sequential faster)
- Medium batches (32-64): ~1.15x (SIMD emerging)
- Large batches (≥64): ~1.30x (SIMD optimal)

## Known Limitations

- **Bipole network**: Layer 2/3 horizontal grouping falls back to sequential (Phase 6B future work)
- **Layer 1 & 2/3 SIMD**: Complex state makes single-pattern SIMD overhead high
- **Mini-batch boundaries**: Last mini-batch may be < 32 (falls back to scalar)
- **Vector API**: Requires Java 24 with preview features

## Future Work

- **Phase 6B** (optional): BipoleCellNetwork SIMD with left/right separation
- **Phase 6D** (optional): Increase mini-batch size to 64
- **GPU acceleration**: Port SIMD operations to GPU kernels
- **Adaptive batching**: Runtime batch size optimization

## License

GNU Affero General Public License v3.0

## Citation

Based on Grossberg, S. (2013). "Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world." Neural Networks, 37, 1-47.

Implementation incorporates latest research from:
- Raizada, R. D. S., & Grossberg, S. (2003). "Towards a theory of the laminar architecture of cerebral cortex."
- Grossberg, S., & Kazerounian, S. (2016). "The LIST PARSE model of serial learning and recognition."
