# ART Laminar Circuits Module

Integration of Grossberg's canonical laminar circuit with temporal dynamics for ART neural networks.

## Overview

This module integrates the temporal dynamics from art-temporal with laminar pathway processing, implementing Grossberg's canonical laminar circuit architecture from "A Canonical Laminar Neocortical Circuit Whose Bottom-Up, Horizontal, and Top-Down Pathways Control Attention, Learning, and Prediction". It provides multi-scale temporal dynamics with proper time scale separation between fast shunting dynamics (~10-100ms) and slow transmitter habituation (~500-5000ms).

## Structure

```
art-laminar/
├── canonical/               # Canonical circuit integration
│   ├── ShuntingPathwayDecorator.java      # Temporal dynamics decorator
│   ├── TemporallyIntegratedPathway.java   # Integration interface
│   ├── SimpleIntegrator.java              # RK4 integrator
│   └── TimeScale.java                     # Time scale enumeration
├── temporal/                # Temporal chunking (LIST PARSE)
│   ├── TemporalChunkingLayer.java         # Chunking interface
│   ├── TemporalChunkingLayerDecorator.java # Chunking decorator
│   ├── TemporalChunk.java                 # Chunk representation
│   ├── ChunkingState.java                 # State management
│   ├── ChunkingParameters.java            # Chunking parameters
│   └── ChunkingStatistics.java            # Statistics tracking
├── core/                    # Core pathway interfaces
├── impl/                    # Pathway implementations
├── performance/             # Vectorized variants
└── test/
    ├── canonical/           # Integration tests
    │   ├── CanonicalCircuitTestBase.java      # Test utilities
    │   ├── ShuntingDynamicsPathwayTest.java   # 13 tests
    │   ├── TransmitterDynamicsTest.java       # 12 tests
    │   └── TimeScaleSeparationTest.java       # 9 tests
    └── temporal/            # Chunking tests
        └── TemporalChunkingTest.java          # 15 tests
```

## Phase 1 Week 1 Integration (Complete)

Successfully integrated temporal dynamics into laminar pathways:

- ✅ Created integration interfaces and decorator pattern
- ✅ Implemented multi-scale RK4 integration
- ✅ Fixed critical bugs in art-temporal core (TransmitterDynamics, ShuntingDynamics)
- ✅ Fixed WorkingMemory transmitter signal management
- ✅ 34 comprehensive tests validating equation correctness
- ✅ 100% test pass rate (54 total tests)

**Key Features:**
- Non-invasive decoration of existing pathways
- Proper time scale separation (50x between fast/slow)
- Transmitter habituation for primacy gradient effects
- Shunting dynamics with lateral inhibition
- Backward compatible with existing pathway implementations

## Phase 1 Week 2 - Temporal Chunking (✅ COMPLETE)

Successfully implemented LIST PARSE model temporal chunking for layers:

- ✅ Temporal chunking infrastructure (6 classes)
- ✅ TemporalChunkingLayer interface
- ✅ Decorator pattern for non-invasive layer enhancement
- ✅ Activation history management (Miller's 7±2)
- ✅ Chunk formation with coherence thresholds
- ✅ Chunk decay and pruning dynamics
- ✅ Temporal context extraction from chunks
- ✅ 15 comprehensive tests (100% passing)
- ✅ LayerState record with activation + context
- ✅ Context weight control for blending
- ✅ 13 layer state tests (100% passing)
- ✅ MultiScaleCoordinator (FAST/MEDIUM/SLOW coordination)
- ✅ MultiScaleLayerProcessor (integrated processing)
- ✅ 12 multi-scale coordination tests (100% passing)
- ✅ WorkingMemoryLayerBridge (STORE 2 + LIST PARSE)
- ✅ Integrated primacy gradient + chunking
- ✅ 10 WorkingMemory integration tests (100% passing)
- ✅ Comprehensive paper validation document
- ✅ 12/12 specifications validated against Grossberg & Kazerounian (2016)

**Chunking Features:**
- Chunk types: SMALL (1-3), MEDIUM (4-5), LARGE (6-7), SUPER (8-12)
- Coherence-based chunk formation (cosine similarity)
- Exponential decay with configurable rates
- Representative pattern computation (weighted averaging)
- Statistics tracking (formation rate, average size, coherence)
- Configurable parameters from Grossberg & Kazerounian (2016)

## Vectorized Variant

The `VectorizedLaminarCircuit` class uses the Java Vector API (incubator) for SIMD operations. Performance improvements vary based on hardware and data characteristics. The implementation targets ARM64/Apple Silicon but should work on other platforms supporting the Vector API.

## Building and Testing

```bash
# Compile
mvn compile -pl art-laminar

# Run tests
mvn test -pl art-laminar

# Specific test
mvn test -pl art-laminar -Dtest=VectorizedLaminarCircuitTest
```

**Current test results: 104/104 tests passing (100%)**
- 3 BasicCompilationTest
- 5 VectorizedLaminarCircuitTest
- 11 BackwardCompatibilityTest
- 13 ShuntingDynamicsPathwayTest
- 12 TransmitterDynamicsTest
- 1 TransmitterIntegrationDebugTest
- 9 TimeScaleSeparationTest
- 18 TemporalChunkingTest (includes layer state tests)
- 10 LayerStateTest
- 12 MultiScaleCoordinationTest
- 10 WorkingMemoryIntegrationTest

## Dependencies

- **art-core**: Core ART interfaces and pattern types
- **art-temporal**: Temporal dynamics (shunting, transmitter gates, integration)
  - temporal-core: DynamicalSystem, ShuntingDynamics, TransmitterDynamics
  - temporal-memory: WorkingMemory with primacy gradient
- **art-performance**: Vectorization interfaces
- **Java 24** with preview features and incubator modules

## Theoretical Foundation

The shunting dynamics and transmitter habituation equations are consistent with Grossberg's canonical laminar circuit framework. These fundamental dynamics appear across multiple Grossberg papers and are implemented in the art-temporal module.

**Shunting Dynamics:**
```
dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij
```

**Transmitter Habituation:**
```
dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
```

**Time Scales:**
- Fast (10-100ms): Neural activation, sensory processing
- Slow (500-5000ms): Transmitter depletion, learning updates

## Validation

- Equation implementations validated against paper specifications
- Numerical stability verified across parameter ranges
- Time scale separation confirmed (>50x ratio)
- Primacy gradient effects demonstrated in WorkingMemory
- Integration accuracy: RK4 with adaptive clamping

## Limitations

- Experimental implementation for research purposes
- Performance depends on hardware and JVM optimizations
- Vector API still in incubator status (Java 24)
- Focus on temporal dynamics; spatial processing ongoing

## License

GNU Affero General Public License v3.0