# ART Cortical Architecture Enhancement Plan

**Version**: 1.2
**Date**: 2025-10-03
**Status**: PHASE 1 COMPLETE - IN PROGRESS (Phase 2 Ready)
**Document Type**: Architecture Design & Implementation Roadmap
**Target Modules**: art-cortical, art-laminar, art-temporal

---

## REVISION HISTORY

### Version 1.2 (2025-10-03) - Phase 1 Completion Update

**Phase 1 Status: ✅ COMPLETE** (via Phase 4 Performance Optimization):

**Delivered** (exceeding original targets):
1. ✅ **Layer4SIMDBatch.java** - SIMD batch processing (380 lines, 13 tests)
2. ✅ **11.81x speedup** on x86-64 AVX-512 (far exceeds 1.40x target)
3. ✅ **Platform-aware fallback** - Automatic sequential fallback on ARM64 (2-lane vectors)
4. ✅ **Bit-exact correctness** - Maintains 1e-10 mathematical precision
5. ✅ **Additional optimizations** not in original plan:
   - ShuntingDynamicsParallel.java (3-6x speedup)
   - HebbianLearningSIMD + BCMLearningSIMD (3-8x speedup)
   - WeightMatrixPool + HebbianLearningPooled (99% memory reduction)
   - CorticalCircuitOptimized (1.2-1.4x speedup)

**Performance Results**:
- x86-64: 15-25x combined speedup (SIMD + parallelism + memory)
- ARM64: 4-7x combined speedup (parallelism + memory, SIMD fallback)
- Test validation: 423 tests passing, 0 failures, 0 regressions
- Production certification: APPROVED

**Documentation**:
- 9 Phase 4 completion reports created
- PLATFORM_PERFORMANCE_NOTES.md added (cross-platform guidance)
- ChromaDB knowledge base updated

**Next Phase**: Phase 2 (Oscillatory Dynamics) ready to begin

### Version 1.1 (2025-10-02) - Post-Audit Revision

**Critical Fixes** (Audit Score: 85/100 → APPROVED):
1. ✅ Added JTransforms 3.1 dependency to art-cortical/pom.xml
2. ✅ Added Pre-Implementation Dependency Validation checklist (Section 3.0)
3. ✅ Revised SIMD performance target: 1.50x → 1.40x (stretch goal: 1.50x)
4. ✅ Adjusted Phase 2 duration: 2-3 weeks → 3-4 weeks (+1 week for FFT learning curve)
5. ✅ Adjusted Phase 3 duration: 4-6 weeks → 6-8 weeks (+2 weeks for complexity)
6. ✅ Updated total timeline: 16-27 weeks → 20-32 weeks (5-8 months)

**Rationale**:
- **JTransforms**: Required for FFT processing in Phase 2, critical blocker identified by audit
- **Dependency Checklist**: Prevents future critical blockers during implementation
- **Performance Target**: Conservative target based on lack of profiling data, stretch goal preserved
- **Time Estimates**: More realistic based on scope (11 SIMD files, 556 tests, FFT integration)

**Audit Findings Addressed**: 4 critical blockers resolved, ready for implementation

### Version 1.0 (2025-10-02) - Initial Release

**Initial comprehensive plan created by java-architect-planner agent**:
- 5-phase implementation roadmap (Phases 1-5)
- 9 major sections covering architecture, testing, risks, timeline
- Test-first methodology with 1e-10 mathematical precision
- Based on unified neocortical theory research synthesis
- 12,000+ words, 3,304 lines

---

## EXECUTIVE SUMMARY

### Vision

Enhance the ART cortical architecture to achieve world-class neurobiological fidelity, research-grade consciousness modeling capabilities, and production-ready performance optimization. This plan integrates cutting-edge research from the unified neocortical theory synthesis while maintaining the project's rigorous standards: 100% test pass rate, 1e-10 mathematical precision, and 95%+ biological fidelity.

### Strategic Objectives

1. **Performance Excellence**: Increase SIMD throughput from 1.30x to 1.40x+ through optimized mini-batch processing (stretch goal: 1.50x)
2. **Consciousness Research**: Enable gamma oscillation analysis (~40 Hz) for computational consciousness studies
3. **Architectural Unity**: Consolidate art-laminar optimizations into art-cortical for maintainable excellence
4. **Perceptual Completeness**: Implement surface filling-in to complement existing boundary completion
5. **Future Readiness**: Establish foundation for GPU acceleration and multi-area hierarchies

### Success Metrics

| Metric | Baseline | Target | Achieved | Status |
|--------|----------|--------|----------|--------|
| SIMD Speedup | 1.30x | 1.40x+ (stretch: 1.50x) | 11.81x (x86-64) | ✅ **EXCEEDED** |
| Combined Speedup | 1.0x | N/A (not planned) | 15-25x (x86-64) | ✅ **BONUS** |
| Memory Reduction | 0% | N/A (not planned) | 99%+ | ✅ **BONUS** |
| Test Coverage | 1,680 tests | 1,850+ tests | 423 (art-cortical) | ⚠️ **CLARIFY** |
| Biological Fidelity | 95% | 97%+ | 95% (maintained) | 🔮 Phase 2 |
| Oscillation Support | None | Full gamma analysis | None | 🔮 Phase 2 |
| Module Count | 3 active | 2 unified | 3 active | 🔮 Phase 3 |

**Notes**:
- ✅ = Achieved/Exceeded
- ⚠️ = Needs clarification (423 art-cortical tests vs 1,680 total project tests)
- 🔮 = Future work (Phases 2-3)
- Phase 1 delivered more than planned (parallelism, memory optimization)

### Timeline Overview

- **Phase 1**: SIMD Optimization - ✅ **COMPLETE** (Oct 3, 2025)
- **Phase 2**: Oscillatory Dynamics (3-4 weeks) - **NEXT** - HIGH PRIORITY
- **Phase 3**: Module Consolidation (6-8 weeks) - HIGH PRIORITY
- **Phase 4**: Surface Filling-In (3-4 weeks) - MEDIUM PRIORITY
- **Phase 5**: Advanced Features (6-12 weeks) - MEDIUM-LOW PRIORITY

**Remaining Duration**: 18-30 weeks (4.5-7.5 months)
**Critical Path**: Phase 2 → Phase 3
**Completed**: Phase 1 (Oct 3, 2025) via Phase 4 Performance Optimization

---

## 1. PROJECT SCOPE & OBJECTIVES

### 1.1 Goals

#### Primary Goals (Phases 1-3)

1. **SIMD Performance Optimization**
   - Increase mini-batch size from 32 to 64 patterns
   - Port SIMD optimizations from art-laminar to art-cortical
   - Achieve 1.40x+ speedup with semantic equivalence (stretch goal: 1.50x)
   - Maintain 0.00e+00 bit-exact difference from sequential

2. **Oscillatory Dynamics Integration**
   - Implement gamma frequency oscillations (~40 Hz)
   - Add phase synchronization detection
   - Enable resonance frequency analysis
   - Support consciousness research metrics

3. **Module Consolidation**
   - Merge art-laminar SIMD optimizations into art-cortical
   - Deprecate art-laminar module (preserve as reference)
   - Consolidate test suites (556 tests → art-cortical)
   - Unified documentation and examples

#### Secondary Goals (Phases 4-5)

4. **Surface Filling-In Completion**
   - Implement Feature Contour System (FCS)
   - Integrate with existing Boundary Contour System (BCS)
   - Enable conscious percept generation
   - Validate brightness/color filling-in

5. **Advanced Features**
   - GPU acceleration foundation (CUDA/OpenCL)
   - Multi-area hierarchies (V1→V2→V4)
   - Adaptive vigilance mechanisms
   - Emotional processing integration (CogEM)

### 1.2 Success Criteria

#### Performance Criteria

- SIMD speedup ≥ 1.40x for batch sizes ≥ 64 (stretch goal: ≥ 1.50x)
- Throughput ≥ 1,150 patterns/sec (up from 1,049.7)
- Memory overhead ≤ 10% increase
- Compilation time ≤ 2 seconds
- Test execution time ≤ 5 seconds for full suite

#### Quality Criteria

- 100% test pass rate (zero failures, zero regressions)
- Mathematical precision: 1e-10 tolerance maintained
- Biological fidelity: ≥ 97% to source papers
- Code coverage: ≥ 90% line coverage
- Documentation completeness: 100% public APIs

#### Functional Criteria

- Gamma oscillation detection with 1 Hz resolution
- Phase synchronization measurement accuracy ≥ 95%
- Surface filling-in accuracy ≥ 90% on standard illusions
- Backward compatibility with existing art-cortical APIs
- Zero breaking changes for existing users

### 1.3 Dependencies & Prerequisites

#### Technical Dependencies

- **Java 24+**: Required for Vector API, records, pattern matching
- **Maven 3.9.1+**: Build system (enforced)
- **Vector API**: Java incubator module (--add-modules jdk.incubator.vector)
- **LWJGL 3.3.6**: Graphics/visualization (optional)
- **JUnit 5**: Testing framework
- **JMH**: Performance benchmarking

#### Module Dependencies

```
art-cortical (target module)
  ├── art-core (base ART interfaces)
  ├── art-performance (vectorization interfaces)
  └── art-temporal (temporal processing - optional after Phase 3)

art-laminar (source for SIMD optimizations)
  ├── batch/ (SIMD implementations to port)
  ├── performance/ (vectorized circuits to port)
  └── benchmarks/ (performance tests to port)
```

#### Knowledge Dependencies

- Unified neocortical theory synthesis (ChromaDB)
- Research papers: Grossberg (1973, 1985, 2013, 2021)
- Implementation documentation: art-cortical, art-laminar READMEs
- SIMD best practices: Java Vector API documentation

### 1.4 Risk Assessment

#### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SIMD optimization breaks semantic equivalence | Medium | High | Comprehensive bit-exact validation, automated testing |
| Oscillation detection has performance overhead | High | Medium | Optional feature flag, lazy initialization |
| Module consolidation introduces regressions | Medium | High | Gradual migration, parallel testing, rollback plan |
| Surface filling-in algorithm instability | Medium | Medium | Extensive parameter tuning, boundary case testing |

#### Medium-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Test suite execution time grows excessively | Medium | Low | Parallel test execution, test categorization |
| Documentation falls behind code changes | High | Low | Documentation-first approach, automated checks |
| API breaking changes required | Low | Medium | Semantic versioning, deprecation warnings |
| Performance regression in edge cases | Medium | Low | Comprehensive benchmarking, performance CI |

#### Low-Risk Items

- Code style inconsistencies (automated formatting)
- Build system compatibility (Maven enforcer)
- Dependency version conflicts (dependency convergence)
- Memory leaks (AutoCloseable pattern, automated detection)

### 1.5 Resource Requirements

#### Human Resources

- **Lead Architect**: Full-time for Phases 1-3 (8-12 weeks)
- **Implementation Engineer**: Part-time for Phases 4-5 (6-12 weeks)
- **Code Reviewer**: 20% time across all phases
- **Documentation Writer**: 10% time across all phases

#### Computational Resources

- **Development Machine**: Java 24, 16GB+ RAM, 8+ cores
- **CI/CD Pipeline**: GitHub Actions or equivalent
- **Performance Testing**: Dedicated benchmark server
- **Code Analysis**: SonarQube or equivalent (optional)

#### Knowledge Resources

- ChromaDB knowledge base (unified theory synthesis)
- Research paper library (DEVONthink)
- Existing codebase documentation
- Community forum access (optional)

---

## 2. ARCHITECTURE DESIGN

### 2.1 SIMD Optimization Enhancement

#### Current State (art-laminar)

```java
// Mini-batch size: 32 patterns
public static final int MINI_BATCH_SIZE = 32;

// Processing loop
for (int batchStart = 0; batchStart < totalPatterns; batchStart += MINI_BATCH_SIZE) {
    int batchEnd = Math.min(batchStart + MINI_BATCH_SIZE, totalPatterns);
    processMiniBlock(patterns, batchStart, batchEnd);
}
```

**Performance**: 1.30x speedup, 1049.7 patterns/sec

#### Target State (art-cortical enhanced)

```java
// Configurable mini-batch size with automatic tuning
public record SIMDConfiguration(
    int miniBatchSize,        // Default: 64 (increased from 32)
    int vectorLaneCount,      // Auto-detected from Vector API
    boolean autoTuning,       // Enable adaptive batch sizing
    double fallbackThreshold  // Switch to sequential if speedup < 1.05x
) {
    public static SIMDConfiguration optimal() {
        return new SIMDConfiguration(
            64,                        // Larger mini-batch for better SIMD utilization
            VectorSpecies.PREFERRED.length(),
            true,                      // Enable auto-tuning
            1.05                       // Require 5% minimum speedup
        );
    }
}

// Enhanced processing with adaptive tuning
public class SIMDBatchProcessor implements AutoCloseable {
    private final SIMDConfiguration config;
    private final PerformanceMonitor monitor;

    public BatchResult processBatch(Pattern[] patterns) {
        var optimalBatchSize = config.autoTuning()
            ? monitor.determineOptimalBatchSize(patterns)
            : config.miniBatchSize();

        return processBatchWithSize(patterns, optimalBatchSize);
    }
}
```

**Expected Performance**: 1.50x+ speedup, 1,200+ patterns/sec

#### Architecture Components

1. **SIMDConfiguration** (record)
   - Immutable configuration object
   - Factory methods for common scenarios
   - Validation logic for parameter constraints

2. **VectorizedLayer Interface**
   ```java
   public sealed interface VectorizedLayer permits
       VectorizedLayer4, VectorizedLayer5, VectorizedLayer6 {

       LayerState processVector(
           VectorSpecies<Double> species,
           VectorMask<Double> mask,
           double[] input,
           LayerParameters params
       );

       default boolean isSIMDSupported(int dimension) {
           return dimension >= species.length();
       }
   }
   ```

3. **PerformanceMonitor**
   - Runtime performance tracking
   - Adaptive batch size tuning
   - Speedup measurement and reporting
   - Fallback to sequential when beneficial

4. **BatchDataLayout**
   - Efficient array-of-structures to structure-of-arrays transpose
   - Cache-friendly memory access patterns
   - SIMD-aligned allocations

#### Integration Points

- **CorticalCircuit**: Add optional SIMD processing mode
- **Layer4, Layer5, Layer6**: Vectorized implementations
- **ShuntingDynamics**: Batch shunting equation solver
- **BipoleCellNetwork**: Sequential (complexity prevents SIMD gains)

#### Design Patterns

- **Strategy Pattern**: SIMDStrategy vs SequentialStrategy
- **Factory Pattern**: VectorizedLayerFactory
- **Builder Pattern**: SIMDConfiguration.builder()
- **Template Method**: AbstractVectorizedLayer

### 2.2 Oscillatory Dynamics Integration

#### Neuroscience Foundation

**Gamma Oscillations** (30-80 Hz, peak ~40 Hz):
- Correlate with conscious perception (Grossberg 2017)
- Emerge from resonance between bottom-up and top-down
- Phase synchronization binds distributed features
- Disrupting gamma disrupts awareness

**Mathematical Model**:
```
Resonance Strength: R(t) = |X(t) · E(t)| / (|X(t)| · |E(t)|)
Oscillation Frequency: f = 1 / (2π√(LC))  where L=layer time constant, C=capacity
Phase Difference: Δφ = arccos(Σ(x_i · y_i) / (||x|| · ||y||))
```

#### Architecture Components

1. **OscillatoryDynamics** (record for immutability)
   ```java
   /**
    * Implements oscillatory neural dynamics for consciousness research.
    * Based on Grossberg (2017) CLEARS framework.
    */
   public record OscillatoryDynamics(
       double baseFrequency,      // Target oscillation frequency (Hz)
       double bandwidth,          // Frequency band width (Hz)
       int historySize,           // Temporal history for FFT (power of 2)
       boolean enablePhaseSync,   // Track phase synchronization
       boolean enablePowerAnalysis // Compute spectral power
   ) {
       public static OscillatoryDynamics gammaDefault() {
           return new OscillatoryDynamics(
               40.0,    // 40 Hz gamma oscillations
               10.0,    // 30-50 Hz bandwidth
               256,     // 256-sample history for FFT
               true,    // Enable phase synchronization
               true     // Enable power analysis
           );
       }
   }
   ```

2. **OscillationAnalyzer**
   ```java
   public class OscillationAnalyzer implements AutoCloseable {
       private final CircularBuffer<double[]> activationHistory;
       private final FFTProcessor fftProcessor;
       private final PhaseDetector phaseDetector;

       /**
        * Analyze oscillatory content of layer activation.
        *
        * @param activation Current layer activation vector
        * @param timestamp Time in milliseconds
        * @return OscillationMetrics including frequency, power, phase
        */
       public OscillationMetrics analyze(double[] activation, double timestamp) {
           activationHistory.add(activation);

           if (!activationHistory.isFull()) {
               return OscillationMetrics.empty();
           }

           var spectrum = fftProcessor.computePowerSpectrum(activationHistory);
           var dominantFreq = spectrum.findPeakFrequency();
           var power = spectrum.getPowerInBand(30.0, 50.0);  // Gamma band
           var phase = phaseDetector.computePhase(activationHistory);

           return new OscillationMetrics(dominantFreq, power, phase, timestamp);
       }
   }
   ```

3. **ResonanceDetector** (enhanced)
   ```java
   public class ResonanceDetector {
       private final OscillationAnalyzer bottomUpAnalyzer;
       private final OscillationAnalyzer topDownAnalyzer;

       /**
        * Detect resonance based on feature-expectation match and
        * phase synchronization.
        */
       public ResonanceState detectResonance(
           double[] bottomUpFeatures,
           double[] topDownExpectations,
           double vigilanceThreshold
       ) {
           // Traditional ART matching
           var matchQuality = computeMatchQuality(bottomUpFeatures, topDownExpectations);
           var artResonance = matchQuality >= vigilanceThreshold;

           // Oscillatory analysis
           var buMetrics = bottomUpAnalyzer.analyze(bottomUpFeatures, currentTime);
           var tdMetrics = topDownAnalyzer.analyze(topDownExpectations, currentTime);

           var phaseDiff = Math.abs(buMetrics.phase() - tdMetrics.phase());
           var phaseSync = phaseDiff < Math.PI / 4;  // Phase-locked within 45°

           var inGammaBand = buMetrics.frequency() >= 30.0
                          && buMetrics.frequency() <= 50.0;

           return new ResonanceState(
               artResonance,
               phaseSync,
               inGammaBand,
               buMetrics,
               tdMetrics,
               matchQuality
           );
       }
   }
   ```

4. **FFTProcessor** (using JTransforms or custom implementation)
   ```java
   public class FFTProcessor {
       private final DoubleFFT_1D fft;

       public PowerSpectrum computePowerSpectrum(CircularBuffer<double[]> history) {
           var data = history.toArray();
           var avgActivation = computeAverageAcrossNeurons(data);

           fft.realForward(avgActivation);

           var power = new double[avgActivation.length / 2];
           for (int i = 0; i < power.length; i++) {
               var real = avgActivation[2 * i];
               var imag = avgActivation[2 * i + 1];
               power[i] = real * real + imag * imag;
           }

           return new PowerSpectrum(power, samplingRate);
       }
   }
   ```

#### Integration Strategy

- **Layer-Level**: Each layer gets optional OscillationAnalyzer
- **Circuit-Level**: CorticalCircuit aggregates oscillation metrics
- **Optional**: Feature flag to disable for performance-critical applications
- **Lazy Initialization**: Only create analyzers when explicitly requested

#### Performance Considerations

- **FFT Complexity**: O(N log N) - use power-of-2 history sizes
- **Memory Overhead**: historySize × layerDimension × sizeof(double)
- **Update Frequency**: Every timestep vs every N timesteps
- **Optimization**: Parallel FFT computation across layers

### 2.3 Module Consolidation Architecture

#### Current Module Structure

```
art-temporal (7 submodules, 145 tests)
├── temporal-core
├── temporal-dynamics
├── temporal-memory
├── temporal-masking
├── temporal-integration
├── temporal-validation
└── temporal-performance

art-laminar (402 tests)
├── batch/ (SIMD implementations)
├── performance/ (vectorized circuits)
├── layers/ (layer implementations)
├── integration/ (ARTLaminarCircuit)
└── canonical/ (temporal integration)

art-cortical (154 tests)
├── dynamics/
├── temporal/
├── layers/
└── network/
```

#### Target Unified Structure

```
art-cortical-unified/
├── dynamics/
│   ├── ShuntingDynamics.java
│   ├── TransmitterDynamics.java
│   ├── OscillatoryDynamics.java        # NEW: Phase 2
│   └── NeuralDynamics.java (interface)
│
├── temporal/
│   ├── WorkingMemory.java
│   ├── MaskingField.java
│   ├── TemporalProcessor.java
│   ├── ItemNode.java
│   └── ListChunk.java
│
├── layers/
│   ├── Layer.java (interface)
│   ├── Layer1.java
│   ├── Layer23.java
│   ├── Layer4.java
│   ├── Layer5.java
│   ├── Layer6.java
│   ├── VectorizedLayer.java (interface)    # NEW: Phase 1
│   ├── VectorizedLayer4.java              # NEW: Phase 1
│   ├── VectorizedLayer5.java              # NEW: Phase 1
│   ├── VectorizedLayer6.java              # NEW: Phase 1
│   └── CorticalCircuit.java (enhanced)
│
├── network/
│   ├── BipoleCell.java
│   ├── BipoleCellNetwork.java
│   ├── SurfaceCell.java                   # NEW: Phase 4
│   └── SurfaceCellNetwork.java            # NEW: Phase 4
│
├── performance/
│   ├── SIMDConfiguration.java             # NEW: Phase 1
│   ├── SIMDBatchProcessor.java            # NEW: Phase 1
│   ├── BatchDataLayout.java               # PORTED: art-laminar
│   ├── PerformanceMonitor.java            # NEW: Phase 1
│   └── VectorizedCircuit.java             # PORTED: art-laminar
│
├── analysis/                               # NEW: Phase 2
│   ├── OscillationAnalyzer.java
│   ├── FFTProcessor.java
│   ├── PhaseDetector.java
│   ├── ResonanceDetector.java (enhanced)
│   └── ConsciousnessMetrics.java
│
└── integration/
    ├── UnifiedCorticalCircuit.java        # NEW: Phase 3
    └── ARTCorticalProcessor.java          # NEW: Phase 3
```

#### Migration Strategy

**Phase 3A: Preparation (Week 1)**
1. Create `art-cortical-unified` branch
2. Copy current art-cortical to new structure
3. Set up parallel testing infrastructure
4. Document API compatibility matrix

**Phase 3B: SIMD Integration (Week 2-3)**
1. Port `batch/` package from art-laminar
2. Port `performance/` vectorized implementations
3. Integrate with existing layers
4. Validate bit-exact equivalence

**Phase 3C: Test Consolidation (Week 4)**
1. Port 402 tests from art-laminar
2. Merge with 154 art-cortical tests
3. Deduplicate overlapping tests
4. Target: 500+ unique tests

**Phase 3D: Documentation & Examples (Week 5)**
1. Unified README with migration guide
2. Update all Javadoc references
3. Create example programs
4. API deprecation warnings

**Phase 3E: Deprecation (Week 6)**
1. Mark art-laminar as deprecated
2. Update parent POM dependencies
3. Create MIGRATION.md guide
4. Release art-cortical v2.0.0

#### Backward Compatibility

**API Preservation**:
```java
// Old art-cortical API (still works)
var circuit = CorticalCircuit.builder()
    .layer1Parameters(params1)
    .build();

// New unified API (preferred)
var circuit = UnifiedCorticalCircuit.builder()
    .layer1Parameters(params1)
    .simdConfiguration(SIMDConfiguration.optimal())
    .oscillatoryDynamics(OscillatoryDynamics.gammaDefault())
    .build();

// Legacy art-laminar compatibility (deprecated)
var laminarCircuit = new ARTLaminarCircuit(params);
// Internally delegates to UnifiedCorticalCircuit
```

### 2.4 Surface Filling-In Architecture

#### Biological Foundation

**Complementary Computing** (Grossberg 1987):
- **Boundary Contour System (BCS)**: Structure (implemented ✓)
  - Inward, oriented, contrast-polarity insensitive
  - BipoleCellNetwork for illusory contours
  - Layer 2/3 horizontal connections

- **Feature Contour System (FCS)**: Quality (to implement)
  - Outward, unoriented, contrast-polarity sensitive
  - Fills regions bounded by BCS
  - Generates conscious brightness/color percepts

**Diffusion Equation**:
```
∂F/∂t = D∇²F - λF + I·(1 - B)

where:
  F = surface feature value (brightness/color)
  D = diffusion coefficient (outward spreading)
  λ = decay rate
  I = direct input (luminance/chrominance)
  B = boundary signal (from BCS, gates diffusion)
```

#### Architecture Components

1. **SurfaceCell** (analogous to BipoleCell)
   ```java
   /**
    * Surface cell implementing diffusive filling-in dynamics.
    * Grossberg & Todorović (1988) FACADE theory.
    */
   public class SurfaceCell implements AutoCloseable {
       private final SurfaceCellParameters params;
       private double activation;
       private double brightness;
       private double[] chromaticity;  // RGB or opponent colors

       /**
        * Evolve surface cell with diffusive dynamics.
        *
        * @param directInput Luminance/color from image
        * @param boundarySignal Gating signal from BCS (0=open, 1=closed)
        * @param neighborInputs Diffusive input from 4/8 neighbors
        * @param dt Timestep
        */
       public void evolve(
           double directInput,
           double boundarySignal,
           double[] neighborInputs,
           double dt
       ) {
           // Diffusion from neighbors (gated by boundary)
           var diffusiveInput = computeWeightedSum(neighborInputs)
                              * (1.0 - boundarySignal);

           // Brightness dynamics (simplified)
           var dB = params.diffusionRate() * diffusiveInput
                  - params.decayRate() * brightness
                  + directInput * (1.0 - boundarySignal);

           brightness += dB * dt;
           brightness = Math.max(0.0, Math.min(1.0, brightness));

           // Similar dynamics for chromaticity
           evolveColor(directInput, boundarySignal, neighborInputs, dt);
       }
   }
   ```

2. **SurfaceCellNetwork**
   ```java
   public class SurfaceCellNetwork implements AutoCloseable {
       private final SurfaceCell[][] cells;  // 2D grid
       private final BipoleCellNetwork boundaryNetwork;  // BCS integration

       /**
        * Process image to generate filled-in surface representation.
        */
       public SurfaceRepresentation process(
           double[][] luminanceImage,
           BoundaryRepresentation boundaries
       ) {
           // Initialize cells with direct input
           initializeFromImage(luminanceImage);

           // Iteratively diffuse until convergence
           for (int iter = 0; iter < maxIterations; iter++) {
               for (int y = 0; y < height; y++) {
                   for (int x = 0; x < width; x++) {
                       var cell = cells[y][x];
                       var boundarySignal = boundaries.getBoundaryStrength(x, y);
                       var neighborInputs = getNeighborBrightness(x, y);

                       cell.evolve(
                           luminanceImage[y][x],
                           boundarySignal,
                           neighborInputs,
                           dt
                       );
                   }
               }

               if (hasConverged()) break;
           }

           return extractSurfaceRepresentation();
       }
   }
   ```

3. **BCS-FCS Integration**
   ```java
   public class ComplementaryVisionSystem {
       private final BipoleCellNetwork bcs;  // Existing
       private final SurfaceCellNetwork fcs;  // New

       public VisualPercept processImage(double[][] image) {
           // BCS: Extract boundaries and illusory contours
           var boundaries = bcs.detectBoundaries(image);

           // FCS: Fill-in surface qualities
           var surfaces = fcs.fillInSurfaces(image, boundaries);

           // Integration: Combine for conscious percept
           return new VisualPercept(boundaries, surfaces);
       }
   }
   ```

#### Test Validation Cases

**Standard Visual Illusions**:
1. **Kanizsa Triangle**: Illusory brightness in subjective contours
2. **Neon Color Spreading**: Color fills past physical boundaries
3. **Brightness Assimilation**: Filling-in creates uniform brightness
4. **Depth Capture**: Surface at nearer depth captures brightness
5. **Craik-O'Brien-Cornsweet**: Edge enhances perceived brightness

---

## 3. PHASED IMPLEMENTATION PLAN

### 3.0 Pre-Implementation Dependency Validation

**Execute BEFORE Phase 1 starts** - This checklist prevents critical blockers during implementation.

#### Dependency Checklist

- [ ] **JTransforms 3.1+** added to art-cortical/pom.xml (for FFT processing in Phase 2)
- [ ] **Java 24 with Vector API** (--add-modules jdk.incubator.vector)
- [ ] **LWJGL 3.3.6+** (macOS ARM64 natives for visualization)
- [ ] **JUnit 5, Mockito 4.8.1+, JMH 1.35+**
- [ ] **Maven 3.9.1+**
- [ ] All dependencies resolve: `mvn dependency:tree`
- [ ] Clean build succeeds: `mvn clean compile`
- [ ] All 1,680 tests pass: `mvn test`

#### Compilation Smoke Tests

Execute these simple smoke tests to verify environment:

```java
// Verify Vector API accessible
import jdk.incubator.vector.*;
var species = DoubleVector.SPECIES_PREFERRED;
System.out.println("Vector length: " + species.length());

// Verify JTransforms accessible
import org.jtransforms.fft.DoubleFFT_1D;
var fft = new DoubleFFT_1D(256);
System.out.println("FFT ready for size 256");
```

**Action if checks fail**: STOP and resolve before proceeding. Do not start Phase 1 with missing dependencies.

---

### Phase 1: SIMD Optimization Enhancement (1-2 weeks)

**Priority**: HIGH
**Duration**: 1-2 weeks
**Dependencies**: None
**Team**: 1 engineer full-time

#### Phase 1A: Preparation & Test Framework (Days 1-2)

**Objectives**:
- Set up performance benchmarking infrastructure
- Create test harnesses for SIMD validation
- Establish baseline performance metrics

**Deliverables**:
1. `SIMDBenchmark.java` - JMH benchmark suite
2. `SIMDValidationTest.java` - Bit-exact equivalence tests
3. Baseline performance report (current 1.30x speedup)

**Test-First Approach**:
```java
@Test
void testMiniBatch64BitExactEquivalence() {
    var sequential = new SequentialProcessor(params);
    var simd = new SIMDBatchProcessor(
        SIMDConfiguration.withBatchSize(64)
    );

    var patterns = generateTestPatterns(128);

    var seqResults = sequential.processBatch(patterns);
    var simdResults = simd.processBatch(patterns);

    assertArrayEquals(seqResults, simdResults, 0.0,
        "SIMD batch-64 must be bit-exact with sequential");
}

@Test
void testAdaptiveBatchSizing() {
    var config = SIMDConfiguration.withAutoTuning(true);
    var processor = new SIMDBatchProcessor(config);

    // Small batch: should fall back to sequential
    var smallBatch = generateTestPatterns(16);
    var result1 = processor.processBatch(smallBatch);
    assertEquals(16, result1.actualBatchSize());  // No mini-batching

    // Large batch: should use 64-pattern mini-batches
    var largeBatch = generateTestPatterns(256);
    var result2 = processor.processBatch(largeBatch);
    assertEquals(64, result2.miniBatchSize());
}
```

**Success Criteria**:
- Benchmarks compile and run successfully
- Baseline measurements recorded
- Test framework validates current implementation

#### Phase 1B: Mini-Batch Size Increase (Days 3-5)

**Objectives**:
- Increase mini-batch size from 32 to 64
- Validate semantic equivalence
- Measure performance improvement

**Implementation**:
```java
// SIMDConfiguration.java
public record SIMDConfiguration(
    int miniBatchSize,
    int vectorLaneCount,
    boolean autoTuning,
    double fallbackThreshold
) {
    public static final int DEFAULT_MINI_BATCH_SIZE = 64;  // Increased from 32

    public static SIMDConfiguration optimal() {
        return new SIMDConfiguration(
            DEFAULT_MINI_BATCH_SIZE,
            VectorSpecies.PREFERRED.length(),
            true,
            1.05
        );
    }

    public SIMDConfiguration validate() {
        if (miniBatchSize < vectorLaneCount) {
            throw new IllegalArgumentException(
                "miniBatchSize (%d) must be >= vectorLaneCount (%d)"
                .formatted(miniBatchSize, vectorLaneCount)
            );
        }
        if (!isPowerOf2(miniBatchSize) && miniBatchSize != 32 && miniBatchSize != 64) {
            // Warn: non-power-of-2 may have alignment issues
        }
        return this;
    }
}
```

**Tests**:
```java
@ParameterizedTest
@ValueSource(ints = {32, 64, 128})
void testMiniBatchSizes(int batchSize) {
    var config = SIMDConfiguration.withBatchSize(batchSize);
    var processor = new SIMDBatchProcessor(config);

    var patterns = generateTestPatterns(256);
    var result = processor.processBatch(patterns);

    // Verify correctness
    validateResults(result);

    // Measure speedup
    var speedup = result.getSpeedup();
    assertTrue(speedup >= 1.0,
        "Batch size %d should not slow down processing".formatted(batchSize));
}
```

**Success Criteria**:
- All tests pass with batch size 64
- Speedup ≥ 1.40x (target 1.50x)
- Zero semantic differences from sequential

#### Phase 1C: art-cortical Integration (Days 6-8)

**Objectives**:
- Add SIMD processing to art-cortical
- Integrate with existing CorticalCircuit
- Maintain backward compatibility

**Implementation**:
```java
// CorticalCircuit.java (enhanced)
public class CorticalCircuit implements AutoCloseable {
    private final Layer1 layer1;
    private final Layer23 layer23;
    private final Layer4 layer4;
    private final Layer5 layer5;
    private final Layer6 layer6;
    private final SIMDBatchProcessor simdProcessor;  // NEW: optional

    public static class Builder {
        private SIMDConfiguration simdConfig = null;  // null = sequential only

        public Builder enableSIMD(SIMDConfiguration config) {
            this.simdConfig = config;
            return this;
        }

        public CorticalCircuit build() {
            var circuit = new CorticalCircuit(/* ... */);
            if (simdConfig != null) {
                circuit.simdProcessor = new SIMDBatchProcessor(
                    simdConfig, circuit.layers
                );
            }
            return circuit;
        }
    }

    /**
     * Process single pattern (existing API, unchanged).
     */
    public PatternResult process(Pattern pattern) {
        return processSequential(pattern);
    }

    /**
     * Process batch with optional SIMD optimization.
     */
    public BatchResult processBatch(Pattern[] patterns) {
        if (simdProcessor != null && patterns.length >= simdProcessor.miniBatchSize()) {
            return simdProcessor.processBatch(patterns);
        } else {
            return processSequentialBatch(patterns);
        }
    }
}
```

**Tests**:
```java
@Test
void testCorticalCircuitSIMDIntegration() {
    var circuit = CorticalCircuit.builder()
        .layer4Parameters(Layer4Parameters.paperDefaults())
        .enableSIMD(SIMDConfiguration.optimal())
        .build();

    var patterns = generateTestPatterns(128);
    var result = circuit.processBatch(patterns);

    assertTrue(result.usedSIMD());
    assertTrue(result.getSpeedup() >= 1.40);
}

@Test
void testBackwardCompatibility() {
    // Old API (no SIMD) still works
    var circuit = CorticalCircuit.builder()
        .layer4Parameters(Layer4Parameters.paperDefaults())
        .build();

    var pattern = generateTestPattern();
    var result = circuit.process(pattern);

    assertNotNull(result);
    // Single-pattern processing unchanged
}
```

**Success Criteria**:
- Backward compatible with existing API
- SIMD batch processing achieves 1.50x+ speedup
- All 154 existing tests still pass

#### Phase 1D: Performance Validation & Documentation (Days 9-10)

**Objectives**:
- Run comprehensive performance benchmarks
- Document performance characteristics
- Create usage examples

**Benchmarks**:
```java
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
public class CorticalCircuitSIMDBenchmark {

    @Param({"32", "64", "128", "256"})
    private int batchSize;

    @Param({"32", "64"})
    private int miniBatchSize;

    private CorticalCircuit circuit;
    private Pattern[] patterns;

    @Setup
    public void setup() {
        circuit = CorticalCircuit.builder()
            .layer4Parameters(Layer4Parameters.paperDefaults())
            .enableSIMD(SIMDConfiguration.withBatchSize(miniBatchSize))
            .build();
        patterns = generateTestPatterns(batchSize);
    }

    @Benchmark
    public BatchResult benchmarkSIMDBatch() {
        return circuit.processBatch(patterns);
    }

    @Benchmark
    public BatchResult benchmarkSequentialBatch() {
        var seqCircuit = CorticalCircuit.builder()
            .layer4Parameters(Layer4Parameters.paperDefaults())
            .build();
        return seqCircuit.processBatch(patterns);
    }
}
```

**Documentation**:
- Performance characteristics table
- Optimal configuration guide
- Troubleshooting common issues

**Success Criteria**:
- Benchmark results show ≥1.50x speedup
- Documentation complete and reviewed
- Examples run successfully

#### Phase 1 Deliverables

- [ ] `SIMDConfiguration.java` - Configuration record
- [ ] `SIMDBatchProcessor.java` - Batch processing engine
- [ ] `VectorizedLayer4/5/6.java` - SIMD layer implementations
- [ ] `PerformanceMonitor.java` - Runtime performance tracking
- [ ] Enhanced `CorticalCircuit` with SIMD support
- [ ] 20+ new SIMD tests (100% pass)
- [ ] Performance benchmark suite
- [ ] Documentation: SIMD_OPTIMIZATION_GUIDE.md

#### Phase 1 Rollback Plan

If performance targets not met:
1. Revert `SIMDConfiguration.DEFAULT_MINI_BATCH_SIZE` to 32
2. Disable auto-tuning by default
3. Mark SIMD as experimental feature
4. Continue with Phase 2 (Oscillatory Dynamics)

---

### Phase 2: Oscillatory Dynamics Integration (3-4 weeks)

**Priority**: HIGH
**Duration**: 3-4 weeks
**Dependencies**: Phase 1 (optional, can run in parallel)
**Team**: 1 engineer full-time

#### Phase 2A: Mathematical Foundation (Days 1-3)

**Objectives**:
- Implement FFT-based frequency analysis
- Create phase detection algorithms
- Validate against synthetic oscillatory signals

**Test-First Development**:
```java
@Test
void testFFTPowerSpectrum() {
    // Generate 40 Hz sine wave
    var samplingRate = 1000.0;  // 1000 Hz
    var duration = 1.0;          // 1 second
    var frequency = 40.0;        // 40 Hz
    var signal = generateSineWave(frequency, samplingRate, duration);

    var fft = new FFTProcessor(samplingRate);
    var spectrum = fft.computePowerSpectrum(signal);

    var peakFreq = spectrum.findPeakFrequency();
    assertEquals(40.0, peakFreq, 1.0,  // 1 Hz resolution
        "FFT should detect 40 Hz oscillation");

    var gammaPower = spectrum.getPowerInBand(30.0, 50.0);
    assertTrue(gammaPower > 0.8,
        "Most power should be in gamma band");
}

@Test
void testPhaseDetection() {
    var signal1 = generateSineWave(40.0, 1000.0, 1.0);
    var signal2 = generateSineWave(40.0, 1000.0, 1.0, Math.PI / 4);  // 45° phase shift

    var detector = new PhaseDetector();
    var phaseDiff = detector.computePhaseDifference(signal1, signal2);

    assertEquals(Math.PI / 4, phaseDiff, 0.1,
        "Phase difference should be π/4 radians");
}

@Test
void testResonanceOscillationEmergence() {
    // This test validates that resonance produces gamma oscillations
    var circuit = CorticalCircuit.builder()
        .oscillatoryDynamics(OscillatoryDynamics.gammaDefault())
        .build();

    var pattern = generateTestPattern();

    // Process for 1 second (enough for oscillations to emerge)
    for (int t = 0; t < 1000; t++) {
        circuit.processTimestep(pattern, t * 0.001);
    }

    var metrics = circuit.getOscillationMetrics();

    assertTrue(metrics.dominantFrequency() >= 30.0
            && metrics.dominantFrequency() <= 50.0,
        "Resonance should produce gamma oscillations");
}
```

**Implementation**:
```java
/**
 * FFT-based spectral analysis for oscillation detection.
 * Uses JTransforms library for efficient FFT computation.
 */
public class FFTProcessor {
    private final DoubleFFT_1D fft;
    private final double samplingRate;

    public FFTProcessor(double samplingRate) {
        this.samplingRate = samplingRate;
        this.fft = new DoubleFFT_1D(/* size */);
    }

    public PowerSpectrum computePowerSpectrum(double[] signal) {
        var padded = zeroPadToPowerOf2(signal);
        fft.realForward(padded);

        var n = padded.length / 2;
        var power = new double[n];
        var frequencies = new double[n];

        for (int i = 0; i < n; i++) {
            var real = padded[2 * i];
            var imag = padded[2 * i + 1];
            power[i] = (real * real + imag * imag) / n;
            frequencies[i] = i * samplingRate / padded.length;
        }

        return new PowerSpectrum(frequencies, power);
    }
}

public record PowerSpectrum(double[] frequencies, double[] power) {
    public double findPeakFrequency() {
        var maxPower = 0.0;
        var peakFreq = 0.0;
        for (int i = 0; i < power.length; i++) {
            if (power[i] > maxPower) {
                maxPower = power[i];
                peakFreq = frequencies[i];
            }
        }
        return peakFreq;
    }

    public double getPowerInBand(double lowFreq, double highFreq) {
        var totalPower = 0.0;
        for (int i = 0; i < frequencies.length; i++) {
            if (frequencies[i] >= lowFreq && frequencies[i] <= highFreq) {
                totalPower += power[i];
            }
        }
        return totalPower;
    }
}
```

**Success Criteria**:
- FFT correctly identifies synthetic oscillations
- Phase detection accurate to 0.1 radians
- All mathematical tests pass (10+ tests)

#### Phase 2B: Layer Integration (Days 4-8)

**Objectives**:
- Add oscillation analysis to each layer
- Implement circular activation buffers
- Create layer-specific oscillation metrics

**Implementation**:
```java
/**
 * Enhanced Layer with oscillatory dynamics tracking.
 */
public class Layer4 implements Layer, AutoCloseable {
    private final Layer4Parameters params;
    private final ShuntingDynamics dynamics;
    private final OscillationAnalyzer oscillationAnalyzer;  // NEW
    private final CircularBuffer<double[]> activationHistory;  // NEW

    public Layer4(Layer4Parameters params, OscillatoryDynamics oscillatoryParams) {
        this.params = params;
        this.dynamics = new ShuntingDynamics(params.shuntingParams());

        if (oscillatoryParams != null) {
            this.oscillationAnalyzer = new OscillationAnalyzer(oscillatoryParams);
            this.activationHistory = new CircularBuffer<>(
                oscillatoryParams.historySize()
            );
        } else {
            this.oscillationAnalyzer = null;
            this.activationHistory = null;
        }
    }

    public LayerState processTimestep(double[] input, double timestamp) {
        // Standard processing
        var activation = dynamics.computeActivation(input);

        // Oscillation analysis (if enabled)
        OscillationMetrics metrics = null;
        if (oscillationAnalyzer != null) {
            activationHistory.add(activation);
            if (activationHistory.isFull()) {
                metrics = oscillationAnalyzer.analyze(
                    activationHistory, timestamp
                );
            }
        }

        return new LayerState(activation, metrics, timestamp);
    }

    public Optional<OscillationMetrics> getOscillationMetrics() {
        return Optional.ofNullable(
            oscillationAnalyzer != null
                ? oscillationAnalyzer.getLatestMetrics()
                : null
        );
    }
}
```

**Tests**:
```java
@Test
void testLayer4OscillationTracking() {
    var oscillatoryParams = OscillatoryDynamics.gammaDefault();
    var layer = new Layer4(
        Layer4Parameters.paperDefaults(),
        oscillatoryParams
    );

    // Process enough timesteps to fill history buffer
    for (int t = 0; t < 256; t++) {
        var input = generateOscillatoryInput(40.0, t * 0.001);
        layer.processTimestep(input, t * 0.001);
    }

    var metrics = layer.getOscillationMetrics();
    assertTrue(metrics.isPresent());

    var freq = metrics.get().dominantFrequency();
    assertTrue(freq >= 35.0 && freq <= 45.0,
        "Layer should track 40 Hz input oscillation");
}

@Test
void testOscillationDisabledByDefault() {
    var layer = new Layer4(
        Layer4Parameters.paperDefaults(),
        null  // No oscillatory dynamics
    );

    layer.processTimestep(generateTestInput(), 0.0);

    var metrics = layer.getOscillationMetrics();
    assertFalse(metrics.isPresent(),
        "Oscillation tracking should be disabled by default");
}
```

**Success Criteria**:
- All layers support optional oscillation tracking
- No performance degradation when disabled
- Oscillation metrics accurate (tested with synthetic signals)

#### Phase 2C: Resonance Enhancement (Days 9-12)

**Objectives**:
- Enhance ResonanceDetector with phase synchronization
- Implement consciousness metrics
- Validate against neuroscience predictions

**Implementation**:
```java
/**
 * Enhanced resonance detection with oscillatory analysis.
 * Grossberg (2017) CLEARS framework implementation.
 */
public class EnhancedResonanceDetector {
    private final OscillationAnalyzer bottomUpAnalyzer;
    private final OscillationAnalyzer topDownAnalyzer;
    private final double vigilanceThreshold;

    public ResonanceState detectResonance(
        double[] bottomUpFeatures,
        double[] topDownExpectations,
        double timestamp
    ) {
        // Traditional ART matching
        var matchQuality = computeMatchQuality(
            bottomUpFeatures,
            topDownExpectations
        );
        var artResonance = matchQuality >= vigilanceThreshold;

        // Oscillatory analysis
        var buMetrics = bottomUpAnalyzer.analyze(bottomUpFeatures, timestamp);
        var tdMetrics = topDownAnalyzer.analyze(topDownExpectations, timestamp);

        // Phase synchronization detection
        var phaseDiff = Math.abs(buMetrics.phase() - tdMetrics.phase());
        var phaseSync = phaseDiff < Math.PI / 4;  // Within 45 degrees

        // Gamma band detection
        var buInGamma = isInGammaBand(buMetrics.dominantFrequency());
        var tdInGamma = isInGammaBand(tdMetrics.dominantFrequency());
        var bothInGamma = buInGamma && tdInGamma;

        // Consciousness likelihood (heuristic)
        var consciousnessLikelihood = computeConsciousnessLikelihood(
            artResonance, phaseSync, bothInGamma, matchQuality
        );

        return new ResonanceState(
            artResonance,
            phaseSync,
            bothInGamma,
            consciousnessLikelihood,
            buMetrics,
            tdMetrics,
            matchQuality,
            timestamp
        );
    }

    private double computeConsciousnessLikelihood(
        boolean artResonance,
        boolean phaseSync,
        boolean gammaOscillations,
        double matchQuality
    ) {
        if (!artResonance) return 0.0;

        var likelihood = matchQuality;  // Base: ART match quality
        if (phaseSync) likelihood += 0.2;
        if (gammaOscillations) likelihood += 0.3;

        return Math.min(1.0, likelihood);
    }

    private boolean isInGammaBand(double frequency) {
        return frequency >= 30.0 && frequency <= 50.0;
    }
}

/**
 * Consciousness metrics for research applications.
 */
public record ConsciousnessMetrics(
    double artMatchQuality,
    boolean phaseSynchronized,
    boolean gammaOscillations,
    double consciousnessLikelihood,
    double timestamp
) {
    public boolean isLikelyConscious() {
        return consciousnessLikelihood >= 0.7;
    }
}
```

**Tests**:
```java
@Test
void testResonanceProducesGammaOscillations() {
    var detector = new EnhancedResonanceDetector(
        OscillatoryDynamics.gammaDefault(),
        0.85  // Vigilance
    );

    // Create matching bottom-up and top-down patterns
    var features = generatePattern(0.0);
    var expectations = generatePattern(0.0);  // Perfect match

    // Process for 1 second to allow oscillations to develop
    ResonanceState finalState = null;
    for (int t = 0; t < 1000; t++) {
        finalState = detector.detectResonance(
            features, expectations, t * 0.001
        );
    }

    assertNotNull(finalState);
    assertTrue(finalState.artResonance());
    assertTrue(finalState.gammaOscillations(),
        "Resonance should produce gamma oscillations");
    assertTrue(finalState.consciousnessLikelihood() >= 0.7,
        "Resonance with gamma should indicate consciousness");
}

@Test
void testMismatchSuppressesOscillations() {
    var detector = new EnhancedResonanceDetector(
        OscillatoryDynamics.gammaDefault(),
        0.85
    );

    // Create mismatched patterns
    var features = generatePattern(0.0);
    var expectations = generatePattern(Math.PI);  // Opposite phase

    ResonanceState finalState = null;
    for (int t = 0; t < 1000; t++) {
        finalState = detector.detectResonance(
            features, expectations, t * 0.001
        );
    }

    assertFalse(finalState.artResonance());
    assertTrue(finalState.consciousnessLikelihood() < 0.3,
        "Mismatch should suppress consciousness indicators");
}
```

**Success Criteria**:
- Resonance correlates with gamma oscillations
- Phase synchronization detected accurately
- Consciousness metrics align with neuroscience predictions

#### Phase 2D: Circuit-Level Integration (Days 13-15)

**Objectives**:
- Add oscillation tracking to CorticalCircuit
- Aggregate metrics across layers
- Create visualization-ready outputs

**Implementation**:
```java
public class CorticalCircuit implements AutoCloseable {
    private final OscillatoryDynamics oscillatoryParams;
    private final Map<LayerType, OscillationMetrics> layerMetrics;

    public CircuitState processWithOscillations(Pattern pattern, double timestamp) {
        // Standard processing
        var state = process(pattern);

        // Collect oscillation metrics from all layers
        if (oscillatoryParams != null) {
            layerMetrics.put(LayerType.LAYER_1,
                layer1.getOscillationMetrics().orElse(null));
            layerMetrics.put(LayerType.LAYER_4,
                layer4.getOscillationMetrics().orElse(null));
            // ... other layers
        }

        return new CircuitState(state, layerMetrics, timestamp);
    }

    public CircuitOscillationSummary getOscillationSummary() {
        var avgFrequency = layerMetrics.values().stream()
            .filter(Objects::nonNull)
            .mapToDouble(OscillationMetrics::dominantFrequency)
            .average()
            .orElse(0.0);

        var inGamma = layerMetrics.values().stream()
            .filter(Objects::nonNull)
            .filter(m -> m.dominantFrequency() >= 30.0
                      && m.dominantFrequency() <= 50.0)
            .count();

        return new CircuitOscillationSummary(
            avgFrequency,
            inGamma,
            layerMetrics.size(),
            layerMetrics
        );
    }
}
```

**Success Criteria**:
- Circuit-level oscillation metrics accurate
- Performance overhead < 10% when enabled
- Visualization data correctly formatted

#### Phase 2E: Documentation & Examples (Days 16-18)

**Deliverables**:
- OSCILLATORY_DYNAMICS_GUIDE.md
- Example: GammaOscillationDemo.java
- Research integration guide
- API documentation

**Example Application**:
```java
/**
 * Demonstrates gamma oscillation detection in conscious vs unconscious states.
 */
public class ConsciousnessResearchDemo {
    public static void main(String[] args) {
        var circuit = CorticalCircuit.builder()
            .layer4Parameters(Layer4Parameters.paperDefaults())
            .oscillatoryDynamics(OscillatoryDynamics.gammaDefault())
            .build();

        // Conscious condition: Attended, resonant stimulus
        System.out.println("Conscious condition:");
        var attendedPattern = generateAtentededPattern();
        for (int t = 0; t < 1000; t++) {
            circuit.processWithOscillations(attendedPattern, t * 0.001);
        }
        var consciousMetrics = circuit.getOscillationSummary();
        System.out.printf("  Gamma power: %.2f%n",
            consciousMetrics.gammaBandPower());
        System.out.printf("  Consciousness likelihood: %.2f%n",
            consciousMetrics.consciousnessLikelihood());

        // Unconscious condition: Unattended, non-resonant
        circuit.reset();
        System.out.println("Unconscious condition:");
        var unattendedPattern = generateUnattendedPattern();
        for (int t = 0; t < 1000; t++) {
            circuit.processWithOscillations(unattendedPattern, t * 0.001);
        }
        var unconsciousMetrics = circuit.getOscillationSummary();
        System.out.printf("  Gamma power: %.2f%n",
            unconsciousMetrics.gammaBandPower());
        System.out.printf("  Consciousness likelihood: %.2f%n",
            unconsciousMetrics.consciousnessLikelihood());
    }
}
```

**Success Criteria**:
- Documentation complete and reviewed
- Examples run successfully
- Research community can use for consciousness studies

#### Phase 2 Deliverables

- [ ] `FFTProcessor.java` - Spectral analysis
- [ ] `PhaseDetector.java` - Phase synchronization detection
- [ ] `OscillationAnalyzer.java` - Per-layer oscillation tracking
- [ ] `EnhancedResonanceDetector.java` - Consciousness metrics
- [ ] `CircularBuffer.java` - Efficient history management
- [ ] Enhanced layers with oscillation support
- [ ] 30+ oscillation tests (100% pass)
- [ ] Documentation: OSCILLATORY_DYNAMICS_GUIDE.md
- [ ] Example: ConsciousnessResearchDemo.java

---

### Phase 3: Module Consolidation (6-8 weeks)

**Priority**: HIGH
**Duration**: 6-8 weeks
**Dependencies**: Phase 1 (SIMD optimizations complete)
**Team**: 1-2 engineers full-time

#### Phase 3A: Preparation & Architecture (Week 1)

**Objectives**:
- Design unified module structure
- Create migration plan
- Set up parallel testing infrastructure

**Deliverables**:
1. Architecture design document
2. API compatibility matrix
3. Test migration strategy
4. Risk mitigation plan

**Tasks**:
- [ ] Create `art-cortical-unified` branch
- [ ] Design package structure
- [ ] Document API changes
- [ ] Create compatibility layer
- [ ] Set up dual build (old + new)

#### Phase 3B: SIMD Integration (Week 2-3)

**Objectives**:
- Port SIMD implementations from art-laminar
- Integrate with art-cortical layers
- Validate bit-exact equivalence

**Port Checklist**:
- [ ] `batch/BatchDataLayout.java`
- [ ] `batch/BatchShuntingDynamics.java`
- [ ] `batch/Layer4SIMDBatch.java`
- [ ] `batch/Layer5SIMDBatch.java`
- [ ] `batch/Layer6SIMDBatch.java`
- [ ] `batch/StatefulBatchProcessor.java`
- [ ] `performance/VectorizedLaminarCircuit.java`
- [ ] `performance/VectorizedLaminarPerformanceStats.java`

**Integration Tests**:
```java
@Test
void testPortedSIMDEquivalence() {
    // Original art-laminar implementation
    var laminarCircuit = new ARTLaminarCircuit(params);

    // New art-cortical implementation
    var corticalCircuit = UnifiedCorticalCircuit.builder()
        .simdConfiguration(SIMDConfiguration.optimal())
        .build();

    var patterns = generateTestPatterns(128);

    var laminarResults = laminarCircuit.processBatch(patterns);
    var corticalResults = corticalCircuit.processBatch(patterns);

    assertArrayEquals(
        laminarResults.getActivations(),
        corticalResults.getActivations(),
        1e-10,
        "Ported SIMD must match art-laminar implementation"
    );
}
```

**Success Criteria**:
- All SIMD code ported successfully
- Performance parity with art-laminar (1.30x → 1.50x)
- Zero regressions in existing tests

#### Phase 3C: Test Consolidation (Week 4)

**Objectives**:
- Merge test suites from art-laminar (402) and art-cortical (154)
- Deduplicate overlapping tests
- Target: 500+ unique, comprehensive tests

**Test Categories**:
1. **Layer Tests** (100 tests)
   - Layer 1: 20 tests
   - Layer 2/3: 20 tests
   - Layer 4: 20 tests
   - Layer 5: 20 tests
   - Layer 6: 20 tests

2. **Dynamics Tests** (50 tests)
   - Shunting dynamics: 20 tests
   - Transmitter dynamics: 15 tests
   - Oscillatory dynamics: 15 tests

3. **Temporal Tests** (80 tests)
   - Working memory: 20 tests
   - Masking field: 30 tests
   - Temporal integration: 30 tests

4. **Network Tests** (40 tests)
   - Bipole cells: 20 tests
   - Surface cells: 20 tests

5. **Integration Tests** (80 tests)
   - Circuit-level: 30 tests
   - Multi-pathway: 20 tests
   - End-to-end: 30 tests

6. **Performance Tests** (100 tests)
   - SIMD validation: 40 tests
   - Benchmarks: 30 tests
   - Stress tests: 30 tests

7. **Biological Validation** (50 tests)
   - Paper fidelity: 30 tests
   - Precision: 20 tests

**Total**: 500 tests

**Deduplication Strategy**:
```java
// Example: Merge overlapping Layer4 tests
// art-laminar/Layer4Test.java (14 tests)
// art-cortical/Layer4Test.java (14 tests)
// → Unified: Layer4Test.java (20 unique tests)

@Test
void testThalamicDrivingInput() {
    // From art-cortical, kept as-is
}

@Test
void testSIMDBatchProcessing() {
    // From art-laminar, kept as-is
}

@Test
void testFastTimeConstant() {
    // Duplicate, merged with testThalamicDrivingInput
    // REMOVED
}
```

**Success Criteria**:
- 500+ unique tests
- 100% pass rate
- No functionality gaps
- Test execution time < 10 seconds

#### Phase 3D: Documentation & Migration Guide (Week 5)

**Objectives**:
- Create comprehensive migration guide
- Update all API documentation
- Write example migration scenarios

**Documentation Deliverables**:

1. **MIGRATION_GUIDE.md**
   - Step-by-step migration from art-laminar
   - Step-by-step migration from art-cortical
   - API mapping table
   - Common pitfalls and solutions

2. **UNIFIED_CORTICAL_ARCHITECTURE.md**
   - Complete architecture overview
   - Component descriptions
   - Design decisions
   - Performance characteristics

3. **API_REFERENCE.md**
   - All public APIs documented
   - Usage examples for each
   - Parameter explanations
   - Return value specifications

4. **EXAMPLES.md**
   - Basic usage
   - Advanced features
   - SIMD optimization
   - Oscillatory dynamics
   - Custom extensions

**Migration Examples**:
```java
// OLD: art-laminar
var laminarCircuit = new ARTLaminarCircuit(
    ARTCircuitParameters.builder(256)
        .vigilance(0.85)
        .build()
);

// NEW: art-cortical-unified
var unifiedCircuit = UnifiedCorticalCircuit.builder()
    .dimension(256)
    .vigilance(0.85)
    .simdConfiguration(SIMDConfiguration.optimal())
    .oscillatoryDynamics(OscillatoryDynamics.gammaDefault())
    .build();

// OLD: art-cortical (basic)
var corticalCircuit = CorticalCircuit.builder()
    .layer4Parameters(Layer4Parameters.paperDefaults())
    .build();

// NEW: art-cortical-unified (backward compatible)
var unifiedCircuit = UnifiedCorticalCircuit.builder()
    .layer4Parameters(Layer4Parameters.paperDefaults())
    .build();
```

**Success Criteria**:
- All documentation complete
- Migration guide tested by independent reviewer
- Examples compile and run

#### Phase 3E: Deprecation & Release (Week 6)

**Objectives**:
- Deprecate art-laminar module
- Release art-cortical v2.0.0
- Update parent POM dependencies

**Deprecation Strategy**:
```java
/**
 * @deprecated As of art-cortical 2.0.0, use {@link UnifiedCorticalCircuit} instead.
 * This class will be removed in art-cortical 3.0.0.
 *
 * <p>Migration example:
 * <pre>{@code
 * // Old
 * var circuit = new ARTLaminarCircuit(params);
 *
 * // New
 * var circuit = UnifiedCorticalCircuit.builder()
 *     .fromLaminarParameters(params)
 *     .build();
 * }</pre>
 */
@Deprecated(since = "2.0.0", forRemoval = true)
public class ARTLaminarCircuit {
    // Internally delegates to UnifiedCorticalCircuit
}
```

**Release Checklist**:
- [ ] Version bump: 1.0.0 → 2.0.0
- [ ] Deprecation warnings added
- [ ] Migration guide published
- [ ] Release notes written
- [ ] CHANGELOG.md updated
- [ ] All tests passing (500+)
- [ ] Performance benchmarks run
- [ ] Documentation reviewed
- [ ] Examples tested
- [ ] Parent POM updated

**Success Criteria**:
- Clean release with no breaking changes
- Deprecation warnings guide users
- Migration path clear and tested

#### Phase 3 Deliverables

- [ ] Unified module: art-cortical v2.0.0
- [ ] 500+ consolidated tests (100% pass)
- [ ] MIGRATION_GUIDE.md
- [ ] UNIFIED_CORTICAL_ARCHITECTURE.md
- [ ] API_REFERENCE.md
- [ ] EXAMPLES.md
- [ ] Deprecated art-laminar with delegation
- [ ] Release notes and CHANGELOG

---

### Phase 4: Surface Filling-In Completion (3-4 weeks)

**Priority**: MEDIUM
**Duration**: 3-4 weeks
**Dependencies**: Phase 3 (unified module)
**Team**: 1 engineer full-time

#### Phase 4A: Surface Cell Implementation (Week 1)

**Objectives**:
- Implement SurfaceCell with diffusion dynamics
- Create SurfaceCellNetwork (2D grid)
- Validate against simple test cases

**Mathematical Foundation**:
```
Diffusion Equation (Grossberg & Todorović 1988):
∂F/∂t = D∇²F - λF + I(1 - B)

where:
  F = surface feature (brightness/color)
  D = diffusion rate (outward spreading)
  λ = decay rate
  I = direct input
  B = boundary signal (gates diffusion)
```

**Implementation**:
```java
public class SurfaceCell {
    private double brightness;
    private double[] chromaticity;  // RGB or opponent colors

    public void evolve(
        double directInput,
        double boundarySignal,
        double[] neighborInputs,
        double dt
    ) {
        // Laplacian (discrete approximation of ∇²F)
        var laplacian = 0.0;
        for (var neighbor : neighborInputs) {
            laplacian += neighbor - brightness;
        }

        // Diffusion (gated by boundary)
        var diffusion = params.diffusionRate() * laplacian
                      * (1.0 - boundarySignal);

        // Decay
        var decay = -params.decayRate() * brightness;

        // Direct input (gated by boundary)
        var input = directInput * (1.0 - boundarySignal);

        // Update
        var dB = diffusion + decay + input;
        brightness += dB * dt;
        brightness = clamp(brightness, 0.0, 1.0);
    }
}
```

**Tests**:
```java
@Test
void testDiffusionWithoutBoundary() {
    var cell = new SurfaceCell(SurfaceCellParameters.paperDefaults());

    // Bright center, dark neighbors
    var centerInput = 1.0;
    var neighborInputs = new double[]{0.0, 0.0, 0.0, 0.0};
    var noBoundary = 0.0;

    // Evolve for several timesteps
    for (int t = 0; t < 100; t++) {
        cell.evolve(centerInput, noBoundary, neighborInputs, 0.01);
    }

    // Brightness should diffuse outward (decrease at center)
    assertTrue(cell.getBrightness() < 1.0,
        "Brightness should diffuse away from center");
}

@Test
void testBoundaryBlocksDiffusion() {
    var cell = new SurfaceCell(SurfaceCellParameters.paperDefaults());

    var centerInput = 1.0;
    var neighborInputs = new double[]{0.0, 0.0, 0.0, 0.0};
    var strongBoundary = 1.0;  // Closed boundary

    for (int t = 0; t < 100; t++) {
        cell.evolve(centerInput, strongBoundary, neighborInputs, 0.01);
    }

    // Brightness should remain high (no diffusion)
    assertTrue(cell.getBrightness() > 0.95,
        "Boundary should block diffusion");
}
```

**Success Criteria**:
- Diffusion dynamics mathematically correct
- Boundary gating works as expected
- Basic tests pass (20+ tests)

#### Phase 4B: BCS-FCS Integration (Week 2)

**Objectives**:
- Integrate SurfaceCellNetwork with BipoleCellNetwork
- Implement complementary vision system
- Validate on standard illusions

**Implementation**:
```java
public class ComplementaryVisionSystem {
    private final BipoleCellNetwork bcs;  // Boundary Contour System
    private final SurfaceCellNetwork fcs;  // Feature Contour System

    public VisualPercept processImage(double[][] luminanceImage) {
        // Stage 1: Boundary detection (existing)
        var boundaries = bcs.detectBoundaries(luminanceImage);

        // Stage 2: Surface filling-in (new)
        var surfaces = fcs.fillInSurfaces(luminanceImage, boundaries);

        // Stage 3: Integration
        return new VisualPercept(boundaries, surfaces);
    }
}

public class SurfaceCellNetwork {
    private final SurfaceCell[][] cells;

    public SurfaceRepresentation fillInSurfaces(
        double[][] image,
        BoundaryRepresentation boundaries
    ) {
        // Initialize from image
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cells[y][x].initialize(image[y][x]);
            }
        }

        // Iterative diffusion until convergence
        for (int iter = 0; iter < maxIterations; iter++) {
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    var cell = cells[y][x];
                    var boundaryStrength = boundaries.get(x, y);
                    var neighbors = getNeighborBrightness(x, y);

                    cell.evolve(
                        image[y][x],
                        boundaryStrength,
                        neighbors,
                        dt
                    );
                }
            }

            if (hasConverged()) break;
        }

        return extractSurfaceRepresentation();
    }

    private boolean hasConverged() {
        // Check if brightness changes < threshold
        return maxBrightnessChange < convergenceThreshold;
    }
}
```

**Validation Illusions**:
1. **Kanizsa Triangle**: Subjective brightness inside triangle
2. **Neon Color Spreading**: Color fills beyond physical boundaries
3. **Craik-O'Brien-Cornsweet**: Uniform brightness from edge contrast

**Tests**:
```java
@Test
void testKanizsaTriangle() {
    var vision = new ComplementaryVisionSystem(
        new BipoleCellNetwork(bcsParams),
        new SurfaceCellNetwork(fcsParams)
    );

    var image = loadKanizsaTriangleImage();
    var percept = vision.processImage(image);

    // Inside triangle should have higher brightness than background
    var triangleCenter = percept.surfaces().getBrightness(128, 128);
    var background = percept.surfaces().getBrightness(10, 10);

    assertTrue(triangleCenter > background + 0.1,
        "Kanizsa triangle should show illusory brightness");
}

@Test
void testNeonColorSpreading() {
    var vision = new ComplementaryVisionSystem(bcsParams, fcsParams);

    var image = loadNeonColorImage();
    var percept = vision.processImage(image);

    // Color should spread beyond inducing lines
    var inducingLine = percept.surfaces().getColor(50, 50);
    var spreadRegion = percept.surfaces().getColor(70, 70);

    assertColorSimilar(inducingLine, spreadRegion, 0.2,
        "Neon color should spread to nearby region");
}
```

**Success Criteria**:
- BCS-FCS integration functional
- Standard illusions reproduced
- 90%+ accuracy on validation set

#### Phase 4C: Performance Optimization (Week 3)

**Objectives**:
- Optimize diffusion computation
- Implement convergence acceleration
- Achieve real-time performance

**Optimizations**:
1. **Sparse Updates**: Only update cells near boundaries
2. **Multi-Grid Methods**: Coarse-to-fine diffusion
3. **SIMD Diffusion**: Vectorize Laplacian computation
4. **Early Termination**: Stop when converged locally

**Success Criteria**:
- Processing time < 100ms for 256x256 image
- No quality degradation
- Performance benchmarks pass

#### Phase 4D: Documentation & Examples (Week 4)

**Deliverables**:
- SURFACE_FILLING_IN_GUIDE.md
- Example: VisualIllusionsDemo.java
- Integration guide for perception research

**Success Criteria**:
- Documentation complete
- Examples demonstrate illusions
- Research community can use for perception studies

#### Phase 4 Deliverables

- [ ] `SurfaceCell.java`
- [ ] `SurfaceCellNetwork.java`
- [ ] `ComplementaryVisionSystem.java`
- [ ] BCS-FCS integration
- [ ] 40+ surface filling-in tests
- [ ] Illusion validation suite
- [ ] Documentation: SURFACE_FILLING_IN_GUIDE.md
- [ ] Example: VisualIllusionsDemo.java

---

### Phase 5: Advanced Features (6-12 weeks)

**Priority**: MEDIUM-LOW
**Duration**: 6-12 weeks
**Dependencies**: Phases 1-4 complete
**Team**: 1 engineer part-time

#### Phase 5A: GPU Acceleration Foundation (Weeks 1-4)

**Objectives**:
- Design GPU architecture
- Implement CUDA/OpenCL kernels
- Benchmark performance gains

**Target Operations**:
- Shunting dynamics (massively parallel)
- FFT computation (cuFFT)
- Matrix operations (cuBLAS)
- Diffusion solver (iterative)

**Expected Speedup**: 10x-100x for large-scale problems

#### Phase 5B: Multi-Area Hierarchies (Weeks 5-8)

**Objectives**:
- Implement V1→V2→V4 visual hierarchy
- Create inter-area communication
- Validate with hierarchical features

**Architecture**:
```
V1 (Primary Visual): Orientation, simple features
  ↓
V2 (Secondary): Complex features, illusory contours
  ↓
V4 (Higher): Object categories, attention modulation
```

#### Phase 5C: Adaptive Vigilance (Weeks 9-10)

**Objectives**:
- Implement dynamic vigilance adjustment
- Context-dependent category granularity
- Validate on category learning tasks

#### Phase 5D: Emotional Processing (Weeks 11-12)

**Objectives**:
- Integrate CogEM (Cognitive-Emotional) model
- Affective modulation of perception
- Emotion-cognition interactions

---

## 4. TESTING STRATEGY

### 4.1 Test Pyramid

```
         /\
        /  \  E2E Tests (5%)
       /____\
      /      \
     / Integ  \ Integration Tests (20%)
    / ration  \
   /___________\
  /             \
 /     Unit      \ Unit Tests (75%)
/________________\
```

**Target Distribution**:
- **Unit Tests**: 75% (375 tests) - Fast, isolated, comprehensive
- **Integration Tests**: 20% (100 tests) - Component interactions
- **End-to-End Tests**: 5% (25 tests) - Full system validation

**Total Target**: 500+ tests (consolidated from 556 current tests)

### 4.2 Test Categories

#### Mathematical Validation Tests

**Purpose**: Verify equations match paper specifications

```java
@Test
void testShuntingDynamicsEquation() {
    // Grossberg (1973) equation: dx/dt = -Ax + (B-x)E - (x+C)I
    var params = new ShuntingParameters(
        0.1,  // A: decay
        1.0,  // B: upper bound
        0.0,  // C: lower bound
        0.5,  // E: excitation
        0.3   // I: inhibition
    );

    var dynamics = new ShuntingDynamics(params);
    var x = 0.5;
    var dxdt = dynamics.computeDerivative(x);

    // Expected: -0.1*0.5 + (1.0-0.5)*0.5 - (0.5+0.0)*0.3
    var expected = -0.05 + 0.25 - 0.15;

    assertEquals(expected, dxdt, 1e-10,
        "Shunting equation must match Grossberg (1973)");
}

@Test
void testARTMatchingRule() {
    // Vigilance test: ρ ≤ |X*| / |I|
    var vigilance = 0.85;
    var input = new double[]{1.0, 0.8, 0.6, 0.4};
    var expectation = new double[]{0.9, 0.7, 0.5, 0.3};

    var matchQuality = computeMatchQuality(input, expectation);
    var resonance = matchQuality >= vigilance;

    // Manual calculation
    var matchedMagnitude = dotProduct(min(input, expectation), ones(4));
    var inputMagnitude = dotProduct(input, ones(4));
    var expectedMatch = matchedMagnitude / inputMagnitude;

    assertEquals(expectedMatch, matchQuality, 1e-10);
}
```

**Coverage**:
- Shunting dynamics: 15 tests
- Transmitter dynamics: 10 tests
- ART matching: 10 tests
- Oscillation analysis: 15 tests
- Surface diffusion: 10 tests

#### Biological Fidelity Tests

**Purpose**: Ensure implementation matches neuroscience

```java
@Test
void testLayer4TimeConstant() {
    // Layer 4 should be fastest (10-50ms time constant)
    var layer4 = new Layer4(Layer4Parameters.paperDefaults());

    var timeConstant = layer4.getTimeConstant();

    assertTrue(timeConstant >= 0.010 && timeConstant <= 0.050,
        "Layer 4 time constant must be 10-50ms (Sherman & Guillery 1998)");
}

@Test
void testBipoleThreeWayFiring() {
    // Von der Heydt et al. (1984): Bipole cells fire in three conditions
    var bipole = new BipoleCell(BipoleCellParameters.paperDefaults());

    // Condition 1: Direct bottom-up
    var bottomUp = 1.0;
    var leftHorizontal = 0.0;
    var rightHorizontal = 0.0;
    assertTrue(bipole.computeActivation(bottomUp, leftHorizontal, rightHorizontal) > 0.5);

    // Condition 2: Bilateral horizontal
    bottomUp = 0.0;
    leftHorizontal = 0.8;
    rightHorizontal = 0.8;
    assertTrue(bipole.computeActivation(bottomUp, leftHorizontal, rightHorizontal) > 0.5);

    // Condition 3: Combined (strongest)
    bottomUp = 0.6;
    leftHorizontal = 0.7;
    rightHorizontal = 0.7;
    var combined = bipole.computeActivation(bottomUp, leftHorizontal, rightHorizontal);
    assertTrue(combined > 0.8);
}

@Test
void testGammaOscillationFrequency() {
    // Grossberg (2017): Conscious resonance produces ~40 Hz gamma
    var circuit = CorticalCircuit.builder()
        .oscillatoryDynamics(OscillatoryDynamics.gammaDefault())
        .build();

    var pattern = generateResonantPattern();
    for (int t = 0; t < 1000; t++) {
        circuit.processTimestep(pattern, t * 0.001);
    }

    var metrics = circuit.getOscillationMetrics();
    var frequency = metrics.dominantFrequency();

    assertTrue(frequency >= 30.0 && frequency <= 50.0,
        "Resonance should produce gamma oscillations (30-50 Hz)");
}
```

**Coverage**:
- Time scale separation: 10 tests
- Neural mechanisms: 20 tests
- Biological constraints: 15 tests
- Cognitive phenomena: 15 tests

#### Performance Tests

**Purpose**: Ensure performance targets met

```java
@Test
void testSIMDSpeedup() {
    var sequential = new SequentialProcessor(params);
    var simd = new SIMDBatchProcessor(SIMDConfiguration.optimal());

    var patterns = generateTestPatterns(256);

    var seqTime = measureExecutionTime(() ->
        sequential.processBatch(patterns)
    );

    var simdTime = measureExecutionTime(() ->
        simd.processBatch(patterns)
    );

    var speedup = seqTime / simdTime;

    assertTrue(speedup >= 1.50,
        "SIMD batch-64 should achieve 1.50x+ speedup (got %.2fx)".formatted(speedup));
}

@Benchmark
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
public long benchmarkFullCircuit() {
    var circuit = CorticalCircuit.builder()
        .simdConfiguration(SIMDConfiguration.optimal())
        .build();

    var pattern = generateTestPattern();
    var result = circuit.process(pattern);

    return result.getProcessingTime();
}
```

**Coverage**:
- SIMD speedup: 20 tests
- Throughput: 15 tests
- Memory usage: 10 tests
- Scalability: 15 tests

#### Integration Tests

**Purpose**: Validate component interactions

```java
@Test
void testTemporalLaminarIntegration() {
    var circuit = UnifiedCorticalCircuit.builder()
        .temporalProcessor(TemporalProcessor.create())
        .layer4Parameters(Layer4Parameters.paperDefaults())
        .build();

    // Temporal sequence
    var sequence = List.of(
        generatePattern("A"),
        generatePattern("B"),
        generatePattern("C")
    );

    // Process sequence
    for (var pattern : sequence) {
        circuit.process(pattern);
    }

    // Verify temporal chunking
    var chunks = circuit.getTemporalProcessor().getChunks();
    assertFalse(chunks.isEmpty());

    // Verify laminar processing
    var layer4Activation = circuit.getLayer4().getActivation();
    assertNotNull(layer4Activation);
}

@Test
void testBCSFCSIntegration() {
    var vision = new ComplementaryVisionSystem(
        new BipoleCellNetwork(bcsParams),
        new SurfaceCellNetwork(fcsParams)
    );

    var image = loadTestImage();
    var percept = vision.processImage(image);

    // Boundaries should gate surface diffusion
    var boundary = percept.boundaries().get(50, 50);
    var surface = percept.surfaces().get(50, 50);

    if (boundary > 0.8) {
        // Strong boundary should preserve surface discontinuity
        var leftSurface = percept.surfaces().get(49, 50);
        var rightSurface = percept.surfaces().get(51, 50);
        assertTrue(Math.abs(leftSurface - rightSurface) > 0.1);
    }
}
```

**Coverage**:
- Multi-pathway: 20 tests
- BCS-FCS: 15 tests
- Temporal-laminar: 15 tests
- Circuit-level: 30 tests

#### Regression Tests

**Purpose**: Prevent reintroduction of known bugs

```java
@Test
void testBug_TemporalChunkingEdgeCase_Issue123() {
    // Regression test for GitHub issue #123
    // Edge case: Single-item sequence should not crash

    var processor = new TemporalProcessor();
    var singleItem = List.of(generatePattern("X"));

    assertDoesNotThrow(() ->
        processor.processSequence(singleItem)
    );

    var chunks = processor.getChunks();
    assertEquals(1, chunks.size());
}

@Test
void testBug_SIMDBoundaryCondition_Issue456() {
    // Regression test for GitHub issue #456
    // SIMD batch size not multiple of mini-batch

    var simd = new SIMDBatchProcessor(
        SIMDConfiguration.withBatchSize(64)
    );

    var patterns = generateTestPatterns(100);  // Not multiple of 64

    assertDoesNotThrow(() ->
        simd.processBatch(patterns)
    );
}
```

**Coverage**:
- Historical bugs: 20 tests
- Edge cases: 15 tests
- Boundary conditions: 15 tests

### 4.3 Test Execution Strategy

#### Parallel Execution

```xml
<!-- Maven Surefire configuration -->
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <configuration>
        <parallel>classes</parallel>
        <threadCount>4</threadCount>
        <perCoreThreadCount>true</perCoreThreadCount>
    </configuration>
</plugin>
```

#### Test Categories

```java
// Fast unit tests (< 100ms each)
@Tag("fast")
public class ShuntingDynamicsTest { }

// Integration tests (100ms - 1s)
@Tag("integration")
public class CorticalCircuitTest { }

// Performance benchmarks (> 1s)
@Tag("benchmark")
public class SIMDBenchmark { }

// Run only fast tests during development
mvn test -Dgroups="fast"

// Run all tests before commit
mvn test

// Run benchmarks separately
mvn test -Dgroups="benchmark"
```

#### Continuous Integration

```yaml
# GitHub Actions workflow
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-java@v2
        with:
          java-version: '24'
      - name: Run unit tests
        run: mvn test -Dgroups="fast,integration"
      - name: Run performance tests
        run: mvn test -Dgroups="benchmark"
      - name: Generate coverage report
        run: mvn jacoco:report
```

### 4.4 Test Coverage Targets

| Component | Line Coverage | Branch Coverage |
|-----------|--------------|-----------------|
| Dynamics | 95%+ | 90%+ |
| Layers | 90%+ | 85%+ |
| Temporal | 90%+ | 85%+ |
| Network | 85%+ | 80%+ |
| SIMD | 85%+ | 80%+ |
| Integration | 80%+ | 75%+ |
| **Overall** | **90%+** | **85%+** |

---

## 5. CODE ORGANIZATION

### 5.1 Package Structure

```
com.hellblazer.art.cortical/
├── dynamics/                    # Neural dynamics
│   ├── NeuralDynamics.java           # Interface
│   ├── ShuntingDynamics.java         # Grossberg (1973)
│   ├── ShuntingParameters.java       # Record
│   ├── TransmitterDynamics.java      # Habituative gating
│   ├── TransmitterParameters.java    # Record
│   ├── OscillatoryDynamics.java      # NEW: Gamma oscillations
│   └── OscillationParameters.java    # NEW: Record
│
├── temporal/                    # Temporal processing (LIST PARSE)
│   ├── TemporalProcessor.java        # Main processor
│   ├── WorkingMemory.java            # STORE 2
│   ├── WorkingMemoryParameters.java  # Record
│   ├── MaskingField.java             # Multi-scale chunking
│   ├── MaskingFieldParameters.java   # Record
│   ├── ItemNode.java                 # Individual items
│   ├── ListChunk.java                # Temporal chunks
│   └── TemporalPattern.java          # Record
│
├── layers/                      # 6-layer laminar circuit
│   ├── Layer.java                    # Sealed interface
│   ├── Layer1.java                   # Sustained attention
│   ├── Layer23.java                  # Horizontal grouping
│   ├── Layer4.java                   # Thalamic input
│   ├── Layer5.java                   # Motor output
│   ├── Layer6.java                   # Top-down feedback
│   ├── VectorizedLayer.java          # NEW: SIMD interface
│   ├── VectorizedLayer4.java         # NEW: SIMD Layer 4
│   ├── VectorizedLayer5.java         # NEW: SIMD Layer 5
│   ├── VectorizedLayer6.java         # NEW: SIMD Layer 6
│   ├── LayerParameters.java          # Record
│   ├── LayerType.java                # Enum
│   ├── LayerState.java               # Record
│   ├── WeightMatrix.java             # Weight management
│   ├── LayerActivationListener.java  # Event listener
│   └── CorticalCircuit.java          # Full circuit integration
│
├── network/                     # Neural networks
│   ├── BipoleCell.java               # Boundary completion
│   ├── BipoleCellNetwork.java        # BCS network
│   ├── BipoleCellParameters.java     # Record
│   ├── SurfaceCell.java              # NEW: Surface filling-in
│   ├── SurfaceCellNetwork.java       # NEW: FCS network
│   ├── SurfaceCellParameters.java    # NEW: Record
│   └── ComplementaryVision.java      # NEW: BCS+FCS integration
│
├── performance/                 # SIMD optimization
│   ├── SIMDConfiguration.java        # NEW: Configuration record
│   ├── SIMDBatchProcessor.java       # NEW: Batch processor
│   ├── BatchDataLayout.java          # PORTED: Transpose utils
│   ├── BatchShuntingDynamics.java    # PORTED: SIMD dynamics
│   ├── PerformanceMonitor.java       # NEW: Runtime tuning
│   ├── PerformanceMetrics.java       # NEW: Record
│   └── VectorizedCircuit.java        # PORTED: Full circuit
│
├── analysis/                    # NEW: Oscillation analysis
│   ├── OscillationAnalyzer.java      # Per-layer analyzer
│   ├── FFTProcessor.java             # Spectral analysis
│   ├── PhaseDetector.java            # Phase synchronization
│   ├── ResonanceDetector.java        # Enhanced resonance
│   ├── ConsciousnessMetrics.java     # Record
│   ├── OscillationMetrics.java       # Record
│   └── PowerSpectrum.java            # Record
│
├── integration/                 # High-level integration
│   ├── UnifiedCorticalCircuit.java   # NEW: Unified circuit
│   ├── ARTCorticalProcessor.java     # NEW: Main processor
│   ├── CircuitBuilder.java           # NEW: Fluent builder
│   └── CircuitConfiguration.java     # NEW: Record
│
└── util/                        # Utilities
    ├── CircularBuffer.java           # NEW: Efficient history
    ├── MathUtils.java                # Math helpers
    ├── VectorUtils.java              # Vector operations
    └── ValidationUtils.java          # Parameter validation
```

### 5.2 Module Dependencies

```
art-cortical (unified)
├── art-core
│   └── Pattern, ART interfaces
├── art-performance
│   └── Vectorization interfaces
├── JTransforms
│   └── FFT computation
├── Vector API (Java 24)
│   └── SIMD operations
└── LWJGL (optional)
    └── Visualization
```

### 5.3 Naming Conventions

#### Classes

- **Interfaces**: Descriptive noun (e.g., `Layer`, `NeuralDynamics`)
- **Implementations**: Interface name + descriptive suffix (e.g., `Layer4`, `ShuntingDynamics`)
- **Records**: Descriptive noun + "Parameters" or "Metrics" (e.g., `ShuntingParameters`, `OscillationMetrics`)
- **Tests**: Class name + "Test" (e.g., `Layer4Test`, `SIMDBatchProcessorTest`)

#### Methods

- **Getters**: `get` prefix (e.g., `getActivation()`, `getOscillationMetrics()`)
- **Setters**: Avoid (use immutable records)
- **Processors**: Verb (e.g., `process()`, `analyze()`, `detect()`)
- **Predicates**: `is`/`has` prefix (e.g., `isResonant()`, `hasConverged()`)

#### Variables

- Use `var` for local variables where type is obvious
- Descriptive names (e.g., `bottomUpActivation`, `topDownExpectation`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_MINI_BATCH_SIZE`)

#### Packages

- Lowercase, singular (e.g., `dynamics`, `layer`, `network`)
- Organized by functionality, not by class type

### 5.4 Documentation Standards

#### Javadoc Requirements

**All public APIs must have**:
1. Purpose summary
2. Parameter descriptions
3. Return value description
4. Exceptions thrown
5. Usage example (for complex APIs)
6. Paper citation (for algorithms)

**Example**:
```java
/**
 * Implements shunting neural dynamics for cortical activation.
 *
 * <p>Based on Grossberg (1973) membrane equation:
 * <pre>
 * dx/dt = -Ax + (B-x)E - (x+C)I
 * </pre>
 *
 * <p>Properties:
 * <ul>
 *   <li>Contrast normalization: Response = E / (A + E + I)</li>
 *   <li>Bounded activation: x ∈ [C, B]</li>
 *   <li>Lyapunov stable: Guaranteed convergence</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * var params = new ShuntingParameters(0.1, 1.0, 0.0, 0.5, 0.3);
 * var dynamics = new ShuntingDynamics(params);
 * var nextState = dynamics.evolve(currentState, 0.01);
 * }</pre>
 *
 * @see Grossberg, S. (1973). "Contour Enhancement, Short Term Memory,
 *      and Constancies in Reverberating Neural Networks"
 * @since 2.0.0
 */
public class ShuntingDynamics implements NeuralDynamics {
    /**
     * Evolves neural state by one timestep using shunting dynamics.
     *
     * @param currentState Current activation state (dimension N)
     * @param dt Timestep in seconds (typically 0.001 - 0.01)
     * @return Next activation state after dt seconds
     * @throws IllegalArgumentException if dt <= 0 or currentState is null
     */
    public double[] evolve(double[] currentState, double dt) {
        // ...
    }
}
```

#### README Standards

Each major component should have README.md:
- Overview
- Key features
- Usage examples
- API reference link
- Performance characteristics
- Related papers

---

## 6. VALIDATION & QUALITY ASSURANCE

### 6.1 Code Review Checklist

#### Functionality
- [ ] All requirements implemented
- [ ] Edge cases handled
- [ ] Error conditions validated
- [ ] Input validation present

#### Testing
- [ ] Unit tests written (test-first)
- [ ] Integration tests added
- [ ] All tests pass (100% pass rate)
- [ ] Coverage targets met (90%+ line coverage)
- [ ] Performance benchmarks run

#### Code Quality
- [ ] Follows Java 24 idioms (var, records, pattern matching)
- [ ] No synchronized keyword (concurrent collections instead)
- [ ] AutoCloseable for resource management
- [ ] Immutable where possible (records)
- [ ] Clean separation of concerns

#### Documentation
- [ ] Javadoc complete (all public APIs)
- [ ] README updated
- [ ] Examples provided
- [ ] Paper citations included
- [ ] Migration guide (if API changes)

#### Performance
- [ ] No obvious performance issues
- [ ] SIMD used where beneficial
- [ ] Memory allocations minimized
- [ ] Benchmark results acceptable

#### Biological Fidelity
- [ ] Equations match papers (1e-10 precision)
- [ ] Time constants within biological ranges
- [ ] Behavioral validation tests pass
- [ ] Paper citations correct

### 6.2 Performance Benchmarks

#### SIMD Optimization Benchmarks

**Baseline**: art-laminar (1.30x speedup, 1049.7 patterns/sec)

**Target**: art-cortical unified (1.50x+ speedup, 1,200+ patterns/sec)

**Benchmark Suite**:
```java
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class PerformanceBenchmarks {

    @Benchmark
    public void sequentialProcessing(Blackhole bh) {
        var circuit = createSequentialCircuit();
        var result = circuit.processBatch(patterns);
        bh.consume(result);
    }

    @Benchmark
    public void simdBatch32(Blackhole bh) {
        var circuit = createSIMDCircuit(32);
        var result = circuit.processBatch(patterns);
        bh.consume(result);
    }

    @Benchmark
    public void simdBatch64(Blackhole bh) {
        var circuit = createSIMDCircuit(64);
        var result = circuit.processBatch(patterns);
        bh.consume(result);
    }
}
```

**Acceptance Criteria**:
| Benchmark | Target Throughput | Max Latency |
|-----------|------------------|-------------|
| Sequential | 800+ patterns/sec | 1.25ms/pattern |
| SIMD Batch-32 | 1,050+ patterns/sec | 0.95ms/pattern |
| SIMD Batch-64 | 1,200+ patterns/sec | 0.83ms/pattern |

#### Memory Benchmarks

**Targets**:
- Heap allocation: < 1MB per 1000 patterns
- GC pressure: < 5% time in GC
- Memory leaks: Zero (validated with profiler)

#### Scalability Benchmarks

**Targets**:
- Linear scaling up to 8 cores
- Batch size 1-1000 patterns (consistent performance per pattern)
- Dimension 64-1024 (performance degrades gracefully)

### 6.3 Biological Validation Tests

#### Precision Validation

**All equations must satisfy**: |computed - expected| < 1e-10

```java
@Test
void testShuntingEquationPrecision() {
    var expected = computeExpectedValue();
    var actual = dynamics.computeActivation();

    assertEquals(expected, actual, 1e-10,
        "Shunting dynamics must have 1e-10 precision");
}
```

#### Paper Fidelity Validation

**Target**: 97%+ fidelity to source papers

**Validation Matrix**:
| Component | Paper | Fidelity | Tests |
|-----------|-------|----------|-------|
| Shunting Dynamics | Grossberg (1973) | 99% | 15 |
| ART Matching | Carpenter & Grossberg (1987) | 98% | 10 |
| Bipole Cells | Grossberg & Mingolla (1985) | 96% | 9 |
| LIST PARSE | Kazerounian & Grossberg (2014) | 95% | 30 |
| Oscillations | Grossberg (2017) | 90% | 15 |
| Surface Filling-In | Grossberg & Todorović (1988) | 90% | 15 |

#### Cognitive Phenomena Validation

**Standard phenomena to reproduce**:
1. Miller's 7±2 working memory capacity
2. Phone number chunking (555-1234)
3. Kanizsa triangle illusory brightness
4. Gamma oscillations during conscious perception
5. Catastrophic forgetting prevention

---

## 7. TIMELINE & MILESTONES

### 7.1 Detailed Timeline

```
Week 1-2: Phase 1 - SIMD Optimization
├── Week 1: Preparation, mini-batch increase, validation
└── Week 2: art-cortical integration, documentation

Week 3-5: Phase 2 - Oscillatory Dynamics
├── Week 3: Mathematical foundation (FFT, phase detection)
├── Week 4: Layer integration, circular buffers
└── Week 5: Resonance enhancement, circuit integration

Week 6-11: Phase 3 - Module Consolidation
├── Week 6: Preparation, architecture design
├── Week 7-8: SIMD integration, porting
├── Week 9: Test consolidation (500+ tests)
├── Week 10: Documentation, migration guide
└── Week 11: Deprecation, release v2.0.0

Week 12-15: Phase 4 - Surface Filling-In
├── Week 12: SurfaceCell implementation
├── Week 13: BCS-FCS integration
├── Week 14: Performance optimization
└── Week 15: Documentation, examples

Week 16-27: Phase 5 - Advanced Features (optional)
├── Week 16-19: GPU acceleration foundation
├── Week 20-23: Multi-area hierarchies
├── Week 24-25: Adaptive vigilance
└── Week 26-27: Emotional processing
```

### 7.2 Milestones

#### Milestone 1: SIMD Optimization Complete (Week 2)
- [ ] Mini-batch size increased to 64
- [ ] 1.50x+ speedup achieved
- [ ] Bit-exact equivalence validated
- [ ] Integrated into art-cortical
- [ ] 20+ SIMD tests passing
- [ ] Documentation complete

**Acceptance Criteria**:
- Performance benchmark shows ≥1.50x speedup
- All tests pass (154 existing + 20 new)
- Zero regressions

#### Milestone 2: Oscillatory Dynamics Complete (Week 5)
- [ ] FFT spectral analysis implemented
- [ ] Phase detection functional
- [ ] Oscillation tracking per layer
- [ ] Enhanced resonance detection
- [ ] Consciousness metrics available
- [ ] 30+ oscillation tests passing
- [ ] Documentation complete

**Acceptance Criteria**:
- Gamma oscillations detected (30-50 Hz)
- Phase synchronization accurate (0.1 rad)
- Consciousness likelihood correlates with resonance

#### Milestone 3: Module Consolidation Complete (Week 11)
- [ ] SIMD code ported from art-laminar
- [ ] 500+ consolidated tests passing
- [ ] art-laminar deprecated with delegation
- [ ] Migration guide complete
- [ ] Release v2.0.0 published

**Acceptance Criteria**:
- Performance parity (1.50x speedup maintained)
- Zero functionality loss
- Clean deprecation path

#### Milestone 4: Surface Filling-In Complete (Week 15)
- [ ] SurfaceCell network implemented
- [ ] BCS-FCS integration functional
- [ ] Standard illusions reproduced (90%+ accuracy)
- [ ] 40+ surface tests passing
- [ ] Documentation complete

**Acceptance Criteria**:
- Kanizsa triangle illusory brightness
- Neon color spreading
- Real-time performance (< 100ms for 256x256)

#### Milestone 5: Advanced Features Complete (Week 27)
- [ ] GPU acceleration foundation
- [ ] Multi-area hierarchies (V1→V2→V4)
- [ ] Adaptive vigilance
- [ ] Emotional processing integration

**Acceptance Criteria**:
- GPU speedup ≥ 10x (if implemented)
- Hierarchical features validated
- All tests passing

### 7.3 Critical Path

```
Phase 1 (SIMD) → Phase 2 (Oscillations) → Phase 3 (Consolidation)
```

**Critical path duration**: 11 weeks

**Phases 4-5 can proceed in parallel** after Phase 3 completion.

### 7.4 Risk Buffers

- **Phase 1**: 2 weeks scheduled, 1 week minimum → 1 week buffer
- **Phase 2**: 3 weeks scheduled, 2 weeks minimum → 1 week buffer
- **Phase 3**: 6 weeks scheduled, 4 weeks minimum → 2 weeks buffer
- **Phase 4**: 4 weeks scheduled, 3 weeks minimum → 1 week buffer
- **Phase 5**: 12 weeks scheduled, 6 weeks minimum → 6 weeks buffer

**Total buffer**: 11 weeks (40% of total duration)

---

## 8. RISK MANAGEMENT

### 8.1 Technical Risks

#### Risk 1: SIMD Optimization Fails to Achieve 1.50x Speedup

**Probability**: Medium
**Impact**: High

**Mitigation**:
- Extensive profiling before implementation
- Incremental optimization with validation
- Fallback: Keep 1.30x speedup, document limitations
- Alternative: Focus on GPU acceleration (Phase 5)

**Contingency**:
If speedup < 1.40x after 2 weeks:
1. Analyze profiling data for bottlenecks
2. Try alternative mini-batch sizes (48, 80, 96)
3. Consider hybrid approach (SIMD + sequential)
4. Document findings, proceed to Phase 2

#### Risk 2: Oscillation Detection Has Excessive Performance Overhead

**Probability**: High
**Impact**: Medium

**Mitigation**:
- Make oscillation tracking optional (feature flag)
- Lazy initialization (only when requested)
- Downsample FFT computation (every N timesteps)
- Use efficient FFT library (JTransforms)

**Contingency**:
If overhead > 20%:
1. Profile to identify bottleneck
2. Reduce FFT frequency (every 10 timesteps instead of every timestep)
3. Use smaller history buffer (128 instead of 256)
4. Implement parallel FFT across layers

#### Risk 3: Module Consolidation Introduces Regressions

**Probability**: Medium
**Impact**: High

**Mitigation**:
- Gradual migration (one component at a time)
- Parallel testing (old vs new)
- Comprehensive regression test suite
- Automated semantic equivalence checking

**Contingency**:
If regressions found:
1. Identify root cause via differential testing
2. Fix immediately (do not proceed to next component)
3. Add regression test to prevent recurrence
4. If unfixable, revert and reassess architecture

#### Risk 4: Surface Filling-In Algorithm Unstable

**Probability**: Medium
**Impact**: Medium

**Mitigation**:
- Start with simple test cases (uniform regions)
- Extensive parameter tuning
- Convergence acceleration techniques (multi-grid)
- Numerical stability analysis

**Contingency**:
If instability persists:
1. Review mathematical formulation (check discretization)
2. Add damping term to diffusion equation
3. Use implicit integration instead of explicit
4. Consult original papers for numerical methods

### 8.2 Schedule Risks

#### Risk 5: Phase Takes Longer Than Estimated

**Probability**: Medium
**Impact**: Medium

**Mitigation**:
- Conservative estimates with buffers
- Weekly progress tracking
- Early warning system (if > 50% time used, < 50% progress)
- Prioritize critical path items

**Contingency**:
If phase running late:
1. Reassess scope (can anything be deferred?)
2. Add resources (if available)
3. Reduce testing depth (but maintain 100% pass rate)
4. Use buffer time from previous phases

#### Risk 6: Critical Path Blocked

**Probability**: Low
**Impact**: High

**Mitigation**:
- Identify dependencies early
- Prepare alternative implementations
- Maintain multiple parallel work streams
- Regular standup meetings to surface blockers

**Contingency**:
If critical path blocked:
1. Identify blocker immediately
2. Assign dedicated resource to unblock
3. Initiate parallel work on non-dependent items
4. Escalate if blocker persists > 2 days

### 8.3 Quality Risks

#### Risk 7: Test Coverage Drops Below 90%

**Probability**: Low
**Impact**: Medium

**Mitigation**:
- Test-first development (write tests before code)
- Automated coverage reports in CI
- Coverage gate (build fails if < 90%)
- Regular coverage review

**Contingency**:
If coverage drops:
1. Identify uncovered code paths
2. Write missing tests immediately
3. Do not merge until coverage restored
4. Review test-first process compliance

#### Risk 8: Biological Fidelity Degrades

**Probability**: Low
**Impact**: High

**Mitigation**:
- Validation tests for all equations (1e-10 precision)
- Paper fidelity tests
- Regular comparison with source papers
- Biological constraints in parameter validation

**Contingency**:
If fidelity drops below 95%:
1. Identify divergence from paper
2. Review mathematical formulation
3. Consult research synthesis in ChromaDB
4. Add tests to prevent future divergence

---

## 9. APPENDICES

### 9.1 Reference Papers

1. **Grossberg, S. (1973)**. "Contour Enhancement, Short Term Memory, and Constancies in Reverberating Neural Networks." *Studies in Applied Mathematics*, 52, 213-257.
   - Shunting dynamics foundation

2. **Grossberg, S., & Mingolla, E. (1985)**. "Neural Dynamics of Form Perception: Boundary Completion, Illusory Figures, and Neon Color Spreading." *Psychological Review*, 92(2), 173-211.
   - Bipole cells, boundary completion

3. **Carpenter, G. A., & Grossberg, S. (1987)**. "A Massively Parallel Architecture for a Self-Organizing Neural Pattern Recognition Machine." *Computer Vision, Graphics, and Image Processing*, 37, 54-115.
   - ART matching rule, stability-plasticity

4. **Grossberg, S., & Todorović, D. (1988)**. "Neural Dynamics of 1-D and 2-D Brightness Perception: A Unified Model of Classical and Recent Phenomena." *Perception & Psychophysics*, 43, 241-277.
   - Surface filling-in, FACADE theory

5. **Sherman, S. M., & Guillery, R. W. (1998)**. "On the Actions that One Nerve Cell Can Have on Another: Distinguishing 'Drivers' from 'Modulators'." *Proceedings of the National Academy of Sciences*, 95, 7121-7126.
   - Driving vs modulatory distinction, layer characteristics

6. **Kazerounian, S., & Grossberg, S. (2014)**. "Real-Time Learning of Predictive Recognition Categories that Chunk Sequences of Items Stored in Working Memory." *Frontiers in Psychology*, 5, 1053.
   - LIST PARSE model, temporal chunking

7. **Grossberg, S. (2017)**. "Towards Solving the Hard Problem of Consciousness: The Varieties of Brain Resonances and the Conscious Experiences that They Support." *Neural Networks*, 87, 38-95.
   - CLEARS framework, consciousness theory

8. **Grossberg, S. (2021)**. "Conscious Mind, Resonant Brain: How Each Brain Makes a Mind." *Oxford University Press*.
   - Comprehensive treatment of unified theory

### 9.2 Glossary

**ART (Adaptive Resonance Theory)**: Neural network architecture solving stability-plasticity dilemma through match-based learning.

**BCS (Boundary Contour System)**: Oriented, contrast-polarity insensitive system for structure determination.

**CLEARS (Consciousness, Learning, Expectation, Attention, Resonance, Synchrony)**: Framework linking resonance to consciousness.

**FCS (Feature Contour System)**: Unoriented, contrast-polarity sensitive system for surface quality representation.

**Gamma Oscillations**: 30-80 Hz neural oscillations correlating with conscious perception.

**LIST PARSE**: Working memory model for temporal sequence learning and chunking.

**Resonance**: State where bottom-up features match top-down expectations, triggering learning.

**SIMD (Single Instruction Multiple Data)**: Parallel processing executing same operation on multiple data elements.

**Shunting Dynamics**: Neural activation dynamics with multiplicative inhibition (Grossberg 1973).

**Vigilance**: ART parameter controlling category granularity (higher = more specific).

### 9.3 Acronym Index

- **API**: Application Programming Interface
- **ART**: Adaptive Resonance Theory
- **BCS**: Boundary Contour System
- **CI/CD**: Continuous Integration/Continuous Deployment
- **CUDA**: Compute Unified Device Architecture
- **FCS**: Feature Contour System
- **FFT**: Fast Fourier Transform
- **GPU**: Graphics Processing Unit
- **JMH**: Java Microbenchmark Harness
- **JVM**: Java Virtual Machine
- **LWJGL**: Lightweight Java Game Library
- **OpenCL**: Open Computing Language
- **SIMD**: Single Instruction Multiple Data
- **STORE**: Short-Term Order REtention (working memory)

---

## IMPLEMENTATION CHECKLIST

### Pre-Implementation
- [ ] Review unified neocortical theory synthesis (ChromaDB)
- [ ] Read key papers (Grossberg 1973, 1985, 2014, 2017)
- [ ] Set up development environment (Java 24, Maven 3.9.1+)
- [ ] Clone repository, verify build succeeds
- [ ] Run existing test suite (1,680+ tests should pass)

### Phase 1: SIMD Optimization (Weeks 1-2)
- [ ] Benchmark baseline performance (art-laminar 1.30x)
- [ ] Implement SIMDConfiguration record
- [ ] Increase mini-batch size to 64
- [ ] Implement PerformanceMonitor with auto-tuning
- [ ] Port to art-cortical with backward compatibility
- [ ] Validate bit-exact equivalence (0.00e+00 difference)
- [ ] Achieve 1.50x+ speedup
- [ ] Write 20+ SIMD tests
- [ ] Document SIMD_OPTIMIZATION_GUIDE.md

### Phase 2: Oscillatory Dynamics (Weeks 3-6)
- [ ] Implement FFTProcessor using JTransforms
- [ ] Implement PhaseDetector for synchronization
- [ ] Create OscillationAnalyzer per-layer tracking
- [ ] Implement CircularBuffer for efficient history
- [ ] Enhance ResonanceDetector with phase sync
- [ ] Create ConsciousnessMetrics record
- [ ] Integrate with all 6 layers (optional feature)
- [ ] Write 30+ oscillation tests
- [ ] Create ConsciousnessResearchDemo example
- [ ] Document OSCILLATORY_DYNAMICS_GUIDE.md

### Phase 3: Module Consolidation (Weeks 7-14)
- [ ] Design unified module architecture
- [ ] Create art-cortical-unified branch
- [ ] Port all SIMD implementations from art-laminar
- [ ] Merge test suites (402 + 154 → 500+ unique)
- [ ] Validate performance parity (1.50x maintained)
- [ ] Write MIGRATION_GUIDE.md
- [ ] Write UNIFIED_CORTICAL_ARCHITECTURE.md
- [ ] Deprecate art-laminar with delegation
- [ ] Release art-cortical v2.0.0
- [ ] Update parent POM dependencies

### Phase 4: Surface Filling-In (Weeks 12-15)
- [ ] Implement SurfaceCell with diffusion dynamics
- [ ] Implement SurfaceCellNetwork (2D grid)
- [ ] Integrate with BipoleCellNetwork (BCS-FCS)
- [ ] Validate on Kanizsa triangle
- [ ] Validate on neon color spreading
- [ ] Optimize for real-time performance (< 100ms)
- [ ] Write 40+ surface filling-in tests
- [ ] Create VisualIllusionsDemo example
- [ ] Document SURFACE_FILLING_IN_GUIDE.md

### Phase 5: Advanced Features (Weeks 16-27, Optional)
- [ ] Design GPU acceleration architecture
- [ ] Implement CUDA/OpenCL kernels
- [ ] Benchmark GPU speedup (target 10x+)
- [ ] Implement V1→V2→V4 hierarchy
- [ ] Implement adaptive vigilance
- [ ] Integrate CogEM emotional processing
- [ ] Write comprehensive tests for each feature
- [ ] Document advanced features

### Post-Implementation
- [ ] Final performance benchmark suite
- [ ] Complete API documentation (100% public APIs)
- [ ] Create comprehensive examples
- [ ] Publish release notes
- [ ] Update CHANGELOG.md
- [ ] Store implementation insights in ChromaDB
- [ ] Create knowledge graph entries
- [ ] Community announcement (if applicable)

---

## DOCUMENT HISTORY

**Version 1.0** (2025-10-02):
- Initial comprehensive plan
- 5 phases, 11 major sections
- Based on unified neocortical theory synthesis
- Targets: 1.50x SIMD speedup, gamma oscillations, module consolidation

**Status**: APPROVED FOR IMPLEMENTATION

**Next Review**: 2025-11-02 (post Phase 1-2 completion)

---

## APPROVAL SIGNATURES

**Plan Author**: Claude Code (Strategic AI Architect)
**Date**: 2025-10-02
**Status**: Awaiting plan-auditor review

---

*This document serves as the authoritative implementation guide for ART cortical architecture enhancements. All implementation decisions should reference this plan. Any deviations must be documented with rationale.*
