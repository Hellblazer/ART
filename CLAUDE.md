# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive **Adaptive Resonance Theory (ART)** implementation in Java 24, featuring 60+ ART algorithm variants with both standard and high-performance vectorized implementations. ART is a neural network architecture for unsupervised and supervised learning, pattern recognition, and clustering. The project is structured as a multi-module Maven build with extensive test coverage (1,408 tests, 100% pass rate), SIMD optimization via Java Vector API, and GPU acceleration support.

## Build System & Commands

### Maven Commands
- **Build the project**: `mvn clean compile`
- **Run tests**: `mvn test` 
- **Run single test**: `mvn test -Dtest=ClassName#methodName`
- **Build with all modules**: `mvn clean install`
- **Generate sources**: `mvn generate-sources` (for Protocol Buffers and JOOQ)
- **Check dependency convergence**: `mvn enforcer:enforce`
- **Update versions**: `mvn versions:display-dependency-updates`

### Requirements
- **Java 24+** (configured for Java 24 features)
- **Maven 3.9.1+** (enforced)
- **macOS ARM64** (LWJGL natives configured for Apple Silicon)

## Architecture & Technology Stack

### Core Technologies
- **Java 24** with modern language features (var, records, pattern matching, virtual threads)
- **Maven multi-module** structure for component organization
- **Protocol Buffers + gRPC** for serialization and communication
- **JOOQ** for type-safe database operations
- **H2 Database** for embedded data storage

### Data Preprocessing & Integration
- **DataPreprocessor** - Comprehensive data preprocessing pipeline
  - Auto-detection of data ranges with normalization to [0,1]
  - Missing value handling with multiple strategies (mean, median, zero)
  - Complement coding for ART algorithms [x, 1-x]
  - L1/L2 normalization options
  - Pipeline builder pattern for composable preprocessing steps
  - Batch processing support
- **SklearnWrapper** - Scikit-learn compatible API wrapper
  - Methods: fit(), predict(), fit_predict(), score(), transform()
  - Parameter management: get_params(), set_params()
  - Incremental learning: partialFit()
  - Factory methods for all ART variants

### Graphics & Visualization
- **JavaFX 24** for GUI applications (use Launcher inner class pattern)
- **LWJGL 3.3.6** for OpenGL graphics and native integration
- **JOML** for 3D math operations (Vector3f, Matrix4f, etc.)

### Testing & Performance
- **JUnit 5** for testing
- **Mockito 4.8.1** for mocking
- **JMH** for micro-benchmarking performance-critical code
- **Surefire** configured with 512MB max heap for test execution

### Vectorization & Performance
- **VectorizedARTAlgorithm Interface** - unified API for all vectorized ART implementations
- **Java Vector API** - SIMD optimizations for pattern processing
- **JOML integration** - hardware-accelerated 3D math operations
- **Parallel processing** - concurrent category search and activation
- **Performance metrics** - comprehensive tracking for optimization analysis

### Key Dependencies
- **Guava** for collections and utilities
- **Apache Commons Lang3** for common utilities  
- **Logback + SLF4J** for logging
- **Prime Mover** custom Maven plugin (snapshot version)

## Code Conventions

### Java Style
- Use `var` for local variable type inference where type is obvious
- Never use `synchronized` - prefer concurrent collections and lock-free patterns
- Follow JavaFX Launcher pattern for Application.launch() calls
- Leverage Java 24 features: records, pattern matching, virtual threads
- Use try-with-resources for resource management

### Maven Module Structure
- Multi-module build with 3 active modules:
  - **art-core**: Core algorithm implementations (818 tests)
  - **art-performance**: Vectorized implementations (582 tests)
  - **gpu-test-framework**: GPU acceleration experiments (8 tests)
- Generated sources go in `target/generated-sources/`
- Protocol Buffers sources in `src/main/proto` and `src/test/proto`
- JOOQ generated classes in `target/generated-sources/jooq`

### Graphics Programming
- Use JOML for all 3D math operations
- LWJGL configured for macOS ARM64 natives
- JavaFX for GUI, LWJGL for low-level graphics
- Consider GPU acceleration for compute-intensive ART algorithms

## Development Workflow

### Module Creation
When creating new modules:
1. Add module to parent `<modules>` section
2. Create module directory with its own `pom.xml`
3. Use parent POM dependency management
4. Follow Maven standard directory layout

### Performance Considerations  
- Use JMH for benchmarking neural network operations
- ART algorithms are compute-intensive - consider parallel processing
- LWJGL enables GPU acceleration via OpenGL/Vulkan
- Virtual threads (Java 24) for concurrent pattern processing

### Vectorized Algorithm Implementation
When working with ART performance algorithms:
- All vectorized algorithms implement `VectorizedARTAlgorithm<T, P>` interface
- Generic type `T` represents performance metrics type
- Generic type `P` represents algorithm-specific parameters type
- Interface provides: `learn()`, `predict()`, `getCategoryCount()`, `getPerformanceStats()`
- Resource management via `AutoCloseable` for proper cleanup
- Performance tracking methods: `resetPerformanceTracking()`, `getPerformanceStats()`

### Algorithm Types (60+ Implementations)

#### Core ART Algorithms
- **ART1** - Binary pattern recognition (original model)
- **ART2/ART2A** - Continuous-valued patterns
- **ARTA/ARTE/ARTSTAR** - Enhanced ART variants
- **FuzzyART** - Fuzzy set theory with complement coding
- **BinaryFuzzyART** - Binary inputs with fuzzy operations
- **DualVigilanceART** - Dual vigilance for boundary/cluster control
- **FusionART** - Multi-channel fusion learning

#### Geometric Models
- **EllipsoidART** - Ellipsoidal category geometry
- **HypersphereART** - Spherical categories
- **QuadraticNeuronART** - Quadratic activation functions

#### Probabilistic Models
- **BayesianART** - Probabilistic category assignment
- **GaussianART** - Gaussian probability distributions

#### Topological & Advanced
- **TopoART** - Topology-preserving dual networks
- **SalienceART** - Feature importance weighting
- **iCVIFuzzyART** - Incremental cluster validity
- **SMART** - Self-organizing map ART hybrid

#### Supervised ARTMAP Family
- **ARTMAP/FuzzyARTMAP** - Supervised classification
- **SimpleARTMAP/BARTMAP** - Simplified variants
- **BinaryFuzzyARTMAP** - Binary supervised learning
- **HypersphereARTMAP** - Geometric supervised
- **DeepARTMAP** - Multi-layer hierarchical

#### Vectorized Performance Versions
All above algorithms have vectorized implementations with "Vectorized" prefix using Java Vector API for 10-100x speedup

### Testing Strategy
- **Total Tests**: 1,408 (100% pass rate)
  - art-core: 818 tests
  - art-performance: 582 tests  
  - gpu-test-framework: 8 tests
- **Test Types**:
  - Unit tests for individual ART components
  - Integration tests for full network behavior
  - Performance tests using JMH for critical paths
  - Visual tests for JavaFX components
  - Regression tests for historical bug prevention
- **BaseVectorizedARTTest** - Shared test framework (83% code reduction)

## Project-Specific Notes

### ART Neural Network Context
- **Adaptive Resonance Theory** is an unsupervised learning architecture
- Focus on real-time pattern recognition and categorization
- Key concepts: vigilance parameter, resonance, competitive learning
- Mathematical operations will be performance-critical

### Graphics Integration
The combination of JavaFX + LWJGL + JOML suggests:
- Real-time visualization of neural network states
- Interactive parameter tuning interfaces  
- 2D/3D visualization of pattern spaces
- Possibly VR/AR applications given LWJGL inclusion

### Custom Repository
Uses `repo-hell` GitHub repository for custom Maven artifacts:
- Prime Mover plugin for advanced build features
- May include other custom neural network utilities
- Check repository for additional documentation