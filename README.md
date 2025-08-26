# Adaptive Resonance Theory (ART) Neural Networks

A comprehensive Java implementation of Adaptive Resonance Theory neural network architectures for unsupervised learning and pattern recognition.

## Overview

This project provides a complete set of ART neural network implementations in Java 24, featuring:

- **Multiple ART Variants**: FuzzyART, ARTE, ART2, GaussianART, BayesianART, HypersphereART, ARTSTAR, ARTA, EllipsoidART, and DeepARTMAP
- **Supervised Learning**: ARTMAP implementations for classification tasks
- **High Performance**: Vectorized implementations using Java Vector API
- **Scikit-learn Compatibility**: Compatible interfaces for clustering metrics
- **Comprehensive Testing**: 394 tests covering all functionality

## Architecture

The project is organized into multiple Maven modules:

- **art-core**: Core ART implementations and base classes
- **art-algorithms**: Vectorized and performance-optimized algorithms  
- **art-supervised**: Supervised learning implementations (ARTMAP variants)
- **art-performance**: Benchmarking and performance testing tools

## Key Features

### Neural Network Variants
- **FuzzyART**: Fuzzy logic-based pattern recognition
- **ARTE**: Enhanced ART with extended capabilities
- **ART2**: Continuous-valued input processing
- **GaussianART**: Gaussian distribution modeling
- **BayesianART**: Uncertainty quantification and probabilistic inference
- **HypersphereART**: Hypersphere-based category representation
- **ARTSTAR**: Star-shaped category regions
- **ARTA**: Attention-based processing
- **EllipsoidART**: Ellipsoidal category boundaries
- **DeepARTMAP**: Hierarchical supervised learning

### Technical Specifications
- **Java 24** with modern language features
- **Maven 3.9.1+** build system
- **Protocol Buffers + gRPC** for serialization
- **JavaFX 24** for visualization
- **LWJGL 3.3.6** for OpenGL graphics
- **JUnit 5** for comprehensive testing

## Quick Start

### Prerequisites
- Java 24+
- Maven 3.9.1+
- macOS ARM64 (LWJGL natives configured for Apple Silicon)

### Build
```bash
mvn clean compile
```

### Test
```bash
mvn test
```

### Run Specific Test
```bash
mvn test -Dtest=ClassName#methodName
```

## Usage Example

```java
// Create and configure a FuzzyART network
var params = new FuzzyParameters.FuzzyParametersBuilder()
    .vigilance(0.8)
    .learningRate(0.1)
    .build();

var art = new FuzzyART(params);

// Train with patterns
Pattern[] trainingData = {
    new DenseVector(new double[]{0.1, 0.2, 0.3}),
    new DenseVector(new double[]{0.8, 0.9, 0.7})
};

art.fit(trainingData);

// Classify new patterns
var result = art.predict(new DenseVector(new double[]{0.15, 0.25, 0.35}));
System.out.println("Category: " + result.categoryIndex());
```

## Performance

The implementation includes several performance optimizations:

- **Vector API**: Utilizing Java's Vector API for SIMD operations
- **Parallel Processing**: Multi-threaded category competition
- **Memory Efficiency**: Optimized weight vector representations
- **JMH Benchmarking**: Micro-benchmarks for critical paths

## Testing

The project includes comprehensive test coverage:
- **394 total tests** across all modules
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Performance tests** using JMH
- **Compatibility tests** for scikit-learn interfaces

## Documentation

- `CLAUDE.md`: Development guidelines and project configuration
- Source code documentation with JavaDoc comments
- Comprehensive test examples demonstrating usage patterns

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the license headers in source files for details.

## Author

Hal Hildebrand - [GitHub](https://github.com/Hellblazer)

## Status

 **Production Ready** - All implementations are complete with full test coverage and no remaining stubs or unfinished code.