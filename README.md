# ART: High-Performance Adaptive Resonance Theory for Java 24

[![Java 24](https://img.shields.io/badge/Java-24-orange.svg)](https://openjdk.java.net/projects/jdk/24/)
[![Maven](https://img.shields.io/badge/Maven-3.9.1+-blue.svg)](https://maven.apache.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Build Status](https://github.com/Hellblazer/ART/workflows/Java%20CI/badge.svg)](https://github.com/Hellblazer/ART/actions)

An implementation of Adaptive Resonance Theory (ART) neural networks for Java 24, utilizing the Vector API, virtual threads, and modern concurrency primitives.

## Features

- **High Performance**: SIMD vectorization via Java 24 Vector API (4-8x speedup)
- **Comprehensive Coverage**: 15+ ART algorithm variants implemented
- **Modern Java**: Full Java 24 feature utilization
- **Production Ready**: Extensive testing (210+ tests), benchmarking, and documentation
- **Data Preprocessing**: Complete data preprocessing pipeline with normalization and missing value handling
- **Scikit-learn Compatible**: Familiar API for Python users transitioning to Java

## Quick Start

### Prerequisites
- Java 24+ ([Download](https://jdk.java.net/24/))
- Maven 3.9.1+

### Installation
```bash
git clone https://github.com/Hellblazer/ART.git
cd ART
mvn clean install
```

### Basic Usage
```java
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.DenseVector;

// Create and train a Fuzzy ART network
var parameters = FuzzyParameters.of(0.85, 0.1, 0.001);  // vigilance, learning rate, bias
var network = new FuzzyART(parameters);

var pattern1 = new DenseVector(new double[]{0.8, 0.2, 0.9, 0.1});
var pattern2 = new DenseVector(new double[]{0.1, 0.9, 0.2, 0.8});

int category1 = network.stepFit(pattern1);
int category2 = network.stepFit(pattern2);
```

### Data Preprocessing
```java
import com.hellblazer.art.core.preprocessing.DataPreprocessor;
import com.hellblazer.art.core.preprocessing.DataPreprocessor.MissingValueStrategy;

// Normalize and complement code your data
var preprocessor = new DataPreprocessor();
double[][] rawData = {{1.5, 2.3}, {3.1, 0.8}, {Double.NaN, 1.2}};

// Handle missing values, normalize, and apply complement coding
var pipeline = DataPreprocessor.createPipeline()
    .handleMissingValues(MissingValueStrategy.MEAN)
    .normalize()
    .complementCode()
    .build();

double[][] processed = pipeline.process(rawData);
```

### Scikit-learn Compatible API
```java
import com.hellblazer.art.core.SklearnWrapper;

// Use familiar sklearn-style interface
var model = SklearnWrapper.fuzzyART(0.85, 0.1, 0.001);

// Fit and predict
double[][] X_train = {{0.1, 0.2}, {0.8, 0.9}, {0.3, 0.4}};
double[][] X_test = {{0.15, 0.25}, {0.75, 0.85}};

model.fit(X_train);
int[] predictions = model.predict(X_test);

// Get clustering score
double score = model.score(X_test, predictions);
```

### High-Performance Version
```java
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;

// Use SIMD-optimized implementation
var parameters = VectorizedParameters.builder()
    .vigilance(0.85)
    .learningRate(0.1)
    .enableSIMD(true)
    .parallelThreads(8)
    .build();

var vectorizedNetwork = new VectorizedFuzzyART(parameters);
```

## Documentation

- **[Algorithm Overview](docs/OVERVIEW.md)** - Detailed algorithm descriptions and usage
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and architecture
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Developer Guide](CLAUDE.md)** - Build configuration and development

## Algorithms

### Unsupervised Learning
| Algorithm | Use Case | Key Feature |
|-----------|----------|-------------|
| FuzzyART | General pattern recognition | Fuzzy set operations |
| DualVigilanceART | Noise-robust clustering | Dual threshold system |
| TopoART | Topology learning | Edge formation, clustering |
| BayesianART | Uncertainty quantification | Confidence estimates |
| GaussianART | Statistical clustering | Gaussian distributions |
| HypersphereART | Geometric clustering | Rotation invariant |
| ART-2 | Preprocessing integration | Noise filtering |

### Supervised Learning
| Algorithm | Use Case | Key Feature |
|-----------|----------|-------------|
| ARTMAP | Classification | Match tracking |
| DeepARTMAP | Hierarchical learning | Multi-layer processing |

## Performance

All algorithms have vectorized implementations with:
- 6-8x performance improvement via SIMD
- Parallel processing support
- Memory-optimized operations

Run benchmarks:
```bash
mvn test -pl art-performance -Dtest=*Benchmark
```

## Project Structure

```
art-core/           # Core algorithms and data structures
art-performance/    # High-performance vectorized implementations  
gpu-test-framework/ # GPU testing infrastructure
docs/              # Documentation
```

## Testing

```bash
# Run all tests
mvn test

# Run specific algorithm tests
mvn test -Dtest=FuzzyARTTest

# Generate coverage report
mvn jacoco:report
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file.

For commercial licensing options, contact [hal.hildebrand@me.com](mailto:hal.hildebrand@me.com).

## Acknowledgments

- Stephen Grossberg and Gail Carpenter for the original ART theory
- [AdaptiveResonanceLib](https://github.com/NiklasMelton/AdaptiveResonanceLib) by Niklas Melton
- OpenJDK Project for Java 24 and the Vector API

---

Built using Java 24. Optimized for Performance. Scientifically Accurate.

*For questions or issues, visit our [GitHub Issues](https://github.com/Hellblazer/ART/issues) page.*
