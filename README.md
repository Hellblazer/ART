# ART: High-Performance Adaptive Resonance Theory Implementation

[![Java 24](https://img.shields.io/badge/Java-24-orange.svg)](https://openjdk.java.net/projects/jdk/24/)
[![Maven](https://img.shields.io/badge/Maven-3.9.1+-blue.svg)](https://maven.apache.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Build Status](https://github.com/Hellblazer/ART/workflows/Java%20CI/badge.svg)](https://github.com/Hellblazer/ART/actions)

A comprehensive implementation of Adaptive Resonance Theory (ART) neural networks, utilizing modern Java features including the Vector API, virtual threads, and advanced concurrency primitives.

## Features

- **High Performance**: SIMD vectorization via Vector API (6-8x average speedup)
- **Comprehensive Coverage**: 60+ ART algorithm variants with vectorized implementations
- **Extensive Testing**: 1,408 total tests with 100% pass rate across all modules
- **Temporal Processing**: Complete temporal ART implementation with 144 tests (14x working memory speedup)
- **Data Preprocessing**: Complete data preprocessing pipeline with normalization and missing value handling
- **Scikit-learn Compatible**: Familiar API for Python users transitioning to Java
- **Production Ready**: Benchmarking, comprehensive documentation, and performance optimization

## Quick Start

### Prerequisites
- Java 24+

### Installation
```bash
git clone https://github.com/Hellblazer/ART.git
cd ART
./mvnw clean install
```

> **Note**: The project includes the Maven wrapper (`mvnw`), so you don't need to install Maven separately.

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
- **[Performance Report](PERFORMANCE_REPORT.md)** - Detailed performance benchmarks

## Algorithms

### Unsupervised Learning
| Algorithm | Use Case | Key Feature | Vectorized |
|-----------|----------|-------------|------------|
| ART1 | Binary pattern recognition | Binary resonance | Yes |
| ART2 | Continuous pattern recognition | Noise filtering | Yes |
| FuzzyART | General pattern recognition | Fuzzy set operations | Yes |
| DualVigilanceART | Noise-robust clustering | Dual threshold system | Yes |
| TopoART | Topology learning | Edge formation, clustering | Yes |
| BayesianART | Uncertainty quantification | Confidence estimates | Yes |
| GaussianART | Statistical clustering | Gaussian distributions | Yes |
| HypersphereART | Geometric clustering | Rotation invariant | Yes |
| EllipsoidART | Ellipsoidal clustering | Orientation adaptive | Yes |
| QuadraticNeuronART | Non-linear boundaries | Quadratic activation | Yes |
| SalienceART | Feature-weighted learning | Dynamic salience weighting | Yes |
| BinaryFuzzyART | Binary fuzzy patterns | Fuzzy binary operations | Yes |
| FusionART | Multi-channel fusion | Channel integration | Yes |
| ARTA | Time-series patterns | Temporal processing | Yes |
| ARTE | Extended ART | Enhanced stability | Yes |
| ARTSTAR | Star topology | Hub-based clustering | Yes |
| iCVIFuzzyART | Incremental validity | CVI optimization | Yes |

### Supervised Learning
| Algorithm | Use Case | Key Feature | Vectorized |
|-----------|----------|-------------|------------|
| ARTMAP | Classification | Match tracking | Yes |
| SimpleARTMAP | Fast classification | Simplified map field | Yes |
| FuzzyARTMAP | Fuzzy classification | Fuzzy match tracking | Yes |
| BinaryFuzzyARTMAP | Binary fuzzy classification | Binary fuzzy tracking | Yes |
| GaussianARTMAP | Statistical classification | Gaussian map field | Yes |
| HypersphereARTMAP | Geometric classification | Spherical boundaries | Yes |
| DeepARTMAP | Hierarchical learning | Multi-layer processing | Yes |
| SalienceARTMAP | Weighted classification | Cross-module salience | Yes |
| BARTMAP | Bayesian classification | Probabilistic map field | Yes |
| SMART | Sequential mapping | Memory-based learning | Yes |

### Reinforcement Learning
| Algorithm | Use Case | Key Feature | Vectorized |
|-----------|----------|-------------|------------|
| FALCON | Q-learning integration | State-action-reward mapping | Yes |
| TD-FALCON | Temporal difference learning | SARSA with eligibility traces | Yes |

## Performance

All algorithms have vectorized implementations with:
- 6-8x average performance improvement via SIMD
- Temporal modules: 14x working memory speedup, 1.53x shunting dynamics speedup
- Parallel processing support
- Memory-optimized operations

Run benchmarks:
```bash
./mvnw test -pl art-performance -Dtest=*Benchmark
```

## Project Structure

```
art-core/           # Core algorithms and data structures (818 tests)
art-performance/    # High-performance vectorized implementations (582 tests)
art-temporal/       # Temporal processing modules (144 tests)
art-markov/         # Markov chain integration
gpu-test-framework/ # GPU testing infrastructure (8 tests)
docs/              # Documentation
```

## Testing

```bash
# Run all tests
./mvnw test

# Run specific algorithm tests
./mvnw test -Dtest=FuzzyARTTest

# Generate coverage report
./mvnw jacoco:report
```

## Contributing

Contributions are welcome. Please submit pull requests with clear descriptions of changes.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file.

For commercial licensing options, contact [hal.hildebrand@me.com](mailto:hal.hildebrand@me.com).

## Acknowledgments

- Stephen Grossberg and Gail Carpenter for the original ART theory
- [AdaptiveResonanceLib](https://github.com/NiklasMelton/AdaptiveResonanceLib) by Niklas Melton
- OpenJDK Project for Java 24 and the Vector API

---

Optimized for Performance. Scientifically Accurate.

*For questions or issues, visit our [GitHub Issues](https://github.com/Hellblazer/ART/issues) page.*
