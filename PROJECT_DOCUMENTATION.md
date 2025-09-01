# ART Neural Network Implementation - Project Documentation

## Project Overview
Adaptive Resonance Theory (ART) implementation in Java 24 with comprehensive performance optimizations and testing.

## Key Implementation Files

### Core Algorithms
- **art-core/** - Core ART algorithm implementations
- **art-performance/** - Vectorized high-performance implementations using Java Vector API

### Testing
- **ComprehensivePerformanceTest.java** - Fixed performance tests with actual measurements
- **ParameterizedPerformanceTest.java** - Scalable testing framework (QUICK/STANDARD/COMPREHENSIVE/FULL)
- **RealWorldPerformanceTest.java** - Large-scale real-world scenario testing

## Performance Results Summary

### Peak Performance Achieved
- **VectorizedHypersphereART**: Up to 5,000,000 patterns/sec (low-dimensional data)
- **VectorizedFuzzyART**: Up to 383,595 patterns/sec (consistent across configurations)

### Test Scales
| Scale | Duration | Use Case |
|-------|----------|----------|
| QUICK | ~1 second | CI/CD pipeline |
| STANDARD | ~5 seconds | Regular testing |
| COMPREHENSIVE | ~30 seconds | Full validation |
| FULL | ~2-5 minutes | Benchmark suite |

### Running Tests
```bash
# Quick validation
mvn test -Dtest=ParameterizedPerformanceTest -Dperformance.test.scale=QUICK

# Comprehensive testing
mvn test -Dtest=ParameterizedPerformanceTest -Dperformance.test.scale=COMPREHENSIVE
```

## Completed Work

### Performance Testing ✅
- Fixed all compilation errors in performance tests
- Created parameterizable test framework
- Obtained actual performance measurements (not projections)
- Validated across diverse real-world scenarios

### Algorithm Implementation ✅
- VectorizedFuzzyART with SIMD optimization
- VectorizedHypersphereART with parallel processing
- VectorizedTopoART for topological preservation
- VectorizedARTMAP variants for supervised learning

### API Integration ✅
- Scikit-learn compatible wrapper (SklearnWrapper)
- Data preprocessing pipeline (DataPreprocessor)
- Complement coding and normalization support

## Key Technologies
- **Java 24** with Vector API (jdk.incubator.vector)
- **Maven 3.9.1+** multi-module build
- **JUnit 5** for testing
- **JMH** for microbenchmarking
- **SIMD** vectorization for performance

## Build & Run
```bash
# Build project
mvn clean compile

# Run all tests
mvn test

# Run specific test
mvn test -Dtest=TestClassName

# Run with specific JVM flags for vector API
java --add-modules jdk.incubator.vector
```

## Documentation Files
- **README.md** - Project overview and quick start
- **CLAUDE.md** - AI assistant configuration
- **PROJECT_DOCUMENTATION.md** - This consolidated documentation (you are here)