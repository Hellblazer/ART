# ART Laminar Circuits Module

An implementation of laminar cortical circuits for ART neural networks, based on Grossberg's work on laminar dynamics.

## Overview

This module provides a Java implementation of laminar processing with layer-based organization and shunting dynamics. It's an experimental implementation intended for learning and experimentation with ART neural networks.

## Structure

The module is organized as follows:

```
art-laminar/
├── core/                    # Interfaces
├── impl/                    # Basic implementations
├── performance/             # Vectorized variant
├── builders/                # Configuration helpers
└── benchmarks/              # JMH performance tests
```

## Implementation Notes

- `LaminarCircuitImpl` extends the BaseART class from art-core (387 lines)
- `AbstractLayer` delegates shunting dynamics to the temporal module (175 lines)
- Achieved approximately 84% code reuse through delegation
- Removed interface naming conventions from earlier versions

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

Current test results: 8 tests passing (3 in BasicCompilationTest, 5 in VectorizedLaminarCircuitTest).

## Dependencies

- art-core: Core ART interfaces
- temporal-dynamics: Shunting dynamics implementation
- art-performance: Vectorization interfaces
- Java 24 with preview features and incubator modules

## Limitations

- This is experimental code for learning purposes
- Performance claims are theoretical and depend on hardware
- The Vector API is still in incubator status
- Limited validation against published research

## License

GNU Affero General Public License v3.0