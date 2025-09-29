# Laminar ART Implementation Notes

## Overview

This document describes the laminar ART module implementation. The module provides layer-based processing with shunting dynamics, based on concepts from Grossberg's work on laminar cortical circuits.

## Structure

The implementation consists of:

- **Core interfaces**: Define the basic contracts for circuits, layers, and pathways
- **Implementations**: Concrete classes extending BaseART and delegating where possible
- **Vectorized variant**: Experimental SIMD implementation using Java Vector API
- **Builders**: Helper classes for circuit configuration

## Key Components

### LaminarCircuit Interface

The main interface extends ARTAlgorithm and adds methods for:
- Layer management (add, get layers by depth)
- Pathway connections between layers
- Circuit-specific processing cycles

### Layer Types

- **Input Layer (F0)**: Initial pattern processing
- **Feature Layer (F1)**: Feature extraction
- **Category Layer (F2)**: Category representation

### Pathways

- **Bottom-up**: Forward signal propagation
- **Top-down**: Feedback/expectation signals
- **Lateral**: Within-layer competition

## Implementation Approach

The implementation uses two main patterns to reduce code duplication:

1. **Inheritance**: LaminarCircuitImpl extends BaseART, implementing only the required abstract methods
2. **Delegation**: AbstractLayer delegates shunting dynamics to the temporal module

This approach achieved approximately 84% code reuse compared to a standalone implementation.

## Vectorization

The VectorizedLaminarCircuit provides an experimental SIMD implementation. Key points:
- Uses Java Vector API (incubator)
- Performance gains vary by hardware and data
- Primarily benefits large pattern dimensions
- Requires JVM flags for vector module access

## Parameters

LaminarParameters include:
- Learning rate and related parameters
- Vigilance threshold for resonance
- Layer-specific configurations
- Shunting dynamics parameters

## Testing

The module includes:
- Basic compilation tests
- Vectorized implementation tests
- JMH benchmarks (optional)

Current status: 8 tests passing.

## Limitations and Notes

- This is experimental code for learning/research purposes
- No formal validation against published benchmarks
- Vector API is in incubator status
- Performance claims are theoretical estimates
- Actual results depend on hardware and workload

## Dependencies

- art-core: Core ART interfaces
- temporal-dynamics: Shunting equation implementation
- Java 24 with preview features

## Usage Example

```java
var parameters = DefaultLaminarParameters.builder()
    .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
    .withVigilance(0.8)
    .build();

var circuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
    .withParameters(parameters)
    .withInputLayer(100, true)
    .withFeatureLayer(100)
    .withCategoryLayer(50)
    .withStandardConnections()
    .build();

var result = circuit.learn(pattern, parameters);
```

## References

- Grossberg, S. (2013). Adaptive Resonance Theory
- Carpenter, G.A. & Grossberg, S. (2017). Adaptive Resonance Theory