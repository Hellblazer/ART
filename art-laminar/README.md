# ART Laminar Circuits Module

## Overview

The `art-laminar` module implements Grossberg's laminar cortical circuit theory within the ART framework. This module provides a refactored, high-performance implementation that achieves ~84% code reuse through delegation to existing ART components.

## Key Features

- **Extends BaseART**: Reduces implementation from 423 lines to ~200 lines
- **Delegates to ShuntingDynamicsImpl**: Leverages existing temporal dynamics
- **Modular Architecture**: Clean separation of layers, pathways, and controllers
- **Event-Driven**: Comprehensive event system for monitoring circuit behavior
- **High Performance**: Prepared for vectorized implementations

## Architecture

### Core Components

- **LaminarCircuit**: Main circuit implementation extending BaseART
- **AbstractLayer**: Base layer using ShuntingDynamicsImpl delegation
- **AbstractPathway**: Base pathway for signal propagation
- **DefaultResonanceController**: ART match function implementation

### Layer Types

- **F0 (Input Layer)**: Initial pattern processing with optional complement coding
- **F1 (Feature Layer)**: Feature extraction and bottom-up/top-down resonance
- **F2 (Category Layer)**: Category representation and selection

### Pathway Types

- **Bottom-Up**: Feedforward signal propagation
- **Top-Down**: Feedback expectation signals
- **Lateral**: Horizontal connections within layers

## Mathematical Foundation

### Shunting Dynamics (Grossberg, 1973)
```
dx/dt = -Ax + (B-x)E - (x+C)I
```
Where:
- A: Decay rate
- B: Upper bound (ceiling)
- C: Lower bound (floor)
- E: Excitatory input
- I: Inhibitory input

### ART Match Function
```
Match = |I ∧ F| / |I|
```
Where:
- I: Input pattern
- F: Feedback expectation
- ∧: Fuzzy AND (min operation)

## Usage Example

```java
// Build a laminar circuit
var circuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
    .withParameters(DefaultLaminarParameters.builder()
        .withLearningParameters(new DefaultLearningParameters(0.5, 0.8, false, 0.01))
        .build())
    .withInputLayer(4, false)
    .withFeatureLayer(4)
    .withCategoryLayer(10)
    .withStandardConnections()
    .withVigilance(0.8)
    .build();

// Train the circuit
var pattern = new DenseVector(new double[]{0.9, 0.1, 0.1, 0.2});
var result = circuit.learn(pattern, parameters);

// Process a cycle
var activation = circuit.processCycle(pattern, parameters);
```

## Performance Characteristics

- **Code Reuse**: ~84% through BaseART extension and delegation
- **Memory Efficiency**: Minimal object allocation
- **Computation**: O(n²) for layer processing, O(n) for pathways
- **Scalability**: Prepared for SIMD vectorization

## Dependencies

- `art-core`: Core ART algorithms and abstractions
- `temporal-dynamics`: Shunting equation implementations
- `art-performance`: Performance utilities (optional)
- `temporal-performance`: Vectorized dynamics (optional)

## Building

```bash
mvn clean compile -pl art-laminar -am
```

## Testing

```bash
mvn test -pl art-laminar
```

## Future Enhancements

- [ ] VectorizedLaminarCircuit with SIMD operations
- [ ] ARTToLaminarAdapter for wrapping existing algorithms
- [ ] GPU acceleration support
- [ ] Advanced resonance controllers
- [ ] Multi-scale temporal dynamics