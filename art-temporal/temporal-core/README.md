# Temporal Core

Core interfaces and types for temporal ART processing.

## Overview

This module defines the fundamental abstractions used throughout the temporal implementation. It provides the contract for evolvable dynamics, state representations, and parameter specifications.

## Key Interfaces

### Evolvable&lt;S extends State&gt;
Base interface for all dynamical systems that evolve over time.

```java
public interface Evolvable<S extends State> {
    S evolve(S currentState, double dt);
    S getState();
    void setState(S state);
    void reset();
}
```

### State
Marker interface for immutable state representations.

```java
public interface State {
    // Marker interface for type safety
}
```

### ActivationState
Represents the activation state of a neural network layer.

```java
public record ActivationState(double[] activations) implements State {
    public double getTotalActivation();
    public double getMaxActivation();
    public int getWinnerIndex();
}
```

### TransmitterState
Represents the state of habituative transmitter gates.

```java
public record TransmitterState(
    double[] levels,
    double[] presynapticSignals,
    double[] depletionHistory
) implements State {
    public double getAverageLevel();
    public boolean isDepleted(double threshold);
}
```

## Parameter Types

### ShuntingParameters
Configuration for shunting dynamics networks.

- Decay rates (uniform or position-dependent)
- Ceiling and floor bounds
- Self-excitation strength
- Lateral inhibition patterns

### TransmitterParameters
Configuration for transmitter habituation.

- Recovery rate (ε)
- Linear depletion rate (λ)
- Quadratic depletion rate (μ)
- Baseline level
- Time constant

### WorkingMemoryParameters
Configuration for STORE 2 working memory.

- Primacy gradient strength (γ)
- Recency boost factor (δ)
- Capacity limit
- Decay parameters

## Design Principles

1. **Immutability**: All state objects are immutable records
2. **Type Safety**: Strong typing with generics
3. **Separation of Concerns**: Dynamics separate from state
4. **Thread Safety**: Immutable design enables safe concurrent access
5. **Testability**: Clean interfaces facilitate testing

## Usage

The core module is a dependency for all other temporal modules. It should not be used directly but through the higher-level implementations.

## Dependencies

None - this is the foundational module with no external dependencies.