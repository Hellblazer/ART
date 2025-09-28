# Temporal Dynamics

Implementation of shunting dynamics and transmitter habituation for temporal processing.

## Overview

This module provides the fundamental dynamical systems that underlie temporal ART processing. It implements the shunting on-center off-surround networks and habituative transmitter gates described in Kazerounian et al. (2014).

## Components

### ShuntingDynamicsImpl

Implements bounded competitive dynamics with self-excitation and lateral inhibition.

**Key Features:**
- Bounded activations (0 ≤ x ≤ B)
- On-center off-surround architecture
- Energy-based convergence
- Winner-take-all competition

**Usage:**
```java
var params = ShuntingParameters.competitiveDefaults(dimension);
var dynamics = new ShuntingDynamicsImpl(params, dimension);

// Set input
dynamics.setExcitatoryInput(inputPattern);

// Evolve dynamics
var state = dynamics.getState();
for (int i = 0; i < iterations; i++) {
    state = dynamics.evolve(state, dt);
}

// Check convergence
boolean converged = dynamics.hasConverged(tolerance);
```

### TransmitterDynamicsImpl

Implements habituative transmitter gates that modulate signal transmission based on activity history.

**Key Features:**
- Activity-dependent depletion
- Passive recovery
- Linear and quadratic depletion terms
- Reset mechanisms

**Usage:**
```java
var params = TransmitterParameters.paperDefaults();
var transmitters = new TransmitterDynamicsImpl(params, dimension);

// Set presynaptic signals
transmitters.setSignals(activations);

// Update transmitters
transmitters.update(dt);

// Apply gating to signals
double[] gatedOutput = transmitters.computeGatedOutput(input);
```

### MultiScaleDynamics

Integrates multiple dynamical systems with proper time scale separation.

**Time Scales:**
- Fast: Shunting dynamics (every timestep)
- Medium: Transmitter dynamics (every 10 timesteps)
- Slow: Timing dynamics (every 100 timesteps)

**Usage:**
```java
var params = MultiScaleParameters.defaults(dimension);
var dynamics = new MultiScaleDynamics(params);

// Process input with multi-scale dynamics
dynamics.update(input, dt);
double[] output = dynamics.getGatedOutput();
```

## Mathematical Foundation

### Shunting Equation
```
dx_i/dt = -A_i * x_i + (B - x_i) * f(x_i) * [I_i + Σ C_ik * g(x_k)] - x_i * Σ D_ij * h(x_j)
```

### Transmitter Equation
```
dz_i/dt = ε(1 - z_i) - z_i * y_i * (λ + μ * y_i)
```

## Test Coverage

27 tests validate:
- Convergence properties
- Boundary conditions
- Energy minimization
- Transmitter depletion and recovery
- Time scale separation
- Numerical stability

## Performance

The module provides efficient implementations with:
- Optimized array operations
- Minimal object allocation
- Cache-friendly memory access
- Support for vectorization (see temporal-performance module)

## Parameters

### Default Parameter Sets

- `competitiveDefaults()`: Strong competition for winner-take-all
- `cooperativeDefaults()`: Weak competition for distributed representation
- `paperDefaults()`: Parameters from Kazerounian et al. (2014)

## Dependencies

- temporal-core: For base interfaces and state types