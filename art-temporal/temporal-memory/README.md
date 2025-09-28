# Temporal Memory

STORE 2 working memory implementation for temporal sequence processing.

## Overview

This module implements the STORE 2 (Sustained Temporal Order REcurrent) model, providing a working memory system that maintains temporal sequences with primacy and recency gradients. Based on the working memory architecture described in Kazerounian et al. (2014).

## Key Features

### Primacy Gradient
Items stored earlier in a sequence receive stronger initial activation, implementing the primacy effect observed in human memory.

### Recency Boost
Recently stored items maintain higher activation through temporal proximity, implementing the recency effect.

### Capacity Constraints
Implements cognitive capacity limits (7±2 items) through competitive dynamics and resource limitations.

### Temporal Pattern Formation
Combines stored items into temporal patterns suitable for downstream processing by masking fields.

## Components

### WorkingMemory

Main working memory implementation with position-dependent storage and retrieval.

**Usage:**
```java
var params = WorkingMemoryParameters.paperDefaults();
var memory = new WorkingMemory(params);

// Store sequence
memory.storeItem(pattern1, 0.0);
memory.storeItem(pattern2, 0.1);
memory.storeItem(pattern3, 0.2);

// Retrieve combined pattern
var temporalPattern = memory.getTemporalPattern();
double[] combined = temporalPattern.getCombinedPattern();

// Check capacity
boolean atCapacity = memory.isAtCapacity();
```

### WorkingMemoryState

Immutable representation of working memory state.

**Contents:**
- Item activations matrix
- Storage timestamps
- Primacy weights
- Total activation

### WorkingMemoryParameters

Configuration parameters for working memory dynamics.

**Key Parameters:**
- `maxItems`: Capacity limit (default: 7)
- `primacyGradientStrength`: γ parameter (default: 0.3)
- `recencyBoost`: δ parameter (default: 0.5)
- `decayRate`: Item decay rate (default: 0.1)

## Serial Position Effects

The implementation reproduces the U-shaped serial position curve:

1. **Primacy Effect**: First items have stronger representation
2. **Recency Effect**: Last items have stronger representation
3. **Middle Items**: Weakest representation

This matches human memory performance in list learning tasks.

## Temporal Pattern Structure

The working memory creates temporal patterns with:
- Combined activation pattern (weighted sum of items)
- Temporal weights (primacy and recency modulated)
- Position information

## Test Coverage

9 tests validate:
- Item storage and retrieval
- Primacy gradient formation
- Recency effects
- Capacity limits
- Serial position curve
- Temporal pattern generation

## Cognitive Phenomena

Successfully reproduces:
- Miller's 7±2 capacity limit
- Serial position effects
- Interference between sequences
- Temporal grouping by proximity

## Integration

Works with:
- `temporal-masking`: Provides input patterns for chunk formation
- `temporal-integration`: Core component of TemporalART
- `temporal-performance`: Vectorized version available

## Dependencies

- temporal-core: For state interfaces and parameters