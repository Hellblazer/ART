# Temporal Masking

Multi-scale masking field implementation for temporal chunking and sequence segmentation.

## Overview

This module implements the masking field architecture from Kazerounian et al. (2014), providing multi-scale competitive dynamics for temporal chunk formation. The masking field processes temporal patterns from working memory to form meaningful chunks at different scales.

## Architecture

### Multi-Scale Structure

The masking field operates at three distinct scales:

| Scale | Preferred Length | Purpose | Examples |
|-------|-----------------|---------|----------|
| Item | 1-2 elements | Individual items | Single phonemes, letters |
| Chunk | 3-4 elements | Small groups | Syllables, area codes |
| List | 5-7 elements | Full sequences | Words, phone numbers |

### Asymmetric Lateral Inhibition

Larger scales inhibit smaller scales more strongly than vice versa, implementing a hierarchical bias toward larger meaningful units when possible.

## Components

### MaskingField

Main masking field implementation with multi-scale competitive dynamics.

**Usage:**
```java
var params = MaskingFieldParameters.listLearningDefaults();
var workingMemory = new WorkingMemory(wmParams);
var maskingField = new MaskingField(params, workingMemory);

// Process temporal pattern
var pattern = createTemporalPattern(items);
maskingField.processTemporalPattern(pattern);

// Get formed chunks
var chunks = maskingField.getListChunks();
for (ListChunk chunk : chunks) {
    System.out.println("Chunk size: " + chunk.getSize());
    System.out.println("Activation: " + chunk.getActivation());
}
```

### ListChunk

Represents a learned temporal chunk with:
- Weight vector (learned pattern)
- Activation level
- Size (number of items)
- Learning history

### MaskingFieldState

Immutable state representation containing:
- Item node activations
- List node activations
- Chunk formations
- Competition results

### MaskingFieldParameters

Configuration for masking field dynamics:
- `numScales`: Number of processing scales (default: 3)
- `asymmetryFactor`: Inhibition asymmetry strength (default: 2.0)
- `integrationTimeStep`: Dynamics time step (default: 0.01)
- `convergenceThreshold`: Competition convergence (default: 0.001)

## Chunking Mechanisms

### Competitive Selection

At each time step:
1. Each scale computes match to current working memory pattern
2. Scales compete through asymmetric lateral inhibition
3. Winner selected based on activation and scale bias
4. Winning scale learns/reinforces chunk

### Adaptive Resonance

New chunks are created when:
- No existing chunk matches above vigilance threshold
- Working memory pattern is stable
- Competition has converged

### Chunk Learning

Uses competitive learning rules:
- Instar learning for bottom-up weights
- Outstar learning for top-down expectations
- Weight normalization for stability

## Phenomena Reproduced

### Phone Number Chunking
Automatically segments 10-digit sequences into 3-3-4 pattern matching human performance.

### Word Segmentation
Identifies word boundaries in continuous phoneme streams through pause detection and familiar chunk recognition.

### Temporal Grouping
Groups items by temporal proximity, with pauses triggering chunk boundaries.

## Test Coverage

38 tests validate:
- Multi-scale competition
- Asymmetric inhibition
- Chunk formation
- Adaptive resonance
- Capacity limits
- Temporal segmentation

## Integration Points

- **Input**: Receives temporal patterns from WorkingMemory
- **Output**: Provides chunk categories to TemporalART
- **Modulation**: Gated by TransmitterDynamics for reset

## Performance Optimizations

- Sparse matrix operations for weight updates
- Lazy evaluation of inactive scales
- Cached competition results
- Vectorized distance computations (in temporal-performance)

## Applications

- Speech recognition and segmentation
- Music phrase detection
- Sequential pattern mining
- Behavioral sequence analysis
- Natural language chunking

## Dependencies

- temporal-core: For interfaces and state types
- temporal-memory: For working memory integration