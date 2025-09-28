# Temporal Integration

Complete TemporalART implementation integrating all temporal processing components.

## Overview

This module provides the complete TemporalART algorithm that integrates working memory, masking field, and transmitter dynamics into a unified temporal processing system. It implements the full model from Kazerounian et al. (2014) for sequence learning and temporal pattern recognition.

## Architecture

TemporalART combines three subsystems:

```
Input Sequence
      ↓
Working Memory (STORE 2)
      ↓
Masking Field (Multi-scale)
      ↓
Category Formation
      ↑
Transmitter Gating
```

## Components

### TemporalART

Main temporal ART implementation coordinating all subsystems.

**Usage:**
```java
// Create temporal ART network
var params = TemporalARTParameters.speechDefaults();
var temporalART = new TemporalART(params);

// Process sequence
List<double[]> sequence = createSequence();
temporalART.processSequence(sequence);

// Get results
var categories = temporalART.getCategories();
int predictedCategory = temporalART.predictSequence(testSequence);

// Get statistics
var stats = temporalART.getStatistics();
System.out.println("Chunks formed: " + stats.chunkCount());
System.out.println("Average chunk size: " + stats.averageChunkSize());
```

### TemporalARTParameters

Comprehensive configuration for all subsystems:

**Parameter Sets:**
- `speechDefaults()`: Optimized for speech processing
- `listLearningDefaults()`: Optimized for list memorization
- `musicDefaults()`: Optimized for musical sequences

**Key Parameters:**
```java
var params = TemporalARTParameters.builder()
    // Working Memory
    .workingMemoryCapacity(7)
    .primacyGradientStrength(0.3)
    .recencyBoostFactor(0.5)

    // Masking Field
    .numScales(3)
    .chunkSizePreference(3)
    .asymmetryFactor(2.0)

    // Transmitters
    .transmitterRecoveryRate(0.01)
    .depletionThreshold(0.3)

    // Learning
    .learningRate(0.1)
    .vigilanceParameter(0.85)
    .build();
```

### TemporalStatistics

Provides metrics on temporal processing:
- Number of chunks formed
- Average chunk size
- Processing time per item
- Memory utilization
- Category distribution

## Processing Pipeline

### Sequence Processing

1. **Input Buffering**: Sequences stored in working memory with primacy gradient
2. **Pattern Formation**: Working memory creates temporal pattern
3. **Chunk Competition**: Masking field identifies best chunk match
4. **Category Learning**: Winning chunk updates or creates category
5. **Transmitter Modulation**: Activity gates future processing
6. **Reset Detection**: Transmitter depletion triggers sequence reset

### Learning Modes

**Supervised Learning:**
```java
temporalART.learnSequence(sequence, targetCategory);
```

**Unsupervised Learning:**
```java
temporalART.processSequence(sequence);
```

**Incremental Learning:**
```java
for (var item : sequence) {
    temporalART.processItem(item, timestamp);
}
```

## Cognitive Phenomena

Successfully reproduces from Kazerounian et al. (2014):

### Miller's 7±2 Rule
Capacity limit emerges from working memory constraints and competitive dynamics.

### Serial Position Effects
U-shaped recall curve with primacy and recency advantages.

### Chunking Strategies
Automatic discovery of optimal chunk sizes for different materials.

### Interference Effects
Proactive and retroactive interference in sequential learning.

## Applications

### Speech Processing
- Word segmentation from continuous speech
- Syllable detection
- Phoneme clustering

### Music Processing
- Phrase detection
- Rhythm pattern learning
- Melodic motif identification

### Sequence Learning
- Motor sequence learning
- Habit formation modeling
- Procedural memory simulation

### Time Series
- Temporal pattern mining
- Anomaly detection in sequences
- Predictive sequence modeling

## Test Coverage

18 tests validate:
- Complete sequence processing
- Category formation
- Chunk learning
- Reset mechanisms
- Parameter sensitivity
- Edge cases

## Performance

- Processes 100-item sequences in < 100ms
- Memory efficient with bounded working memory
- Supports streaming input for real-time processing

## Integration with Base ART

TemporalART can be combined with other ART algorithms:
```java
// Temporal preprocessing for FuzzyART
var temporal = new TemporalART(temporalParams);
var fuzzy = new FuzzyART(fuzzyParams);

var chunks = temporal.processSequence(sequence);
for (var chunk : chunks) {
    fuzzy.learn(chunk.getPattern());
}
```

## Dependencies

- temporal-core: Base interfaces
- temporal-memory: Working memory
- temporal-masking: Masking field
- temporal-dynamics: Transmitter dynamics