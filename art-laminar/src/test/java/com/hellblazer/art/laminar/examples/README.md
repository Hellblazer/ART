# ART Laminar Circuit Examples

Comprehensive demonstrations of the ART laminar circuit capabilities using non-visual, real-world examples.

## Overview

These demos showcase the key features of the ART laminar module:
- **Unsupervised category learning** (Adaptive Resonance Theory)
- **Temporal sequence learning** (LIST PARSE model)
- **SIMD batch processing** (1.30x speedup with Phase 6C optimization)
- **Multi-scale temporal processing** (fast/medium/slow time scales)
- **Online learning** (stability-plasticity balance)

All examples are runnable JUnit tests that serve as both documentation and validation.

## Examples

### 1. IrisClassificationDemo.java

**Classic machine learning benchmark**: Iris flower classification

**Demonstrates**:
- Unsupervised clustering of the famous Iris dataset (150 samples, 3 species, 4 features)
- Vigilance parameter effects on category granularity
- Batch processing with SIMD optimization (1.30x speedup)
- Online learning and category stability
- Category analysis and interpretation

**Run**:
```bash
mvn test -Dtest=IrisClassificationDemo
```

**Key Demos**:
- **Demo 1**: Basic classification (forms 3-5 categories for 3 species)
- **Demo 2**: Vigilance tuning (0.5 → 0.95 vigilance range)
- **Demo 3**: Batch processing performance (Phase 6C SIMD)
- **Demo 4**: Online learning stability (new samples don't destabilize)
- **Demo 5**: Category purity analysis (species → category mapping)

**Expected Output**:
```
=== DEMO 1: Basic Iris Classification ===

Training on 150 iris samples...

Categories formed: 4
Expected: 3 (one per species) or 4-5 (split Versicolor/Virginica)

Category Distribution:
  Category 0: 50 samples (33.3%) - Setosa
  Category 1: 35 samples (23.3%) - Versicolor
  Category 2: 15 samples (10.0%) - Virginica
  Category 3: 50 samples (33.3%) - Versicolor/Virginica mix
```

---

### 2. SequenceLearningDemo.java

**Temporal chunking**: Shows how similar consecutive patterns form chunks

**Demonstrates**:
- Chunking repeated/similar patterns (coherence-based)
- Sensor reading chunking (stable temperature periods)
- Activity burst detection (network traffic patterns)
- Rhythmic pattern chunking (XXX YYY XXX)
- Working memory capacity (Miller's 7±2 with decay)

**Run**:
```bash
mvn test -Dtest=SequenceLearningDemo
```

**Key Demos**:
- **Demo 1**: Repeated pattern chunking (AAA BBB CCC → 8 chunks, coherence 0.771)
- **Demo 2**: Sensor reading chunking (stable periods → 13 chunks, coherence 0.998)
- **Demo 3**: Activity burst detection (high/low traffic → 10 chunks, coherence 0.916)
- **Demo 4**: Rhythmic patterns (XXX YYY XXX → 8 chunks with perfect internal coherence)
- **Demo 5**: Working memory capacity (3 groups → 8 active chunks, coherence 0.948)

**Expected Output**:
```
=== DEMO 1: Repeated Pattern Chunking ===

Processing sequence: A A A  B B B  C C C
(Repeated patterns should form chunks)

Chunk Formation Statistics:
  Total chunks formed: 8
  Average chunk size: 4.3 items
  Average coherence: 0.771

Expected Behavior:
- AAA → 1 chunk (3 identical patterns, coherence = 1.0)
- BBB → 1 chunk (3 identical patterns, coherence = 1.0)
- CCC → 1 chunk (3 identical patterns, coherence = 1.0)
- Total: 3 chunks with high coherence
```

---

### 3. AnomalyDetectionDemo.java

**Real-world application**: Network traffic anomaly detection

**Demonstrates**:
- Training on normal patterns, detecting anomalies
- Vigilance tuning for precision/recall balance
- Online adaptive detection (evolving normal patterns)
- Batch anomaly scanning (fast log file analysis with SIMD)
- Multi-category anomaly type classification

**Run**:
```bash
mvn test -Dtest=AnomalyDetectionDemo
```

**Key Demos**:
- **Demo 1**: Basic anomaly detection (train on normal, test on anomalous)
- **Demo 2**: Vigilance tuning (finding optimal false positive/true positive balance)
- **Demo 3**: Online adaptive detection (learning new normal patterns while detecting anomalies)
- **Demo 4**: Batch anomaly scanning (500 log entries with 1.30x SIMD speedup)
- **Demo 5**: Anomaly type classification (port scan vs DDoS vs data exfiltration)

**Expected Output**:
```
=== DEMO 1: Basic Anomaly Detection ===

Training on normal traffic patterns...
Normal traffic categories: 5

Testing anomaly detection:

Normal traffic:
  ✓ Normal (match: 0.912, category: 2)
  ✓ Normal (match: 0.887, category: 1)
  ✓ Normal (match: 0.901, category: 3)
  ...

Anomalous traffic:
  ✗ ANOMALY DETECTED (match: 0.623)
  ✗ ANOMALY DETECTED (match: 0.541)
  ✗ ANOMALY DETECTED (match: 0.707)
  ...

Detection Rate: 90.0% of normal patterns recognized
Anomaly Detection Rate: 80.0% of anomalies caught
```

---

## Running All Examples

```bash
# Run all example demos
mvn test -Dtest=*Demo

# Run specific demo
mvn test -Dtest=IrisClassificationDemo#demo1_BasicIrisClassification
mvn test -Dtest=SequenceLearningDemo#demo2_PrimacyGradient
mvn test -Dtest=AnomalyDetectionDemo#demo4_BatchAnomalyScanning
```

## Key Concepts Demonstrated

### Adaptive Resonance Theory (ART)

**Core Properties**:
- **Unsupervised Learning**: No labeled data required
- **Vigilance Parameter**: Controls category granularity (high → specific, low → general)
- **Stability-Plasticity**: Stable categories while learning new patterns
- **Fast Learning**: One-shot or few-shot category formation
- **Match-Based Learning**: Only learns if pattern matches category well enough

**ART Matching Rule**:
```
Match Score = |X ∩ E| / |X|
if Match ≥ Vigilance: Learn (update category)
else: Search (try next category or create new)
```

### Temporal Dynamics

**LIST PARSE Model** (Grossberg & Kazerounian 2016):
- **Working Memory**: Items stored with primacy gradient
- **Temporal Chunking**: Groups 7±2 items into coherent chunks
- **Chunk Types**: SMALL (1-3), MEDIUM (4-5), LARGE (6-7), SUPER (8-12)
- **Coherence**: Similarity-based chunk formation

**Multi-Scale Time Scales**:
- **FAST** (10-100ms): Sensory processing (Layer 4)
- **MEDIUM** (100-1000ms): Working memory (Layer 2/3)
- **SLOW** (1000ms+): Long-term learning (Layer 6)

### SIMD Batch Processing (Phase 6C)

**Performance**:
- **Speedup**: 1.30x over sequential baseline
- **Throughput**: 1050 patterns/sec (vs 807 baseline)
- **Semantic Equivalence**: 0.00e+00 max difference (bit-exact)

**Architecture**:
- **Mini-Batch Size**: 32 patterns (optimal SIMD threshold)
- **Stateful Processing**: Sequential pattern processing with layer-level SIMD
- **Automatic Fallback**: Uses sequential when SIMD not beneficial

## Implementation Notes

### Iris Classification

**Dataset Statistics** (normalized [0,1]):
- **Setosa**: Small petals (0.1-0.3), wide sepals (0.6-0.8) - easily separable
- **Versicolor**: Medium size (0.4-0.6) - some overlap with Virginica
- **Virginica**: Large petals (0.7-0.9), narrow sepals (0.5-0.7) - some overlap

**Recommended Vigilance**: 0.75-0.85 (balances purity and parsimony)

### Temporal Chunking

**Pattern Encoding**: Various dimensions (5-10 features)
**Chunking Mechanism**: Cosine similarity between consecutive patterns
**Chunk Parameters**:
- History size: 7-15 (Miller's 7±2 working memory capacity)
- Coherence threshold: 0.3-0.5 (similarity-based chunk formation)
- Formation threshold: 0.3 (activation strength required)
- Decay rate: 0.05-0.1 (slow chunk decay)

**Key Principle**: Chunks form when consecutive patterns are SIMILAR (high cosine similarity).
This works for:
- Repeated patterns (AAA, BBB, CCC)
- Stable periods (sensor readings with noise)
- Activity bursts (similar high/low load patterns)
- Rhythmic repetitions (XXX YYY XXX)

### Anomaly Detection

**Traffic Features** (8 dimensions):
1. Packets/sec (normalized 0-1000)
2. Bytes/sec (normalized 0-1M)
3. Source ports (normalized 0-100)
4. Destination ports (normalized 0-100)
5. TCP flags (normalized 0-255)
6. Duration (normalized 0-300 sec)
7. Protocol (0=TCP, 1=UDP, 2=ICMP)
8. Error rate (normalized 0-1)

**Anomaly Types**:
- **Port Scan**: Many destination ports, low bytes
- **DDoS**: Very high packet rate, many sources
- **Data Exfiltration**: Very high bytes, long duration

**Recommended Vigilance**: 0.85-0.90 (balance false positives/true positives)

## Theoretical Foundation

Based on research papers:
- **Grossberg, S. (2013)**: "Adaptive Resonance Theory" - Neural Networks
- **Grossberg, S. & Kazerounian, S. (2016)**: "LIST PARSE model" - Frontiers in Psychology
- **Carpenter & Grossberg (1987)**: "ART 2" - Applied Optics

## Performance Characteristics

**Test Environment**: macOS ARM64, Java 24, Vector API

**Typical Performance**:
- Single pattern: ~1.2ms processing time
- Batch (32 patterns): ~0.95ms per pattern (1.30x speedup)
- Large batch (100 patterns): ~0.92ms per pattern
- Throughput: 800-1100 patterns/sec (depending on batch size)

**Memory Usage**:
- Circuit: ~2-5 MB (depending on max categories)
- Temporal chunking: +1-2 MB (history buffers)
- Batch processing: +0.5-1 MB (transpose buffers)

## Citation

If you use these examples in your research, please cite:

```
@software{art_laminar_2025,
  title = {ART Laminar Circuit: SIMD-Optimized Implementation},
  author = {Hildebrand, Hal},
  year = {2025},
  note = {Phase 6C Complete: 1.30x speedup with 402 tests passing}
}
```

---

*Last Updated: 2025-10-01*
*Module: art-laminar v0.0.1-SNAPSHOT*
*Phase: 6C Complete (1.30x speedup)*
