# Technical Architecture

## Core Components Overview

The ART Cognitive Architecture implements six key innovations for unlimited sequence processing while respecting the 7±2 working memory constraint.

## 1. Recursive Hierarchical Memory

**Location**: `src/main/java/com/art/textgen/memory/RecursiveHierarchicalMemory.java`

### Design Principles
- **7±2 Chunking**: Each level maintains exactly 7 items (Miller's constraint)
- **Exponential Capacity**: Level L can represent 7^L base tokens
- **Lossy Compression**: Pattern-based compression with decompression capability
- **Dynamic Activation**: Relevance-based retrieval and expansion

### Implementation Details
```java
public class RecursiveHierarchicalMemory {
    private static final int ITEMS_PER_LEVEL = 7;  // Miller's magic number
    private static final int DEFAULT_LEVELS = 5;
    
    // Effective capacity calculation
    // Level 0: 7 tokens
    // Level 1: 49 compressed tokens (7^1 * 7)
    // Level 2: 343 compressed phrases (7^2 * 7)  
    // Level 3: 2,401 compressed paragraphs (7^3 * 7)
    // Level 4: 16,807 compressed sections (7^4 * 7)
    // Total: ~20,000 token effective capacity
}
```

### Key Methods
- `addToken(Object token)`: Bottom-up compression cascade
- `compressItems(List<Chunk> items)`: Pattern extraction and compression
- `getActiveContext(int queryDepth)`: Relevance-based retrieval with decompression
- `getEffectiveCapacity()`: Calculate total representational capacity

## 2. Multi-Timescale Memory Bank

**Location**: `src/main/java/com/art/textgen/memory/MultiTimescaleMemoryBank.java`

### Temporal Hierarchy
Each memory operates at different time constants:
```java
memories = {
    'phoneme': WorkingMemory(capacity=7, tau=0.1),    // ~100ms
    'word':    WorkingMemory(capacity=7, tau=1.0),    // ~1s
    'phrase':  WorkingMemory(capacity=7, tau=10.0),   // ~10s  
    'sentence':WorkingMemory(capacity=7, tau=60.0),   // ~1min
    'paragraph':WorkingMemory(capacity=7, tau=600.0), // ~10min
    'document':WorkingMemory(capacity=7, tau=3600.0)  // ~1hour
}
```

### Cross-Timescale Integration
- **Bottom-up**: Lower levels feed completed units upward
- **Top-down**: Higher levels provide expectations/context
- **Horizontal**: Same-level lateral interactions
- **Shunting Combination**: Grossberg dynamics for multi-scale integration

## 3. Enhanced Pattern Generator

**Location**: `src/main/java/com/art/textgen/generation/EnhancedPatternGenerator.java`

### Advanced Sampling
- **Top-k Sampling**: Select from k highest probability tokens
- **Top-p (Nucleus) Sampling**: Dynamic vocabulary based on cumulative probability
- **Temperature Scaling**: Control randomness/creativity
- **Repetition Penalty**: Prevent loops and repetition

### Generation Modes
```java
public enum GenerationMode {
    CONSERVATIVE(0.7, 20, 0.5, 1.5),  // Low temp, focused
    BALANCED(1.0, 40, 0.9, 1.2),      // Default settings
    CREATIVE(1.2, 50, 0.95, 1.0),     // High temp, diverse
    PRECISE(0.5, 10, 0.3, 2.0);       // Very focused
}
```

### Context-Aware Scoring
- **Syntactic Continuations**: Grammar-based next token prediction
- **Semantic Coherence**: Topic consistency scoring
- **Pattern-based Candidates**: Sequence pattern completion
- **Bigram/Trigram Scoring**: Local coherence optimization

## 4. Neural Dynamics Components

**Location**: `src/main/java/com/art/textgen/dynamics/`

### Grossberg Shunting Equations
```java
// Implemented in ShuntingEquations.java
dx/dt = -Ax + (B - x)E - (x + C)I

Where:
- x: Neural activation  
- A: Decay rate
- B: Upper bound
- C: Lower bound  
- E: Excitatory input
- I: Inhibitory input
```

### ART Resonance Detection
```java
// ResonanceDetector.java
- Pattern matching with vigilance parameter
- Category formation and learning
- Bottom-up/top-down matching
- Reset when vigilance violated
```

### Attention Dynamics  
```java
// AttentionalDynamics.java
- Focus shifting mechanisms
- Competitive activation
- Winner-take-all dynamics
```

## 5. Training Pipeline Architecture

**Location**: `src/main/java/com/art/textgen/training/`

### Incremental Learning
- **No Catastrophic Forgetting**: ART resonance preserves existing patterns
- **Pattern Extraction**: N-gram, syntactic, and semantic patterns
- **Corpus Loading**: Support for multiple text formats
- **Vocabulary Building**: Dynamic vocabulary expansion

### Training Components
- `IncrementalTrainer.java`: Core training without forgetting
- `PatternExtractor.java`: Multi-level pattern discovery
- `CorpusLoader.java`: Text processing and tokenization
- `TrainingPipeline.java`: Orchestrated training process

## 6. Memory Management

### Adaptive Compression
- **Pattern-based**: Frequent sequences compressed to single units
- **Importance Weighting**: Retain high-value information longer
- **Temporal Decay**: Natural forgetting based on access patterns
- **Strategic Reset**: ART reset/search for memory reorganization

### Capacity Management
```java
// Effective memory usage
Working Memory: 7 items × 6 timescales = 42 items
Hierarchical: 7 items × 5 levels = 35 chunks
Skip Connections: Variable based on reset frequency
Total: Bounded memory with unbounded effective capacity
```

## Performance Characteristics

### Computational Complexity
- **Update Time**: O(log n) for hierarchical insertion
- **Retrieval Time**: O(k log n) where k is context window  
- **Generation Time**: O(1) for local generation
- **Memory Usage**: O(n log n) for unlimited sequences

### Biological Plausibility
- **7±2 Constraint**: Respected at all levels
- **Neural Dynamics**: Based on proven Grossberg equations
- **Incremental Learning**: Matches human learning patterns
- **Hierarchical Organization**: Mirrors cognitive chunking

## Integration Points

### Transformer Replacement
This architecture can replace transformer output layers:
```java
// Instead of transformer output layer
TransformerOutput output = transformer.generate(input);

// Use ART cognitive architecture  
EnhancedPatternGenerator generator = new EnhancedPatternGenerator(vocabulary);
String output = generator.generate(input, maxTokens);
```

### API Compatibility
- Standard text generation interface
- Configurable sampling parameters  
- Batch processing support
- Real-time streaming generation

## Configuration Guidelines

### Memory Tuning
- Increase hierarchy levels for longer contexts
- Adjust timescale constants for different domains
- Tune vigilance for pattern specificity
- Configure compression ratios for memory/quality tradeoff

### Generation Tuning  
- Temperature: 0.1-2.0 (higher = more creative)
- Top-k: 10-50 (smaller = more focused)
- Top-p: 0.3-0.95 (smaller = more focused)
- Repetition penalty: 1.0-2.0 (higher = less repetition)

This architecture successfully solves the fundamental challenge of scaling Grossberg's 7±2 constraint to unlimited sequence processing while maintaining biological plausibility and computational efficiency.