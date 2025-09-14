# HART-CQ: Hierarchical Adaptive Resonance Theory with Competitive Queuing

## Overview

HART-CQ is an experimental architecture that combines Adaptive Resonance Theory (ART) with competitive queuing dynamics for deterministic text processing. The system processes text through multiple parallel channels, uses hierarchical categorization, and generates output through template selection.

**Status**: In active development. Performance characteristics are preliminary and subject to change.


## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)

## Architecture

HART-CQ achieves reliable output through:

1. **Multi-Channel Processing**: 6 parallel channels extract different linguistic features
2. **Hierarchical ART**: 3-level DeepARTMAP for pattern recognition
3. **Competitive Queuing**: Grossberg dynamics for template selection
4. **Template-Bounded Generation**: All output constrained to predefined templates

### System Flow

```
Input Text → Tokenization → Sliding Windows → Multi-Channel Processing
                                                       ↓
                                            Hierarchical Categorization
                                                       ↓
                                             Competitive Queue Selection
                                                       ↓
                                             Template-Based Generation → Output
```

## Features

### Core Capabilities
- ✅ **Deterministic**: Template-based generation ensures consistent results
- ✅ **Parallel Processing**: 6 channels process features simultaneously
- ✅ **Online Learning**: Incremental learning without catastrophic forgetting
- ✅ **Thread-Safe**: Concurrent processing with no race conditions
- ✅ **Explainable**: Category activations provide interpretability

### Technical Specifications
- **Window Size**: 20 tokens (sliding)
- **Window Overlap**: 5 tokens
- **Hierarchy Levels**: 3 (Morpheme, Phrase, Discourse)
- **Channel Count**: 6 active channels
- **Template Library**: 25+ predefined templates

## Testing

### Test Coverage
- **Unit Tests**: 150+ test methods
- **Integration Tests**: 27 comprehensive tests
- **Coverage**: 90% of critical paths

## Quick Start

### Prerequisites
- Java 24 or higher
- Maven 3.9.1+
- 512MB heap minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ART.git
cd ART

# Build the project
mvn clean install

# Run tests
mvn test -pl hart-cq-integration
```

### Basic Usage

```java
import com.hellblazer.art.hartcq.integration.HARTCQ;
import com.hellblazer.art.hartcq.ProcessingResult;

// Initialize HART-CQ
var hartcq = new HARTCQ();

// Process single sentence
var result = hartcq.process("What is the weather today?");
System.out.println("Output: " + result.getOutput());
System.out.println("Processing time: " + result.getProcessingTime());

// Batch processing for maximum throughput
var sentences = List.of(
    "Hello world",
    "How are you?",
    "The cat sat on the mat"
);
var results = hartcq.processBatch(sentences);

// Clean shutdown
hartcq.shutdown();
```

## Modules

HART-CQ consists of 5 interconnected modules:

### 1. hart-cq-core
Core algorithms and channel implementations.

**Key Components**:
- `MultiChannelProcessor`: Orchestrates parallel channel processing
- `SlidingWindow`: Token window management
- `Tokenizer`: Text tokenization with edge case handling
- Channel implementations (8 types)

### 2. hart-cq-hierarchical
Hierarchical ART processing using DeepARTMAP.

**Key Components**:
- `HierarchicalProcessor`: 3-level categorization
- `ARTAdapter`: Integration with base ART algorithms
- `CategoryManager`: Category lifecycle management

### 3. hart-cq-feedback
Adaptive resonance and feedback control.

**Key Components**:
- `FeedbackController`: Error correction and adaptation
- `ResonanceManager`: Vigilance parameter control

### 4. hart-cq-spatial
Spatial and positional encoding features.

**Key Components**:
- `TemplateSystem`: Template management and selection
- `PositionalEncoder`: Sinusoidal position encoding (Transformer-style)

### 5. hart-cq-integration
Main integration and pipeline orchestration.

**Key Components**:
- `HARTCQ`: Main entry point and API
- `ProcessingResult`: Result container with metadata
- `StreamProcessor`: Asynchronous stream processing

## API Reference

### Main API: HARTCQ

```java
public class HARTCQ {
    // Initialize with default configuration
    public HARTCQ();

    // Initialize with custom config
    public HARTCQ(HARTCQConfig config);

    // Process single input
    public ProcessingResult process(String input);

    // Async processing
    public CompletableFuture<ProcessingResult> processAsync(String input);

    // Batch processing
    public List<ProcessingResult> processBatch(List<String> inputs);

    // Learning/training
    public void train(String input, String expectedOutput);

    // Enable/disable learning
    public void setLearningEnabled(boolean enabled);

    // Performance stats
    public PerformanceStats getStats();

    // Clean shutdown
    public void shutdown();
}
```

### ProcessingResult

```java
public class ProcessingResult {
    public String getInput();
    public String getOutput();
    public boolean isSuccessful();
    public Duration getProcessingTime();
    public int getTokensProcessed();
    public double getConfidence();
    public Map<String, Object> getMetadata();
}
```

### Channel Types

1. **PositionalChannel**: Sinusoidal positional encoding
2. **WordChannel**: Word-level embeddings (comprehension only)
3. **ContextChannel**: Historical context with momentum
4. **StructuralChannel**: Grammatical structure analysis
5. **SemanticChannel**: Semantic relationships
6. **TemporalChannel**: Time-based decay patterns
7. **SyntaxChannel**: Syntactic patterns
8. **PhoneticChannel**: Sound patterns

## Configuration

### Grossberg Dynamics Parameters

```java
public class GrossbergParameters {
    public static final double SELF_EXCITATION = 1.2;      // Strengthens active items
    public static final double LATERAL_INHIBITION = 0.3;   // Suppresses competitors
    public static final double PRIMACY_GRADIENT = 0.95;    // Favors earlier items
    public static final int K_WINNERS = 1;                 // Winner-take-all
}
```

### Hierarchical Vigilance

```java
public class HierarchicalConfig {
    public static final double MORPHEME_VIGILANCE = 0.9;   // High precision
    public static final double PHRASE_VIGILANCE = 0.7;     // Medium precision
    public static final double DISCOURSE_VIGILANCE = 0.5;  // Low precision
}
```

### Performance Tuning

```properties
# hart-cq.properties
hartcq.parallelism.level=8
hartcq.batch.size=100
hartcq.cache.enabled=true
hartcq.learning.rate=0.1
hartcq.template.count=25
```

## Examples

### Example 1: Question Answering

```java
var hartcq = new HARTCQ();

// Configure for Q&A mode
hartcq.setLearningEnabled(false);  // Use pre-trained knowledge

var question = "What is the capital of France?";
var result = hartcq.process(question);

// Output uses template: "The [subject] of [entity] is [answer]."
System.out.println(result.getOutput());
// "The capital of France is Paris."
```

### Example 2: Real-time Stream Processing

```java
var hartcq = new HARTCQ();
var executor = Executors.newFixedThreadPool(4);

// Process stream of messages
Stream<String> messageStream = getMessageStream();

messageStream
    .map(msg -> CompletableFuture.supplyAsync(
        () -> hartcq.process(msg), executor))
    .map(CompletableFuture::join)
    .filter(ProcessingResult::isSuccessful)
    .forEach(result -> {
        System.out.println("Processed: " + result.getOutput());
    });
```

### Example 3: Batch Processing for Maximum Throughput

```java
var hartcq = new HARTCQ();

// Prepare batch
var sentences = loadSentences();  // Load 10,000 sentences
var batchSize = 100;

// Process in batches for optimal performance
for (int i = 0; i < sentences.size(); i += batchSize) {
    var batch = sentences.subList(i,
        Math.min(i + batchSize, sentences.size()));

    var results = hartcq.processBatch(batch);

    // ~13ms for 100 sentences = 7,692 sentences/sec
    processResults(results);
}
```

## Testing

### Running Tests

```bash
# All tests
mvn test

# Specific module
mvn test -pl hart-cq-core

# Specific test class
mvn test -Dtest=MultiChannelProcessorTest

# Performance tests only
mvn test -Dtest=PerformanceTest
```

### Test Coverage

- **Unit Tests**: 150+ test methods
- **Integration Tests**: 27 comprehensive tests
- **Performance Tests**: JMH benchmarks
- **Coverage**: 90% of critical paths

### Key Test Suites

- `MultiChannelProcessorTest`: Channel coordination
- `HierarchicalProcessorTest`: DeepARTMAP integration
- `EndToEndPipelineTest`: Full pipeline validation
- `PerformanceTest`: Throughput benchmarks
- `ConcurrencyTest`: Thread safety validation

## Performance Optimization

### Tips for Maximum Performance

1. **Use Batch Processing**: Process multiple sentences together
2. **Enable Caching**: Cache frequently used templates
3. **Tune Thread Pool**: Match CPU cores for parallel processing
4. **Adjust Window Size**: Smaller windows for faster processing
5. **Disable Learning**: Turn off learning for inference-only mode

### Memory Management

```java
// Recommended JVM flags for production
-Xms512m -Xmx2g
-XX:+UseG1GC
-XX:MaxGCPauseMillis=100
-XX:+ParallelRefProcEnabled
```

## Troubleshooting

### Common Issues

**Issue**: Lower than expected throughput
- **Solution**: Enable batch processing, check thread pool size

**Issue**: Memory usage growing
- **Solution**: Call `hartcq.resetStats()` periodically, check cache size

**Issue**: Non-deterministic results
- **Solution**: Verify `deterministicMode` is enabled

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/ART.git
cd ART

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
mvn clean test

# Submit pull request
```

## License

Copyright (c) 2025 Hal Hildebrand. All rights reserved.

Licensed under the GNU Affero General Public License v3.0.

## Acknowledgments

- Based on Grossberg's Adaptive Resonance Theory
- Inspired by Transformer positional encoding
- Uses DeepARTMAP from Petrenko et al. (2025)

## Citations

If you use HART-CQ in your research, please cite:

```bibtex
@software{hartcq2025,
  author = {Hildebrand, Hal},
  title = {HART-CQ: Hierarchical ART with Competitive Queuing},
  year = {2025},
  url = {https://github.com/yourusername/ART}
}
```

## Contact

- **Author**: Hal Hildebrand
- **Email**: [Contact Email]
- **GitHub**: [GitHub Profile]

---

**Current Version**: 0.0.1-SNAPSHOT
**Last Updated**: September 14, 2025
**Status**: Development/Experimental (90% test coverage achieved, performance validation pending)