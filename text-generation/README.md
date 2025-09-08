# ART Text Generation Module

This module implements a cognitive architecture for text generation based on Adaptive Resonance Theory (ART) and Grossberg's neural dynamics. The system addresses sequence length limitations through hierarchical memory compression while maintaining biological plausibility.

## Overview

The implementation provides unlimited sequence processing capabilities by solving the fundamental 7±2 working memory constraint through recursive hierarchical chunking, multi-timescale processing, and ART-based pattern learning without catastrophic forgetting.

## Architecture Components

### Core Systems
- **Neural Dynamics** (`dynamics/`): Grossberg shunting equations, resonance detection, attention mechanisms
- **Memory Systems** (`memory/`): Recursive hierarchical memory, multi-timescale memory bank
- **Pattern Generation** (`generation/`): Enhanced pattern generator with advanced sampling strategies
- **Training Pipeline** (`training/`): Incremental learning without catastrophic forgetting

### Key Features
- Recursive hierarchical memory with 7±2 chunking achieving approximately 20,000 token capacity
- Multi-timescale parallel working memories (100ms to 1hour time constants)
- ART resonance-based learning that preserves existing patterns
- Advanced sampling methods including top-k, top-p, temperature scaling, and repetition penalty
- Real-time incremental learning capability

## Performance Characteristics

- Training Speed: 28 seconds for 42MB corpus (8.27M tokens)
- Pattern Learning: 13.9M n-gram patterns, 231K syntactic patterns
- Memory Efficiency: 113,405 unique tokens with hierarchical compression
- Generation Speed: Approximately 100 tokens per second
- No catastrophic forgetting during continuous learning

## Quick Start

### Build
```bash
mvn clean install
```

### Run Interactive Application
```bash
mvn exec:java -Dexec.mainClass="com.art.textgen.ARTTextGenerationApp"
```

### Basic Usage
```java
Vocabulary vocabulary = new Vocabulary(64);
EnhancedPatternGenerator generator = new EnhancedPatternGenerator(vocabulary);
TrainingPipeline pipeline = new TrainingPipeline(vocabulary, generator);

// Train on sample corpus
pipeline.trainFromSamples();

// Generate text
String result = generator.generate("The future of AI", 50);
```

## Documentation

Complete documentation is available in the [`docs/`](docs/) directory:

- **[Architecture Guide](docs/architecture/)** - Technical implementation details
- **[Performance Analysis](docs/performance/)** - Benchmarks and metrics
- **[User Guide](docs/user-guides/)** - Usage instructions and examples
- **[Original Requirements](docs/original-requirements/)** - Cognitive architecture specifications

## Transformer Integration

This system can serve as a replacement for transformer output layers, providing:
- Logarithmic memory complexity vs. quadratic attention
- No catastrophic forgetting during incremental learning
- Real-time adaptation without expensive retraining
- Explainable generation through pattern activation traces
- Memory-efficient processing for unlimited sequence lengths

## Requirements

- Java 24+
- Maven 3.9.1+
- 4GB RAM recommended
- macOS ARM64 (LWJGL configured for Apple Silicon)

## Testing

```bash
# Run all tests
mvn test

# Test training pipeline
mvn test -Dtest=TestTrainingPipeline

# Test core generation
mvn test -Dtest=TestTextGeneration
```

## License

Part of the ART project - see main LICENSE file.