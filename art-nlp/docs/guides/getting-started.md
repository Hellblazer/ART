# Getting Started with ART-NLP

## Quick Start Guide

This guide will help you get up and running with the ART-NLP module in minutes. Follow these steps to set up your development environment and process your first text.

## Prerequisites

### System Requirements
- **Java 24+** with Vector API support enabled
- **Maven 3.9.1+** for building
- **6GB+ RAM** (12GB recommended for production)
- **10GB disk space** for models and working data

### Environment Setup
1. **Verify Java Version**
   ```bash
   java --version
   # Should show Java 24 or higher
   ```

2. **Check Maven**
   ```bash
   mvn --version
   # Should show Maven 3.9.1 or higher
   ```

3. **Verify Memory**
   ```bash
   # On macOS/Linux
   free -h
   # Ensure you have at least 6GB available
   ```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/ART.git
cd ART/art-nlp
```

### 2. Build the Module
```bash
mvn clean compile
```

This will:
- Download the 4.7GB FastText model automatically
- Compile all source files
- Set up the development environment

### 3. Run Tests (Optional)
```bash
mvn test
```

## First Steps

### Hello World Example

Create a simple Java class to test the basic functionality:

```java
package com.example;

import com.hellblazer.art.nlp.processor.MultiChannelProcessor;
import com.hellblazer.art.nlp.core.ProcessingResult;

public class HelloArtNLP {
    public static void main(String[] args) {
        // Create processor with default channels
        var processor = MultiChannelProcessor.createWithDefaults();
        
        // Process some text
        var result = processor.process("Hello, ART-NLP! This is my first text analysis.");
        
        // Print results
        System.out.println("=== ART-NLP Analysis Results ===");
        System.out.println("Original text: " + result.getText());
        System.out.println("Category: " + result.getCategory());
        System.out.printf("Confidence: %.2f%n", result.getConfidence());
        System.out.println("Processing time: " + result.getProcessingTimeMs() + "ms");
        System.out.println("Token count: " + result.getTokenCount());
        
        // Show channel results
        System.out.println("\n=== Channel Results ===");
        result.getChannelCategories().forEach((channel, category) -> 
            System.out.printf("%s: category %d%n", channel, category));
        
        // Show entities
        if (!result.getEntities().isEmpty()) {
            System.out.println("\n=== Extracted Entities ===");
            result.getEntities().forEach(entity -> 
                System.out.printf("'%s' (%s) [%d:%d] confidence=%.2f%n",
                    entity.getText(), entity.getType(),
                    entity.getStartIndex(), entity.getEndIndex(),
                    entity.getConfidence()));
        }
        
        // Clean up
        processor.close();
        System.out.println("\nDone!");
    }
}
```

### Running the Example

1. **Compile**
   ```bash
   javac -cp target/classes:target/dependency/* src/main/java/com/example/HelloArtNLP.java
   ```

2. **Run**
   ```bash
   java -cp target/classes:target/dependency/*:src/main/java com.example.HelloArtNLP
   ```

Expected output:
```
=== ART-NLP Analysis Results ===
Original text: Hello, ART-NLP! This is my first text analysis.
Category: 15
Confidence: 0.85
Processing time: 147ms
Token count: 9

=== Channel Results ===
semantic: category 15
syntactic: category 8
entity: category 2
context: category 12
sentiment: category 5

Done!
```

## Common Usage Patterns

### 1. Basic Text Processing
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    var texts = List.of(
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process natural language.",
        "John Smith works at OpenAI in San Francisco."
    );
    
    for (var text : texts) {
        var result = processor.process(text);
        System.out.printf("Text: %s -> Category: %d (confidence: %.2f)%n",
            text, result.getCategory(), result.getConfidence());
    }
}
```

### 2. Asynchronous Processing
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    var futures = new ArrayList<CompletableFuture<ProcessingResult>>();
    
    // Submit multiple texts for async processing
    for (var text : textList) {
        var future = processor.processAsync(text);
        futures.add(future);
    }
    
    // Collect results
    var results = futures.stream()
        .map(CompletableFuture::join)
        .collect(Collectors.toList());
    
    System.out.println("Processed " + results.size() + " texts asynchronously");
}
```

### 3. Document Analysis
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    var document = Document.builder()
        .withContent("This is a multi-paragraph document.\n\nIt contains multiple sentences. Each sentence provides different information.")
        .withMetadata("source", "user_input")
        .withMetadata("language", "en")
        .build();
    
    var analysis = processor.processDocument(document);
    
    System.out.println("Document Analysis:");
    System.out.println("- Sentences: " + analysis.getSentences().size());
    System.out.println("- Paragraphs: " + analysis.getParagraphs().size());
    System.out.println("- Category: " + analysis.getProcessingResult().getCategory());
}
```

### 4. Stream Processing
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    var textStream = new ByteArrayInputStream(
        "Line 1: First text to process\nLine 2: Second text\nLine 3: Third text"
        .getBytes(StandardCharsets.UTF_8));
    
    processor.processStream(textStream, new NLPProcessor.ResultCallback() {
        @Override
        public void onResult(ProcessingResult result) {
            System.out.printf("Processed: '%s' -> Category %d%n", 
                result.getText().trim(), result.getCategory());
        }
        
        @Override
        public void onError(Throwable error) {
            System.err.println("Error: " + error.getMessage());
        }
        
        @Override
        public void onComplete() {
            System.out.println("Stream processing complete");
        }
    });
}
```

## Configuration

### Basic Channel Configuration

Create a custom processor with specific channels:

```java
// Create processor with custom configuration
var processor = MultiChannelProcessor.builder()
    .enableParallelProcessing(true)
    .threadPoolSize(4)
    .consensusStrategy(new WeightedVotingConsensus())
    .fusionStrategy(new ConcatenationFusion())
    .build();

// Add semantic channel with custom settings
var semanticChannel = new FastTextChannel("semantic", 0.90, fastTextModelPath);
processor.addChannel("semantic", semanticChannel, 1.0);

// Add entity channel
var entityChannel = new EntityChannel("entity", 0.85);
processor.addChannel("entity", entityChannel, 0.8);

// Process text
var result = processor.process("Custom configuration example");
```

### Vigilance Parameter Tuning

Vigilance controls category formation sensitivity:
- **Low vigilance (0.1-0.5)**: Fewer, broader categories
- **Medium vigilance (0.6-0.8)**: Balanced categorization
- **High vigilance (0.9-0.99)**: Many specific categories

```java
// High vigilance for fine-grained categorization
var semanticChannel = new FastTextChannel("semantic", 0.95, modelPath);

// Low vigilance for broad categorization  
var contextChannel = new ContextChannel("context", 0.3);
```

## Monitoring and Debugging

### Performance Monitoring
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    // Process some texts
    for (int i = 0; i < 100; i++) {
        processor.process("Sample text " + i);
    }
    
    // Get statistics
    var stats = processor.getStatistics();
    System.out.printf("Processed: %d texts%n", stats.getTotalProcessed());
    System.out.printf("Success rate: %.1f%%%n", stats.getOverallSuccessRate() * 100);
    System.out.printf("Average time: %.1fms%n", stats.getAverageProcessingTime());
    System.out.printf("Throughput: %.1f ops/sec%n", stats.getThroughput());
    
    // Channel-specific stats
    stats.getChannelStatistics().forEach((name, channelStats) -> {
        System.out.printf("Channel %s: %d categories, %.1f%% success%n",
            name, channelStats.categoryCount(), channelStats.successRate() * 100);
    });
}
```

### Channel Management
```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    // Check available channels
    System.out.println("Available channels: " + processor.getChannelNames());
    System.out.println("Enabled channels: " + processor.getEnabledChannelNames());
    
    // Enable/disable channels
    processor.setChannelEnabled("sentiment", false);
    System.out.println("Disabled sentiment channel");
    
    // Check readiness
    if (processor.isReady()) {
        System.out.println("Processor ready for processing");
    } else {
        System.out.println("Processor not ready - check channel initialization");
    }
}
```

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**
   - Increase JVM heap size: `-Xmx8G`
   - Reduce parallel processing threads
   - Check FastText model loading (1.2GB runtime)

2. **Channel Initialization Failed**
   - Verify FastText model exists and is readable
   - Check OpenNLP models in resources
   - Ensure proper file permissions

3. **Poor Performance**
   - Enable parallel processing
   - Tune thread pool size
   - Check vigilance parameters (too high = too many categories)

4. **Low Accuracy**
   - Adjust vigilance parameters
   - Check channel weights in consensus strategy
   - Verify model quality and relevance

### Debug Logging

Enable debug logging to see internal operations:

```java
// Add to your logback.xml or application properties
Logger logger = LoggerFactory.getLogger("com.hellblazer.art.nlp");
logger.setLevel(Level.DEBUG);

// Or programmatically
System.setProperty("org.slf4j.simpleLogger.log.com.hellblazer.art.nlp", "debug");
```

### Model Verification

Check if models are properly loaded:

```java
try (var processor = MultiChannelProcessor.createWithDefaults()) {
    var stats = processor.getStatistics();
    
    // Check if all expected channels are present
    var expectedChannels = Set.of("semantic", "syntactic", "entity", "context", "sentiment");
    var actualChannels = processor.getChannelNames();
    
    var missingChannels = Sets.difference(expectedChannels, actualChannels);
    if (!missingChannels.isEmpty()) {
        System.err.println("Missing channels: " + missingChannels);
        System.err.println("Check model files and dependencies");
    }
}
```

## Next Steps

Now that you have basic functionality working:

1. **Read the [API Reference](../api/)** for comprehensive interface documentation
2. **Explore [Channel Configuration](channel-configuration.md)** for advanced channel setup
3. **Check [Performance Guide](../performance/)** for optimization strategies
4. **Review [Best Practices](best-practices.md)** for production deployment
5. **See [Advanced Usage](advanced-usage.md)** for complex scenarios

## Example Projects

Check out these complete examples:
- **Document Classifier**: Build a document classification system
- **Entity Extractor**: Extract and classify named entities
- **Sentiment Analyzer**: Analyze emotional content
- **Semantic Search**: Vector-based text similarity search

All examples are available in the `/examples` directory of the repository.

---

*Happy processing with ART-NLP! 🚀*