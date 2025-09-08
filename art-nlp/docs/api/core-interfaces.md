# Core Interfaces API Reference

## Overview

The ART-NLP module provides a comprehensive set of interfaces for natural language processing using Adaptive Resonance Theory. This document covers the core interfaces that form the foundation of the system.

## NLPProcessor Interface

The primary interface for all NLP processing operations.

### Interface Definition

```java
package com.hellblazer.art.nlp.core;

public interface NLPProcessor extends AutoCloseable {
    // Primary processing methods
    ProcessingResult process(String text);
    CompletableFuture<ProcessingResult> processAsync(String text);
    void processStream(InputStream stream, ResultCallback callback);
    DocumentAnalysis processDocument(Document document);
    
    // Statistics and monitoring
    ProcessingStats getStatistics();
    
    // Channel management
    boolean setChannelEnabled(String channelName, boolean enabled);
    Set<String> getChannelNames();
    Set<String> getEnabledChannelNames();
    
    // System state
    boolean isReady();
    void reset();
    void resetChannel(String channelName);
    
    // Persistence
    void saveState();
    void loadState();
    
    // Lifecycle
    void shutdown();
    void close(); // AutoCloseable implementation
}
```

### Key Methods

#### process(String text)
Processes a single text input through all enabled channels.

**Parameters:**
- `text`: Input text to process (required, non-null, non-blank)

**Returns:**
- `ProcessingResult`: Comprehensive analysis results

**Throws:**
- `IllegalArgumentException`: If text is null or blank
- `RuntimeException`: If processing fails critically

**Example:**
```java
var processor = MultiChannelProcessor.createWithDefaults();
var result = processor.process("The quick brown fox jumps over the lazy dog.");

System.out.println("Category: " + result.getCategory());
System.out.println("Confidence: " + result.getConfidence());
System.out.println("Entities: " + result.getEntities().size());
```

#### processAsync(String text)
Asynchronous version of process() method.

**Parameters:**
- `text`: Input text to process

**Returns:**
- `CompletableFuture<ProcessingResult>`: Future containing processing results

**Example:**
```java
var processor = MultiChannelProcessor.createWithDefaults();
var future = processor.processAsync("Analyze this text asynchronously");

future.thenAccept(result -> {
    System.out.println("Async result: " + result.getCategory());
}).exceptionally(throwable -> {
    System.err.println("Processing failed: " + throwable.getMessage());
    return null;
});
```

#### processStream(InputStream stream, ResultCallback callback)
Processes streaming text with callback for each processed chunk.

**Parameters:**
- `stream`: Input stream containing text data (UTF-8 encoded)
- `callback`: Result callback implementing `ResultCallback` interface

**Throws:**
- `IllegalArgumentException`: If parameters are null

**Example:**
```java
var processor = MultiChannelProcessor.createWithDefaults();
var inputStream = new ByteArrayInputStream(textData.getBytes(StandardCharsets.UTF_8));

processor.processStream(inputStream, new ResultCallback() {
    @Override
    public void onResult(ProcessingResult result) {
        System.out.println("Chunk processed: " + result.getText());
    }
    
    @Override
    public void onError(Throwable error) {
        System.err.println("Processing error: " + error.getMessage());
    }
    
    @Override
    public void onComplete() {
        System.out.println("Stream processing complete");
    }
});
```

#### processDocument(Document document)
Processes a complete document with metadata enrichment.

**Parameters:**
- `document`: Document object containing text and metadata

**Returns:**
- `DocumentAnalysis`: Enhanced analysis with document structure

**Example:**
```java
var document = Document.builder()
    .withContent("This is a document with metadata.")
    .withMetadata("author", "John Doe")
    .withMetadata("title", "Sample Document")
    .build();

var analysis = processor.processDocument(document);
System.out.println("Sentences: " + analysis.getSentences().size());
System.out.println("Paragraphs: " + analysis.getParagraphs().size());
```

### ResultCallback Interface

Functional interface for streaming processing callbacks.

```java
@FunctionalInterface
public interface ResultCallback {
    void onResult(ProcessingResult result);
    
    default void onError(Throwable error) {
        System.err.println("Processing error: " + error.getMessage());
    }
    
    default void onComplete() {
        // Default: no action
    }
}
```

## ProcessingResult Class

Core data structure containing comprehensive processing results.

### Class Definition

```java
public final class ProcessingResult {
    private final String text;
    private final double confidence;
    private final int category;
    private final long processingTimeMs;
    private final Map<String, Integer> channelCategories;
    private final List<Entity> entities;
    private final int tokenCount;
    private final Map<String, ChannelResult> channelResults;
    private final Map<String, Object> consensusMetadata;
    private final DenseVector fusedFeatures;
    private final boolean success;
    private final String errorMessage;
}
```

### Key Methods

#### Basic Information
```java
public String getText()                              // Original input text
public double getConfidence()                        // Overall confidence [0.0, 1.0]
public int getCategory()                             // Consensus category ID
public long getProcessingTimeMs()                    // Processing duration
public boolean isSuccess()                           // Processing success flag
public String getErrorMessage()                      // Error message if failed
```

#### Channel-Specific Results
```java
public Map<String, Integer> getChannelCategories()   // Category per channel
public Map<String, ChannelResult> getChannelResults() // Detailed channel results
public boolean hasChannelResult(String channelName) // Check channel participation
public ChannelResult getChannelResult(String channelName) // Get specific channel result
```

#### Linguistic Analysis
```java
public List<Entity> getEntities()                   // Extracted entities
public int getTokenCount()                          // Token count
public DenseVector getFusedFeatures()               // Combined feature vector
public Map<String, Object> getConsensusMetadata()   // Consensus algorithm metadata
```

### Builder Pattern
```java
var result = ProcessingResult.builder()
    .text("Input text")
    .confidence(0.85)
    .category(42)
    .processingTimeMs(150)
    .withChannelCategories(channelCategories)
    .withEntities(entityList)
    .withTokenCount(15)
    .fusedFeatures(featureVector)
    .consensusMetadata(metadata)
    .build();
```

### Factory Methods
```java
// Success result
ProcessingResult result = ProcessingResult.success(text, category, confidence, channelResults);

// Failure result
ProcessingResult result = ProcessingResult.failed("Processing error occurred");
ProcessingResult result = ProcessingResult.failed(text, "Specific error message");
```

## Entity Class

Represents extracted named entities with metadata.

### Class Definition

```java
public final class Entity {
    private final String text;          // Entity text
    private final EntityType type;      // Entity type (PERSON, LOCATION, etc.)
    private final int startIndex;       // Start position in original text
    private final int endIndex;         // End position in original text
    private final double confidence;    // Entity confidence [0.0, 1.0]
    private final String source;        // Extracting channel name
    private final Map<String, Object> metadata; // Additional metadata
}
```

### Usage Examples
```java
// Create entity
var entity = Entity.builder()
    .text("John Doe")
    .type(EntityType.PERSON)
    .startIndex(0)
    .endIndex(8)
    .confidence(0.95)
    .source("entity")
    .withMetadata("gender", "unknown")
    .build();

// Entity processing
var entities = result.getEntities();
for (var entity : entities) {
    System.out.printf("Entity: %s (%s) at [%d, %d] confidence=%.2f%n",
        entity.getText(), entity.getType(), 
        entity.getStartIndex(), entity.getEndIndex(), entity.getConfidence());
}
```

## Document and DocumentAnalysis

### Document Class
```java
public final class Document {
    private final String content;                    // Document text content
    private final Map<String, Object> metadata;     // Document metadata
    private final String id;                        // Optional document ID
    private final Instant timestamp;                // Document timestamp
}
```

### DocumentAnalysis Class
```java
public final class DocumentAnalysis {
    private final Document document;                 // Original document
    private final ProcessingResult processingResult; // NLP analysis
    private final List<String> sentences;          // Sentence segmentation
    private final List<String> paragraphs;         // Paragraph structure
    private final Map<String, Object> analysisMetadata; // Analysis metadata
}
```

### Usage Example
```java
var document = Document.builder()
    .withContent("Multi-paragraph document.\n\nSecond paragraph here.")
    .withMetadata("source", "user_input")
    .withId("doc_001")
    .build();

var analysis = processor.processDocument(document);

System.out.println("Document analysis:");
System.out.println("- Sentences: " + analysis.getSentences().size());
System.out.println("- Paragraphs: " + analysis.getParagraphs().size());
System.out.println("- Processing result: " + analysis.getProcessingResult().getCategory());
```

## ProcessingStats Interface

Comprehensive processing statistics and metrics.

### Interface Definition
```java
public interface ProcessingStats {
    // Overall statistics
    Instant getStartTime();
    Instant getLastUpdate();
    int getTotalProcessed();
    int getSuccessfulProcessed();
    int getFailedProcessed();
    double getOverallSuccessRate();
    double getAverageProcessingTime();
    double getThroughput(); // operations per second
    
    // Channel-specific statistics
    Map<String, ChannelStats> getChannelStatistics();
    ChannelStats getChannelStats(String channelName);
    
    // System metrics
    Map<String, Object> getSystemMetrics();
}
```

### ChannelStats Record
```java
public record ChannelStats(
    String channelName,
    int totalProcessed,
    int successfulProcessed,
    int categoryCount,
    double averageProcessingTime,
    double successRate,
    Map<String, Object> channelSpecificMetrics
) {
    public int getFailedProcessed() {
        return totalProcessed - successfulProcessed;
    }
}
```

### Usage Example
```java
var stats = processor.getStatistics();

System.out.println("Overall Statistics:");
System.out.printf("- Total processed: %d%n", stats.getTotalProcessed());
System.out.printf("- Success rate: %.2f%%%n", stats.getOverallSuccessRate() * 100);
System.out.printf("- Throughput: %.1f ops/sec%n", stats.getThroughput());

System.out.println("\nChannel Statistics:");
for (var entry : stats.getChannelStatistics().entrySet()) {
    var channelStats = entry.getValue();
    System.out.printf("- %s: %d processed, %.2f%% success%n",
        channelStats.channelName(),
        channelStats.totalProcessed(),
        channelStats.successRate() * 100);
}
```

## Error Handling

### Exception Hierarchy
```java
// Base exception for ART-NLP operations
public class NLPException extends RuntimeException {
    public NLPException(String message) { super(message); }
    public NLPException(String message, Throwable cause) { super(message, cause); }
}

// Specific exception types
public class ChannelInitializationException extends NLPException { }
public class ModelLoadingException extends NLPException { }
public class ProcessingException extends NLPException { }
public class ConfigurationException extends NLPException { }
```

### Error Result Patterns
```java
// Checking for errors in results
if (!result.isSuccess()) {
    System.err.println("Processing failed: " + result.getErrorMessage());
    return;
}

// Handling channel-specific errors
for (var entry : result.getChannelResults().entrySet()) {
    var channelResult = entry.getValue();
    if (!channelResult.success()) {
        System.err.printf("Channel %s failed: %s%n", 
            entry.getKey(), channelResult.errorMessage());
    }
}
```

## Thread Safety and Concurrency

### Thread Safety Guarantees
- **NLPProcessor**: All methods are thread-safe
- **ProcessingResult**: Immutable, safe for concurrent access
- **Entity**: Immutable, safe for concurrent access
- **Document**: Immutable, safe for concurrent access

### Concurrency Best Practices
```java
// Safe concurrent usage
var processor = MultiChannelProcessor.createWithDefaults();

// Multiple threads can safely call process()
var executor = Executors.newFixedThreadPool(10);
var futures = new ArrayList<CompletableFuture<ProcessingResult>>();

for (String text : textInputs) {
    var future = CompletableFuture.supplyAsync(() -> processor.process(text), executor);
    futures.add(future);
}

// Wait for all results
var results = futures.stream()
    .map(CompletableFuture::join)
    .collect(Collectors.toList());
```

---

*This API reference covers the core interfaces. See [Channel APIs](channel-apis.md) for channel-specific interfaces and [Configuration](configuration.md) for configuration management.*