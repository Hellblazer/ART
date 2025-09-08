# Channel APIs Reference

## Overview

This document provides detailed API reference for all ART-NLP channel implementations. Each channel specializes in a different aspect of natural language processing while sharing a common interface through the `BaseChannel` abstract class.

## BaseChannel Abstract Class

### Core Interface

```java
package com.hellblazer.art.nlp.channels.base;

public abstract class BaseChannel {
    protected final String channelName;
    protected final double vigilance;
    protected final DataPreprocessor preprocessor;
    protected final CategoryPersistence persistence;
    protected final ChannelMetrics metrics;
    protected final ReadWriteLock lock;
    
    // CRITICAL: Main classification method - must be implemented
    public abstract int classify(DenseVector input);
    
    // Lifecycle management
    public abstract void saveState();
    public abstract void loadState();
    protected abstract void performInitialization();
    protected abstract void performCleanup();
    
    // Category management
    public abstract int getCategoryCount();
    public abstract int pruneCategories(double threshold);
}
```

### Common Methods

#### classify(DenseVector input)
**Primary classification method** - all channels must implement this.

**Parameters:**
- `input`: DenseVector to classify (preprocessed)

**Returns:**
- `int`: Category ID (0-based), or -1 if classification fails

**Thread Safety**: Uses ReadWriteLock for concurrent access

#### Lifecycle Methods
```java
public final void initialize()              // Initialize channel (load state, setup)
public final void shutdown()                // Shutdown channel (save state, cleanup)
public final boolean isInitialized()       // Check initialization status
public final boolean isLearningEnabled()   // Check if learning is active
public final void setLearningEnabled(boolean enabled) // Enable/disable learning
```

#### Metrics and Monitoring
```java
public final ChannelMetrics getMetrics()   // Get performance metrics
protected final void recordClassification(long timeMs, boolean categoryCreated)
protected final void recordError()         // Record processing error
```

## Semantic Channel (FastTextChannel)

Processes semantic meaning using FastText word embeddings and FuzzyART clustering.

### Class Definition

```java
public final class FastTextChannel extends BaseChannel {
    // Configuration
    private final FastTextModel fastTextModel;
    private final PreprocessingPipeline preprocessingPipeline;
    private final OOVStrategy oovStrategy;
    private final boolean useSubwordFallback;
    private final int maxTokensPerInput;
    
    // ART algorithm
    private final FuzzyART fuzzyART;
    private final FuzzyParameters artParameters;
}
```

### Constructor Options

#### Default Configuration
```java
public FastTextChannel(String channelName, double vigilance, Path fastTextModelPath) 
    throws IOException
```

**Parameters:**
- `channelName`: Channel identifier
- `vigilance`: ART vigilance parameter [0.0, 1.0]
- `fastTextModelPath`: Path to FastText model file

**Example:**
```java
var semanticChannel = new FastTextChannel("semantic", 0.85, 
    Path.of("models/cc.en.300.vec.gz"));
```

#### Custom Configuration
```java
public FastTextChannel(String channelName, double vigilance, Path fastTextModelPath, 
                      int dimensions, OOVStrategy oovStrategy, boolean useSubwordFallback, 
                      int maxTokensPerInput, PreprocessingPipeline preprocessingPipeline) 
    throws IOException
```

### Key Methods

#### classifyText(String text)
**High-level text classification method.**

```java
public int classifyText(String text)
```

**Processing Steps:**
1. Tokenize input text
2. Get word vectors from FastText model
3. Handle out-of-vocabulary (OOV) words
4. Aggregate vectors using mean pooling
5. Apply preprocessing pipeline
6. Classify using underlying ART algorithm

**Example:**
```java
var semanticChannel = new FastTextChannel("semantic", 0.85, modelPath);
var category = semanticChannel.classifyText("Machine learning is fascinating!");
System.out.println("Semantic category: " + category);
```

#### getTextSimilarity(String text1, String text2)
**Calculate semantic similarity between two texts.**

```java
public double getTextSimilarity(String text1, String text2)
```

**Returns:** Cosine similarity [0.0, 1.0]

**Example:**
```java
var similarity = semanticChannel.getTextSimilarity(
    "I love machine learning", 
    "Artificial intelligence is amazing"
);
System.out.printf("Similarity: %.3f%n", similarity);
```

#### getTextVector(String text)
**Get processed vector representation of text.**

```java
public DenseVector getTextVector(String text)
```

### OOV Strategy Options

```java
public enum OOVStrategy {
    SKIP,              // Skip unknown words
    ZERO_VECTOR,       // Use zero vector
    RANDOM_VECTOR,     // Use random vector  
    AVERAGE_FALLBACK   // Use average of known words
}
```

**Configuration Example:**
```java
var channel = new FastTextChannel("semantic", 0.85, modelPath, 300,
    OOVStrategy.RANDOM_VECTOR, true, 100, preprocessingPipeline);
```

### Performance Metrics

#### FastTextMetrics
```java
public record FastTextMetrics(
    long totalClassifications,
    long successfulClassifications,
    int categoryCount,
    double averageProcessingTime,
    int totalTokens,
    int oovTokens,
    int successfulTextClassifications,
    double cacheHitRate,
    double oovRate,
    int cacheSize
) {
    public double successRate() { /* ... */ }
    public double textSuccessRate() { /* ... */ }
    public double oovRateByTokens() { /* ... */ }
}
```

**Usage:**
```java
var metrics = semanticChannel.getFastTextMetrics();
System.out.printf("OOV rate: %.1f%% (%d/%d tokens)%n",
    metrics.oovRateByTokens() * 100, metrics.oovTokens(), metrics.totalTokens());
```

## Syntactic Channel

Analyzes grammatical structure using part-of-speech tagging and SalienceAware ART.

### Class Definition

```java
public final class SyntacticChannel extends BaseChannel {
    private final TokenizerME tokenizer;
    private final POSTaggerME posTagger;
    private final SalienceART salienceART;
    private final POSSequenceProcessor sequenceProcessor;
}
```

### Constructor
```java
public SyntacticChannel(String channelName, double vigilance) throws IOException
```

**Example:**
```java
var syntacticChannel = new SyntacticChannel("syntactic", 0.75);
```

### Key Methods

#### classifyText(String text)
```java
public int classifyText(String text)
```

**Processing Steps:**
1. Tokenize input text
2. Apply POS tagging
3. Extract syntactic patterns
4. Convert to feature vector
5. Classify using SalienceART

#### extractPOSSequence(String text)
```java
public List<String> extractPOSSequence(String text)
```

**Returns:** Sequence of POS tags

**Example:**
```java
var posSequence = syntacticChannel.extractPOSSequence("The cat sits on the mat");
// Result: [DT, NN, VBZ, IN, DT, NN]
```

## Entity Channel

Performs named entity recognition using OpenNLP NER and FuzzyARTMAP supervised learning.

### Class Definition

```java
public final class EntityChannel extends BaseChannel {
    private final Map<EntityType, NameFinderME> entityFinders;
    private final FuzzyARTMAP fuzzyARTMAP;
    private final BIOTagProcessor bioProcessor;
}
```

### Constructor
```java
public EntityChannel(String channelName, double vigilance) throws IOException
```

**Example:**
```java
var entityChannel = new EntityChannel("entity", 0.80);
```

### Key Methods

#### extractEntities(String text)
```java
public List<Entity> extractEntities(String text)
```

**Returns:** List of extracted entities with metadata

**Example:**
```java
var entities = entityChannel.extractEntities(
    "John Smith works at Google in Mountain View");

for (var entity : entities) {
    System.out.printf("'%s' → %s [%d:%d] confidence=%.2f%n",
        entity.getText(), entity.getType(),
        entity.getStartIndex(), entity.getEndIndex(),
        entity.getConfidence());
}
```

#### classifyText(String text)
```java
public int classifyText(String text)
```

**Processing Steps:**
1. Extract named entities
2. Convert entities to feature vectors
3. Apply BIO tagging
4. Classify using FuzzyARTMAP

### Supported Entity Types

```java
public enum EntityType {
    PERSON,
    ORGANIZATION,
    LOCATION,
    MONEY,
    PERCENT,
    DATE,
    TIME,
    DISCOURSE_CONTRAST,    // e.g., "however", "but"
    DISCOURSE_CONCLUSION,  // e.g., "therefore", "thus"
    RELATIONSHIP           // e.g., "related to", "caused by"
}
```

## Context Channel

Analyzes contextual relationships and discourse patterns using TopoART.

### Class Definition

```java
public final class ContextChannel extends BaseChannel {
    private final TopoART topoART;
    private final WindowingManager windowingManager;
    private final DiscourseMarkerDetector discourseDetector;
    private final RelationshipExtractor relationshipExtractor;
}
```

### Constructor
```java
public ContextChannel() throws IOException
```

**Default Configuration:**
- Vigilance: 0.70
- Window size: 5 tokens
- Discourse marker detection enabled

### Key Methods

#### classifyText(String text)
```java
public int classifyText(String text)
```

**Processing Steps:**
1. Apply sliding window tokenization
2. Detect discourse markers
3. Extract relationships
4. Create contextual feature vectors
5. Classify using TopoART

#### extractEntities(String text)
```java
public List<Entity> extractEntities(String text)
```

**Specialized for discourse analysis:**
- Detects contrast markers ("however", "but", "although")
- Identifies conclusion markers ("therefore", "thus", "consequently")
- Extracts relationship indicators ("related to", "caused by")

**Example:**
```java
var contextEntities = contextChannel.extractEntities(
    "The weather was bad, however, the game continued. " +
    "Therefore, the fans were disappointed.");

// Expected entities:
// - "however" → DISCOURSE_CONTRAST
// - "therefore" → DISCOURSE_CONCLUSION
```

## Sentiment Channel

Analyzes emotional content using emotion lexicons and FuzzyART classification.

### Class Definition

```java
public final class SentimentChannel extends BaseChannel {
    private final EmotionLexicon emotionLexicon;
    private final FuzzyART fuzzyART;
    private final SentimentFeatureExtractor featureExtractor;
}
```

### Constructor
```java
public SentimentChannel() throws IOException
```

**Default Configuration:**
- Vigilance: 0.60
- Multi-dimensional emotion analysis (valence, arousal, dominance)
- Built-in emotion lexicons

### Key Methods

#### classifyText(String text)
```java
public int classifyText(String text)
```

**Processing Steps:**
1. Extract emotion-bearing words
2. Calculate sentiment scores
3. Create multi-dimensional feature vector
4. Classify using FuzzyART

#### getSentimentScores(String text)
```java
public SentimentScore getSentimentScores(String text)
```

**Returns:** Multi-dimensional sentiment analysis

```java
public record SentimentScore(
    double valence,    // Positive/negative [-1.0, 1.0]
    double arousal,    // Calm/excited [0.0, 1.0]
    double dominance,  // Submissive/dominant [0.0, 1.0]
    double confidence  // Analysis confidence [0.0, 1.0]
) { }
```

**Example:**
```java
var scores = sentimentChannel.getSentimentScores("I love this amazing product!");
System.out.printf("Valence: %.2f, Arousal: %.2f, Dominance: %.2f%n",
    scores.valence(), scores.arousal(), scores.dominance());
```

## Channel Configuration Patterns

### Creating Custom Channels
```java
// Extend BaseChannel for custom functionality
public class CustomChannel extends BaseChannel {
    private final CustomART customART;
    
    public CustomChannel(String name, double vigilance) {
        super(name, vigilance);
        this.customART = new CustomART();
    }
    
    @Override
    public int classify(DenseVector input) {
        // Custom classification logic
        return customART.classify(preprocessInput(input));
    }
    
    @Override
    public void saveState() {
        // Custom state persistence
        customART.saveWeights();
    }
    
    @Override
    public void loadState() {
        // Custom state loading
        customART.loadWeights();
    }
    
    // Implement other abstract methods...
}
```

### Channel Registration
```java
// Add custom channel to processor
var processor = MultiChannelProcessor.builder().build();
var customChannel = new CustomChannel("custom", 0.8);
processor.addChannel("custom", customChannel, 1.0);
```

### Advanced Configuration
```java
// Fine-tune channel parameters
var semanticChannel = new FastTextChannel("semantic", 0.90, modelPath, 300,
    OOVStrategy.AVERAGE_FALLBACK, true, 200,
    VectorPreprocessor.pipeline()
        .normalize(NormalizationType.L2)
        .complementCode()
        .dimensionalityReduction(150)
        .build());

// Add with custom weight
processor.addChannel("semantic", semanticChannel, 1.5);
```

## Error Handling and Resilience

### Channel-Level Error Handling
```java
// Channels handle errors gracefully
try {
    var result = channel.classifyText(text);
    if (result >= 0) {
        // Success
    } else {
        // Classification failed
    }
} catch (Exception e) {
    // Channel threw exception - handle appropriately
    logger.warn("Channel {} failed: {}", channel.getChannelName(), e.getMessage());
}
```

### Processor-Level Resilience
```java
// Processor continues with available channels
var result = processor.process(text);

// Check which channels succeeded
for (var entry : result.getChannelResults().entrySet()) {
    var channelResult = entry.getValue();
    if (!channelResult.isSuccess()) {
        System.err.printf("Channel %s failed: %s%n", 
            entry.getKey(), channelResult.errorMessage());
    }
}
```

## Performance Optimization

### Channel-Specific Optimizations

#### Semantic Channel
```java
// Optimize for throughput
var semanticChannel = new FastTextChannel("semantic", 0.70, modelPath, 300,
    OOVStrategy.SKIP,        // Skip OOV words for speed
    false,                   // Disable subword fallback
    50,                      // Limit token count
    simplePreprocessing);    // Minimal preprocessing
```

#### Entity Channel
```java
// Optimize for accuracy vs speed
var entityChannel = new EntityChannel("entity", 0.90); // High vigilance
entityChannel.setLearningEnabled(false); // Disable learning for consistency
```

### Memory Management
```java
// Periodic category pruning
processor.getChannelNames().forEach(channelName -> {
    var channel = processor.getChannel(channelName);
    var prunedCount = channel.pruneCategories(0.01); // Remove categories with <1% usage
    if (prunedCount > 0) {
        logger.info("Pruned {} categories from channel {}", prunedCount, channelName);
    }
});
```

---

*This API reference covers all channel implementations. See [Core Interfaces](core-interfaces.md) for base interfaces and [Configuration](configuration.md) for system-wide configuration options.*