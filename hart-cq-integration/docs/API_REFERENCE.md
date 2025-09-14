# HART-CQ API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Channel Interfaces](#channel-interfaces)
3. [Processing Components](#processing-components)
4. [Template System](#template-system)
5. [Competitive Queue](#competitive-queue)
6. [Feedback Control](#feedback-control)
7. [Performance Monitoring](#performance-monitoring)
8. [Configuration](#configuration)

## Core Classes

### HARTCQ

Main entry point for the HART-CQ system.

```java
package com.hellblazer.art.hartcq.integration;

public class HARTCQ implements AutoCloseable {

    /**
     * Initialize HART-CQ with default configuration.
     */
    public HARTCQ()

    /**
     * Initialize HART-CQ with custom configuration.
     * @param config Custom configuration settings
     */
    public HARTCQ(HARTCQConfig config)

    /**
     * Process a single input sentence.
     * @param input Text to process
     * @return Processing result with output and metadata
     */
    public ProcessingResult process(String input)

    /**
     * Process input asynchronously.
     * @param input Text to process
     * @return Future containing processing result
     */
    public CompletableFuture<ProcessingResult> processAsync(String input)

    /**
     * Process multiple inputs in batch for maximum throughput.
     * @param inputs List of sentences to process
     * @return List of processing results
     */
    public List<ProcessingResult> processBatch(List<String> inputs)

    /**
     * Train the system with labeled data.
     * @param input Training input
     * @param expectedOutput Expected output for training
     */
    public void train(String input, String expectedOutput)

    /**
     * Enable or disable online learning.
     * @param enabled True to enable learning, false to disable
     */
    public void setLearningEnabled(boolean enabled)

    /**
     * Get current performance statistics.
     * @return Performance statistics object
     */
    public PerformanceStats getStats()

    /**
     * Reset performance statistics.
     */
    public void resetStats()

    /**
     * Clean shutdown of resources.
     */
    public void shutdown()

    /**
     * AutoCloseable implementation.
     */
    @Override
    public void close()
}
```

### ProcessingResult

Container for processing results with metadata.

```java
package com.hellblazer.art.hartcq;

public class ProcessingResult {

    /**
     * Get the original input text.
     * @return Input text
     */
    public String getInput()

    /**
     * Get the processed output text.
     * @return Output text from template
     */
    public String getOutput()

    /**
     * Check if processing was successful.
     * @return true if successful, false otherwise
     */
    public boolean isSuccessful()

    /**
     * Get processing duration.
     * @return Duration of processing
     */
    public Duration getProcessingTime()

    /**
     * Get number of tokens processed.
     * @return Token count
     */
    public int getTokensProcessed()

    /**
     * Get confidence score of output.
     * @return Confidence value between 0.0 and 1.0
     */
    public double getConfidence()

    /**
     * Get selected template ID.
     * @return Template identifier
     */
    public String getTemplateId()

    /**
     * Get processing metadata.
     * @return Map of metadata key-value pairs
     */
    public Map<String, Object> getMetadata()

    /**
     * Get category activations from hierarchical processing.
     * @return Array of category labels [morpheme, phrase, discourse]
     */
    public String[] getCategories()

    /**
     * Get channel activations.
     * @return Map of channel name to activation values
     */
    public Map<String, float[]> getChannelActivations()
}
```

## Channel Interfaces

### Channel

Base interface for all channel implementations.

```java
package com.hellblazer.art.hartcq.core.channels;

public interface Channel {

    /**
     * Process a window of tokens through this channel.
     * @param tokens Array of tokens to process
     * @return Feature vector output
     */
    float[] processWindow(Token[] tokens);

    /**
     * Get the output dimension of this channel.
     * @return Number of output features
     */
    int getOutputDimension();

    /**
     * Get the channel name.
     * @return Channel identifier
     */
    String getName();

    /**
     * Reset channel state.
     */
    void reset();

    /**
     * Check if channel is deterministic.
     * @return true if deterministic, false if stochastic
     */
    boolean isDeterministic();

    /**
     * Get channel type.
     * @return Channel type enumeration
     */
    ChannelType getChannelType();

    /**
     * Set channel parameters.
     * @param params Map of parameter names to values
     */
    void setParameters(Map<String, Object> params);

    /**
     * Get current channel parameters.
     * @return Map of parameter names to values
     */
    Map<String, Object> getParameters();
}
```

### Channel Implementations

#### PositionalChannel
```java
public class PositionalChannel implements Channel {
    // Output dimension: 64
    // Encoding: Sinusoidal (Transformer-style)
    // Parameters: maxPosition, baseFrequency
}
```

#### WordChannel
```java
public class WordChannel implements Channel {
    // Output dimension: 128
    // Encoding: Word2Vec embeddings
    // Mode: COMPREHENSION_ONLY (no generation)
    // Parameters: embeddingDim, vocabularySize
}
```

#### ContextChannel
```java
public class ContextChannel implements Channel {
    // Output dimension: 40
    // Features: Historical context with momentum
    // Parameters: historySize, decayFactor, momentum
}
```

#### StructuralChannel
```java
public class StructuralChannel implements Channel {
    // Output dimension: 56
    // Features: Grammatical patterns, POS tags
    // Parameters: maxDepth, featureTypes
}
```

#### SemanticChannel
```java
public class SemanticChannel implements Channel {
    // Output dimension: 48
    // Features: Topic coherence, entity tracking
    // Parameters: topicCount, coherenceThreshold
}
```

#### TemporalChannel
```java
public class TemporalChannel implements Channel {
    // Output dimension: 32
    // Features: Time-based patterns, decay
    // Parameters: decayRate, windowSize
}
```

Note: SyntaxChannel (32 dimensions) and PhoneticChannel (24 dimensions) are implemented but not currently active in the processing pipeline.
```

## Processing Components

### MultiChannelProcessor

Orchestrates parallel channel processing.

```java
package com.hellblazer.art.hartcq.core;

public class MultiChannelProcessor {

    /**
     * Initialize with default channels.
     */
    public MultiChannelProcessor()

    /**
     * Initialize with custom channels.
     * @param channels List of channel implementations
     */
    public MultiChannelProcessor(List<Channel> channels)

    /**
     * Process window through all channels.
     * @param window Token window to process
     * @return Combined feature vector
     */
    public float[] processWindow(Token[] window)

    /**
     * Process windows in batch.
     * @param windows List of token windows
     * @return List of feature vectors
     */
    public List<float[]> processBatch(List<Token[]> windows)

    /**
     * Get total output dimension.
     * @return Sum of all channel dimensions
     */
    public int getTotalOutputDimension()

    /**
     * Enable/disable specific channel.
     * @param channelName Name of channel
     * @param enabled Enable state
     */
    public void setChannelEnabled(String channelName, boolean enabled)

    /**
     * Get channel by name.
     * @param channelName Channel identifier
     * @return Channel instance or null
     */
    public Channel getChannel(String channelName)

    /**
     * Add new channel.
     * @param channel Channel to add
     */
    public void addChannel(Channel channel)

    /**
     * Remove channel.
     * @param channelName Channel to remove
     */
    public void removeChannel(String channelName)
}
```

### HierarchicalProcessor

Manages DeepARTMAP hierarchical categorization.

```java
package com.hellblazer.art.hartcq.hierarchical;

public class HierarchicalProcessor {

    /**
     * Initialize with default vigilance levels.
     */
    public HierarchicalProcessor()

    /**
     * Initialize with custom vigilance levels.
     * @param vigilanceLevels Array of vigilance parameters
     */
    public HierarchicalProcessor(double[] vigilanceLevels)

    /**
     * Process features through hierarchy.
     * @param features Input feature vector
     * @return Category labels for each level
     */
    public String[] process(float[] features)

    /**
     * Set vigilance for specific level.
     * @param level Hierarchy level (0-2)
     * @param vigilance Vigilance parameter value
     */
    public void setVigilance(int level, double vigilance)

    /**
     * Get current vigilance parameters.
     * @return Array of vigilance values
     */
    public double[] getVigilanceLevels()

    /**
     * Get category count at level.
     * @param level Hierarchy level
     * @return Number of categories
     */
    public int getCategoryCount(int level)

    /**
     * Reset all categories.
     */
    public void reset()

    /**
     * Enable/disable learning.
     * @param enabled Learning state
     */
    public void setLearningEnabled(boolean enabled)
}
```

### Tokenizer

Text tokenization with edge case handling.

```java
package com.hellblazer.art.hartcq.core;

public class Tokenizer {

    /**
     * Tokenize input text.
     * @param text Input text
     * @return Array of tokens
     */
    public Token[] tokenize(String text)

    /**
     * Tokenize with custom delimiters.
     * @param text Input text
     * @param delimiters Delimiter pattern
     * @return Array of tokens
     */
    public Token[] tokenize(String text, String delimiters)

    /**
     * Set tokenization rules.
     * @param rules Tokenization configuration
     */
    public void setRules(TokenizationRules rules)

    /**
     * Handle special tokens.
     * @param handlers Map of token type to handler
     */
    public void setSpecialHandlers(Map<TokenType, TokenHandler> handlers)
}
```

## Template System

### Template

Template structure for deterministic output generation.

```java
package com.hellblazer.art.hartcq.spatial;

public class Template {

    /**
     * Get template ID.
     * @return Unique identifier
     */
    public String getId()

    /**
     * Get template pattern.
     * @return Template string with placeholders
     */
    public String getPattern()

    /**
     * Get template type.
     * @return Type enumeration
     */
    public TemplateType getType()

    /**
     * Get template slots.
     * @return List of slot definitions
     */
    public List<Slot> getSlots()

    /**
     * Fill template with values.
     * @param values Map of slot names to values
     * @return Filled template string
     */
    public String fill(Map<String, String> values)

    /**
     * Validate slot values.
     * @param values Map of slot names to values
     * @return Validation result
     */
    public ValidationResult validate(Map<String, String> values)

    /**
     * Get required slots.
     * @return List of required slot names
     */
    public List<String> getRequiredSlots()

    /**
     * Get optional slots.
     * @return List of optional slot names
     */
    public List<String> getOptionalSlots()
}
```

### TemplateLibrary

Manages template collection and selection.

```java
package com.hellblazer.art.hartcq.spatial;

public class TemplateLibrary {

    /**
     * Initialize with default templates.
     */
    public TemplateLibrary()

    /**
     * Load templates from file.
     * @param path Path to template file
     */
    public void loadTemplates(String path)

    /**
     * Add template to library.
     * @param template Template to add
     */
    public void addTemplate(Template template)

    /**
     * Get template by ID.
     * @param id Template identifier
     * @return Template or null
     */
    public Template getTemplate(String id)

    /**
     * Get templates by type.
     * @param type Template type
     * @return List of matching templates
     */
    public List<Template> getTemplatesByType(TemplateType type)

    /**
     * Select best template for categories.
     * @param categories Category labels
     * @return Best matching template
     */
    public Template selectTemplate(String[] categories)

    /**
     * Get template count.
     * @return Number of templates
     */
    public int getTemplateCount()

    /**
     * Clear all templates.
     */
    public void clear()
}
```

## Competitive Queue

### CompetitiveQueue

Grossberg dynamics for template selection.

```java
package com.hellblazer.art.hartcq.core;

public class CompetitiveQueue {

    /**
     * Initialize with default parameters.
     */
    public CompetitiveQueue()

    /**
     * Initialize with custom parameters.
     * @param params Grossberg dynamics parameters
     */
    public CompetitiveQueue(GrossbergParameters params)

    /**
     * Select winning template.
     * @param categories Input categories
     * @return Selected template
     */
    public Template select(String[] categories)

    /**
     * Select top-k templates.
     * @param categories Input categories
     * @param k Number of winners
     * @return List of top templates
     */
    public List<Template> selectTopK(String[] categories, int k)

    /**
     * Update activation for item.
     * @param itemId Item identifier
     * @param activation New activation value
     */
    public void updateActivation(String itemId, double activation)

    /**
     * Get current activations.
     * @return Map of item IDs to activations
     */
    public Map<String, Double> getActivations()

    /**
     * Set self-excitation parameter.
     * @param value Self-excitation strength
     */
    public void setSelfExcitation(double value)

    /**
     * Set lateral inhibition parameter.
     * @param value Inhibition strength
     */
    public void setLateralInhibition(double value)

    /**
     * Set primacy gradient.
     * @param value Primacy factor
     */
    public void setPrimacyGradient(double value)

    /**
     * Reset queue state.
     */
    public void reset()

    /**
     * Run dynamics for specified iterations.
     * @param iterations Number of iterations
     */
    public void runDynamics(int iterations)
}
```

### GrossbergParameters

Configuration for Grossberg dynamics.

```java
package com.hellblazer.art.hartcq.core;

public class GrossbergParameters {

    public static final double DEFAULT_SELF_EXCITATION = 1.2;
    public static final double DEFAULT_LATERAL_INHIBITION = 0.3;
    public static final double DEFAULT_PRIMACY_GRADIENT = 0.95;
    public static final double DEFAULT_DECAY = 0.1;
    public static final double DEFAULT_UPPER_BOUND = 1.0;

    private double selfExcitation;
    private double lateralInhibition;
    private double primacyGradient;
    private double decay;
    private double upperBound;

    // Getters and setters for all parameters

    /**
     * Create default parameters.
     * @return Default configuration
     */
    public static GrossbergParameters defaults()

    /**
     * Create parameters for winner-take-all.
     * @return WTA configuration
     */
    public static GrossbergParameters winnerTakeAll()

    /**
     * Create parameters for k-winners.
     * @param k Number of winners
     * @return k-WTA configuration
     */
    public static GrossbergParameters kWinners(int k)
}
```

## Feedback Control

### FeedbackController

Manages adaptive feedback and error correction.

```java
package com.hellblazer.art.hartcq.feedback;

public class FeedbackController {

    /**
     * Initialize feedback controller.
     */
    public FeedbackController()

    /**
     * Process feedback signal.
     * @param expected Expected output
     * @param actual Actual output
     * @return Correction signal
     */
    public CorrectionSignal processFeedback(String expected, String actual)

    /**
     * Apply correction to system.
     * @param signal Correction signal
     */
    public void applyCorrection(CorrectionSignal signal)

    /**
     * Set learning rate.
     * @param rate Learning rate value
     */
    public void setLearningRate(double rate)

    /**
     * Get error history.
     * @return List of recent errors
     */
    public List<ErrorRecord> getErrorHistory()

    /**
     * Clear error history.
     */
    public void clearHistory()

    /**
     * Enable/disable feedback.
     * @param enabled Feedback state
     */
    public void setEnabled(boolean enabled)
}
```

## Performance Monitoring

### PerformanceStats

Performance statistics tracking.

```java
package com.hellblazer.art.hartcq;

public class PerformanceStats {

    /**
     * Get total sentences processed.
     * @return Sentence count
     */
    public long getTotalSentencesProcessed()

    /**
     * Get average processing time.
     * @return Average duration per sentence
     */
    public Duration getAverageProcessingTime()

    /**
     * Get throughput.
     * @return Sentences per second
     */
    public double getThroughput()

    /**
     * Get minimum processing time.
     * @return Fastest processing duration
     */
    public Duration getMinProcessingTime()

    /**
     * Get maximum processing time.
     * @return Slowest processing duration
     */
    public Duration getMaxProcessingTime()

    /**
     * Get percentile latency.
     * @param percentile Percentile value (0-100)
     * @return Latency at percentile
     */
    public Duration getPercentileLatency(double percentile)

    /**
     * Get category statistics.
     * @return Map of category to usage count
     */
    public Map<String, Long> getCategoryStats()

    /**
     * Get template usage statistics.
     * @return Map of template ID to usage count
     */
    public Map<String, Long> getTemplateStats()

    /**
     * Get channel performance.
     * @return Map of channel to average processing time
     */
    public Map<String, Duration> getChannelPerformance()

    /**
     * Export statistics to JSON.
     * @return JSON representation
     */
    public String toJson()

    /**
     * Reset all statistics.
     */
    public void reset()
}
```

## Configuration

### HARTCQConfig

Main configuration class.

```java
package com.hellblazer.art.hartcq;

public class HARTCQConfig {

    /**
     * Create default configuration.
     */
    public HARTCQConfig()

    /**
     * Load configuration from properties.
     * @param properties Properties object
     */
    public HARTCQConfig(Properties properties)

    /**
     * Load configuration from file.
     * @param path Path to config file
     */
    public static HARTCQConfig fromFile(String path)

    // Window configuration
    public int getWindowSize()
    public void setWindowSize(int size)
    public int getWindowOverlap()
    public void setWindowOverlap(int overlap)

    // Channel configuration
    public boolean isChannelEnabled(String channelName)
    public void setChannelEnabled(String channelName, boolean enabled)
    public Map<String, Map<String, Object>> getChannelParameters()

    // Hierarchy configuration
    public double[] getVigilanceLevels()
    public void setVigilanceLevels(double[] levels)
    public boolean isLearningEnabled()
    public void setLearningEnabled(boolean enabled)

    // Grossberg configuration
    public GrossbergParameters getGrossbergParameters()
    public void setGrossbergParameters(GrossbergParameters params)

    // Performance configuration
    public int getBatchSize()
    public void setBatchSize(int size)
    public int getThreadPoolSize()
    public void setThreadPoolSize(int size)
    public boolean isCachingEnabled()
    public void setCachingEnabled(boolean enabled)

    // Template configuration
    public String getTemplatePath()
    public void setTemplatePath(String path)
    public int getMaxTemplateCount()
    public void setMaxTemplateCount(int count)

    /**
     * Validate configuration.
     * @return Validation result with any errors
     */
    public ValidationResult validate()

    /**
     * Export configuration to properties.
     * @return Properties representation
     */
    public Properties toProperties()
}
```

### ConfigurationBuilder

Fluent builder for configuration.

```java
package com.hellblazer.art.hartcq;

public class ConfigurationBuilder {

    /**
     * Start building configuration.
     * @return New builder instance
     */
    public static ConfigurationBuilder create()

    /**
     * Set window parameters.
     */
    public ConfigurationBuilder withWindow(int size, int overlap)

    /**
     * Enable specific channels.
     */
    public ConfigurationBuilder withChannels(String... channelNames)

    /**
     * Set vigilance levels.
     */
    public ConfigurationBuilder withVigilance(double... levels)

    /**
     * Set Grossberg parameters.
     */
    public ConfigurationBuilder withGrossbergDynamics(double selfExcitation,
                                                     double lateralInhibition)

    /**
     * Set performance parameters.
     */
    public ConfigurationBuilder withPerformance(int batchSize, int threads)

    /**
     * Enable caching.
     */
    public ConfigurationBuilder withCaching()

    /**
     * Set template path.
     */
    public ConfigurationBuilder withTemplates(String path)

    /**
     * Build configuration.
     * @return Configuration instance
     */
    public HARTCQConfig build()
}
```

## Exception Handling

### HARTCQException

Base exception for HART-CQ errors.

```java
package com.hellblazer.art.hartcq;

public class HARTCQException extends Exception {

    public HARTCQException(String message)
    public HARTCQException(String message, Throwable cause)

    /**
     * Get error code.
     * @return Error code enumeration
     */
    public ErrorCode getErrorCode()

    /**
     * Get error context.
     * @return Map of context values
     */
    public Map<String, Object> getContext()
}
```

### ProcessingException

Exception during processing pipeline.

```java
package com.hellblazer.art.hartcq;

public class ProcessingException extends HARTCQException {

    /**
     * Get pipeline stage where error occurred.
     * @return Pipeline stage enumeration
     */
    public PipelineStage getStage()

    /**
     * Get partial result if available.
     * @return Partial processing result or null
     */
    public ProcessingResult getPartialResult()
}
```

---

**Document Version**: 1.0
**Last Updated**: September 14, 2025
**API Stability**: Stable (v0.0.1-SNAPSHOT)