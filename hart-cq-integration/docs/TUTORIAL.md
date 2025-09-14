# HART-CQ Tutorial & Examples

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Real-World Examples](#real-world-examples)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

#### Prerequisites
- Java 24 or higher
- Maven 3.9.1+
- 512MB RAM minimum (2GB recommended for production)

#### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ART.git
cd ART

# Build all modules
mvn clean install

# Run tests to verify installation
mvn test -pl hart-cq-integration
```

#### Adding as Dependency

```xml
<dependency>
    <groupId>com.hellblazer.art</groupId>
    <artifactId>hart-cq-integration</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>
```

### First Program

```java
import com.hellblazer.art.hartcq.integration.HARTCQ;
import com.hellblazer.art.hartcq.ProcessingResult;

public class HelloHARTCQ {
    public static void main(String[] args) {
        // Initialize HART-CQ
        try (var hartcq = new HARTCQ()) {

            // Process a simple sentence
            var result = hartcq.process("Hello, world!");

            // Print the output
            System.out.println("Input: " + result.getInput());
            System.out.println("Output: " + result.getOutput());
            System.out.println("Processing time: " + result.getProcessingTime());
            System.out.println("Confidence: " + result.getConfidence());
        }
    }
}
```

## Basic Usage

### Processing Single Sentences

```java
public class SingleSentenceExample {
    public static void main(String[] args) {
        var hartcq = new HARTCQ();

        // Question processing
        var question = "What is the weather today?";
        var result = hartcq.process(question);
        System.out.println("Q: " + question);
        System.out.println("A: " + result.getOutput());

        // Statement processing
        var statement = "The cat sat on the mat.";
        result = hartcq.process(statement);
        System.out.println("Statement: " + statement);
        System.out.println("Analysis: " + result.getOutput());

        // Command processing
        var command = "Please open the door.";
        result = hartcq.process(command);
        System.out.println("Command: " + command);
        System.out.println("Response: " + result.getOutput());

        hartcq.shutdown();
    }
}
```

### Batch Processing

```java
public class BatchProcessingExample {
    public static void main(String[] args) {
        var hartcq = new HARTCQ();

        // Prepare batch of sentences
        var sentences = List.of(
            "The quick brown fox jumps over the lazy dog.",
            "How are you today?",
            "Machine learning is transforming technology.",
            "Please send the report by tomorrow.",
            "The meeting has been rescheduled to 3 PM."
        );

        // Process batch
        var startTime = System.nanoTime();
        var results = hartcq.processBatch(sentences);
        var totalTime = System.nanoTime() - startTime;

        // Display results
        for (int i = 0; i < results.size(); i++) {
            var result = results.get(i);
            System.out.printf("%d. Input: %s%n", i + 1, sentences.get(i));
            System.out.printf("   Output: %s%n", result.getOutput());
            System.out.printf("   Template: %s%n", result.getTemplateId());
            System.out.println();
        }

        // Processing complete
        System.out.printf("Processed %d sentences in %.2f ms%n",
                         sentences.size(), totalTime / 1_000_000.0);

        hartcq.shutdown();
    }
}
```

### Asynchronous Processing

```java
public class AsyncProcessingExample {
    public static void main(String[] args) throws Exception {
        var hartcq = new HARTCQ();
        var executor = Executors.newFixedThreadPool(4);

        // Process multiple sentences asynchronously
        var inputs = List.of(
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks.",
            "What are transformers in AI?"
        );

        // Submit async tasks
        var futures = inputs.stream()
            .map(input -> hartcq.processAsync(input))
            .toList();

        // Wait for all results
        var results = futures.stream()
            .map(future -> {
                try {
                    return future.get(1, TimeUnit.SECONDS);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            })
            .toList();

        // Display results
        for (var result : results) {
            System.out.println("Q: " + result.getInput());
            System.out.println("A: " + result.getOutput());
            System.out.println();
        }

        executor.shutdown();
        hartcq.shutdown();
    }
}
```

## Advanced Features

### Custom Configuration

```java
public class CustomConfigurationExample {
    public static void main(String[] args) {
        // Build custom configuration
        var config = ConfigurationBuilder.create()
            .withWindow(25, 7)  // Larger window with more overlap
            .withChannels("PositionalChannel", "WordChannel",
                         "SemanticChannel", "StructuralChannel")
            .withVigilance(0.95, 0.75, 0.55)  // Stricter vigilance
            .withGrossbergDynamics(1.5, 0.4)  // Stronger competition
            .withPerformance(200, 8)  // Larger batch, more threads
            .withCaching()  // Enable caching
            .build();

        // Initialize with custom config
        var hartcq = new HARTCQ(config);

        // Test with different inputs
        var inputs = List.of(
            "Complex scientific hypotheses require careful validation.",
            "Quantum computing promises exponential speedup.",
            "Climate change affects global weather patterns."
        );

        for (var input : inputs) {
            var result = hartcq.process(input);
            System.out.println("Input: " + input);
            System.out.println("Output: " + result.getOutput());
            System.out.println("Categories: " + Arrays.toString(result.getCategories()));
            System.out.println();
        }

        hartcq.shutdown();
    }
}
```

### Channel Customization

```java
public class ChannelCustomizationExample {
    public static void main(String[] args) {
        var hartcq = new HARTCQ();

        // Get multi-channel processor
        var processor = hartcq.getMultiChannelProcessor();

        // Disable specific channels for testing
        processor.setChannelEnabled("PhoneticChannel", false);
        processor.setChannelEnabled("TemporalChannel", false);

        // Configure channel parameters
        var wordChannel = processor.getChannel("WordChannel");
        wordChannel.setParameters(Map.of(
            "embeddingDim", 100,  // Increase embedding dimension
            "vocabularySize", 50000
        ));

        var contextChannel = processor.getChannel("ContextChannel");
        contextChannel.setParameters(Map.of(
            "historySize", 50,
            "decayFactor", 0.9,
            "momentum", 0.7
        ));

        // Test with different channel configurations
        var testSentences = List.of(
            "The algorithm converged after 100 iterations.",
            "Neural networks learn hierarchical representations.",
            "Backpropagation optimizes network weights."
        );

        for (var sentence : testSentences) {
            var result = hartcq.process(sentence);

            // Get channel activations
            var activations = result.getChannelActivations();
            System.out.println("Input: " + sentence);
            System.out.println("Active channels:");
            for (var entry : activations.entrySet()) {
                if (entry.getValue() != null && entry.getValue().length > 0) {
                    System.out.printf("  %s: %.3f avg activation%n",
                                    entry.getKey(),
                                    Arrays.stream(entry.getValue()).average().orElse(0));
                }
            }
            System.out.println();
        }

        hartcq.shutdown();
    }
}
```

### Training and Learning

```java
public class TrainingExample {
    public static void main(String[] args) {
        var hartcq = new HARTCQ();

        // Enable learning mode
        hartcq.setLearningEnabled(true);

        // Training data
        var trainingPairs = List.of(
            new TrainingPair("What is Java?",
                           "Java is a programming language."),
            new TrainingPair("Who invented Python?",
                           "Python was created by Guido van Rossum."),
            new TrainingPair("When was C++ released?",
                           "C++ was first released in 1985."),
            new TrainingPair("What is functional programming?",
                           "Functional programming is a paradigm based on functions."),
            new TrainingPair("How does recursion work?",
                           "Recursion involves a function calling itself.")
        );

        // Train the system
        System.out.println("Training HART-CQ...");
        for (var pair : trainingPairs) {
            hartcq.train(pair.input, pair.expectedOutput);
            System.out.println("Trained: " + pair.input);
        }

        // Disable learning for inference
        hartcq.setLearningEnabled(false);

        // Test with similar questions
        var testQuestions = List.of(
            "What is JavaScript?",
            "Who created Ruby?",
            "When was Go released?",
            "What is object-oriented programming?",
            "How does iteration work?"
        );

        System.out.println("\nTesting learned patterns:");
        for (var question : testQuestions) {
            var result = hartcq.process(question);
            System.out.println("Q: " + question);
            System.out.println("A: " + result.getOutput());
            System.out.println("Confidence: " + result.getConfidence());
            System.out.println();
        }

        hartcq.shutdown();
    }

    record TrainingPair(String input, String expectedOutput) {}
}
```

## Real-World Examples

### Chat Application Integration

```java
public class ChatBotExample {
    private final HARTCQ hartcq;
    private final Map<String, List<String>> conversationHistory;

    public ChatBotExample() {
        this.hartcq = new HARTCQ();
        this.conversationHistory = new ConcurrentHashMap<>();
    }

    public String processMessage(String userId, String message) {
        // Get user history
        var history = conversationHistory.computeIfAbsent(userId,
                                                         k -> new ArrayList<>());

        // Add context from history
        var contextualMessage = buildContext(history, message);

        // Process with HART-CQ
        var result = hartcq.process(contextualMessage);

        // Update history
        history.add(message);
        history.add(result.getOutput());

        // Keep history size manageable
        if (history.size() > 20) {
            history.subList(0, history.size() - 20).clear();
        }

        return result.getOutput();
    }

    private String buildContext(List<String> history, String message) {
        if (history.isEmpty()) {
            return message;
        }

        // Include last 2 exchanges for context
        var contextSize = Math.min(4, history.size());
        var context = String.join(" ",
                                 history.subList(history.size() - contextSize,
                                               history.size()));
        return context + " " + message;
    }

    public void shutdown() {
        hartcq.shutdown();
    }

    public static void main(String[] args) {
        var chatBot = new ChatBotExample();

        // Simulate conversation
        var userId = "user123";
        var messages = List.of(
            "Hello!",
            "What's the weather like?",
            "Tell me about machine learning.",
            "How does it work?",
            "Thanks for the explanation!"
        );

        for (var message : messages) {
            System.out.println("User: " + message);
            var response = chatBot.processMessage(userId, message);
            System.out.println("Bot: " + response);
            System.out.println();
        }

        chatBot.shutdown();
    }
}
```

### Stream Processing Pipeline

```java
public class StreamProcessingExample {
    public static void main(String[] args) throws Exception {
        var hartcq = new HARTCQ();

        // Create processing pipeline
        var pipeline = ProcessingPipeline.builder()
            .source(KafkaSource.fromTopic("input-texts"))
            .processor(text -> hartcq.process(text))
            .filter(result -> result.getConfidence() > 0.8)
            .transform(result -> new EnrichedResult(
                result.getOutput(),
                result.getTemplateId(),
                result.getConfidence(),
                Instant.now()
            ))
            .sink(KafkaSink.toTopic("processed-texts"))
            .build();

        // Start pipeline
        pipeline.start();

        // Monitor performance
        var monitor = new PerformanceMonitor(hartcq);
        monitor.startMonitoring(Duration.ofSeconds(10));

        // Run for some time
        Thread.sleep(60_000);

        // Shutdown
        pipeline.stop();
        monitor.stopMonitoring();
        hartcq.shutdown();

        // Print final stats
        var stats = hartcq.getStats();
        System.out.println("Final Statistics:");
        System.out.println("Total processed: " + stats.getTotalSentencesProcessed());
        System.out.println("Average throughput: " + stats.getThroughput());
        System.out.println("P99 latency: " + stats.getPercentileLatency(99));
    }

    record EnrichedResult(String output, String templateId,
                         double confidence, Instant timestamp) {}
}

class PerformanceMonitor {
    private final HARTCQ hartcq;
    private ScheduledExecutorService scheduler;

    public PerformanceMonitor(HARTCQ hartcq) {
        this.hartcq = hartcq;
    }

    public void startMonitoring(Duration interval) {
        scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(this::printStats, 0,
                                     interval.toSeconds(), TimeUnit.SECONDS);
    }

    private void printStats() {
        var stats = hartcq.getStats();
        System.out.printf("[%s] Throughput: %.0f/s, Avg latency: %.2f ms%n",
                         Instant.now(),
                         stats.getThroughput(),
                         stats.getAverageProcessingTime().toNanos() / 1_000_000.0);
    }

    public void stopMonitoring() {
        if (scheduler != null) {
            scheduler.shutdown();
        }
    }
}
```

### REST API Service

```java
@RestController
@RequestMapping("/api/hartcq")
public class HARTCQController {

    private final HARTCQ hartcq;

    public HARTCQController() {
        var config = ConfigurationBuilder.create()
            .withPerformance(100, 8)
            .withCaching()
            .build();
        this.hartcq = new HARTCQ(config);
    }

    @PostMapping("/process")
    public ProcessingResponse process(@RequestBody ProcessingRequest request) {
        var result = hartcq.process(request.text());

        return new ProcessingResponse(
            result.getOutput(),
            result.getConfidence(),
            result.getProcessingTime().toMillis(),
            result.getTemplateId()
        );
    }

    @PostMapping("/batch")
    public List<ProcessingResponse> processBatch(@RequestBody BatchRequest request) {
        var results = hartcq.processBatch(request.texts());

        return results.stream()
            .map(result -> new ProcessingResponse(
                result.getOutput(),
                result.getConfidence(),
                result.getProcessingTime().toMillis(),
                result.getTemplateId()
            ))
            .toList();
    }

    @GetMapping("/stats")
    public StatsResponse getStats() {
        var stats = hartcq.getStats();

        return new StatsResponse(
            stats.getTotalSentencesProcessed(),
            stats.getThroughput(),
            stats.getAverageProcessingTime().toMillis(),
            stats.getPercentileLatency(50).toMillis(),
            stats.getPercentileLatency(95).toMillis(),
            stats.getPercentileLatency(99).toMillis()
        );
    }

    @PostMapping("/train")
    public void train(@RequestBody TrainingRequest request) {
        hartcq.train(request.input(), request.expectedOutput());
    }

    @PreDestroy
    public void shutdown() {
        hartcq.shutdown();
    }

    record ProcessingRequest(String text) {}
    record BatchRequest(List<String> texts) {}
    record TrainingRequest(String input, String expectedOutput) {}
    record ProcessingResponse(String output, double confidence,
                             long processingTimeMs, String templateId) {}
    record StatsResponse(long totalProcessed, double throughput,
                        long avgLatencyMs, long p50Ms, long p95Ms, long p99Ms) {}
}
```

## Configuration

### Configuration Examples

```java
public class ConfigurationExample {
    public static void main(String[] args) {
        // Custom configuration
        var config = ConfigurationBuilder.create()
            .withWindow(15, 3)  // Smaller window for speed
            .withChannels("PositionalChannel", "WordChannel", "SemanticChannel")
            .withVigilance(0.8, 0.6, 0.4)  // Relaxed vigilance
            .withPerformance(100, 8)  // Batch size and thread pool
            .withCaching()  // Enable all caching
            .build();

        var hartcq = new HARTCQ(config);

        // Prepare large dataset
        var sentences = generateTestSentences(10_000);

        // Warm up JVM
        System.out.println("Warming up...");
        for (int i = 0; i < 100; i++) {
            hartcq.processBatch(sentences.subList(0, 100));
        }
        hartcq.resetStats();

        // Benchmark
        System.out.println("Starting benchmark...");
        var startTime = System.nanoTime();

        // Process in optimal batch sizes
        var batchSize = 500;
        for (int i = 0; i < sentences.size(); i += batchSize) {
            var batch = sentences.subList(i,
                                         Math.min(i + batchSize, sentences.size()));
            hartcq.processBatch(batch);
        }

        var totalTime = System.nanoTime() - startTime;

        // Results
        var stats = hartcq.getStats();
        System.out.printf("Processed %d sentences in %.2f seconds%n",
                         sentences.size(), totalTime / 1_000_000_000.0);
        // Stats available through getStats() method
        System.out.printf("Average latency: %.3f ms%n",
                         stats.getAverageProcessingTime().toNanos() / 1_000_000.0);
        System.out.printf("P99 latency: %.3f ms%n",
                         stats.getPercentileLatency(99).toNanos() / 1_000_000.0);

        hartcq.shutdown();
    }

    private static List<String> generateTestSentences(int count) {
        var templates = List.of(
            "The %s %s the %s.",
            "How does %s affect %s?",
            "%s is important for %s.",
            "Please %s the %s.",
            "When will %s be %s?"
        );

        var subjects = List.of("system", "algorithm", "network", "database", "model");
        var verbs = List.of("processes", "analyzes", "optimizes", "validates", "transforms");
        var objects = List.of("data", "requests", "patterns", "results", "connections");

        var random = new Random(42);
        var sentences = new ArrayList<String>(count);

        for (int i = 0; i < count; i++) {
            var template = templates.get(random.nextInt(templates.size()));
            var subject = subjects.get(random.nextInt(subjects.size()));
            var verb = verbs.get(random.nextInt(verbs.size()));
            var object = objects.get(random.nextInt(objects.size()));

            sentences.add(String.format(template, subject, verb, object));
        }

        return sentences;
    }
}
```

### Memory Management

```java
public class MemoryManagementExample {
    public static void main(String[] args) {
        // Configure for memory efficiency
        var config = new HARTCQConfig();
        config.setWindowSize(10);  // Smaller windows
        config.setBatchSize(50);   // Moderate batch size
        config.setCachingEnabled(false);  // Disable caching if memory constrained

        var hartcq = new HARTCQ(config);

        // Monitor memory usage
        var runtime = Runtime.getRuntime();
        var initialMemory = runtime.totalMemory() - runtime.freeMemory();

        // Process with periodic cleanup
        var sentences = generateTestSentences(1000);
        var processedCount = 0;

        for (int i = 0; i < sentences.size(); i += 50) {
            var batch = sentences.subList(i, Math.min(i + 50, sentences.size()));
            hartcq.processBatch(batch);
            processedCount += batch.size();

            // Periodic cleanup
            if (processedCount % 500 == 0) {
                hartcq.resetStats();  // Clear statistics
                System.gc();  // Suggest garbage collection

                var currentMemory = runtime.totalMemory() - runtime.freeMemory();
                var memoryUsed = (currentMemory - initialMemory) / 1_048_576.0;
                System.out.printf("Processed %d sentences, Memory delta: %.2f MB%n",
                                processedCount, memoryUsed);
            }
        }

        hartcq.shutdown();
    }

    private static List<String> generateTestSentences(int count) {
        // Simple sentence generation
        return IntStream.range(0, count)
            .mapToObj(i -> "Test sentence number " + i)
            .toList();
    }
}
```

## Troubleshooting

### Common Issues and Solutions

#### Processing Issues

**Problem**: Processing errors or unexpected results.

**Solution**:
```java
// Verify configuration
var config = ConfigurationBuilder.create()
    .withWindow(20, 5)  // Standard window configuration
    .withChannels("PositionalChannel", "WordChannel", "ContextChannel")
    .withPerformance(100, Runtime.getRuntime().availableProcessors())
    .build();
```

#### High Memory Usage

**Problem**: Application uses too much memory.

**Solution**:
```java
// Memory-efficient configuration
var config = new HARTCQConfig();
config.setCachingEnabled(false);
config.setBatchSize(25);  // Smaller batches
config.setThreadPoolSize(4);  // Fewer threads

// Periodic cleanup
hartcq.resetStats();  // Clear accumulated stats
```

#### Non-Deterministic Results

**Problem**: Same input produces different outputs.

**Solution**:
```java
// Ensure deterministic mode
var config = new HARTCQConfig();
config.setDeterministicMode(true);
config.setSeed(42);  // Fixed random seed

// Disable learning during inference
hartcq.setLearningEnabled(false);
```

#### Template Selection Issues

**Problem**: Wrong templates being selected.

**Solution**:
```java
// Adjust Grossberg dynamics
var params = new GrossbergParameters();
params.setSelfExcitation(1.5);  // Stronger self-excitation
params.setLateralInhibition(0.5);  // Stronger competition
config.setGrossbergParameters(params);

// Adjust vigilance for better matching
config.setVigilanceLevels(new double[]{0.95, 0.8, 0.6});
```

### Debugging Tips

```java
public class DebuggingExample {
    public static void main(String[] args) {
        // Enable debug logging
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");

        var hartcq = new HARTCQ();

        // Process with detailed inspection
        var input = "Complex sentence for debugging.";
        var result = hartcq.process(input);

        // Inspect all aspects
        System.out.println("=== Debug Information ===");
        System.out.println("Input: " + input);
        System.out.println("Output: " + result.getOutput());
        System.out.println("Template ID: " + result.getTemplateId());
        System.out.println("Confidence: " + result.getConfidence());
        System.out.println("Processing time: " + result.getProcessingTime());
        System.out.println("Categories: " + Arrays.toString(result.getCategories()));

        // Channel activations
        System.out.println("\nChannel Activations:");
        var activations = result.getChannelActivations();
        for (var entry : activations.entrySet()) {
            System.out.printf("  %s: %d dimensions%n",
                            entry.getKey(),
                            entry.getValue().length);
        }

        // Metadata
        System.out.println("\nMetadata:");
        for (var entry : result.getMetadata().entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }

        hartcq.shutdown();
    }
}
```

---

**Document Version**: 1.0
**Last Updated**: September 14, 2025
**Tutorial Level**: Beginner to Advanced