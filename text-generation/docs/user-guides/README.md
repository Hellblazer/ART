# User Guide

## Getting Started with ART Text Generation

This guide covers everything you need to know to use the ART Cognitive Architecture for text generation.

## 🚀 **Quick Start**

### Prerequisites
- Java 24+
- Maven 3.9.1+  
- 4GB RAM recommended
- macOS ARM64 (LWJGL configured for Apple Silicon)

### 1. Build the System
```bash
cd /Users/hal.hildebrand/git/ART/text-generation
mvn clean install
```

### 2. Run the Interactive Application
```bash
mvn exec:java -Dexec.mainClass="com.art.textgen.ARTTextGenerationApp"
```

This launches an interactive menu with options for:
- Training on sample corpus
- Training on custom corpus
- Text generation
- Performance benchmarking

### 3. Basic Text Generation
```java
// Initialize the cognitive architecture
Vocabulary vocabulary = new Vocabulary(64);
EnhancedPatternGenerator generator = new EnhancedPatternGenerator(vocabulary);

// Simple generation
String result = generator.generate("The future of AI", 50);
System.out.println(result);
```

## 📖 **Training the System**

### Using Built-in Corpus
The system comes with a comprehensive corpus downloader:
```bash
mvn exec:java -Dexec.mainClass="com.art.textgen.training.CorpusDownloader"
```

This downloads:
- Project Gutenberg classics
- Technical documentation  
- Philosophy texts
- Scientific articles
- News articles
- Conversation samples

### Training from Custom Corpus
```java
// Setup training pipeline
TrainingPipeline pipeline = new TrainingPipeline(vocabulary, generator);

// Train from directory of text files
pipeline.trainFromDirectory("path/to/your/corpus");

// Or train from individual files
pipeline.trainFromFile("document.txt");
```

### Supported File Formats
- `.txt` - Plain text files
- `.md` - Markdown files
- `.json` - JSON documents (text fields extracted)

### Incremental Training
```java
// Initial training
pipeline.trainFromSamples();

// Add new documents without forgetting previous learning
pipeline.trainFromFile("new_document.txt");  // No catastrophic forgetting!

// Continue adding documents
pipeline.trainFromDirectory("additional_corpus");
```

## 🎛️ **Configuration Options**

### Generation Modes
The system provides pre-configured modes for different use cases:

```java
// Conservative - focused, deterministic output
generator.configureMode(GenerationMode.CONSERVATIVE);

// Balanced - good creativity/coherence balance  
generator.configureMode(GenerationMode.BALANCED);

// Creative - high diversity, surprising output
generator.configureMode(GenerationMode.CREATIVE);

// Precise - very focused, minimal randomness
generator.configureMode(GenerationMode.PRECISE);
```

### Fine-grained Control
```java
// Manual parameter tuning
generator.setTemperature(0.8);        // 0.1-2.0, higher = more creative
generator.setTopK(40);                // 10-50, smaller = more focused  
generator.setTopP(0.9);               // 0.3-0.95, smaller = more focused
generator.setRepetitionPenalty(1.2);  // 1.0-2.0, higher = less repetition
```

### Memory Configuration
```java
// Adjust hierarchical memory
RecursiveHierarchicalMemory memory = new RecursiveHierarchicalMemory(5); // 5 levels
memory.setCompressionRatio(10.0);  // Higher = more compression

// Configure multi-timescale processing
MultiTimescaleMemoryBank bank = new MultiTimescaleMemoryBank();
bank.setTimescales(new double[]{0.1, 1.0, 10.0, 60.0, 600.0}); // Custom time constants
```

## 💻 **API Reference**

### Core Classes

#### `EnhancedPatternGenerator`
Main text generation class with advanced sampling.

```java
// Construction
EnhancedPatternGenerator(Vocabulary vocab)
EnhancedPatternGenerator(Vocabulary vocab, double temperature)

// Primary generation methods
String generate(String prompt, int maxTokens)
String generateNext(List<String> context)
GenerationResult generateWithMetrics(String prompt, int length)

// Configuration
void configureMode(GenerationMode mode)
void setSamplingConfig(SamplingStrategies.SamplingConfig config)
Map<String, Object> getGenerationStats()
```

#### `TrainingPipeline`
Orchestrates the training process with no catastrophic forgetting.

```java
// Construction
TrainingPipeline(Vocabulary vocab, PatternGenerator generator)

// Training methods
void trainFromSamples()
void trainFromDirectory(String path)  
void trainFromFile(String filename)
TrainingMetrics getTrainingMetrics()
```

#### `RecursiveHierarchicalMemory`
Implements the 7±2 hierarchical memory system.

```java
// Construction  
RecursiveHierarchicalMemory()           // 5 levels default
RecursiveHierarchicalMemory(int levels) // Custom levels

// Memory operations
void addToken(Object token)
List<Object> getActiveContext(int queryDepth)
double getEffectiveCapacity()           // ~20,000 tokens with 5 levels
```

### Generation Result Analysis
```java
GenerationResult result = generator.generateWithMetrics(prompt, 100);

// Access generated text
String text = result.getFullText();

// Access performance metrics  
Map<String, Object> metrics = result.metrics;
System.out.println("Generation time: " + metrics.get("generation_time_ms"));
System.out.println("Tokens per second: " + metrics.get("tokens_per_second"));
System.out.println("Diversity: " + metrics.get("diversity"));
```

## 🔧 **Advanced Usage**

### Beam Search Generation
For higher quality output at the cost of speed:
```java
List<String> context = Arrays.asList("The", "future", "of");
List<String> beamResult = generator.generateWithBeamSearch(
    context, 
    50,     // length
    5       // beam width
);
```

### Custom Sampling Strategies
```java
SamplingStrategies.SamplingConfig config = new SamplingStrategies.SamplingConfig();
config.temperature = 0.9;
config.topK = 30;
config.topP = 0.85;
config.repetitionPenalty = 1.3;
config.adaptiveTemp = true;  // Dynamic temperature adjustment

generator.setSamplingConfig(config);
```

### Memory Introspection
```java
// Examine hierarchical memory state
RecursiveHierarchicalMemory memory = textGen.getHierarchicalMemory();
double capacity = memory.getEffectiveCapacity();
List<Object> activeContext = memory.getActiveContext(1000);

// Multi-timescale memory analysis
MultiTimescaleMemoryBank bank = textGen.getMemoryBank();
for (String scale : bank.getTimescales()) {
    WorkingMemory scaleMemory = bank.getMemory(scale);
    System.out.println(scale + ": " + scaleMemory.size() + " items");
}
```

## 🎯 **Use Case Examples**

### Creative Writing Assistant
```java
generator.configureMode(GenerationMode.CREATIVE);
generator.setTemperature(1.3);
String story = generator.generate("In a world where AI has emotions", 200);
```

### Technical Documentation Generation
```java
generator.configureMode(GenerationMode.PRECISE);
generator.setTemperature(0.6);
String docs = generator.generate("The API endpoint accepts parameters", 150);
```

### Conversational Text Completion
```java
generator.configureMode(GenerationMode.BALANCED);
String response = generator.generate("User: What do you think about climate change?", 100);
```

### Code Comment Generation
```java  
generator.configureMode(GenerationMode.CONSERVATIVE);
generator.setTopK(20);
String comment = generator.generate("This function calculates the", 50);
```

## 📊 **Monitoring & Debugging**

### Performance Monitoring
```java
// Get detailed generation statistics
Map<String, Object> stats = generator.getGenerationStats();
System.out.println("History size: " + stats.get("historySize"));
System.out.println("Current temperature: " + stats.get("temperature"));

// Training metrics
TrainingMetrics metrics = pipeline.getTrainingMetrics();
System.out.println("Patterns learned: " + metrics.getTotalPatterns());
System.out.println("Training time: " + metrics.getTrainingTimeMs() + "ms");
```

### Debugging Generation
```java
// Enable detailed logging for generation process
Logger logger = LoggerFactory.getLogger(EnhancedPatternGenerator.class);
logger.setLevel(Level.DEBUG);

// Generate with detailed trace
GenerationResult result = generator.generateWithMetrics(prompt, length);
System.out.println(result.toString()); // Includes full metrics
```

## ⚙️ **Performance Tuning**

### For Speed
```java
generator.configureMode(GenerationMode.CONSERVATIVE);
generator.setTopK(10);              // Small search space
generator.setRepetitionPenalty(1.0); // Disable penalty computation
```

### For Quality
```java
generator.configureMode(GenerationMode.PRECISE);
// Use beam search for best results (slower)
List<String> highQuality = generator.generateWithBeamSearch(context, length, 10);
```

### For Creativity
```java
generator.configureMode(GenerationMode.CREATIVE);
generator.setTemperature(1.5);      // High randomness
generator.setTopP(0.95);            // Large vocabulary
```

### Memory Optimization
```java
// Reduce hierarchy levels for lower memory usage
RecursiveHierarchicalMemory memory = new RecursiveHierarchicalMemory(3); // vs default 5

// Increase compression for longer contexts  
memory.setCompressionRatio(15.0);   // Higher compression ratio
```

## 🚨 **Troubleshooting**

### Common Issues

**Out of Memory Errors**
```bash
# Increase JVM heap size
export MAVEN_OPTS="-Xmx8g"
mvn exec:java -Dexec.mainClass="com.art.textgen.ARTTextGenerationApp"
```

**Slow Training**
- Reduce corpus size for initial testing
- Use sample corpus: `pipeline.trainFromSamples()`
- Check disk I/O performance

**Poor Generation Quality**
- Train on more/better data: `pipeline.trainFromDirectory("larger_corpus")`
- Adjust temperature: `generator.setTemperature(0.8)`
- Use beam search: `generateWithBeamSearch()`

**Repetitive Output**
- Increase repetition penalty: `generator.setRepetitionPenalty(1.5)`
- Use creative mode: `configureMode(GenerationMode.CREATIVE)`
- Check vocabulary coverage

### Getting Help
- Check the [Architecture Documentation](../architecture/README.md)
- Review [Performance Analysis](../performance/README.md)
- See [Original Requirements](../original-requirements/) for design rationale
- File issues on the main ART project repository

This user guide provides everything needed to effectively use the ART Cognitive Architecture for practical text generation applications.