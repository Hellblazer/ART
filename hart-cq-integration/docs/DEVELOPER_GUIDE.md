# HART-CQ Developer Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Environment](#development-environment)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Contributing Code](#contributing-code)
6. [Testing Guidelines](#testing-guidelines)
7. [Performance Guidelines](#performance-guidelines)
8. [Debugging & Troubleshooting](#debugging--troubleshooting)
9. [Release Process](#release-process)

## Getting Started

### Prerequisites

- **Java 24+** (with preview features enabled)
- **Maven 3.9.1+**
- **Git 2.30+**
- **IDE**: IntelliJ IDEA 2024.2+ or VS Code with Java extensions
- **Hardware**: 8GB RAM minimum, 16GB recommended

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ART.git
cd ART

# Set Java 24
export JAVA_HOME=/path/to/java24
export PATH=$JAVA_HOME/bin:$PATH

# Verify environment
java -version  # Should show Java 24
mvn -version   # Should show Maven 3.9.1+

# Build the project
mvn clean install

# Run tests
mvn test

# Run specific module tests
mvn test -pl hart-cq-core
mvn test -pl hart-cq-integration
```

### IDE Configuration

#### IntelliJ IDEA

1. Open project as Maven project
2. Set Project SDK to Java 24
3. Enable preview features: Settings → Build → Compiler → Java Compiler
4. Add VM options: `--enable-preview`
5. Configure code style: Import `config/intellij-codestyle.xml`

#### VS Code

1. Install Java Extension Pack
2. Configure `settings.json`:
```json
{
    "java.configuration.runtimes": [{
        "name": "JavaSE-24",
        "path": "/path/to/java24",
        "default": true
    }],
    "java.compile.args": ["--enable-preview"],
    "maven.executable.path": "/path/to/maven/bin/mvn"
}
```

## Project Structure

### Module Organization

```
ART/
├── hart-cq-core/              # Core algorithms and channels
│   ├── src/main/java/
│   │   └── com/hellblazer/art/hartcq/core/
│   │       ├── channels/      # Channel implementations
│   │       ├── gpu/           # GPU acceleration
│   │       └── *.java         # Core classes
│   └── src/test/
├── hart-cq-hierarchical/      # Hierarchical processing
│   └── src/main/java/
│       └── com/hellblazer/art/hartcq/hierarchical/
├── hart-cq-feedback/          # Feedback control
│   └── src/main/java/
│       └── com/hellblazer/art/hartcq/feedback/
├── hart-cq-spatial/           # Template system
│   └── src/main/java/
│       └── com/hellblazer/art/hartcq/spatial/
├── hart-cq-integration/       # Main integration
│   ├── src/main/java/
│   │   └── com/hellblazer/art/hartcq/integration/
│   ├── docs/                  # Documentation
│   └── benchmarks/            # JMH benchmarks
└── pom.xml                    # Parent POM
```

### Package Conventions

```java
// Core functionality
com.hellblazer.art.hartcq.core.*

// Channel implementations
com.hellblazer.art.hartcq.core.channels.*

// Hierarchical processing
com.hellblazer.art.hartcq.hierarchical.*

// Feedback mechanisms
com.hellblazer.art.hartcq.feedback.*

// Template and spatial
com.hellblazer.art.hartcq.spatial.*

// Integration and API
com.hellblazer.art.hartcq.integration.*
```

## Architecture Guidelines

### Design Principles

1. **Template-Based**: All output comes from predefined templates
2. **Determinism**: Same input always produces same output
3. **Performance**: Maintain sub-millisecond latency
4. **Thread Safety**: All public APIs must be thread-safe
5. **Resource Management**: Use try-with-resources and AutoCloseable

### Coding Standards

#### Java Style

```java
// Use var for local variables
var processor = new MultiChannelProcessor();
var result = processor.processWindow(tokens);

// Use records for data carriers
public record ProcessingResult(String output, double confidence,
                              Duration processingTime) {}

// Use sealed classes for type hierarchies
public sealed interface Channel
    permits PositionalChannel, WordChannel, ContextChannel {
}

// Virtual threads for concurrency
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    futures = channels.stream()
        .map(ch -> executor.submit(() -> ch.processWindow(tokens)))
        .toList();
}

// Pattern matching
switch (token.getType()) {
    case WORD -> processWord(token);
    case PUNCTUATION -> processPunctuation(token);
    case NUMBER -> processNumber(token);
    default -> throw new IllegalArgumentException("Unknown type");
}
```

#### Naming Conventions

```java
// Classes: PascalCase
public class MultiChannelProcessor {}

// Interfaces: PascalCase, no 'I' prefix
public interface Channel {}

// Methods: camelCase, verb phrases
public ProcessingResult processWindow(Token[] tokens) {}

// Constants: UPPER_SNAKE_CASE
public static final int MAX_WINDOW_SIZE = 20;

// Packages: lowercase
package com.hellblazer.art.hartcq.core.channels;
```

### Adding New Channels

#### Step 1: Implement Channel Interface

```java
package com.hellblazer.art.hartcq.core.channels;

public class CustomChannel implements Channel {
    private static final int OUTPUT_DIMENSION = 50;
    private final Map<String, Object> parameters;

    public CustomChannel() {
        this.parameters = new HashMap<>();
        initializeDefaults();
    }

    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[OUTPUT_DIMENSION];

        // Channel-specific processing
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != null) {
                // Extract features
                var features = extractFeatures(tokens[i]);
                // Update output vector
                updateOutput(output, features, i);
            }
        }

        return output;
    }

    @Override
    public int getOutputDimension() {
        return OUTPUT_DIMENSION;
    }

    @Override
    public String getName() {
        return "CustomChannel";
    }

    @Override
    public void reset() {
        // Reset any state
    }

    @Override
    public boolean isDeterministic() {
        return true;  // Or false if using randomness
    }

    @Override
    public ChannelType getChannelType() {
        return ChannelType.FEATURE_EXTRACTION;
    }

    @Override
    public void setParameters(Map<String, Object> params) {
        this.parameters.putAll(params);
    }

    @Override
    public Map<String, Object> getParameters() {
        return new HashMap<>(parameters);
    }

    private void initializeDefaults() {
        parameters.put("threshold", 0.5);
        parameters.put("windowSize", 10);
    }

    private Features extractFeatures(Token token) {
        // Implementation
    }

    private void updateOutput(float[] output, Features features, int position) {
        // Implementation
    }
}
```

#### Step 2: Add Tests

```java
package com.hellblazer.art.hartcq.core.channels;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CustomChannelTest {

    @Test
    void testProcessWindow() {
        var channel = new CustomChannel();
        var tokens = createTestTokens();

        var output = channel.processWindow(tokens);

        assertNotNull(output);
        assertEquals(50, output.length);
        // Verify specific features
    }

    @Test
    void testDeterminism() {
        var channel = new CustomChannel();
        var tokens = createTestTokens();

        var output1 = channel.processWindow(tokens);
        var output2 = channel.processWindow(tokens);

        assertArrayEquals(output1, output2, 0.001f);
    }

    @Test
    void testParameterConfiguration() {
        var channel = new CustomChannel();
        var params = Map.of("threshold", 0.8, "windowSize", 15);

        channel.setParameters(params);

        assertEquals(0.8, channel.getParameters().get("threshold"));
        assertEquals(15, channel.getParameters().get("windowSize"));
    }
}
```

#### Step 3: Register Channel

```java
// In MultiChannelProcessor.java
private void initializeChannels() {
    channels.add(new PositionalChannel());
    channels.add(new WordChannel());
    channels.add(new CustomChannel());  // Add your channel
    // ... other channels
}
```

### Adding New Templates

```java
public class CustomTemplate extends Template {

    public CustomTemplate() {
        super(
            "custom.pattern",
            "The [SUBJECT] [VERB] [OBJECT] with [MODIFIER].",
            List.of(
                new Slot("SUBJECT", SlotType.REQUIRED),
                new Slot("VERB", SlotType.REQUIRED),
                new Slot("OBJECT", SlotType.REQUIRED),
                new Slot("MODIFIER", SlotType.OPTIONAL)
            ),
            TemplateType.STATEMENT
        );
    }

    @Override
    public ValidationResult validate(Map<String, String> values) {
        // Custom validation logic
        if (!values.containsKey("SUBJECT")) {
            return ValidationResult.error("Missing required SUBJECT");
        }
        // Additional validation
        return ValidationResult.success();
    }

    @Override
    public String fill(Map<String, String> values) {
        // Custom filling logic with defaults
        var filled = super.fill(values);
        if (!values.containsKey("MODIFIER")) {
            filled = filled.replace(" with [MODIFIER]", "");
        }
        return filled;
    }
}
```

## Contributing Code

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Write tests first** (TDD approach)
4. **Implement feature**
5. **Run tests**: `mvn test`
6. **Check style**: `mvn checkstyle:check`
7. **Submit pull request**

### Pull Request Guidelines

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks run

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Commit Message Format

```
type(scope): subject

body

footer
```

Examples:
```
feat(channels): add quantum entanglement channel

Implements a new channel that uses quantum entanglement
for feature extraction. Achieves 2x performance improvement
on certain workloads.

Closes #123
```

```
fix(core): prevent NPE in token processing

Adds null check in TokenProcessor to handle empty tokens
gracefully.

Fixes #456
```

## Testing Guidelines

### Test Structure

```java
class ComponentTest {

    private Component component;

    @BeforeEach
    void setUp() {
        component = new Component();
    }

    @Test
    @DisplayName("Should process valid input correctly")
    void testValidInput() {
        // Given
        var input = createValidInput();

        // When
        var result = component.process(input);

        // Then
        assertNotNull(result);
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("Should handle null input gracefully")
    void testNullInput() {
        // Given
        String input = null;

        // When/Then
        assertThrows(IllegalArgumentException.class,
                    () -> component.process(input));
    }

    @ParameterizedTest
    @ValueSource(strings = {"test1", "test2", "test3"})
    void testMultipleInputs(String input) {
        var result = component.process(input);
        assertNotNull(result);
    }
}
```

### Test Coverage Requirements

- **Unit Tests**: 80% minimum coverage
- **Integration Tests**: All public APIs
- **Performance Tests**: Critical paths
- **Edge Cases**: Null, empty, boundary values

### Performance Testing

```java
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
public class ChannelBenchmark {

    private Channel channel;
    private Token[] tokens;

    @Setup
    public void setup() {
        channel = new CustomChannel();
        tokens = generateTokens(20);
    }

    @Benchmark
    public float[] benchmarkProcessing() {
        return channel.processWindow(tokens);
    }

    @TearDown
    public void tearDown() {
        channel.reset();
    }
}
```

Run benchmarks:
```bash
mvn clean install
java -jar benchmarks/target/benchmarks.jar
```

## Performance Guidelines

### Optimization Checklist

- [ ] Profile before optimizing
- [ ] Use appropriate data structures
- [ ] Minimize object allocation
- [ ] Leverage parallelism where beneficial
- [ ] Cache frequently accessed data
- [ ] Use primitive arrays over collections
- [ ] Avoid premature optimization

### Memory Management

```java
// Good: Reuse buffers
class ChannelProcessor {
    private final float[] buffer = new float[MAX_SIZE];

    public void process() {
        Arrays.fill(buffer, 0);  // Reset
        // Use buffer
    }
}

// Bad: Allocate per call
class ChannelProcessor {
    public void process() {
        float[] buffer = new float[MAX_SIZE];  // Allocation
        // Use buffer
    }  // GC pressure
}
```

### Concurrency Best Practices

```java
// Use concurrent collections
private final ConcurrentLinkedDeque<Token> queue = new ConcurrentLinkedDeque<>();

// Use atomic operations
private final AtomicLong counter = new AtomicLong();

// Use CompletableFuture for async
public CompletableFuture<Result> processAsync(Input input) {
    return CompletableFuture.supplyAsync(
        () -> process(input),
        executor
    );
}

// Use virtual threads (Java 24)
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    // Process with virtual threads
}
```

## Debugging & Troubleshooting

### Logging Configuration

```xml
<!-- logback.xml -->
<configuration>
    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <!-- Debug specific components -->
    <logger name="com.hellblazer.art.hartcq.core" level="DEBUG"/>
    <logger name="com.hellblazer.art.hartcq.core.channels" level="TRACE"/>

    <root level="INFO">
        <appender-ref ref="STDOUT"/>
    </root>
</configuration>
```

### Common Issues

#### Issue: Slow Channel Processing

```java
// Enable profiling
var profiler = new ChannelProfiler();
profiler.start();

var result = channel.processWindow(tokens);

profiler.stop();
logger.info("Channel {} took {} ms",
           channel.getName(),
           profiler.getElapsedMillis());
```

#### Issue: Memory Leaks

```java
// Use weak references for caches
private final WeakHashMap<String, float[]> cache = new WeakHashMap<>();

// Clear collections when done
@Override
public void close() {
    cache.clear();
    queue.clear();
    // Release resources
}
```

#### Issue: Non-Deterministic Results

```java
// Ensure deterministic initialization
public class DeterministicChannel implements Channel {
    private final Random random = new Random(42);  // Fixed seed

    @Override
    public boolean isDeterministic() {
        return true;
    }
}
```

### Debugging Tools

```bash
# Thread dump
jstack <pid>

# Heap dump
jmap -dump:format=b,file=heap.bin <pid>

# GC logs
java -Xlog:gc* -jar app.jar

# Flight recorder
java -XX:StartFlightRecording=filename=recording.jfr -jar app.jar
```

## Release Process

### Version Management

```xml
<!-- Update version in all POMs -->
<version>1.0.0</version>  <!-- Release version -->
<version>1.1.0-SNAPSHOT</version>  <!-- Next development version -->
```

### Release Checklist

1. **Update version numbers**
   ```bash
   mvn versions:set -DnewVersion=1.0.0
   ```

2. **Run full test suite**
   ```bash
   mvn clean test
   mvn verify
   ```

3. **Run performance benchmarks**
   ```bash
   mvn clean install -pl benchmarks
   java -jar benchmarks/target/benchmarks.jar
   ```

4. **Update documentation**
   - README.md
   - API_REFERENCE.md
   - CHANGELOG.md

5. **Create release branch**
   ```bash
   git checkout -b release/1.0.0
   ```

6. **Tag release**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   ```

7. **Deploy to repository**
   ```bash
   mvn clean deploy
   ```

8. **Create GitHub release**
   - Upload artifacts
   - Add release notes
   - Publish

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        java: [ 24 ]

    steps:
    - uses: actions/checkout@v3
    - name: Set up JDK ${{ matrix.java }}
      uses: actions/setup-java@v3
      with:
        java-version: ${{ matrix.java }}
        distribution: 'oracle'

    - name: Cache Maven dependencies
      uses: actions/cache@v3
      with:
        path: ~/.m2
        key: ${{ runner.os }}-m2-${{ hashFiles('**/pom.xml') }}

    - name: Build with Maven
      run: mvn clean compile

    - name: Run tests
      run: mvn test

    - name: Generate test report
      uses: dorny/test-reporter@v1
      if: success() || failure()
      with:
        name: Maven Tests
        path: target/surefire-reports/*.xml
        reporter: java-junit
```

## Support & Resources

### Documentation
- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Performance Guide](PERFORMANCE.md)
- [Tutorial](TUTORIAL.md)

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and ideas
- Wiki: Additional documentation

### Useful Links
- [Adaptive Resonance Theory Papers](https://scholar.google.com/scholar?q=adaptive+resonance+theory)
- [Grossberg Dynamics](https://en.wikipedia.org/wiki/Stephen_Grossberg)
- [Java 24 Features](https://openjdk.org/projects/jdk/24/)
- [Maven Documentation](https://maven.apache.org/guides/)

---

**Document Version**: 1.0
**Last Updated**: September 14, 2025
**Maintainer**: Hal Hildebrand