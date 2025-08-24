# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Adaptive Resonance Theory (ART)** implementation in Java 24. ART is a neural network architecture for unsupervised learning and pattern recognition. The project is structured as a multi-module Maven build with focus on performance, graphics visualization, and mathematical operations.

## Build System & Commands

### Maven Commands
- **Build the project**: `mvn clean compile`
- **Run tests**: `mvn test` 
- **Run single test**: `mvn test -Dtest=ClassName#methodName`
- **Build with all modules**: `mvn clean install`
- **Generate sources**: `mvn generate-sources` (for Protocol Buffers and JOOQ)
- **Check dependency convergence**: `mvn enforcer:enforce`
- **Update versions**: `mvn versions:display-dependency-updates`

### Requirements
- **Java 24+** (configured for Java 24 features)
- **Maven 3.9.1+** (enforced)
- **macOS ARM64** (LWJGL natives configured for Apple Silicon)

## Architecture & Technology Stack

### Core Technologies
- **Java 24** with modern language features (var, records, pattern matching, virtual threads)
- **Maven multi-module** structure for component organization
- **Protocol Buffers + gRPC** for serialization and communication
- **JOOQ** for type-safe database operations
- **H2 Database** for embedded data storage

### Graphics & Visualization
- **JavaFX 24** for GUI applications (use Launcher inner class pattern)
- **LWJGL 3.3.6** for OpenGL graphics and native integration
- **JOML** for 3D math operations (Vector3f, Matrix4f, etc.)

### Testing & Performance
- **JUnit 5** for testing
- **Mockito 4.8.1** for mocking
- **JMH** for micro-benchmarking performance-critical code
- **Surefire** configured with 512MB max heap for test execution

### Key Dependencies
- **Guava** for collections and utilities
- **Apache Commons Lang3** for common utilities  
- **Logback + SLF4J** for logging
- **Prime Mover** custom Maven plugin (snapshot version)

## Code Conventions

### Java Style
- Use `var` for local variable type inference where type is obvious
- Never use `synchronized` - prefer concurrent collections and lock-free patterns
- Follow JavaFX Launcher pattern for Application.launch() calls
- Leverage Java 24 features: records, pattern matching, virtual threads
- Use try-with-resources for resource management

### Maven Module Structure
- Multi-module build with `<modules>` in parent POM
- Generated sources go in `target/generated-sources/`
- Protocol Buffers sources in `src/main/proto` and `src/test/proto`
- JOOQ generated classes in `target/generated-sources/jooq`

### Graphics Programming
- Use JOML for all 3D math operations
- LWJGL configured for macOS ARM64 natives
- JavaFX for GUI, LWJGL for low-level graphics
- Consider GPU acceleration for compute-intensive ART algorithms

## Development Workflow

### Module Creation
When creating new modules:
1. Add module to parent `<modules>` section
2. Create module directory with its own `pom.xml`
3. Use parent POM dependency management
4. Follow Maven standard directory layout

### Performance Considerations  
- Use JMH for benchmarking neural network operations
- ART algorithms are compute-intensive - consider parallel processing
- LWJGL enables GPU acceleration via OpenGL/Vulkan
- Virtual threads (Java 24) for concurrent pattern processing

### Testing Strategy
- Unit tests for individual ART components
- Integration tests for full network behavior
- Performance tests using JMH for critical paths
- Visual tests for JavaFX components

## Project-Specific Notes

### ART Neural Network Context
- **Adaptive Resonance Theory** is an unsupervised learning architecture
- Focus on real-time pattern recognition and categorization
- Key concepts: vigilance parameter, resonance, competitive learning
- Mathematical operations will be performance-critical

### Graphics Integration
The combination of JavaFX + LWJGL + JOML suggests:
- Real-time visualization of neural network states
- Interactive parameter tuning interfaces  
- 2D/3D visualization of pattern spaces
- Possibly VR/AR applications given LWJGL inclusion

### Custom Repository
Uses `repo-hell` GitHub repository for custom Maven artifacts:
- Prime Mover plugin for advanced build features
- May include other custom neural network utilities
- Check repository for additional documentation