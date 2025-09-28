# Goal-Seeking Module

Advanced trajectory planning and state transition system powered by ART-Temporal sequence learning.

## Overview

This module provides intelligent goal-seeking capabilities by integrating temporal sequence learning from the ART-Temporal system. It learns from successful trajectories and generates new sequences to achieve goals, combining multi-scale processing with adaptive pattern recognition.

### Key Features

- **Temporal Sequence Learning**: Learns successful state transition sequences using ART-Temporal
- **Adaptive Trajectory Generation**: Generates new trajectories based on learned patterns
- **Multi-Scale Dynamics**: Processes information at multiple temporal scales (Goal, Strategic, Tactical, Execution)
- **Pattern Adaptation**: Adapts learned patterns to new situations
- **Working Memory**: Maintains trajectory context during planning

### Core Components

- `TemporalGoalSeeker` - Main integration class combining ART-Temporal with goal-seeking
- `StateTrajectoryPlanner` - Plans trajectories through state space with learning capabilities
- `TransitionLibrary` - Stores and recalls successful state transitions
- `StateTransitionGenerator` - Multi-scale alignment and trajectory generation
- `ExecutionFeedbackSystem` - Manages execution with feedback loops

## ART-Temporal Integration

The goal-seeking module now leverages the full power of ART-Temporal's sequence learning:

### Temporal Modules Used

- **temporal-integration**: Main TemporalART implementation for sequence learning
- **temporal-memory**: Working memory for maintaining trajectory context
- **temporal-masking**: Selective attention to relevant transitions
- **temporal-dynamics**: Multi-scale temporal processing
- **temporal-performance**: High-performance vectorized implementations

## Installation

```bash
# Clone repository
git clone https://github.com/hellblazer/ART.git
cd ART

# Build with temporal dependencies
mvn clean install -pl art-temporal -am
mvn clean install -pl goal-seeking -am

# Run tests
mvn test -pl goal-seeking
```

## Basic Usage

### Using TemporalGoalSeeker

```java
import com.hellblazer.art.goal.temporal.TemporalGoalSeeker;
import com.hellblazer.art.goal.State;
import com.hellblazer.art.temporal.integration.TemporalARTParameters;

// Create temporal goal seeker
var parameters = TemporalARTParameters.builder()
    .vigilance(0.85f)
    .learningRate(0.1f)
    .build();
var goalSeeker = new TemporalGoalSeeker(parameters);

// Learn from successful trajectory
List<State> successfulTrajectory = getSuccessfulPath();
goalSeeker.learnTrajectory(successfulTrajectory, 0.95f); // 95% success rate

// Generate new trajectory
State currentState = new State(new double[]{0.0, 0.0, 0.0});
State goalState = new State(new double[]{1.0, 1.0, 1.0});
List<State> generatedTrajectory = goalSeeker.generateTrajectory(currentState, goalState);

// The system will either:
// 1. Adapt a learned pattern if similar trajectory exists
// 2. Generate a novel sequence using temporal dynamics
```

### Legacy Components

```java
import com.hellblazer.art.goal.*;

// Traditional state trajectory planning
var planner = new StateTrajectoryPlanner();
var trajectory = planner.planTrajectory(current, goal);

// Transition library for storing successes
var library = new TransitionLibrary();
library.learnTransition(from, to, action, success);
```

## Architecture

### Multi-Scale Temporal Processing

The system operates at multiple temporal scales matching goal-seeking layers:

```java
// Time scales initialized in TemporalGoalSeeker
timeScaleOrchestrator.addTimeScale("goal", 1.0f);      // Slowest - high-level goals
timeScaleOrchestrator.addTimeScale("strategic", 5.0f);  // Medium-slow - strategic planning
timeScaleOrchestrator.addTimeScale("tactical", 10.0f);  // Medium-fast - tactical decisions
timeScaleOrchestrator.addTimeScale("execution", 40.0f); // Fastest - execution details
```

### Sequence Learning Process

1. **Learning Phase**:
   - Convert successful trajectories to temporal patterns
   - Train TemporalART network on sequences
   - Store patterns with success metrics

2. **Generation Phase**:
   - Search for similar learned patterns
   - Adapt patterns to new start/goal states
   - Generate novel sequences when no pattern matches

3. **Adaptation Mechanism**:
   - Use temporal dynamics for smooth interpolation
   - Blend learned templates with current context
   - Maintain temporal coherence across scales

## API Reference

### TemporalGoalSeeker

```java
// Learn from trajectory
public void learnTrajectory(List<State> trajectory, float success)

// Generate new trajectory
public List<State> generateTrajectory(State current, State goal)

// Get statistics
public TemporalARTStatistics getStatistics()

// Reset the system
public void reset()
```

### StateTrajectoryPlanner

```java
// Plan trajectory with learning
public Trajectory planTrajectory(State current, State goal)

// Adapt existing trajectory
public Trajectory adaptTrajectory(Trajectory cached, State newStart, State newGoal)
```

### TransitionLibrary

```java
// Learn successful transitions
public void learnTransition(State from, State to, Action action, float success)

// Recall similar transitions
public Action recallTransition(State from, State to)
```

## Configuration

### Temporal Parameters

```java
var parameters = TemporalARTParameters.builder()
    .vigilance(0.85f)          // Pattern matching threshold
    .learningRate(0.1f)         // Learning speed
    .workingMemorySize(100)     // Trajectory context size
    .multiScaleDepth(4)         // Number of temporal scales
    .build();
```

### Planning Parameters

```java
// Trajectory planning
planner.setMaxTrajectoryLength(100);
planner.setStepSize(0.1f);
planner.setConvergenceThreshold(0.01f);

// Transition library
library.setSimilarityThreshold(0.85f);
library.setMaxTransitionsPerPair(10);
```

## Testing

```bash
# Run all tests
mvn test -pl goal-seeking

# Run temporal integration tests
mvn test -pl goal-seeking -Dtest=TemporalGoalSeekerTest

# Run with coverage
mvn test -pl goal-seeking jacoco:report
```

### Test Coverage

- `TemporalGoalSeekerTest` - Tests temporal sequence learning and generation
- `StateTrajectoryPlannerTest` - Tests basic trajectory planning
- `TransitionLibraryTest` - Tests transition storage and recall

## Implementation Status

### Completed ✅
- ART-Temporal integration via `TemporalGoalSeeker`
- Maven dependencies configured for all temporal modules
- Multi-scale temporal dynamics matching goal-seeking layers
- Test demonstrations of sequence learning and generation

### In Progress 🚧
- Fixing legacy component compilation issues
- Full integration with StateTrajectoryPlanner
- Performance benchmarking

### Planned 📋
- Online learning during execution
- GPU acceleration for large-scale planning
- Hierarchical goal decomposition

## Performance Considerations

- Temporal sequence learning scales with trajectory length
- Working memory maintains fixed-size context window
- Vectorized implementations available for high-performance scenarios
- Pattern matching uses efficient similarity metrics

## License

GNU Affero General Public License v3.0

## References

- Kazerounian, S., & Grossberg, S. (2014). Real-time learning of predictive recognition categories that chunk sequences of items stored in working memory
- Grossberg, S. (2013). Adaptive Resonance Theory
- Multi-scale hierarchical processing in neural systems