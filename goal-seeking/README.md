# Goal-Seeking Module

A state transition and action selection system for the ART library.

## Overview

This module provides methods for generating state transitions and selecting actions. It processes information at multiple scales and uses ART networks for pattern learning.

### Components

- `StateTransitionGenerator` - Generates transitions between states using multi-scale processing
- `MultiCriteriaActionSelector` - Selects actions based on weighted criteria
- `LearningFeedbackStack` - Learns feedback patterns using FuzzyART

## How It Works

![Goal-Seeking Conceptual Flow](./docs/diagrams/conceptual-flow.png)

### State Transition Generation

The `StateTransitionGenerator` searches for alignment between processing layers operating at different scales (1x, 5x, 10x, 20x, 40x). It runs an iterative numerical search using phase calculations - not a real-time system.

### Action Selection

The `MultiCriteriaActionSelector` scores actions based on:
- Goal alignment (35%)
- State compatibility (25%)
- Historical performance (25%)
- Context appropriateness (15%)

Actions compete through lateral inhibition, with the highest-scoring action selected.

### Feedback Learning

The `LearningFeedbackStack` uses FuzzyART to categorize feedback patterns and learn appropriate responses. It adjusts timing and strength parameters based on outcomes.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ART.git
cd ART

# Build module
mvn clean install -pl goal-seeking -am

# Run tests
mvn test -pl goal-seeking
```

## Basic Usage

```java
import com.hellblazer.art.goal.*;

// Create components
var transitionGen = new StateTransitionGenerator();
var actionSelector = new MultiCriteriaActionSelector();
var feedbackStack = new LearningFeedbackStack("system", "goal");

// Define states
State current = new MyState(x, y, velocity);
State goal = new MyGoalState(targetX, targetY);

// Generate state transition
StateTransition transition = transitionGen.generateTransition(current, goal);

// Create action candidates
List<ActionCandidate> candidates = generateCandidates(transition);

// Select action
SelectedAction action = actionSelector.selectAction(current, goal, candidates);

// Process feedback
FeedbackData feedback = new FeedbackData(sensorData, timestamp);
ModulationSignal modulation = feedbackStack.processFeedback(feedback);

// Learn from outcome
Effect effect = new Effect(success, magnitude);
feedbackStack.learnFromEffect(modulation, effect);
```

## Architecture

### Multi-Scale Processing

Processing occurs at five different scale factors:
- 1x: High-level goals
- 5x: Strategic planning
- 10x: Tactical decisions
- 20x: State transitions
- 40x: Execution details

### Alignment Search

The system iteratively searches for states where the different scales align, using:
- Phase alignment checks between scales
- Correlation of layer outputs
- Cross-scale modulation verification

### Pattern Learning

FuzzyART is used for pattern categorization and learning. The system forms categories for similar patterns and adjusts its responses based on feedback.

## API Reference

### StateTransitionGenerator

```java
public StateTransition generateTransition(State from, State to)
```
Generates a transition between states by searching for alignment across multiple processing scales.

### MultiCriteriaActionSelector

```java
public SelectedAction selectAction(SystemState current, GoalState goal,
                                  List<ActionCandidate> candidates)
```
Selects an action from candidates based on weighted scoring criteria.

### LearningFeedbackStack

```java
public ModulationSignal processFeedback(FeedbackData raw)
public void learnFromEffect(ModulationSignal signal, Effect effect)
```
Processes feedback and adjusts parameters based on outcomes.

## Configuration

Parameters can be adjusted for different behaviors:

```java
// Transition generation
transitionGen.setAlignmentThreshold(0.8f);
transitionGen.setMaxIterations(1000);

// Action selection
actionSelector.setExplorationRate(0.1f);
actionSelector.setLateralInhibitionStrength(0.3f);

// Feedback learning
feedbackStack.setVigilance(0.7f);
feedbackStack.setLearningRate(0.1f);
```

## Implementation Notes

- Scale factors are multipliers in the phase calculation, not timing parameters
- Processing runs as fast as the CPU allows
- Uses Java's CompletableFuture for parallel processing
- Pattern memory automatically prunes old entries to prevent unbounded growth

## Testing

```bash
# Run all tests
mvn test -pl goal-seeking

# Run specific test
mvn test -pl goal-seeking -Dtest=LearningFeedbackStackTest

# Run with coverage
mvn test -pl goal-seeking jacoco:report
```

## License

GNU Affero General Public License v3.0

## References

- Grossberg, S. (2013). Adaptive Resonance Theory
- Multi-scale hierarchical processing
- Winner-take-all selection mechanisms