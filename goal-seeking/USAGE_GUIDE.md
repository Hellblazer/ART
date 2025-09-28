# Goal-Seeking Neural Architecture - Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [ART-Temporal Integration](#art-temporal-integration) **NEW**
4. [Advanced Scenarios](#advanced-scenarios)
5. [Real-World Examples](#real-world-examples)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

Add the goal-seeking module with temporal dependencies to your Maven project:

```xml
<dependency>
    <groupId>com.hellblazer.art</groupId>
    <artifactId>goal-seeking</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>

<!-- For ART-Temporal integration -->
<dependency>
    <groupId>com.hellblazer.art</groupId>
    <artifactId>temporal-integration</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>
```

### Minimal Example with Temporal Learning

```java
import com.hellblazer.art.goal.temporal.TemporalGoalSeeker;
import com.hellblazer.art.goal.State;
import com.hellblazer.art.temporal.integration.TemporalARTParameters;

public class QuickStart {
    public static void main(String[] args) {
        // Create temporal goal seeker
        var parameters = TemporalARTParameters.builder()
            .vigilance(0.85f)
            .learningRate(0.1f)
            .build();
        var goalSeeker = new TemporalGoalSeeker(parameters);

        // Define states
        State current = new State(new double[]{0.0, 0.0, 0.0});
        State goal = new State(new double[]{1.0, 1.0, 1.0});

        // Generate trajectory using temporal patterns
        var trajectory = goalSeeker.generateTrajectory(current, goal);

        System.out.println("Generated trajectory with " + trajectory.size() + " steps");
    }
}
```

---

## Basic Usage

### Step 1: Define Your State Space

Implement the State interface for your domain:

```java
public class NavigationState extends State {
    // State can be any n-dimensional vector
    public NavigationState(double x, double y, double z, double vx, double vy, double vz) {
        super(new double[]{x, y, z, vx, vy, vz});
    }

    // Convenience getters
    public double getX() { return data[0]; }
    public double getY() { return data[1]; }
    public double getZ() { return data[2]; }
    public double getVx() { return data[3]; }
    public double getVy() { return data[4]; }
    public double getVz() { return data[5]; }
}
```

### Step 2: Learn from Successful Trajectories

```java
// Create temporal goal seeker
var goalSeeker = new TemporalGoalSeeker();

// Generate or load a successful trajectory
List<State> successfulPath = new ArrayList<>();
for (int i = 0; i <= 10; i++) {
    double t = i / 10.0;
    successfulPath.add(new State(new double[]{t, t*t, Math.sin(t)}));
}

// Learn from the successful trajectory
goalSeeker.learnTrajectory(successfulPath, 0.95f); // 95% success rate

// Now the system can generate similar trajectories
```

### Step 3: Generate New Trajectories

```java
// Define new start and goal
State newStart = new State(new double[]{0.1, 0.0, 0.0});
State newGoal = new State(new double[]{0.9, 0.8, 0.7});

// Generate trajectory using learned patterns
List<State> generatedPath = goalSeeker.generateTrajectory(newStart, newGoal);

// The system will:
// 1. Search for similar learned patterns
// 2. Adapt them to the new situation
// 3. Or generate novel sequences if no pattern matches
```

---

## ART-Temporal Integration

### Understanding the Architecture

The goal-seeking module now integrates with ART-Temporal for advanced sequence learning:

```java
// TemporalGoalSeeker internally uses:
// - TemporalART: For sequence learning and generation
// - WorkingMemory: To maintain trajectory context
// - TimeScaleOrchestrator: For multi-scale processing
// - MaskingField: For selective attention to transitions
```

### Multi-Scale Processing

The system processes trajectories at multiple temporal scales:

```java
// Automatically configured time scales
Goal Layer:      1.0x - High-level strategic goals
Strategic Layer: 5.0x - Medium-term planning
Tactical Layer:  10.0x - Short-term decisions
Execution Layer: 40.0x - Immediate actions
```

### Learning Multiple Patterns

```java
var goalSeeker = new TemporalGoalSeeker();

// Learn different types of trajectories
for (TrajectoryType type : TrajectoryType.values()) {
    List<State> trajectory = generateTrajectoryOfType(type);
    float success = evaluateTrajectory(trajectory);

    if (success > 0.5f) {
        goalSeeker.learnTrajectory(trajectory, success);
    }
}

// The system builds a repertoire of patterns
```

### Pattern Adaptation

```java
// The system adapts learned patterns to new situations
State currentState = getCurrentState();
State desiredGoal = getDesiredGoal();

// Generate adapted trajectory
List<State> adaptedPath = goalSeeker.generateTrajectory(currentState, desiredGoal);

// The adaptation process:
// 1. Finds similar learned patterns based on start/goal similarity
// 2. Blends template patterns with current context
// 3. Uses temporal dynamics for smooth interpolation
```

---

## Advanced Scenarios

### Scenario 1: Robot Navigation with Learning

```java
public class RobotNavigationExample {
    private TemporalGoalSeeker navigator;
    private List<State> currentPath;

    public void initialize() {
        var params = TemporalARTParameters.builder()
            .vigilance(0.8f)      // Pattern matching threshold
            .learningRate(0.15f)  // Faster learning for real-time
            .workingMemorySize(50) // Shorter context window
            .build();
        navigator = new TemporalGoalSeeker(params);
    }

    public void navigateToGoal(State goal) {
        State current = getRobotState();

        // Generate path using learned patterns
        currentPath = navigator.generateTrajectory(current, goal);

        // Execute path
        for (State waypoint : currentPath) {
            moveRobotTo(waypoint);

            // Check for obstacles
            if (obstacleDetected()) {
                // Replan from current position
                current = getRobotState();
                currentPath = navigator.generateTrajectory(current, goal);
            }
        }

        // Learn from successful navigation
        if (goalReached(goal)) {
            navigator.learnTrajectory(currentPath, calculateSuccess());
        }
    }
}
```

### Scenario 2: Market Trading Strategy

```java
public class TradingStrategyExample {
    private TemporalGoalSeeker strategyPlanner;

    public void optimizePortfolio() {
        // Define market state (prices, volumes, indicators)
        State currentMarket = new State(new double[]{
            getPrice("AAPL"), getVolume("AAPL"), getRSI("AAPL"),
            getPrice("GOOGL"), getVolume("GOOGL"), getRSI("GOOGL"),
            // ... more securities
        });

        // Define target portfolio state
        State targetPortfolio = new State(new double[]{
            0.3, // 30% AAPL
            0.2, // 20% GOOGL
            // ... target allocations
        });

        // Generate trading sequence
        List<State> tradingPlan = strategyPlanner.generateTrajectory(
            currentMarket, targetPortfolio
        );

        // Execute trades following the plan
        for (int i = 1; i < tradingPlan.size(); i++) {
            State action = computeAction(tradingPlan.get(i-1), tradingPlan.get(i));
            executeTrade(action);
        }
    }
}
```

### Scenario 3: Game AI Decision Making

```java
public class GameAIExample {
    private TemporalGoalSeeker aiPlanner;

    public void planStrategy() {
        // Current game state
        State gameState = new State(new double[]{
            getPlayerHealth(), getPlayerMana(), getPlayerPosition(),
            getEnemyHealth(), getEnemyPosition(), getObjectiveDistance()
        });

        // Winning state
        State winState = new State(new double[]{
            1.0, // Full health
            1.0, // Full mana
            1.0, // At objective
            0.0, // Enemy defeated
            999, // Enemy far away
            0.0  // At objective
        });

        // Generate strategy sequence
        List<State> strategy = aiPlanner.generateTrajectory(gameState, winState);

        // Learn from victories
        if (gameWon()) {
            aiPlanner.learnTrajectory(recordedStates, 1.0f);
        }
    }
}
```

---

## Performance Optimization

### Using Vectorized Implementations

```java
// For high-performance scenarios, use vectorized temporal modules
import com.hellblazer.art.temporal.performance.VectorizedTemporalART;

// The TemporalGoalSeeker automatically uses vectorized implementations
// when available for better performance
```

### Batch Learning

```java
// Learn multiple trajectories in batch for efficiency
List<List<State>> successfulTrajectories = loadTrajectories();

for (var trajectory : successfulTrajectories) {
    goalSeeker.learnTrajectory(trajectory, evaluateSuccess(trajectory));
}
```

### Memory Management

```java
// Configure working memory size for your use case
var params = TemporalARTParameters.builder()
    .workingMemorySize(100)  // Larger for complex trajectories
    .maxCategories(1000)     // Limit pattern storage
    .pruneThreshold(0.5f)    // Remove low-success patterns
    .build();
```

### Parallel Processing

```java
// Generate multiple trajectories in parallel
CompletableFuture<List<State>> future1 =
    CompletableFuture.supplyAsync(() -> goalSeeker.generateTrajectory(start1, goal1));
CompletableFuture<List<State>> future2 =
    CompletableFuture.supplyAsync(() -> goalSeeker.generateTrajectory(start2, goal2));

CompletableFuture.allOf(future1, future2).join();
```

---

## Real-World Examples

### Example 1: Drone Path Planning

```java
public class DronePathPlanner {
    private TemporalGoalSeeker pathPlanner;
    private List<State> noFlyZones;

    public List<State> planFlightPath(GPS start, GPS destination) {
        // Convert GPS to state representation
        State startState = gpsToState(start);
        State goalState = gpsToState(destination);

        // Generate initial path
        List<State> path = pathPlanner.generateTrajectory(startState, goalState);

        // Validate and adjust for no-fly zones
        path = avoidNoFlyZones(path);

        // Optimize for battery efficiency
        path = optimizeForBattery(path);

        return path;
    }

    private State gpsToState(GPS gps) {
        return new State(new double[]{
            gps.latitude, gps.longitude, gps.altitude,
            0, 0, 0  // Initially no velocity
        });
    }
}
```

### Example 2: Manufacturing Process Optimization

```java
public class ManufacturingOptimizer {
    private TemporalGoalSeeker processPlanner;

    public void optimizeProductionLine() {
        // Current production state
        State current = new State(new double[]{
            getThroughput(), getQualityScore(), getEfficiency(),
            getWasteRate(), getEnergyUsage(), getCost()
        });

        // Target optimal state
        State optimal = new State(new double[]{
            1.0,  // Maximum throughput
            0.99, // High quality
            0.95, // High efficiency
            0.01, // Minimal waste
            0.3,  // Low energy usage
            0.2   // Low cost
        });

        // Generate optimization trajectory
        List<State> optimizationPlan = processPlanner.generateTrajectory(current, optimal);

        // Implement changes gradually
        for (State targetState : optimizationPlan) {
            adjustProductionParameters(targetState);
            waitForStabilization();

            // Learn from successful adjustments
            if (improvementDetected()) {
                processPlanner.learnTrajectory(getRecentStates(), getImprovement());
            }
        }
    }
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: No trajectory generated
```java
// Solution: Check if patterns have been learned
if (goalSeeker.getStatistics().categoriesLearned == 0) {
    // No patterns learned yet - train with examples
    trainWithExamples(goalSeeker);
}
```

#### Issue: Generated trajectories are too long
```java
// Solution: Adjust parameters for more direct paths
var params = TemporalARTParameters.builder()
    .stepSize(0.2f)           // Larger steps
    .convergenceThreshold(0.05f) // Less precision
    .build();
```

#### Issue: System not learning from trajectories
```java
// Solution: Check success threshold
// Only trajectories with success > 0.5 are learned
float success = evaluateTrajectory(trajectory);
if (success > 0.5f) {
    goalSeeker.learnTrajectory(trajectory, success);
} else {
    System.out.println("Trajectory not successful enough: " + success);
}
```

#### Issue: Pattern adaptation not working
```java
// Solution: Adjust similarity threshold
var params = TemporalARTParameters.builder()
    .vigilance(0.7f)  // Lower threshold for more flexible matching
    .build();
```

### Performance Monitoring

```java
// Monitor system performance
var stats = goalSeeker.getStatistics();
System.out.println("Categories learned: " + stats.categoriesLearned);
System.out.println("Sequences generated: " + stats.sequencesGenerated);
System.out.println("Average success: " + stats.averageSuccess);
```

### Debug Output

```java
// Enable detailed logging
Logger logger = LoggerFactory.getLogger(TemporalGoalSeeker.class);
((ch.qos.logback.classic.Logger)logger).setLevel(Level.DEBUG);
```

---

## Best Practices

1. **Start with simple trajectories** - Train on easy examples first
2. **Learn incrementally** - Add complexity gradually
3. **Monitor success rates** - Track what works
4. **Tune parameters for your domain** - Adjust vigilance, learning rate
5. **Use appropriate state representations** - Normalize values to [0,1]
6. **Provide diverse training examples** - Cover various scenarios
7. **Clean up old patterns** - Reset periodically if needed

---

## Further Resources

- [API Documentation](./docs/api)
- [Architecture Overview](./README.md)
- [Test Examples](./src/test/java/com/hellblazer/art/goal/temporal)
- [Research Papers](./docs/papers)