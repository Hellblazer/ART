# Goal-Seeking Neural Architecture - Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Scenarios](#advanced-scenarios)
4. [Real-World Examples](#real-world-examples)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

Add the goal-seeking module to your Maven project:

```xml
<dependency>
    <groupId>com.hellblazer.art</groupId>
    <artifactId>goal-seeking</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>
```

### Minimal Example

```java
import com.hellblazer.art.goal.*;

public class QuickStart {
    public static void main(String[] args) {
        // Create core components
        var oscillator = new StateTransitionOscillator();
        var selector = new ResonanceActionSelector();
        var feedbackStack = new LearningFeedbackStack("system", "goal");

        // Define simple states
        State current = () -> new float[]{0.0f, 0.0f};
        State goal = () -> new float[]{1.0f, 1.0f};

        // Generate transition
        var transition = oscillator.generateTransition(current, goal);

        System.out.println("Transition generated from " + current + " to " + goal);
    }
}
```

---

## Basic Usage

### Step 1: Define Your State Space

First, implement the state interfaces for your domain:

```java
public class NavigationState implements SystemState, State {
    private final float x, y;          // Position
    private final float vx, vy;        // Velocity
    private final float energy;        // Energy level

    public NavigationState(float x, float y, float vx, float vy, float energy) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.energy = energy;
    }

    @Override
    public float distanceTo(State other) {
        if (other instanceof NavigationState nav) {
            float dx = x - nav.x;
            float dy = y - nav.y;
            return (float)Math.sqrt(dx * dx + dy * dy);
        }
        return Float.MAX_VALUE;
    }

    @Override
    public State interpolate(State other, float t) {
        if (other instanceof NavigationState nav) {
            return new NavigationState(
                x + (nav.x - x) * t,
                y + (nav.y - y) * t,
                vx + (nav.vx - vx) * t,
                vy + (nav.vy - vy) * t,
                energy + (nav.energy - energy) * t
            );
        }
        return this;
    }

    @Override
    public OscillationSignature getOscillationSignature() {
        // Convert state to oscillatory signature
        float phase = (float)Math.atan2(vy, vx);
        float frequency = (float)Math.sqrt(vx * vx + vy * vy);
        float[] harmonics = {energy * 0.1f, energy * 0.05f};

        return new OscillationSignature(phase, frequency, harmonics);
    }

    @Override
    public float[] encode() {
        return new float[]{x, y, vx, vy, energy};
    }
}
```

### Step 2: Define Goals

```java
public class NavigationGoal implements GoalState {
    private final float targetX, targetY;
    private final float desiredSpeed;
    private final float priority;

    public NavigationGoal(float x, float y, float speed, float priority) {
        this.targetX = x;
        this.targetY = y;
        this.desiredSpeed = speed;
        this.priority = priority;
    }

    @Override
    public OscillationPattern getOscillationPattern() {
        // Goal pattern with low frequency
        float phase = (float)Math.atan2(targetY, targetX);
        float frequency = 1.0f; // Low frequency for long-term goals
        float amplitude = priority;

        return new OscillationPattern(phase, frequency, amplitude);
    }

    @Override
    public float[] encode() {
        return new float[]{targetX, targetY, desiredSpeed, priority};
    }
}
```

### Step 3: Define Actions

```java
public class NavigationAction implements ActionCandidate {
    private final String name;
    private final float thrust;     // Forward thrust
    private final float rotation;   // Rotation angle
    private final float confidence;

    public NavigationAction(String name, float thrust, float rotation, float confidence) {
        this.name = name;
        this.thrust = thrust;
        this.rotation = rotation;
        this.confidence = confidence;
    }

    @Override
    public OscillationPattern getOscillationPattern() {
        // Actions create high frequency patterns
        float phase = rotation;
        float frequency = 40.0f; // High frequency for rapid execution
        float amplitude = thrust;

        return new OscillationPattern(phase, frequency, amplitude);
    }

    @Override
    public float getExpectedResonance() {
        // Estimate how well this action resonates
        return Math.abs(thrust) * confidence;
    }

    @Override
    public float getConfidence() {
        return confidence;
    }

    @Override
    public float[] encode() {
        return new float[]{thrust, rotation, confidence};
    }

    public void execute(Robot robot) {
        robot.applyThrust(thrust);
        robot.rotate(rotation);
    }
}
```

### Step 4: Complete Action Cycle

```java
public class NavigationSystem {
    private final StateTransitionOscillator oscillator;
    private final ResonanceActionSelector selector;
    private final LearningFeedbackStack feedbackStack;
    private final Robot robot;

    public NavigationSystem(Robot robot) {
        this.robot = robot;
        this.oscillator = new StateTransitionOscillator();
        this.selector = new ResonanceActionSelector();
        this.feedbackStack = new LearningFeedbackStack("navigation", "goal");
    }

    public void navigate(NavigationGoal goal) {
        while (!isGoalReached(goal)) {
            // 1. Get current state
            NavigationState current = getCurrentState();

            // 2. Generate state transition
            StateTransition transition = oscillator.generateTransition(current, goal);

            // 3. Generate action candidates
            List<ActionCandidate> candidates = generateActionCandidates(transition);

            // 4. Select best action through resonance
            SelectedAction selected = selector.selectAction(current, goal, candidates);

            // 5. Execute action
            NavigationAction action = (NavigationAction) selected.action;
            action.execute(robot);

            // 6. Get feedback
            float[] sensors = robot.getSensorReadings();
            FeedbackData feedback = new FeedbackData(sensors, System.currentTimeMillis());

            // 7. Process feedback
            ModulationSignal modulation = feedbackStack.processFeedback(feedback);

            // 8. Evaluate success
            float progress = measureProgress(current, goal);
            Effect effect = new Effect(progress > 0, Math.abs(progress));

            // 9. Learn from outcome
            feedbackStack.learnFromEffect(modulation, effect);
            selector.learnFromOutcome(selected, progress);

            // 10. Apply modulation
            oscillator.applyModulation(modulation);

            // Small delay for real-time systems
            Thread.sleep(50); // 20 Hz update rate
        }
    }

    private List<ActionCandidate> generateActionCandidates(StateTransition transition) {
        List<ActionCandidate> candidates = new ArrayList<>();

        // Generate discrete action options
        candidates.add(new NavigationAction("forward", 1.0f, 0.0f, 0.9f));
        candidates.add(new NavigationAction("backward", -0.5f, 0.0f, 0.7f));
        candidates.add(new NavigationAction("turn_left", 0.3f, -0.5f, 0.8f));
        candidates.add(new NavigationAction("turn_right", 0.3f, 0.5f, 0.8f));
        candidates.add(new NavigationAction("stop", 0.0f, 0.0f, 0.6f));

        // Add exploration action if needed
        if (Math.random() < 0.1) { // 10% exploration
            float randomThrust = (float)(Math.random() * 2 - 1);
            float randomRotation = (float)(Math.random() * Math.PI - Math.PI/2);
            candidates.add(new NavigationAction("explore", randomThrust, randomRotation, 0.5f));
        }

        return candidates;
    }
}
```

---

## Advanced Scenarios

### Hierarchical Goal Decomposition

```java
public class HierarchicalGoalSystem {
    private final Map<String, StateTransitionOscillator> oscillators;
    private final Map<String, LearningFeedbackStack> feedbackStacks;

    public HierarchicalGoalSystem() {
        // Create oscillators for different levels
        oscillators = Map.of(
            "strategic", new StateTransitionOscillator(), // Long-term planning
            "tactical", new StateTransitionOscillator(),  // Medium-term
            "operational", new StateTransitionOscillator() // Immediate execution
        );

        // Create feedback stacks between levels
        feedbackStacks = Map.of(
            "strategic->tactical", new LearningFeedbackStack("strategic", "tactical"),
            "tactical->operational", new LearningFeedbackStack("tactical", "operational"),
            "operational->strategic", new LearningFeedbackStack("operational", "strategic")
        );
    }

    public void pursueComplexGoal(ComplexGoal goal) {
        // Decompose into subgoals
        List<SubGoal> strategicGoals = goal.getStrategicGoals();

        for (SubGoal strategic : strategicGoals) {
            // Strategic level generates tactical goals
            List<SubGoal> tacticalGoals = generateTacticalGoals(strategic);

            for (SubGoal tactical : tacticalGoals) {
                // Tactical level generates operational actions
                List<Action> operations = generateOperations(tactical);

                // Execute operations with feedback
                for (Action op : operations) {
                    executeWithFeedback(op);
                }
            }
        }
    }

    private void executeWithFeedback(Action action) {
        // Execute action
        action.execute();

        // Collect multi-level feedback
        FeedbackData operationalFeedback = collectOperationalFeedback();
        FeedbackData tacticalFeedback = collectTacticalFeedback();
        FeedbackData strategicFeedback = collectStrategicFeedback();

        // Process feedback at each level
        var opMod = feedbackStacks.get("operational->strategic")
            .processFeedback(operationalFeedback);
        var tacMod = feedbackStacks.get("tactical->operational")
            .processFeedback(tacticalFeedback);
        var stratMod = feedbackStacks.get("strategic->tactical")
            .processFeedback(strategicFeedback);

        // Apply modulations
        oscillators.get("operational").applyModulation(opMod);
        oscillators.get("tactical").applyModulation(tacMod);
        oscillators.get("strategic").applyModulation(stratMod);
    }
}
```

### Multi-Agent Coordination

```java
public class MultiAgentSystem {
    private final List<Agent> agents;
    private final ResonanceActionSelector globalSelector;

    public MultiAgentSystem(int numAgents) {
        this.agents = new ArrayList<>();
        this.globalSelector = new ResonanceActionSelector();

        for (int i = 0; i < numAgents; i++) {
            agents.add(new Agent("Agent-" + i));
        }
    }

    public void coordinatedAction(GlobalGoal goal) {
        // Collect all agent states
        List<SystemState> states = agents.stream()
            .map(Agent::getState)
            .collect(Collectors.toList());

        // Generate collective oscillation
        CollectiveOscillation collective = computeCollectiveOscillation(states);

        // Each agent selects action based on collective resonance
        for (Agent agent : agents) {
            List<ActionCandidate> localCandidates = agent.generateCandidates();

            // Filter candidates by collective resonance
            List<ActionCandidate> filtered = localCandidates.stream()
                .filter(c -> resonatesWithCollective(c, collective))
                .collect(Collectors.toList());

            // Select action
            SelectedAction selected = agent.selector.selectAction(
                agent.getState(),
                goal,
                filtered
            );

            // Execute with awareness of other agents
            agent.executeWithCoordination(selected, agents);
        }
    }

    private boolean resonatesWithCollective(
            ActionCandidate candidate,
            CollectiveOscillation collective) {

        OscillationPattern pattern = candidate.getOscillationPattern();

        // Check phase alignment
        float phaseDiff = Math.abs(pattern.phase - collective.dominantPhase);
        boolean phaseAligned = phaseDiff < Math.PI / 4; // Within 45 degrees

        // Check frequency harmony
        float freqRatio = pattern.frequency / collective.dominantFrequency;
        boolean harmonious = Math.abs(freqRatio - Math.round(freqRatio)) < 0.1;

        return phaseAligned && harmonious;
    }
}

class Agent {
    final String id;
    final StateTransitionOscillator oscillator;
    final ResonanceActionSelector selector;
    final LearningFeedbackStack feedbackStack;
    SystemState state;

    Agent(String id) {
        this.id = id;
        this.oscillator = new StateTransitionOscillator();
        this.selector = new ResonanceActionSelector();
        this.feedbackStack = new LearningFeedbackStack(id, "collective");
    }

    void executeWithCoordination(SelectedAction action, List<Agent> others) {
        // Execute own action
        execute(action);

        // Send feedback to nearby agents
        for (Agent other : others) {
            if (isNearby(other)) {
                FeedbackData feedback = createFeedbackFor(other);
                ModulationSignal signal = other.feedbackStack.processFeedback(feedback);
                other.oscillator.applyModulation(signal);
            }
        }
    }
}
```

### Dynamic Environment Adaptation

```java
public class AdaptiveGoalSeeker {
    private final StateTransitionOscillator oscillator;
    private final ResonanceActionSelector selector;
    private final Map<String, LearningFeedbackStack> environmentStacks;
    private EnvironmentProfile currentProfile;

    public AdaptiveGoalSeeker() {
        this.oscillator = new StateTransitionOscillator();
        this.selector = new ResonanceActionSelector();
        this.environmentStacks = new HashMap<>();
    }

    public void adaptToEnvironment(Environment env) {
        // Detect environment type
        EnvironmentProfile profile = profileEnvironment(env);

        // Load or create feedback stack for this environment
        String envKey = profile.getKey();
        LearningFeedbackStack stack = environmentStacks.computeIfAbsent(
            envKey,
            k -> new LearningFeedbackStack("system", envKey)
        );

        // Adjust oscillator frequencies for environment
        adjustOscillatorForEnvironment(profile);

        // Set exploration based on familiarity
        float familiarity = calculateFamiliarity(profile);
        selector.setExplorationRate(1.0f - familiarity);

        currentProfile = profile;
    }

    private void adjustOscillatorForEnvironment(EnvironmentProfile profile) {
        switch (profile.getType()) {
            case STATIC:
                // Lower frequencies for stable environment
                oscillator.setFrequency("Band1", 0.5f);
                oscillator.setFrequency("Band2", 2.0f);
                break;

            case DYNAMIC:
                // Higher frequencies for changing environment
                oscillator.setFrequency("Band1", 2.0f);
                oscillator.setFrequency("Band2", 10.0f);
                break;

            case ADVERSARIAL:
                // High frequencies for rapid response
                oscillator.setFrequency("Band4", 30.0f);
                oscillator.setFrequency("Band5", 60.0f);
                break;

            case COOPERATIVE:
                // Synchronized frequencies for coordination
                oscillator.setFrequency("Band3", 10.0f);
                oscillator.setCoupling(0.8f); // Strong coupling
                break;
        }
    }

    private float calculateFamiliarity(EnvironmentProfile profile) {
        String key = profile.getKey();
        if (!environmentStacks.containsKey(key)) {
            return 0.0f; // Completely unfamiliar
        }

        FeedbackStackMetrics metrics = environmentStacks.get(key).getMetrics();

        // Familiarity based on success rate and experience
        float successComponent = metrics.getSuccessRate();
        float experienceComponent = Math.min(1.0f, metrics.total / 1000.0f);

        return 0.7f * successComponent + 0.3f * experienceComponent;
    }
}
```

---

## Real-World Examples

### Example 1: Drone Navigation

```java
public class DroneNavigationSystem {
    private final StateTransitionOscillator oscillator;
    private final ResonanceActionSelector selector;
    private final LearningFeedbackStack feedbackStack;
    private final DroneController drone;

    public void navigateToTarget(GPS target) {
        DroneState current = new DroneState(
            drone.getGPS(),
            drone.getIMU(),
            drone.getBatteryLevel()
        );

        DroneGoal goal = new DroneGoal(
            target,
            FlightMode.EFFICIENT,
            Priority.HIGH
        );

        while (!hasReachedTarget(target)) {
            // Generate transition
            StateTransition transition = oscillator.generateTransition(current, goal);

            // Generate flight commands
            List<ActionCandidate> commands = generateFlightCommands(transition);

            // Select command
            SelectedAction selected = selector.selectAction(current, goal, commands);

            // Execute flight command
            FlightCommand cmd = (FlightCommand) selected.action;
            drone.execute(cmd);

            // Process sensor feedback
            SensorPacket sensors = drone.getSensorPacket();
            FeedbackData feedback = new FeedbackData(
                sensors.toArray(),
                sensors.timestamp
            );

            ModulationSignal modulation = feedbackStack.processFeedback(feedback);

            // Check for obstacles or wind
            if (sensors.hasObstacle() || sensors.windSpeed > threshold) {
                // Rapid adaptation needed
                modulation = amplifyModulation(modulation, 2.0f);
            }

            // Learn from flight performance
            float efficiency = calculateFlightEfficiency(sensors);
            Effect effect = new Effect(efficiency > 0.7f, efficiency);
            feedbackStack.learnFromEffect(modulation, effect);

            // Apply modulation
            oscillator.applyModulation(modulation);

            // Update current state
            current = new DroneState(
                drone.getGPS(),
                drone.getIMU(),
                drone.getBatteryLevel()
            );
        }
    }

    private List<ActionCandidate> generateFlightCommands(StateTransition transition) {
        List<ActionCandidate> commands = new ArrayList<>();

        // Basic movements
        commands.add(new FlightCommand("ascend", 0, 0, 1, 0));
        commands.add(new FlightCommand("descend", 0, 0, -1, 0));
        commands.add(new FlightCommand("forward", 1, 0, 0, 0));
        commands.add(new FlightCommand("backward", -1, 0, 0, 0));
        commands.add(new FlightCommand("left", 0, 1, 0, 0));
        commands.add(new FlightCommand("right", 0, -1, 0, 0));
        commands.add(new FlightCommand("rotate_cw", 0, 0, 0, 1));
        commands.add(new FlightCommand("rotate_ccw", 0, 0, 0, -1));
        commands.add(new FlightCommand("hover", 0, 0, 0, 0));

        // Complex maneuvers based on transition
        if (transition.strategy.intensity > 0.8f) {
            commands.add(new FlightCommand("spiral_ascend", 1, 1, 1, 0.5f));
            commands.add(new FlightCommand("quick_dodge", 2, 0, 0, 0));
        }

        return commands;
    }
}
```

### Example 2: Game AI - NPC Behavior

```java
public class NPCBehaviorSystem {
    private final Map<String, StateTransitionOscillator> behaviorOscillators;
    private final ResonanceActionSelector actionSelector;
    private final Map<String, LearningFeedbackStack> emotionStacks;

    public NPCBehaviorSystem() {
        // Different oscillators for different behaviors
        behaviorOscillators = Map.of(
            "combat", new StateTransitionOscillator(),
            "exploration", new StateTransitionOscillator(),
            "social", new StateTransitionOscillator(),
            "survival", new StateTransitionOscillator()
        );

        actionSelector = new ResonanceActionSelector();

        // Emotion-driven feedback
        emotionStacks = Map.of(
            "fear", new LearningFeedbackStack("state", "fear"),
            "anger", new LearningFeedbackStack("state", "anger"),
            "curiosity", new LearningFeedbackStack("state", "curiosity"),
            "joy", new LearningFeedbackStack("state", "joy")
        );
    }

    public NPCAction decideBehavior(NPCState npc, GameWorld world) {
        // Determine primary goal based on context
        NPCGoal primaryGoal = determinePrimaryGoal(npc, world);

        // Get relevant oscillator
        StateTransitionOscillator oscillator = behaviorOscillators.get(
            primaryGoal.getBehaviorType()
        );

        // Generate state transition
        StateTransition transition = oscillator.generateTransition(
            npc,
            primaryGoal
        );

        // Generate possible actions
        List<ActionCandidate> actions = new ArrayList<>();

        // Combat actions
        if (npc.isInCombat()) {
            actions.add(new CombatAction("attack", npc.getTarget()));
            actions.add(new CombatAction("defend", npc));
            actions.add(new CombatAction("flee", findEscape(npc, world)));
            actions.add(new CombatAction("call_help", npc.getAllies()));
        }

        // Social actions
        if (npc.hasNearbyNPCs()) {
            actions.add(new SocialAction("greet", npc.getNearestNPC()));
            actions.add(new SocialAction("trade", npc.getTradePartner()));
            actions.add(new SocialAction("follow", npc.getLeader()));
        }

        // Exploration actions
        if (npc.isExploring()) {
            actions.add(new ExploreAction("investigate", npc.getPointOfInterest()));
            actions.add(new ExploreAction("patrol", npc.getPatrolRoute()));
            actions.add(new ExploreAction("search", npc.getSearchArea()));
        }

        // Select action based on resonance
        SelectedAction selected = actionSelector.selectAction(
            npc,
            primaryGoal,
            actions
        );

        // Apply emotional modulation
        applyEmotionalFeedback(npc, selected);

        return (NPCAction) selected.action;
    }

    private void applyEmotionalFeedback(NPCState npc, SelectedAction action) {
        // Get emotional state
        EmotionalState emotions = npc.getEmotionalState();

        // Process each emotion
        for (Emotion emotion : emotions.getActiveEmotions()) {
            float[] emotionVector = emotion.toVector();
            FeedbackData feedback = new FeedbackData(
                emotionVector,
                System.currentTimeMillis()
            );

            LearningFeedbackStack stack = emotionStacks.get(emotion.getType());
            ModulationSignal modulation = stack.processFeedback(feedback);

            // Apply emotional modulation to relevant oscillator
            String behaviorType = mapEmotionToBehavior(emotion);
            behaviorOscillators.get(behaviorType).applyModulation(modulation);
        }
    }
}
```

### Example 3: Financial Trading

```java
public class TradingSystem {
    private final StateTransitionOscillator marketOscillator;
    private final ResonanceActionSelector tradeSelector;
    private final Map<String, LearningFeedbackStack> assetFeedback;

    public TradingSystem(Portfolio portfolio) {
        this.marketOscillator = new StateTransitionOscillator();
        this.tradeSelector = new ResonanceActionSelector();
        this.assetFeedback = new HashMap<>();

        // Configure for market frequencies
        marketOscillator.setFrequency("Band1", 0.01f);  // Daily trends
        marketOscillator.setFrequency("Band2", 0.1f);   // Hourly patterns
        marketOscillator.setFrequency("Band3", 1.0f);   // Minute changes
        marketOscillator.setFrequency("Band4", 10.0f);  // Second ticks
        marketOscillator.setFrequency("Band5", 100.0f); // Millisecond updates
    }

    public TradeDecision analyzeMarket(MarketState market, TradingGoal goal) {
        // Generate market transition prediction
        StateTransition prediction = marketOscillator.generateTransition(
            market,
            goal
        );

        // Generate trade candidates
        List<ActionCandidate> trades = generateTradeCandidates(market, prediction);

        // Select trade based on resonance
        SelectedAction selected = tradeSelector.selectAction(
            market,
            goal,
            trades
        );

        // Process market feedback for each asset
        for (Asset asset : market.getAssets()) {
            processAssetFeedback(asset);
        }

        return new TradeDecision(selected);
    }

    private void processAssetFeedback(Asset asset) {
        String symbol = asset.getSymbol();

        LearningFeedbackStack stack = assetFeedback.computeIfAbsent(
            symbol,
            k -> new LearningFeedbackStack("market", symbol)
        );

        // Create feedback from price action
        float[] priceData = {
            asset.getPrice(),
            asset.getVolume(),
            asset.getVolatility(),
            asset.getMomentum()
        };

        FeedbackData feedback = new FeedbackData(
            priceData,
            asset.getTimestamp()
        );

        ModulationSignal modulation = stack.processFeedback(feedback);

        // Learn from profit/loss
        float pnl = calculatePnL(asset);
        Effect effect = new Effect(pnl > 0, Math.abs(pnl));
        stack.learnFromEffect(modulation, effect);

        // Apply market-specific modulation
        marketOscillator.applyModulation(modulation);
    }

    private List<ActionCandidate> generateTradeCandidates(
            MarketState market,
            StateTransition prediction) {

        List<ActionCandidate> trades = new ArrayList<>();

        // Generate trades based on prediction confidence
        float confidence = prediction.confidence;

        if (confidence > 0.8f) {
            // High confidence - larger positions
            trades.add(new TradeAction("buy_large", 1000, market.getBestAsk()));
            trades.add(new TradeAction("sell_large", 1000, market.getBestBid()));
        }

        if (confidence > 0.6f) {
            // Medium confidence - normal positions
            trades.add(new TradeAction("buy_medium", 500, market.getBestAsk()));
            trades.add(new TradeAction("sell_medium", 500, market.getBestBid()));
        }

        // Always include small positions and hold
        trades.add(new TradeAction("buy_small", 100, market.getBestAsk()));
        trades.add(new TradeAction("sell_small", 100, market.getBestBid()));
        trades.add(new TradeAction("hold", 0, 0));

        // Market making
        trades.add(new MarketMakeAction(
            "make_market",
            market.getMidPrice(),
            market.getSpread()
        ));

        return trades;
    }
}
```

---

## Performance Optimization

### Parallel Processing

```java
public class OptimizedGoalSeeker {
    private final ForkJoinPool executorPool;
    private final int parallelism;

    public OptimizedGoalSeeker() {
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.executorPool = new ForkJoinPool(parallelism);
    }

    public List<SelectedAction> selectMultipleActions(
            List<SystemState> states,
            List<GoalState> goals,
            List<List<ActionCandidate>> candidateSets) {

        return executorPool.submit(() ->
            IntStream.range(0, states.size())
                .parallel()
                .mapToObj(i -> {
                    ResonanceActionSelector selector = new ResonanceActionSelector();
                    return selector.selectAction(
                        states.get(i),
                        goals.get(i),
                        candidateSets.get(i)
                    );
                })
                .collect(Collectors.toList())
        ).join();
    }
}
```

### Caching and Memoization

```java
public class CachedGoalSeeker {
    private final Map<StateTransitionKey, StateTransition> transitionCache;
    private final Map<SelectionKey, SelectedAction> selectionCache;
    private final int maxCacheSize = 10000;

    public StateTransition getCachedTransition(State from, State to) {
        StateTransitionKey key = new StateTransitionKey(from, to);

        return transitionCache.computeIfAbsent(key, k -> {
            // Evict if cache too large
            if (transitionCache.size() > maxCacheSize) {
                evictOldestTransitions();
            }
            return oscillator.generateTransition(from, to);
        });
    }
}
```

### Batch Processing

```java
public class BatchProcessor {
    public void processFeedbackBatch(List<FeedbackData> feedbacks) {
        // Process in parallel batches
        int batchSize = 100;

        for (int i = 0; i < feedbacks.size(); i += batchSize) {
            int end = Math.min(i + batchSize, feedbacks.size());
            List<FeedbackData> batch = feedbacks.subList(i, end);

            CompletableFuture<List<ModulationSignal>> future =
                CompletableFuture.supplyAsync(() ->
                    batch.parallelStream()
                        .map(feedbackStack::processFeedback)
                        .collect(Collectors.toList())
                );

            // Apply modulations when ready
            future.thenAccept(modulations -> {
                for (ModulationSignal mod : modulations) {
                    oscillator.applyModulation(mod);
                }
            });
        }
    }
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Poor Action Selection
```java
// Symptom: Actions don't achieve goals
// Solution: Increase resonance sensitivity
selector.setResonanceAmplification(1.5f);
selector.setExplorationRate(0.2f); // More exploration
```

#### Issue: Oscillator Instability
```java
// Symptom: Oscillations grow unbounded
// Solution: Add damping
oscillator.setDamping(0.1f); // 10% damping
oscillator.setCouplingLimit(0.5f); // Limit coupling strength
```

#### Issue: Slow Learning
```java
// Symptom: Feedback doesn't improve performance
// Solution: Adjust learning parameters
feedbackStack.setLearningRate(0.2f); // Increase from default
feedbackStack.setVigilance(0.5f); // Less selective
```

#### Issue: Memory Overflow
```java
// Symptom: OutOfMemoryError with large pattern memory
// Solution: Implement memory management
selector.setMaxPatternMemory(1000);
if (selector.getPatternMemorySize() > 900) {
    selector.pruneOldestPatterns(100);
}
```

### Debugging Tools

```java
public class GoalSeekingDebugger {
    public void debugOscillatorState(StateTransitionOscillator oscillator) {
        Map<String, OscillatorState> states = oscillator.getOscillatorStates();

        for (Map.Entry<String, OscillatorState> entry : states.entrySet()) {
            System.out.printf("%s: phase=%.2f, amp=%.2f, freq=%.2f%n",
                entry.getKey(),
                entry.getValue().phase,
                entry.getValue().amplitude,
                entry.getValue().frequency
            );
        }
    }

    public void debugResonance(SelectedAction action) {
        System.out.println("Selected: " + action.action);
        System.out.println("Resonance: " + action.resonanceScore);
        System.out.println("Confidence: " + action.confidence);

        for (Map.Entry<String, Float> detail : action.evaluationDetails.entrySet()) {
            System.out.printf("  %s: %.3f%n", detail.getKey(), detail.getValue());
        }
    }

    public void debugLearning(LearningFeedbackStack stack) {
        FeedbackStackMetrics metrics = stack.getMetrics();
        System.out.printf("Success rate: %.1f%%%n", metrics.getSuccessRate() * 100);
        System.out.println("Categories: " + metrics.categories);
        System.out.println("Total trials: " + metrics.total);
        System.out.println("Learning rate: " + metrics.learningRate);
        System.out.println("Vigilance: " + metrics.vigilance);
    }
}
```

### Performance Profiling

```java
public class PerformanceProfiler {
    private final Map<String, Long> timings = new HashMap<>();

    public void profile(String operation, Runnable task) {
        long start = System.nanoTime();
        task.run();
        long duration = System.nanoTime() - start;

        timings.merge(operation, duration, Long::sum);
    }

    public void printProfile() {
        System.out.println("Performance Profile:");
        timings.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .forEach(e -> System.out.printf("%s: %.2f ms%n",
                e.getKey(),
                e.getValue() / 1_000_000.0
            ));
    }

    // Usage
    public void profiledGoalSeeking() {
        profile("transition", () -> oscillator.generateTransition(current, goal));
        profile("selection", () -> selector.selectAction(current, goal, candidates));
        profile("feedback", () -> feedbackStack.processFeedback(feedback));
        profile("learning", () -> feedbackStack.learnFromEffect(modulation, effect));
    }
}
```

---

*This guide provides comprehensive examples for using the Goal-Seeking Neural Architecture. For additional support, see the [API Documentation](API.md) or [Architecture Guide](ARCHITECTURE.md).*