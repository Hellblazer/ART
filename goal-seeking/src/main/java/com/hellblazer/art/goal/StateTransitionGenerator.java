package com.hellblazer.art.goal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DEPRECATED: State Transition Generator (Legacy)
 *
 * This class is being phased out in favor of the new decomposed architecture:
 * - MultiScaleProcessor: Handles multi-scale processing and alignment
 * - GoalSeekingCoordinator: Orchestrates the overall goal-seeking process
 * - ExecutionFeedbackSystem: Manages execution with feedback loops
 *
 * Maintained for backward compatibility during migration.
 *
 * @deprecated Use {@link GoalSeekingCoordinator} instead
 */
@Deprecated
public class StateTransitionGenerator {
    private static final Logger log = LoggerFactory.getLogger(StateTransitionGenerator.class);

    // Delegate to new components
    private final GoalSeekingCoordinator coordinator;
    private final MultiScaleProcessor multiScaleProcessor;

    // Legacy layer references for compatibility
    private final GoalLayer goalLayer;
    private final StrategicLayer strategicLayer;
    private final TacticalLayer tacticalLayer;
    private final TransitionLayer transitionLayer;
    private final ExecutionLayer executionLayer;

    // State management for compatibility
    private State currentState;
    private State goalState;
    private float iterationCount = 0;

    // Legacy fields for compatibility
    private float alignmentThreshold = 0.8f;
    private float phaseAlignmentWeight = 0.4f;
    private float amplitudeCorrelationWeight = 0.3f;
    private float crossScaleCouplingWeight = 0.3f;
    private final AdaptiveTrajectoryPlanner trajectoryPlanner = new AdaptiveTrajectoryPlanner();
    private final GoalChangeRecorder goalRecorder = new GoalChangeRecorder();

    public StateTransitionGenerator() {
        log.warn("StateTransitionGenerator is deprecated. Consider using GoalSeekingCoordinator instead.");

        // Initialize new architecture components
        this.coordinator = new GoalSeekingCoordinator();
        this.multiScaleProcessor = new MultiScaleProcessor();

        // Keep legacy layers for backward compatibility
        this.goalLayer = new GoalLayer(1.0f);
        this.strategicLayer = new StrategicLayer(5.0f);
        this.tacticalLayer = new TacticalLayer(10.0f);
        this.transitionLayer = new TransitionLayer(20.0f);
        this.executionLayer = new ExecutionLayer(40.0f);
    }

    /**
     * Generate state transition by finding multi-scale alignment
     * and planning a trajectory through state space
     *
     * @deprecated Use {@link GoalSeekingCoordinator#seekGoal} instead
     */
    @Deprecated
    public StateTransition generateTransition(State from, State to) {
        return generateTransition(from, to, "default");
    }

    /**
     * Generate state transition with context for goal change tracking
     *
     * @deprecated Use {@link GoalSeekingCoordinator#seekGoal} instead
     */
    @Deprecated
    public StateTransition generateTransition(State from, State to, String context) {
        log.info("Delegating to GoalSeekingCoordinator for transition generation");

        this.currentState = from;
        this.goalState = to;

        // Adapt legacy State to new unified State interface
        var adaptedFrom = new LegacyStateAdapter(from);
        var adaptedTo = new LegacyStateAdapter(to);

        // Use new architecture with synchronous wrapper
        var controller = new LegacyExecutionController();
        var futureResult = coordinator.seekGoal(adaptedFrom, adaptedTo, context, controller);

        try {
            var result = futureResult.get();

            // Convert back to legacy format
            return convertToLegacyTransition(result, from, to);
        } catch (Exception e) {
            log.error("Failed to generate transition using new architecture, falling back to legacy", e);

            // Fallback to legacy implementation
            return legacyGenerateTransition(from, to, context);
        }
    }

    private StateTransition legacyGenerateTransition(State from, State to, String context) {
        // Keep original implementation as fallback
        initializeLayers(from, to);
        LayerState alignedState = searchForAlignment();

        // Create simple trajectory for compatibility
        var trajectory = createSimpleTrajectory(from, to);

        return extractTransition(alignedState, trajectory);
    }

    private StateTransition convertToLegacyTransition(
            GoalSeekingCoordinator.ExecutionResult result,
            State from, State to) {

        List<Action> actions = new ArrayList<>();
        StateTrajectoryPlanner.Trajectory trajectory = null;
        Strategy strategy = null;

        if (result.plan != null) {
            // Convert new actions to legacy actions
            for (var newAction : result.plan.actions) {
                var tactic = new TacticalSequence(from, newAction.confidence);
                actions.add(new Action(tactic, 0, newAction.cost));
            }

            trajectory = result.plan.trajectory;
            strategy = new Strategy(from, to, result.successScore);
        }

        return new StateTransition(from, to, actions, trajectory, strategy);
    }

    private StateTrajectoryPlanner.Trajectory createSimpleTrajectory(State from, State to) {
        // Create simple two-state trajectory for legacy compatibility
        var states = List.<StateTrajectoryPlanner.State>of(
            new TrajectoryStateAdapter(from),
            new TrajectoryStateAdapter(to)
        );
        return new StateTrajectoryPlanner.Trajectory(states);
    }

    private void initializeLayers(State from, State to) {
        goalLayer.setGoalState(to);
        strategicLayer.setStateContext(from, to);
        tacticalLayer.setCurrentState(from);
        transitionLayer.setTransitionVector(from.vectorTo(to));
        executionLayer.setExecutionContext(from);
    }

    private LayerState searchForAlignment() {
        int maxIterations = 1000;
        int iteration = 0;

        while (iteration < maxIterations) {
            // Process all layers in parallel
            LayerState state = processAllLayers();

            // Check for alignment between layers
            if (detectAlignment(state)) {
                return state;
            }

            // Adapt based on partial alignment
            adaptLayers(state);

            // Update iteration counter
            iterationCount += 0.001f; // Increment for phase calculations
            iteration++;
        }

        // Return best state even if full alignment not achieved
        return getCurrentLayerState();
    }

    private LayerState processAllLayers() {
        // Run all layers in parallel
        CompletableFuture<LayerOutput> goalFuture =
            CompletableFuture.supplyAsync(() -> goalLayer.process(iterationCount));

        CompletableFuture<LayerOutput> stratFuture =
            CompletableFuture.supplyAsync(() -> strategicLayer.process(iterationCount));

        CompletableFuture<LayerOutput> tacFuture =
            CompletableFuture.supplyAsync(() -> tacticalLayer.process(iterationCount));

        CompletableFuture<LayerOutput> transFuture =
            CompletableFuture.supplyAsync(() -> transitionLayer.process(iterationCount));

        CompletableFuture<LayerOutput> execFuture =
            CompletableFuture.supplyAsync(() -> executionLayer.process(iterationCount));

        // Wait for all to complete
        CompletableFuture.allOf(goalFuture, stratFuture, tacFuture, transFuture, execFuture).join();

        try {
            return new LayerState(
                goalFuture.get(),
                stratFuture.get(),
                tacFuture.get(),
                transFuture.get(),
                execFuture.get(),
                iterationCount
            );
        } catch (Exception e) {
            throw new RuntimeException("Layer processing failed", e);
        }
    }

    private boolean detectAlignment(LayerState state) {
        float phaseAlignment = calculatePhaseAlignment(state);
        float amplitudeCorrelation = calculateAmplitudeCorrelation(state);
        float crossScaleCoupling = calculateCrossScaleCoupling(state);

        float alignmentScore =
            phaseAlignment * phaseAlignmentWeight +
            amplitudeCorrelation * amplitudeCorrelationWeight +
            crossScaleCoupling * crossScaleCouplingWeight;

        return alignmentScore > alignmentThreshold;
    }

    private float calculatePhaseAlignment(LayerState state) {
        // Check if phases align across scales (accounting for scale factors)
        float goalStratPhase = phaseDifference(state.goal.phase, state.strategic.phase * 5);
        float stratTacPhase = phaseDifference(state.strategic.phase, state.tactical.phase * 2);
        float tacTransPhase = phaseDifference(state.tactical.phase, state.transition.phase * 2);

        // Convert phase differences to coherence (1 = perfect alignment, 0 = opposite)
        float goalStratCoherence = (1 + (float)Math.cos(goalStratPhase)) / 2;
        float stratTacCoherence = (1 + (float)Math.cos(stratTacPhase)) / 2;
        float tacTransCoherence = (1 + (float)Math.cos(tacTransPhase)) / 2;

        // Average coherence across all layer pairs
        return (goalStratCoherence + stratTacCoherence + tacTransCoherence) / 3.0f;
    }

    private float phaseDifference(float phase1, float phase2) {
        float diff = (phase1 - phase2) % (2 * (float)Math.PI);
        if (diff > Math.PI) diff -= 2 * Math.PI;
        if (diff < -Math.PI) diff += 2 * Math.PI;
        return diff;
    }

    private float calculateAmplitudeCorrelation(LayerState state) {
        // Calculate correlation between layer amplitudes
        float[] amplitudes = {
            state.goal.amplitude,
            state.strategic.amplitude,
            state.tactical.amplitude,
            state.transition.amplitude,
            state.execution.amplitude
        };

        float mean = 0;
        for (float amp : amplitudes) mean += amp;
        mean /= amplitudes.length;

        float variance = 0;
        for (float amp : amplitudes) {
            variance += (amp - mean) * (amp - mean);
        }
        variance /= amplitudes.length;

        // Low variance = high correlation
        return 1.0f / (1.0f + variance);
    }

    private float calculateCrossScaleCoupling(LayerState state) {
        // Check if slower layers modulate faster layers appropriately
        // Goal phase should influence strategic amplitude
        float goalStratCoupling = Math.abs(
            (float)Math.cos(state.goal.phase) - state.strategic.amplitude
        );

        // Strategic phase should influence tactical amplitude
        float stratTacCoupling = Math.abs(
            (float)Math.cos(state.strategic.phase) - state.tactical.amplitude
        );

        // Average coupling strength (inverted - lower difference = stronger coupling)
        float avgCoupling = 1.0f - (goalStratCoupling + stratTacCoupling) / 2.0f;

        return Math.max(0, avgCoupling);
    }

    private void adaptLayers(LayerState state) {
        // Adapt processing based on partial alignment
        float alignmentQuality = calculateAlignmentQuality(state);

        if (alignmentQuality < 0.5f) {
            // Poor alignment - increase exploration
            strategicLayer.increaseExploration(0.1f);
            tacticalLayer.increaseExploration(0.1f);
        } else if (alignmentQuality > 0.7f) {
            // Good alignment - stabilize
            strategicLayer.decreaseExploration(0.05f);
            tacticalLayer.decreaseExploration(0.05f);
        }

        // Adapt convergence based on stability
        if (isUnstable(state)) {
            // System is unstable - increase damping
            transitionLayer.increaseDamping(0.1f);
        }
    }

    private float calculateAlignmentQuality(LayerState state) {
        float phaseAlignment = calculatePhaseAlignment(state);
        float amplitudeCorrelation = calculateAmplitudeCorrelation(state);
        float crossScaleCoupling = calculateCrossScaleCoupling(state);

        return (phaseAlignment + amplitudeCorrelation + crossScaleCoupling) / 3.0f;
    }

    private boolean isUnstable(LayerState state) {
        // Detect if system is not converging
        // (Implementation would track history of states)
        return false; // Simplified
    }

    private LayerState getCurrentLayerState() {
        return processAllLayers();
    }

    private StateTransition extractTransition(LayerState state, StateTrajectoryPlanner.Trajectory trajectory) {
        // Extract coherent action from aligned layer state
        List<Action> actions = new ArrayList<>();

        // Strategic layer provides high-level plan
        Strategy strategy = strategicLayer.extractStrategy();

        // Tactical layer provides action sequences
        List<TacticalSequence> tactics = tacticalLayer.extractTactics();

        // Execution layer provides action details
        for (TacticalSequence tactic : tactics) {
            actions.addAll(executionLayer.detailActions(tactic));
        }

        return new StateTransition(
            currentState,
            goalState,
            actions,
            trajectory,
            strategy
        );
    }

    // Inner classes for processing layers

    static class GoalLayer extends BaseLayer {
        private State goalState;

        GoalLayer(float scale) {
            super(scale);
        }

        void setGoalState(State goal) {
            this.goalState = goal;
        }

        @Override
        protected float computeActivation(float phase) {
            // Goal activation based on goal state properties
            return goalState != null ? goalState.getImportance() : 1.0f;
        }
    }

    static class StrategicLayer extends BaseLayer {
        private State fromState;
        private State toState;
        private float explorationRate = 0.1f;

        StrategicLayer(float scale) {
            super(scale);
        }

        void setStateContext(State from, State to) {
            this.fromState = from;
            this.toState = to;
        }

        void increaseExploration(float delta) {
            explorationRate = Math.min(1.0f, explorationRate + delta);
        }

        void decreaseExploration(float delta) {
            explorationRate = Math.max(0.0f, explorationRate - delta);
        }

        @Override
        protected float computeActivation(float phase) {
            // Strategic activation based on state distance
            if (fromState != null && toState != null) {
                float distance = fromState.distanceTo(toState);
                return 1.0f / (1.0f + distance) + explorationRate * (float)Math.random();
            }
            return 1.0f;
        }

        Strategy extractStrategy() {
            return new Strategy(fromState, toState, amplitude);
        }
    }

    static class TacticalLayer extends BaseLayer {
        private State currentState;
        private float explorationRate = 0.1f;

        TacticalLayer(float scale) {
            super(scale);
        }

        void setCurrentState(State state) {
            this.currentState = state;
        }

        void increaseExploration(float delta) {
            explorationRate = Math.min(1.0f, explorationRate + delta);
        }

        void decreaseExploration(float delta) {
            explorationRate = Math.max(0.0f, explorationRate - delta);
        }

        @Override
        protected float computeActivation(float phase) {
            // Tactical activation based on current state
            return currentState != null ?
                currentState.getActionReadiness() + explorationRate * (float)Math.random() : 1.0f;
        }

        List<TacticalSequence> extractTactics() {
            List<TacticalSequence> tactics = new ArrayList<>();
            // Generate tactical sequences based on current amplitude
            int numTactics = Math.max(1, (int)(amplitude * 5));
            for (int i = 0; i < numTactics; i++) {
                tactics.add(new TacticalSequence(currentState, amplitude / numTactics));
            }
            return tactics;
        }
    }

    static class TransitionLayer extends BaseLayer {
        private float[] transitionVector;
        private float damping = 0.1f;

        TransitionLayer(float scale) {
            super(scale);
        }

        void setTransitionVector(float[] vector) {
            this.transitionVector = vector;
        }

        void increaseDamping(float delta) {
            damping = Math.min(0.9f, damping + delta);
        }

        @Override
        protected float computeActivation(float phase) {
            // Transition activation with damping
            float baseActivation = 1.0f;
            if (transitionVector != null) {
                float magnitude = 0;
                for (float v : transitionVector) magnitude += v * v;
                baseActivation = (float)Math.sqrt(magnitude);
            }
            return baseActivation * (1.0f - damping);
        }

        float[] getTransitionVector() {
            return transitionVector;
        }

        float getDamping() {
            return damping;
        }
    }

    static class ExecutionLayer extends BaseLayer {
        private State executionContext;

        ExecutionLayer(float scale) {
            super(scale);
        }

        void setExecutionContext(State context) {
            this.executionContext = context;
        }

        @Override
        protected float computeActivation(float phase) {
            // Execution details based on context
            return executionContext != null ?
                executionContext.getExecutionReadiness() : 1.0f;
        }

        List<Action> detailActions(TacticalSequence tactic) {
            List<Action> actions = new ArrayList<>();
            // Generate detailed actions for tactical sequence
            for (int i = 0; i < tactic.getActionCount(); i++) {
                actions.add(new Action(tactic, i, amplitude));
            }
            return actions;
        }
    }

    // Base layer class
    abstract static class BaseLayer {
        protected final float scaleFactor;
        protected float phase = 0;
        protected float amplitude = 1.0f;

        BaseLayer(float scaleFactor) {
            this.scaleFactor = scaleFactor;
        }

        LayerOutput process(float iterationCount) {
            // Phase progresses based on scale factor
            phase = (2 * (float)Math.PI * scaleFactor * iterationCount) % (2 * (float)Math.PI);
            amplitude = computeActivation(phase);
            return new LayerOutput(phase, amplitude, scaleFactor);
        }

        protected abstract float computeActivation(float phase);
    }

    // Data structures

    static class LayerState {
        final LayerOutput goal, strategic, tactical, transition, execution;
        final float iteration;

        LayerState(LayerOutput goal, LayerOutput strategic, LayerOutput tactical,
                   LayerOutput transition, LayerOutput execution, float iteration) {
            this.goal = goal;
            this.strategic = strategic;
            this.tactical = tactical;
            this.transition = transition;
            this.execution = execution;
            this.iteration = iteration;
        }
    }

    static class LayerOutput {
        final float phase;
        final float amplitude;
        final float scale;

        LayerOutput(float phase, float amplitude, float scale) {
            this.phase = phase;
            this.amplitude = amplitude;
            this.scale = scale;
        }
    }

    // Legacy interface definitions (kept for backward compatibility)

    /**
     * @deprecated Use {@link com.hellblazer.art.goal.State} instead
     */
    @Deprecated
    public interface State {
        float distanceTo(State other);
        State interpolate(State other, float t);
        float[] vectorTo(State other);
        float getImportance();
        float getActionReadiness();
        float getExecutionReadiness();
    }

    public static class StateTransition {
        public final State from, to;
        public final List<Action> actions;
        public final StateTrajectoryPlanner.Trajectory trajectory;
        public final Strategy strategy;

        StateTransition(State from, State to, List<Action> actions,
                       StateTrajectoryPlanner.Trajectory trajectory, Strategy strategy) {
            this.from = from;
            this.to = to;
            this.actions = actions;
            this.trajectory = trajectory;
            this.strategy = strategy;
        }
    }

    public static class Action {
        public final TacticalSequence tactic;
        public final int index;
        public final float strength;

        Action(TacticalSequence tactic, int index, float strength) {
            this.tactic = tactic;
            this.index = index;
            this.strength = strength;
        }
    }

    public static class Strategy {
        public final State from, to;
        public final float intensity;

        Strategy(State from, State to, float intensity) {
            this.from = from;
            this.to = to;
            this.intensity = intensity;
        }
    }

    public static class TacticalSequence {
        public final State context;
        public final float intensity;
        private final int actionCount;

        TacticalSequence(State context, float intensity) {
            this.context = context;
            this.intensity = intensity;
            this.actionCount = Math.max(1, (int)(intensity * 10));
        }

        public int getActionCount() {
            return actionCount;
        }
    }

    /**
     * Adapter to bridge between our State and StateTrajectoryPlanner.State
     */
    static class TrajectoryStateAdapter implements StateTrajectoryPlanner.State {
        private final State wrappedState;

        TrajectoryStateAdapter(State state) {
            this.wrappedState = state;
        }

        @Override
        public float distanceTo(StateTrajectoryPlanner.State other) {
            if (other instanceof TrajectoryStateAdapter adapter) {
                return wrappedState.distanceTo(adapter.wrappedState);
            }
            return Float.MAX_VALUE;
        }

        @Override
        public StateTrajectoryPlanner.State interpolate(StateTrajectoryPlanner.State other, float t) {
            if (other instanceof TrajectoryStateAdapter adapter) {
                return new TrajectoryStateAdapter(wrappedState.interpolate(adapter.wrappedState, t));
            }
            return this;
        }

        @Override
        public float[] vectorTo(StateTrajectoryPlanner.State other) {
            if (other instanceof TrajectoryStateAdapter adapter) {
                return wrappedState.vectorTo(adapter.wrappedState);
            }
            return new float[0];
        }

        State getWrappedState() {
            return wrappedState;
        }
    }

    /**
     * Record goal change for pattern learning
     */
    private void recordGoalChange(State current, State oldGoal, State newGoal, String context) {
        var metadata = new HashMap<String, Object>();
        metadata.put("alignmentThreshold", alignmentThreshold);
        metadata.put("iterationCount", iterationCount);

        goalRecorder.recordGoalChange(
            new TrajectoryStateAdapter(current),
            new TrajectoryStateAdapter(oldGoal),
            new TrajectoryStateAdapter(newGoal),
            context,
            metadata
        );

        // Adapt strategy based on learned patterns
        var strategy = goalRecorder.learnAdaptationStrategy(context);
        adaptAlignmentParameters(strategy);
    }

    /**
     * Plan trajectory with adaptive goal change handling
     */
    private StateTrajectoryPlanner.Trajectory planAdaptiveTrajectory(State from, State to) {
        var fromAdapter = new TrajectoryStateAdapter(from);
        var toAdapter = new TrajectoryStateAdapter(to);

        // Get goal change analysis
        var analysis = goalRecorder.analyzeRecent(10000); // Last 10 seconds

        if (analysis.isHighVolatility() || analysis.isRapidChanges()) {
            // Use predictive planning if goals are changing frequently
            var prediction = goalRecorder.predictNextChange(fromAdapter, toAdapter, "default");

            if (prediction.confidence > 0.5f && prediction.predictedGoal != null) {
                // Plan with uncertainty
                var possibleGoals = List.<StateTrajectoryPlanner.State>of(
                    toAdapter,
                    prediction.predictedGoal
                );
                float[] probabilities = {1.0f - prediction.confidence, prediction.confidence};

                return trajectoryPlanner.planWithUncertainty(
                    fromAdapter, toAdapter, possibleGoals, probabilities
                );
            }
        }

        // Start new trajectory or handle goal change
        if (trajectoryPlanner.getNextState() == null) {
            return trajectoryPlanner.startTrajectory(fromAdapter, toAdapter);
        } else {
            return trajectoryPlanner.handleGoalChange(toAdapter, fromAdapter);
        }
    }

    /**
     * Adapt alignment parameters based on goal change patterns
     */
    private void adaptAlignmentParameters(GoalChangeRecorder.AdaptationStrategy strategy) {
        switch (strategy) {
            case SMOOTH_BLEND:
                // Relax alignment for smooth transitions
                alignmentThreshold = Math.max(0.6f, alignmentThreshold * 0.95f);
                break;
            case FULL_REPLAN:
                // Tighten alignment for major changes
                alignmentThreshold = Math.min(0.95f, alignmentThreshold * 1.05f);
                break;
            case PREDICTIVE:
                // Balance for predictive planning
                alignmentThreshold = 0.75f;
                break;
            default:
                // Keep current settings
                break;
        }
    }

    /**
     * Get goal change statistics
     *
     * @deprecated Use {@link GoalSeekingCoordinator#getGoalChangeStatistics} instead
     */
    @Deprecated
    public GoalChangeRecorder.RecorderStatistics getGoalChangeStatistics() {
        return coordinator.getGoalChangeStatistics();
    }

    /**
     * Adapter to bridge legacy State to new unified State interface
     */
    private static class LegacyStateAdapter implements com.hellblazer.art.goal.State {
        private final StateTransitionGenerator.State legacyState;

        LegacyStateAdapter(StateTransitionGenerator.State legacy) {
            this.legacyState = legacy;
        }

        @Override
        public float distanceTo(com.hellblazer.art.goal.State other) {
            if (other instanceof LegacyStateAdapter adapter) {
                return legacyState.distanceTo(adapter.legacyState);
            }
            return Float.MAX_VALUE;
        }

        @Override
        public com.hellblazer.art.goal.State interpolate(com.hellblazer.art.goal.State other, float t) {
            if (other instanceof LegacyStateAdapter adapter) {
                return new LegacyStateAdapter(legacyState.interpolate(adapter.legacyState, t));
            }
            return this;
        }

        @Override
        public float[] vectorTo(com.hellblazer.art.goal.State other) {
            if (other instanceof LegacyStateAdapter adapter) {
                return legacyState.vectorTo(adapter.legacyState);
            }
            return new float[0];
        }

        @Override
        public float getImportance() {
            return legacyState.getImportance();
        }

        @Override
        public float getActionReadiness() {
            return legacyState.getActionReadiness();
        }

        @Override
        public float getExecutionReadiness() {
            return legacyState.getExecutionReadiness();
        }
    }

    /**
     * Legacy execution controller for compatibility
     */
    private static class LegacyExecutionController implements ExecutionFeedbackSystem.ExecutionController {
        @Override
        public ExecutionFeedbackSystem.ExecutionStep executeStep(
                com.hellblazer.art.goal.State targetState) {
            // Simple execution that returns target as actual
            return new ExecutionFeedbackSystem.ExecutionStep(targetState, false);
        }
    }
}