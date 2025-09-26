package com.hellblazer.art.goal;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Execution feedback system that closes the loop between planning and execution.
 *
 * Monitors execution of planned trajectories, detects divergence,
 * triggers replanning when necessary, and learns from execution outcomes.
 */
public class ExecutionFeedbackSystem {
    private static final Logger log = LoggerFactory.getLogger(ExecutionFeedbackSystem.class);

    // Core components
    private final StateTrajectoryPlanner trajectoryPlanner;
    private final TransitionLibrary transitionLibrary;
    private final LearningFeedbackStack feedbackStack;
    private final GoalChangeRecorder goalRecorder;

    // Execution monitoring
    private ExecutionMonitor currentMonitor;
    private final List<ExecutionHistory> history;
    private final Map<String, ExecutionStatistics> statistics;

    // Configuration
    private float divergenceThreshold = 0.1f;
    private float successThreshold = 0.8f;
    private int maxHistorySize = 1000;

    public ExecutionFeedbackSystem(
            StateTrajectoryPlanner planner,
            TransitionLibrary library,
            LearningFeedbackStack feedback,
            GoalChangeRecorder recorder) {
        this.trajectoryPlanner = planner;
        this.transitionLibrary = library;
        this.feedbackStack = feedback;
        this.goalRecorder = recorder;
        this.history = new ArrayList<>();
        this.statistics = new HashMap<>();
    }

    /**
     * Execute a planned trajectory with feedback and monitoring.
     */
    public CompletableFuture<ExecutionResult> executeTrajectory(
            StateTrajectoryPlanner.Trajectory planned,
            ExecutionController controller) {

        // Create execution monitor
        currentMonitor = new ExecutionMonitor(planned, controller);

        // Start asynchronous execution
        return CompletableFuture.supplyAsync(() -> {
            try {
                return executeWithMonitoring(currentMonitor);
            } catch (Exception e) {
                log.error("Execution failed", e);
                return ExecutionResult.failure(e);
            }
        });
    }

    private ExecutionResult executeWithMonitoring(ExecutionMonitor monitor) {
        var planned = monitor.plannedTrajectory;
        var controller = monitor.controller;
        var actualStates = new ArrayList<State>();

        State currentState = null;
        int stepCount = 0;

        for (int i = 0; i < planned.length(); i++) {
            StateTrajectoryPlanner.State wrappedPlannedState = planned.getState(i);
            State plannedState = unwrapState(wrappedPlannedState);

            // Execute step
            ExecutionStep step = controller.executeStep(plannedState);
            actualStates.add(step.actualState);
            currentState = step.actualState;

            // Monitor divergence
            float divergence = step.actualState.distanceTo(plannedState);
            monitor.recordDivergence(i, divergence);

            // Check if replanning needed
            if (shouldReplan(divergence, monitor)) {
                log.info("Replanning triggered at step {} with divergence {}",
                        i, divergence);

                // Trigger replanning
                var replanned = replan(currentState, planned, i);
                if (replanned != null) {
                    // Update plan and continue
                    planned = replanned;
                    monitor.updatePlan(replanned);
                }
            }

            // Check for early termination
            if (step.shouldTerminate) {
                log.info("Early termination at step {}", i);
                break;
            }

            stepCount++;
        }

        // Create actual trajectory
        var actualTrajectory = new StateTrajectoryPlanner.Trajectory(
            wrapStates(actualStates)
        );

        // Calculate success metric
        float success = calculateSuccess(planned, actualTrajectory, currentState);

        // Learn from execution
        learnFromExecution(planned, actualTrajectory, success);

        // Record history
        recordExecution(planned, actualTrajectory, success);

        return new ExecutionResult(
            actualTrajectory,
            success,
            stepCount,
            monitor.getAverageDivergence()
        );
    }

    private boolean shouldReplan(float divergence, ExecutionMonitor monitor) {
        // Replan if divergence exceeds threshold
        if (divergence > divergenceThreshold) {
            return true;
        }

        // Replan if divergence is increasing rapidly
        if (monitor.isDivergenceIncreasing()) {
            return true;
        }

        return false;
    }

    private StateTrajectoryPlanner.Trajectory replan(
            State currentState,
            StateTrajectoryPlanner.Trajectory originalPlan,
            int currentIndex) {

        // Get original goal
        StateTrajectoryPlanner.State wrappedGoal = originalPlan.getState(originalPlan.length() - 1);

        // Replan from current position
        return trajectoryPlanner.planTrajectory(
            wrapState(currentState),
            wrappedGoal
        );
    }

    private void learnFromExecution(
            StateTrajectoryPlanner.Trajectory planned,
            StateTrajectoryPlanner.Trajectory actual,
            float success) {

        // Update trajectory planner's dynamics model
        trajectoryPlanner.learnFromExecution(planned, actual, success);

        // Update transition library
        for (int i = 0; i < Math.min(planned.length() - 1, actual.length() - 1); i++) {
            State from = unwrapState(actual.getState(i));
            State to = unwrapState(actual.getState(i + 1));

            // Create action representation
            var action = new TransitionLibrary.Action(
                "execution",
                from.vectorTo(to),
                1.0f / success  // Cost inversely proportional to success
            );

            // Learn transition
            transitionLibrary.learnTransition(
                wrapLibraryState(from),
                wrapLibraryState(to),
                action,
                success
            );
        }

        // Update feedback stack
        float[] feedbackVector = createFeedbackVector(planned, actual, success);
        var feedbackData = new LearningFeedbackStack.FeedbackData(
            feedbackVector,
            System.currentTimeMillis()
        );

        var modulation = feedbackStack.processFeedback(feedbackData);
        var effect = new LearningFeedbackStack.Effect(success > successThreshold, success);
        feedbackStack.learnFromEffect(modulation, effect);
    }

    private float calculateSuccess(
            StateTrajectoryPlanner.Trajectory planned,
            StateTrajectoryPlanner.Trajectory actual,
            State finalState) {

        // Goal achievement
        State goal = unwrapState(planned.getState(planned.length() - 1));
        float goalDistance = finalState.distanceTo(goal);
        float goalScore = 1.0f / (1.0f + goalDistance);

        // Path efficiency
        float plannedCost = planned.cost();
        float actualCost = actual.cost();
        float efficiencyScore = plannedCost / (actualCost + 0.01f);
        efficiencyScore = Math.min(1.0f, efficiencyScore);

        // Smoothness
        float smoothnessScore = calculateSmoothness(actual);

        // Combined success metric
        return 0.5f * goalScore + 0.3f * efficiencyScore + 0.2f * smoothnessScore;
    }

    private float calculateSmoothness(StateTrajectoryPlanner.Trajectory trajectory) {
        if (trajectory.length() < 3) return 1.0f;

        float totalJerk = 0;
        for (int i = 2; i < trajectory.length(); i++) {
            State prev = unwrapState(trajectory.getState(i - 2));
            State curr = unwrapState(trajectory.getState(i - 1));
            State next = unwrapState(trajectory.getState(i));

            float[] v1 = prev.vectorTo(curr);
            float[] v2 = curr.vectorTo(next);

            // Calculate jerk (change in acceleration)
            float jerk = 0;
            for (int j = 0; j < Math.min(v1.length, v2.length); j++) {
                float accelChange = v2[j] - v1[j];
                jerk += accelChange * accelChange;
            }

            totalJerk += Math.sqrt(jerk);
        }

        // Convert to smoothness score
        float avgJerk = totalJerk / (trajectory.length() - 2);
        return 1.0f / (1.0f + avgJerk);
    }

    private void recordExecution(
            StateTrajectoryPlanner.Trajectory planned,
            StateTrajectoryPlanner.Trajectory actual,
            float success) {

        var record = new ExecutionHistory(
            System.currentTimeMillis(),
            planned,
            actual,
            success
        );

        history.add(record);

        // Prune old history
        while (history.size() > maxHistorySize) {
            history.remove(0);
        }

        // Update statistics
        updateStatistics(record);
    }

    private void updateStatistics(ExecutionHistory record) {
        String context = "default"; // Could be parameterized

        var stats = statistics.computeIfAbsent(context, k -> new ExecutionStatistics());
        stats.update(record);
    }

    private float[] createFeedbackVector(
            StateTrajectoryPlanner.Trajectory planned,
            StateTrajectoryPlanner.Trajectory actual,
            float success) {

        // Create feedback vector from execution characteristics
        return new float[]{
            success,
            planned.cost(),
            actual.cost(),
            calculateSmoothness(actual),
            actual.length() / (float) planned.length()
        };
    }

    // Wrapper methods for state interface compatibility

    private StateTrajectoryPlanner.State wrapState(State state) {
        return new StateWrapper(state);
    }

    private State unwrapState(StateTrajectoryPlanner.State state) {
        if (state instanceof StateWrapper wrapper) {
            return wrapper.wrapped;
        }
        throw new IllegalArgumentException("Cannot unwrap non-wrapper state");
    }

    private List<StateTrajectoryPlanner.State> wrapStates(List<State> states) {
        return states.stream().map(this::wrapState).toList();
    }

    private TransitionLibrary.State wrapLibraryState(State state) {
        return new LibraryStateWrapper(state);
    }

    // Inner classes

    /**
     * Monitors execution and tracks divergence.
     */
    static class ExecutionMonitor {
        final StateTrajectoryPlanner.Trajectory plannedTrajectory;
        final ExecutionController controller;
        final List<Float> divergences;

        StateTrajectoryPlanner.Trajectory currentPlan;

        ExecutionMonitor(StateTrajectoryPlanner.Trajectory planned, ExecutionController controller) {
            this.plannedTrajectory = planned;
            this.controller = controller;
            this.divergences = new ArrayList<>();
            this.currentPlan = planned;
        }

        void recordDivergence(int step, float divergence) {
            divergences.add(divergence);
        }

        void updatePlan(StateTrajectoryPlanner.Trajectory newPlan) {
            this.currentPlan = newPlan;
        }

        boolean isDivergenceIncreasing() {
            if (divergences.size() < 3) return false;

            int n = divergences.size();
            float recent = divergences.get(n - 1);
            float prev = divergences.get(n - 2);
            float prevPrev = divergences.get(n - 3);

            return recent > prev && prev > prevPrev;
        }

        float getAverageDivergence() {
            if (divergences.isEmpty()) return 0;

            float sum = 0;
            for (float d : divergences) {
                sum += d;
            }
            return sum / divergences.size();
        }
    }

    /**
     * Interface for execution control.
     */
    public interface ExecutionController {
        ExecutionStep executeStep(State targetState);
    }

    /**
     * Result of executing a single step.
     */
    public static class ExecutionStep {
        public final State actualState;
        public final boolean shouldTerminate;
        public final Map<String, Object> metadata;

        public ExecutionStep(State actual, boolean terminate) {
            this(actual, terminate, new HashMap<>());
        }

        public ExecutionStep(State actual, boolean terminate, Map<String, Object> metadata) {
            this.actualState = actual;
            this.shouldTerminate = terminate;
            this.metadata = metadata;
        }
    }

    /**
     * Result of trajectory execution.
     */
    public static class ExecutionResult {
        public final StateTrajectoryPlanner.Trajectory actualTrajectory;
        public final float successScore;
        public final int stepsExecuted;
        public final float averageDivergence;
        public final boolean successful;

        ExecutionResult(StateTrajectoryPlanner.Trajectory actual, float success,
                       int steps, float avgDivergence) {
            this.actualTrajectory = actual;
            this.successScore = success;
            this.stepsExecuted = steps;
            this.averageDivergence = avgDivergence;
            this.successful = success > 0.5f;
        }

        static ExecutionResult failure(Exception e) {
            return new ExecutionResult(null, 0, 0, Float.MAX_VALUE);
        }
    }

    /**
     * Historical record of execution.
     */
    static class ExecutionHistory {
        final long timestamp;
        final StateTrajectoryPlanner.Trajectory planned;
        final StateTrajectoryPlanner.Trajectory actual;
        final float successScore;

        ExecutionHistory(long time, StateTrajectoryPlanner.Trajectory planned,
                        StateTrajectoryPlanner.Trajectory actual, float success) {
            this.timestamp = time;
            this.planned = planned;
            this.actual = actual;
            this.successScore = success;
        }
    }

    /**
     * Execution statistics.
     */
    static class ExecutionStatistics {
        int totalExecutions = 0;
        int successfulExecutions = 0;
        float totalSuccess = 0;
        float totalDivergence = 0;

        void update(ExecutionHistory record) {
            totalExecutions++;
            totalSuccess += record.successScore;

            if (record.successScore > 0.5f) {
                successfulExecutions++;
            }

            // Calculate divergence
            for (int i = 0; i < Math.min(record.planned.length(), record.actual.length()); i++) {
                float div = record.planned.getState(i).distanceTo(record.actual.getState(i));
                totalDivergence += div;
            }
        }

        float getSuccessRate() {
            return totalExecutions > 0 ? (float) successfulExecutions / totalExecutions : 0;
        }

        float getAverageSuccess() {
            return totalExecutions > 0 ? totalSuccess / totalExecutions : 0;
        }

        float getAverageDivergence() {
            return totalExecutions > 0 ? totalDivergence / totalExecutions : 0;
        }
    }

    // State wrapper classes for interface compatibility

    static class StateWrapper implements StateTrajectoryPlanner.State {
        final State wrapped;

        StateWrapper(State state) {
            this.wrapped = state;
        }

        @Override
        public float distanceTo(StateTrajectoryPlanner.State other) {
            if (other instanceof StateWrapper wrapper) {
                return wrapped.distanceTo(wrapper.wrapped);
            }
            return Float.MAX_VALUE;
        }

        @Override
        public StateTrajectoryPlanner.State interpolate(StateTrajectoryPlanner.State other, float t) {
            if (other instanceof StateWrapper wrapper) {
                return new StateWrapper(wrapped.interpolate(wrapper.wrapped, t));
            }
            return this;
        }

        @Override
        public float[] vectorTo(StateTrajectoryPlanner.State other) {
            if (other instanceof StateWrapper wrapper) {
                return wrapped.vectorTo(wrapper.wrapped);
            }
            return new float[0];
        }
    }

    static class LibraryStateWrapper implements TransitionLibrary.State {
        final State wrapped;

        LibraryStateWrapper(State state) {
            this.wrapped = state;
        }

        @Override
        public float distanceTo(TransitionLibrary.State other) {
            if (other instanceof LibraryStateWrapper wrapper) {
                return wrapped.distanceTo(wrapper.wrapped);
            }
            return Float.MAX_VALUE;
        }
    }

    // Configuration methods

    public void setDivergenceThreshold(float threshold) {
        this.divergenceThreshold = Math.max(0.01f, threshold);
    }

    public void setSuccessThreshold(float threshold) {
        this.successThreshold = Math.max(0.1f, Math.min(1.0f, threshold));
    }

    public ExecutionStatistics getStatistics(String context) {
        return statistics.getOrDefault(context, new ExecutionStatistics());
    }
}