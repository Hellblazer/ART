package com.hellblazer.art.goal;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plans trajectories through abstract state spaces.
 *
 * This planner generates sequences of states that form a path from current to goal.
 * It learns from experience to generate better trajectories over time.
 */
public class StateTrajectoryPlanner {

    // Configuration
    private final int maxTrajectoryLength;
    private final float stepSize;
    private final int planningIterations;
    private final float convergenceThreshold;

    // Learned dynamics model - predicts next state given current state and action
    private final DynamicsModel dynamics;

    // Trajectory optimization
    private final TrajectoryOptimizer optimizer;

    // Memory of successful trajectories
    private final TrajectoryLibrary library;

    public StateTrajectoryPlanner() {
        this(100, 0.1f, 50, 0.01f);
    }

    public StateTrajectoryPlanner(int maxLength, float stepSize, int iterations, float convergence) {
        this.maxTrajectoryLength = maxLength;
        this.stepSize = stepSize;
        this.planningIterations = iterations;
        this.convergenceThreshold = convergence;

        this.dynamics = new DynamicsModel();
        this.optimizer = new TrajectoryOptimizer();
        this.library = new TrajectoryLibrary();
    }

    /**
     * Plan a trajectory from current state to goal state.
     * Returns a sequence of states that form the path.
     */
    public Trajectory planTrajectory(State current, State goal) {
        // First check if we have a similar trajectory in memory
        var cached = library.findSimilarTrajectory(current, goal);
        if (cached != null && cached.isValid()) {
            return adaptTrajectory(cached, current, goal);
        }

        // Generate initial trajectory estimate
        var trajectory = generateInitialTrajectory(current, goal);

        // Optimize the trajectory
        for (int iter = 0; iter < planningIterations; iter++) {
            var improved = optimizer.optimize(trajectory, goal);

            // Check convergence
            float improvement = trajectory.cost() - improved.cost();
            trajectory = improved;

            if (improvement < convergenceThreshold) {
                break; // Converged
            }
        }

        // Store successful trajectory for future use
        if (trajectory.reachesGoal(goal)) {
            library.store(trajectory);
        }

        return trajectory;
    }

    /**
     * Generate initial trajectory using simple interpolation or learned dynamics
     */
    private Trajectory generateInitialTrajectory(State current, State goal) {
        var states = new ArrayList<State>();
        states.add(current);

        State state = current;
        float remainingDistance = current.distanceTo(goal);

        for (int step = 0; step < maxTrajectoryLength && remainingDistance > convergenceThreshold; step++) {
            // Compute direction toward goal
            float[] direction = state.vectorTo(goal);

            // Apply dynamics model if we have learned one
            State nextState;
            if (dynamics.isTrained()) {
                // Use learned dynamics to predict next state
                nextState = dynamics.predictNextState(state, direction, stepSize);
            } else {
                // Simple interpolation
                float t = Math.min(stepSize, remainingDistance / state.distanceTo(goal));
                nextState = state.interpolate(goal, t);
            }

            states.add(nextState);

            // Update for next iteration
            state = nextState;
            remainingDistance = state.distanceTo(goal);
        }

        return new Trajectory(states);
    }

    /**
     * Adapt a cached trajectory to new start/goal states
     */
    private Trajectory adaptTrajectory(Trajectory cached, State newStart, State newGoal) {
        var adapted = new ArrayList<State>();
        adapted.add(newStart);

        // Transform cached waypoints to new coordinate frame
        State cachedStart = cached.states.get(0);
        State cachedGoal = cached.states.get(cached.states.size() - 1);

        for (int i = 1; i < cached.states.size(); i++) {
            State cachedState = cached.states.get(i);

            // Compute relative position in cached trajectory
            float progress = (float)i / cached.states.size();

            // Interpolate in new trajectory
            State adaptedState = newStart.interpolate(newGoal, progress);

            // Blend with cached dynamics
            if (dynamics.isTrained()) {
                adaptedState = dynamics.blendStates(adaptedState, cachedState, 0.3f);
            }

            adapted.add(adaptedState);
        }

        return new Trajectory(adapted);
    }

    /**
     * Learn from executed trajectory - update dynamics model
     */
    public void learnFromExecution(Trajectory planned, Trajectory actual, float success) {
        // Update dynamics model with actual vs planned
        for (int i = 0; i < Math.min(planned.length() - 1, actual.length() - 1); i++) {
            State plannedCurrent = planned.states.get(i);
            State plannedNext = planned.states.get(i + 1);
            State actualNext = actual.states.get(i + 1);

            dynamics.updateModel(plannedCurrent, plannedNext, actualNext);
        }

        // Store successful trajectories
        if (success > 0.8f) {
            library.store(actual);
        }

        // Update optimizer parameters based on success
        optimizer.adaptParameters(success);
    }

    // ============= Inner Classes =============

    /**
     * Represents a trajectory through state space
     */
    public static class Trajectory {
        public final List<State> states;
        private Float cachedCost = null;

        public Trajectory(List<State> states) {
            this.states = new ArrayList<>(states);
        }

        public int length() {
            return states.size();
        }

        public State getState(int index) {
            return states.get(index);
        }

        public boolean reachesGoal(State goal) {
            if (states.isEmpty()) return false;
            State last = states.get(states.size() - 1);
            return last.distanceTo(goal) < 0.01f;
        }

        public float cost() {
            if (cachedCost == null) {
                cachedCost = computeCost();
            }
            return cachedCost;
        }

        private float computeCost() {
            float cost = 0;

            // Path length cost
            for (int i = 0; i < states.size() - 1; i++) {
                cost += states.get(i).distanceTo(states.get(i + 1));
            }

            // Smoothness cost (acceleration)
            for (int i = 1; i < states.size() - 1; i++) {
                State prev = states.get(i - 1);
                State curr = states.get(i);
                State next = states.get(i + 1);

                float[] v1 = prev.vectorTo(curr);
                float[] v2 = curr.vectorTo(next);

                // Acceleration = change in velocity
                float accel = 0;
                for (int j = 0; j < Math.min(v1.length, v2.length); j++) {
                    float diff = v2[j] - v1[j];
                    accel += diff * diff;
                }

                cost += Math.sqrt(accel) * 0.1f; // Weight smoothness
            }

            return cost;
        }

        public boolean isValid() {
            return states.size() > 1 && cost() < Float.MAX_VALUE;
        }
    }

    /**
     * Learns and predicts system dynamics
     */
    class DynamicsModel {
        private final Map<StateTransition, StateUpdate> model = new ConcurrentHashMap<>();
        private boolean trained = false;

        public boolean isTrained() {
            return trained && model.size() > 100;
        }

        public State predictNextState(State current, float[] action, float stepSize) {
            // Find similar transitions in model
            var similar = findSimilarTransitions(current, action);

            if (similar.isEmpty()) {
                // No model available - use simple integration
                return simpleIntegration(current, action, stepSize);
            }

            // Weighted average of predictions
            State prediction = null;
            float totalWeight = 0;

            for (var entry : similar.entrySet()) {
                float weight = entry.getValue();
                State predicted = entry.getKey();

                if (prediction == null) {
                    prediction = predicted;
                } else {
                    // Blend predictions
                    float t = weight / (totalWeight + weight);
                    prediction = prediction.interpolate(predicted, t);
                }
                totalWeight += weight;
            }

            return prediction;
        }

        public void updateModel(State from, State plannedNext, State actualNext) {
            var key = new StateTransition(from, plannedNext);
            var update = new StateUpdate(actualNext, System.currentTimeMillis());

            model.merge(key, update, (old, new_) -> {
                // Exponential moving average
                return old.blend(new_, 0.3f);
            });

            trained = true;
        }

        public State blendStates(State s1, State s2, float t) {
            return s1.interpolate(s2, t);
        }

        private Map<State, Float> findSimilarTransitions(State current, float[] action) {
            var similar = new HashMap<State, Float>();

            for (var entry : model.entrySet()) {
                var transition = entry.getKey();
                float similarity = computeSimilarity(current, transition.from);

                if (similarity > 0.8f) {
                    similar.put(entry.getValue().resultState, similarity);
                }
            }

            return similar;
        }

        private float computeSimilarity(State s1, State s2) {
            float distance = s1.distanceTo(s2);
            return Math.max(0, 1.0f - distance);
        }

        private State simpleIntegration(State current, float[] direction, float stepSize) {
            // Create a temporary goal in the direction of action
            // This is a fallback when we have no learned dynamics

            // Normalize direction
            float magnitude = 0;
            for (float d : direction) {
                magnitude += d * d;
            }
            magnitude = (float) Math.sqrt(magnitude);

            if (magnitude < 0.001f) {
                return current; // No movement
            }

            // Scale by step size
            float scale = stepSize / magnitude;

            // For now, just move a small step in that direction
            // This would be replaced by actual dynamics in a real system
            return current.interpolate(current, 1.0f + scale);
        }
    }

    /**
     * Optimizes trajectories for smoothness and efficiency
     */
    class TrajectoryOptimizer {
        private float smoothingWeight = 0.5f;
        private float efficiencyWeight = 0.5f;

        public Trajectory optimize(Trajectory initial, State goal) {
            var optimized = new ArrayList<>(initial.states);

            // Apply smoothing
            for (int iter = 0; iter < 5; iter++) {
                optimized = new ArrayList<>(smooth(optimized));
            }

            // Apply shortcutting
            optimized = new ArrayList<>(shortcut(optimized, goal));

            return new Trajectory(optimized);
        }

        private List<State> smooth(List<State> states) {
            if (states.size() < 3) return states;

            var smoothed = new ArrayList<State>();
            smoothed.add(states.get(0)); // Keep start

            for (int i = 1; i < states.size() - 1; i++) {
                State prev = states.get(i - 1);
                State curr = states.get(i);
                State next = states.get(i + 1);

                // Weighted average for smoothing
                State smooth1 = prev.interpolate(curr, 0.75f);
                State smooth2 = smooth1.interpolate(next, 0.5f);

                smoothed.add(smooth2);
            }

            smoothed.add(states.get(states.size() - 1)); // Keep goal
            return smoothed;
        }

        private List<State> shortcut(List<State> states, State goal) {
            if (states.size() < 3) return states;

            var shortened = new ArrayList<State>();
            shortened.add(states.get(0));

            int i = 0;
            while (i < states.size() - 1) {
                // Try to skip ahead
                int bestSkip = 1;
                for (int skip = 2; skip <= Math.min(5, states.size() - i - 1); skip++) {
                    State from = states.get(i);
                    State to = states.get(i + skip);

                    // Check if direct path is valid (no obstacles in real system)
                    if (isValidPath(from, to)) {
                        bestSkip = skip;
                    }
                }

                i += bestSkip;
                if (i < states.size()) {
                    shortened.add(states.get(i));
                }
            }

            return shortened;
        }

        private boolean isValidPath(State from, State to) {
            // In real system, check for obstacles/constraints
            // For now, always valid
            return true;
        }

        public void adaptParameters(float success) {
            if (success < 0.5f) {
                // Poor performance - more smoothing
                smoothingWeight = Math.min(0.8f, smoothingWeight * 1.1f);
            } else if (success > 0.9f) {
                // Good performance - can be more aggressive
                efficiencyWeight = Math.min(0.8f, efficiencyWeight * 1.1f);
                smoothingWeight = Math.max(0.3f, smoothingWeight * 0.95f);
            }
        }
    }

    /**
     * Stores and retrieves successful trajectories
     */
    class TrajectoryLibrary {
        private final List<StoredTrajectory> trajectories = new ArrayList<>();
        private final int maxStored = 1000;

        public void store(Trajectory trajectory) {
            if (trajectory.isValid()) {
                trajectories.add(new StoredTrajectory(trajectory, System.currentTimeMillis()));

                // Prune old trajectories
                if (trajectories.size() > maxStored) {
                    trajectories.sort((a, b) -> Long.compare(b.timestamp, a.timestamp));
                    trajectories.subList(maxStored, trajectories.size()).clear();
                }
            }
        }

        public Trajectory findSimilarTrajectory(State start, State goal) {
            float bestScore = Float.MAX_VALUE;
            Trajectory best = null;

            for (var stored : trajectories) {
                var traj = stored.trajectory;
                if (traj.states.size() < 2) continue;

                State trajStart = traj.states.get(0);
                State trajEnd = traj.states.get(traj.states.size() - 1);

                float startDist = start.distanceTo(trajStart);
                float goalDist = goal.distanceTo(trajEnd);
                float score = startDist + goalDist;

                if (score < bestScore && score < 0.2f) {
                    bestScore = score;
                    best = traj;
                }
            }

            return best;
        }
    }

    // ============= Helper Classes =============

    static class StateTransition {
        final State from;
        final State to;

        StateTransition(State from, State to) {
            this.from = from;
            this.to = to;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof StateTransition that)) return false;
            return from.equals(that.from) && to.equals(that.to);
        }

        @Override
        public int hashCode() {
            return Objects.hash(from, to);
        }
    }

    static class StateUpdate {
        final State resultState;
        final long timestamp;

        StateUpdate(State result, long time) {
            this.resultState = result;
            this.timestamp = time;
        }

        StateUpdate blend(StateUpdate other, float t) {
            State blended = resultState.interpolate(other.resultState, t);
            return new StateUpdate(blended, System.currentTimeMillis());
        }
    }

    static class StoredTrajectory {
        final Trajectory trajectory;
        final long timestamp;
        int useCount = 0;

        StoredTrajectory(Trajectory traj, long time) {
            this.trajectory = traj;
            this.timestamp = time;
        }
    }

    // State interface (matches StateTransitionGenerator.State)
    public interface State {
        float distanceTo(State other);
        State interpolate(State other, float t);
        float[] vectorTo(State other);
    }
}