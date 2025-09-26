package com.hellblazer.art.goal;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Trajectory planner that respects state and transition constraints.
 *
 * Extends the base trajectory planner with:
 * - State space constraints (boundaries, obstacles)
 * - Transition constraints (velocity limits, forbidden transitions)
 * - Path constraints (smoothness, energy)
 * - Collision avoidance
 */
public class ConstrainedTrajectoryPlanner extends AdaptiveTrajectoryPlanner {
    private static final Logger log = LoggerFactory.getLogger(ConstrainedTrajectoryPlanner.class);

    // Constraint management
    private final List<TrajectoryConstraint> constraints;
    private final ObstacleMap obstacleMap;
    private final StateValidator validator;

    // Configuration
    private float constraintViolationPenalty = 10.0f;
    private int maxProjectionIterations = 10;
    private float projectionTolerance = 0.01f;

    public ConstrainedTrajectoryPlanner() {
        super();
        this.constraints = new ArrayList<>();
        this.obstacleMap = new ObstacleMap();
        this.validator = new StateValidator();
    }

    /**
     * Add a trajectory constraint.
     */
    public void addConstraint(TrajectoryConstraint constraint) {
        constraints.add(constraint);
        log.debug("Added constraint: {}", constraint.getName());
    }

    /**
     * Add an obstacle to avoid.
     */
    public void addObstacle(Obstacle obstacle) {
        obstacleMap.addObstacle(obstacle);
    }

    @Override
    public Trajectory planTrajectory(State current, State goal) {
        // First check if goal is reachable
        if (!isReachable(current, goal)) {
            log.warn("Goal state is not reachable from current state");
            return createFailureTrajectory(current);
        }

        // Plan initial trajectory
        var trajectory = super.planTrajectory(current, goal);

        // Apply constraints
        trajectory = enforceConstraints(trajectory);

        // Avoid obstacles
        trajectory = avoidObstacles(trajectory);

        // Validate final trajectory
        if (!validateTrajectory(trajectory)) {
            log.warn("Generated trajectory violates constraints");
            trajectory = repairTrajectory(trajectory);
        }

        return trajectory;
    }

    /**
     * Check if goal is reachable considering constraints.
     */
    private boolean isReachable(State from, State to) {
        // For now, assume states are valid if they exist
        // Actual constraint checking happens during trajectory execution
        // TODO: Implement proper reachability check with unwrapping
        return from != null && to != null;
    }

    /**
     * Enforce constraints on trajectory.
     */
    private Trajectory enforceConstraints(Trajectory trajectory) {
        var states = new ArrayList<com.hellblazer.art.goal.State>();

        for (int i = 0; i < trajectory.length(); i++) {
            com.hellblazer.art.goal.State state = unwrapState(trajectory.getState(i));

            // Project to valid state if needed
            com.hellblazer.art.goal.State validState = projectToValid(state);

            // Check transition constraints
            if (i > 0 && !states.isEmpty()) {
                com.hellblazer.art.goal.State prevState = states.get(states.size() - 1);
                if (!canTransition(prevState, validState)) {
                    // Insert intermediate states if needed
                    var intermediates = createValidTransition(prevState, validState);
                    states.addAll(intermediates);
                } else {
                    states.add(validState);
                }
            } else {
                states.add(validState);
            }
        }

        return new Trajectory(wrapStates(states));
    }

    /**
     * Avoid obstacles in trajectory.
     */
    private Trajectory avoidObstacles(Trajectory trajectory) {
        var states = new ArrayList<com.hellblazer.art.goal.State>();
        com.hellblazer.art.goal.State lastValid = null;

        for (int i = 0; i < trajectory.length(); i++) {
            com.hellblazer.art.goal.State state = unwrapState(trajectory.getState(i));

            if (obstacleMap.intersects(state)) {
                // State is inside obstacle - find way around
                if (lastValid != null && i < trajectory.length() - 1) {
                    com.hellblazer.art.goal.State nextValid = findNextValidState(trajectory, i);
                    if (nextValid != null) {
                        var avoidancePath = planAvoidancePath(lastValid, nextValid);
                        states.addAll(avoidancePath);
                        // Skip to next valid state
                        while (i < trajectory.length() - 1 &&
                               obstacleMap.intersects(unwrapState(trajectory.getState(i)))) {
                            i++;
                        }
                        i--; // Adjust for loop increment
                    }
                }
            } else {
                states.add(state);
                lastValid = state;
            }
        }

        return new Trajectory(wrapStates(states));
    }

    /**
     * Project state to satisfy constraints.
     */
    private com.hellblazer.art.goal.State projectToValid(com.hellblazer.art.goal.State state) {
        com.hellblazer.art.goal.State projected = state;
        int iterations = 0;

        while (!validator.isValid(projected) && iterations < maxProjectionIterations) {
            com.hellblazer.art.goal.State newProjected = projected;

            // Apply each constraint projection
            for (var constraint : constraints) {
                if (!constraint.isSatisfied(projected)) {
                    newProjected = constraint.project(newProjected);
                }
            }

            // Check convergence
            if (projected.distanceTo(newProjected) < projectionTolerance) {
                break;
            }

            projected = newProjected;
            iterations++;
        }

        return projected;
    }

    /**
     * Check if transition between states is valid.
     */
    private boolean canTransition(State from, State to) {
        // Check state constraints
        if (!from.canTransitionTo(to)) {
            return false;
        }

        // Check trajectory constraints
        for (var constraint : constraints) {
            if (!constraint.canTransition(from, to)) {
                return false;
            }
        }

        // Check for obstacle intersection
        if (isPathBlocked(from, to)) {
            return false;
        }

        return true;
    }

    /**
     * Create valid transition between states.
     */
    private List<State> createValidTransition(State from, State to) {
        var states = new ArrayList<State>();

        // Simple interpolation with projection
        int steps = 10;
        for (int i = 1; i < steps; i++) {
            float t = (float) i / steps;
            State interpolated = from.interpolate(to, t);
            State valid = projectToValid(interpolated);
            states.add(valid);
        }

        return states;
    }

    /**
     * Check if path is blocked by obstacles.
     */
    private boolean isPathBlocked(State from, State to) {
        // Sample points along path
        int samples = 20;
        for (int i = 0; i <= samples; i++) {
            float t = (float) i / samples;
            State point = from.interpolate(to, t);
            if (obstacleMap.intersects(point)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Check for indirect path when direct is blocked.
     */
    private boolean checkIndirectPath(State from, State to) {
        // Simplified - in practice would use proper path planning
        // Try a few random intermediate points
        Random rand = new Random();
        for (int attempt = 0; attempt < 10; attempt++) {
            State intermediate = generateRandomValidState(from, to, rand);
            if (!isPathBlocked(from, intermediate) &&
                !isPathBlocked(intermediate, to)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Plan path to avoid obstacles.
     */
    private List<State> planAvoidancePath(State from, State to) {
        // Simplified avoidance - in practice use RRT* or similar
        var path = new ArrayList<State>();

        // Try to go around obstacle
        State detour = findDetourPoint(from, to);
        if (detour != null && !obstacleMap.intersects(detour)) {
            // Path through detour
            path.addAll(createValidTransition(from, detour));
            path.add(detour);
            path.addAll(createValidTransition(detour, to));
        } else {
            // Fallback to simple interpolation
            path.addAll(createValidTransition(from, to));
        }

        return path;
    }

    /**
     * Find detour point around obstacle.
     */
    private com.hellblazer.art.goal.State findDetourPoint(com.hellblazer.art.goal.State from, com.hellblazer.art.goal.State to) {
        // Find perpendicular direction
        float[] direction = from.vectorTo(to);
        float[] perpendicular = computePerpendicular(direction);

        // Try detour points
        for (float offset : new float[]{1.0f, -1.0f, 2.0f, -2.0f}) {
            com.hellblazer.art.goal.State detour = createOffsetState(from, to, perpendicular, offset);
            if (!obstacleMap.intersects(detour) && validator.isValid(detour)) {
                return detour;
            }
        }

        return null;
    }

    /**
     * Find next valid state in trajectory.
     */
    private com.hellblazer.art.goal.State findNextValidState(Trajectory trajectory, int startIndex) {
        for (int i = startIndex + 1; i < trajectory.length(); i++) {
            com.hellblazer.art.goal.State state = unwrapState(trajectory.getState(i));
            if (!obstacleMap.intersects(state) && validator.isValid(state)) {
                return state;
            }
        }
        return null;
    }

    /**
     * Validate entire trajectory.
     */
    private boolean validateTrajectory(Trajectory trajectory) {
        for (int i = 0; i < trajectory.length(); i++) {
            com.hellblazer.art.goal.State state = unwrapState(trajectory.getState(i));

            // Check state validity
            if (!validator.isValid(state)) {
                return false;
            }

            // Check transition validity
            if (i > 0) {
                State prev = unwrapState(trajectory.getState(i - 1));
                if (!canTransition(prev, state)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Repair invalid trajectory.
     */
    private Trajectory repairTrajectory(Trajectory trajectory) {
        log.debug("Repairing invalid trajectory");

        var repaired = new ArrayList<com.hellblazer.art.goal.State>();
        com.hellblazer.art.goal.State lastValid = null;

        for (int i = 0; i < trajectory.length(); i++) {
            com.hellblazer.art.goal.State state = unwrapState(trajectory.getState(i));
            com.hellblazer.art.goal.State valid = projectToValid(state);

            if (lastValid != null && !canTransition(lastValid, valid)) {
                // Insert valid transition
                var transition = createValidTransition(lastValid, valid);
                repaired.addAll(transition);
            }

            repaired.add(valid);
            lastValid = valid;
        }

        return new Trajectory(wrapStates(repaired));
    }

    /**
     * Create failure trajectory when planning fails.
     */
    private Trajectory createFailureTrajectory(State current) {
        // Single state trajectory indicating failure
        return new Trajectory(List.of(current));
    }

    /**
     * Generate random valid state for path exploration.
     */
    private com.hellblazer.art.goal.State generateRandomValidState(com.hellblazer.art.goal.State from, com.hellblazer.art.goal.State to, Random rand) {
        // Random point in space between from and to
        float t = rand.nextFloat();
        com.hellblazer.art.goal.State random = from.interpolate(to, t);

        // Add some perpendicular offset
        float[] direction = from.vectorTo(to);
        float[] perpendicular = computePerpendicular(direction);
        float offset = (rand.nextFloat() - 0.5f) * 2.0f;

        return createOffsetState(from, to, perpendicular, offset);
    }

    /**
     * Compute perpendicular vector.
     */
    private float[] computePerpendicular(float[] vector) {
        if (vector.length < 2) {
            return new float[]{0};
        }

        // Simple 2D perpendicular
        float[] perp = new float[vector.length];
        perp[0] = -vector[1];
        perp[1] = vector[0];

        // Zero out other dimensions
        for (int i = 2; i < perp.length; i++) {
            perp[i] = 0;
        }

        return perp;
    }

    /**
     * Create state with perpendicular offset.
     */
    private com.hellblazer.art.goal.State createOffsetState(com.hellblazer.art.goal.State from, com.hellblazer.art.goal.State to,
                                   float[] perpendicular, float offset) {
        // Midpoint
        com.hellblazer.art.goal.State mid = from.interpolate(to, 0.5f);

        // Create offset state (simplified - assumes vector space)
        final float[] offsetVector = perpendicular.clone();
        for (int i = 0; i < offsetVector.length; i++) {
            offsetVector[i] *= offset;
        }

        // Return offset state
        return new OffsetState(mid, offsetVector);
    }

    // Wrapper/unwrapper methods
    private com.hellblazer.art.goal.State unwrapState(StateTrajectoryPlanner.State state) {
        if (state instanceof StateWrapper wrapper) {
            return wrapper.wrapped;
        }
        throw new IllegalArgumentException("Cannot unwrap non-wrapper state");
    }

    private StateTrajectoryPlanner.State wrapState(com.hellblazer.art.goal.State state) {
        return new StateWrapper(state);
    }

    private List<StateTrajectoryPlanner.State> wrapStates(List<com.hellblazer.art.goal.State> states) {
        return states.stream().map(this::wrapState).toList();
    }

    /**
     * Trajectory constraint interface.
     */
    public interface TrajectoryConstraint {
        String getName();
        boolean isSatisfied(com.hellblazer.art.goal.State state);
        boolean canTransition(com.hellblazer.art.goal.State from, com.hellblazer.art.goal.State to);
        com.hellblazer.art.goal.State project(com.hellblazer.art.goal.State state);
        float getViolation(com.hellblazer.art.goal.State state);
    }

    /**
     * Obstacle representation.
     */
    public interface Obstacle {
        boolean contains(com.hellblazer.art.goal.State state);
        float distanceTo(com.hellblazer.art.goal.State state);
    }

    /**
     * Map of obstacles.
     */
    static class ObstacleMap {
        private final List<Obstacle> obstacles = new ArrayList<>();

        void addObstacle(Obstacle obstacle) {
            obstacles.add(obstacle);
        }

        boolean intersects(com.hellblazer.art.goal.State state) {
            for (var obstacle : obstacles) {
                if (obstacle.contains(state)) {
                    return true;
                }
            }
            return false;
        }

        float nearestDistance(com.hellblazer.art.goal.State state) {
            float minDist = Float.MAX_VALUE;
            for (var obstacle : obstacles) {
                minDist = Math.min(minDist, obstacle.distanceTo(state));
            }
            return minDist;
        }
    }

    /**
     * State validator.
     */
    class StateValidator {
        boolean isValid(com.hellblazer.art.goal.State state) {
            // Check basic validity
            if (!state.isValid()) {
                return false;
            }

            // Check constraints
            for (var constraint : constraints) {
                if (!constraint.isSatisfied(state)) {
                    return false;
                }
            }

            // Check obstacles
            if (obstacleMap.intersects(state)) {
                return false;
            }

            return true;
        }
    }

    /**
     * State wrapper for interface compatibility.
     */
    static class StateWrapper implements StateTrajectoryPlanner.State {
        final com.hellblazer.art.goal.State wrapped;

        StateWrapper(com.hellblazer.art.goal.State state) {
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

    /**
     * Offset state implementation.
     */
    static class OffsetState implements com.hellblazer.art.goal.State {
        private final com.hellblazer.art.goal.State base;
        private final float[] offset;

        OffsetState(com.hellblazer.art.goal.State base, float[] offset) {
            this.base = base;
            this.offset = offset;
        }

        @Override
        public float distanceTo(com.hellblazer.art.goal.State other) {
            return base.distanceTo(other);
        }

        @Override
        public com.hellblazer.art.goal.State interpolate(com.hellblazer.art.goal.State other, float t) {
            return base.interpolate(other, t);
        }

        @Override
        public float[] vectorTo(com.hellblazer.art.goal.State other) {
            float[] baseVector = base.vectorTo(other);
            float[] adjusted = baseVector.clone();
            for (int i = 0; i < Math.min(adjusted.length, offset.length); i++) {
                adjusted[i] -= offset[i] * (1 - i / (float)adjusted.length);
            }
            return adjusted;
        }
    }
}