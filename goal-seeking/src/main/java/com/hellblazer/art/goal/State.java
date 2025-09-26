package com.hellblazer.art.goal;

/**
 * Unified state interface for the goal-seeking system.
 *
 * Represents any state in an abstract state space that can be:
 * - Measured for distance
 * - Interpolated for smooth transitions
 * - Vectorized for direction computation
 * - Evaluated for importance and readiness
 * - Constrained by domain-specific rules
 */
public interface State {

    /**
     * Calculate distance to another state.
     * Must satisfy metric space properties:
     * - Non-negative: d(x,y) >= 0
     * - Identity: d(x,x) = 0
     * - Symmetric: d(x,y) = d(y,x)
     * - Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
     */
    float distanceTo(State other);

    /**
     * Interpolate between this state and another.
     * @param other Target state
     * @param t Interpolation factor [0,1] where 0=this, 1=other
     * @return Interpolated state
     */
    State interpolate(State other, float t);

    /**
     * Compute vector from this state to another.
     * Used for direction and gradient calculations.
     */
    float[] vectorTo(State other);

    /**
     * Get the importance/salience of this state.
     * Higher values indicate more important states.
     * @return Importance value [0,1]
     */
    default float getImportance() {
        return 1.0f;
    }

    /**
     * Get readiness for action execution.
     * Indicates how ready the system is to take action from this state.
     * @return Readiness value [0,1]
     */
    default float getActionReadiness() {
        return 0.8f;
    }

    /**
     * Get readiness for execution.
     * Indicates execution capability in this state.
     * @return Execution readiness [0,1]
     */
    default float getExecutionReadiness() {
        return 0.9f;
    }

    /**
     * Check if this state is valid according to domain constraints.
     * @return true if state satisfies all constraints
     */
    default boolean isValid() {
        return true;
    }

    /**
     * Check if transition to another state is allowed.
     * @param other Target state
     * @return true if transition is permitted
     */
    default boolean canTransitionTo(State other) {
        return other != null && other.isValid();
    }

    /**
     * Get the dimensionality of this state's vector representation.
     * @return Number of dimensions
     */
    default int getDimensions() {
        return vectorTo(this).length;
    }

    /**
     * Create a copy of this state.
     * @return Deep copy of the state
     */
    default State copy() {
        return this; // Override for mutable states
    }

    /**
     * Get a normalized version of this state.
     * @return Normalized state (unit length in vector space)
     */
    default State normalize() {
        float[] self = vectorTo(this);
        float magnitude = 0;
        for (float v : self) {
            magnitude += v * v;
        }

        if (magnitude < 0.0001f) {
            return this; // Already at origin
        }

        magnitude = (float) Math.sqrt(magnitude);
        final float scale = 1.0f / magnitude;

        // Create normalized state by scaling
        return new State() {
            @Override
            public float distanceTo(State other) {
                return State.this.distanceTo(other) * scale;
            }

            @Override
            public State interpolate(State other, float t) {
                return State.this.interpolate(other, t);
            }

            @Override
            public float[] vectorTo(State other) {
                float[] vec = State.this.vectorTo(other);
                float[] normalized = new float[vec.length];
                for (int i = 0; i < vec.length; i++) {
                    normalized[i] = vec[i] * scale;
                }
                return normalized;
            }
        };
    }

    /**
     * Get constraints applicable to this state.
     * @return Array of constraints, empty if unconstrained
     */
    default StateConstraint[] getConstraints() {
        return new StateConstraint[0];
    }

    /**
     * Project this state onto constraint-satisfying manifold.
     * @return Closest valid state that satisfies constraints
     */
    default State projectToValid() {
        if (isValid()) {
            return this;
        }

        // Default: no projection, override in implementations
        return this;
    }

    /**
     * Constraint that can be applied to states.
     */
    interface StateConstraint {
        /**
         * Check if state satisfies this constraint.
         */
        boolean isSatisfied(State state);

        /**
         * Project state to satisfy constraint.
         */
        State project(State state);

        /**
         * Get constraint violation amount.
         * @return 0 if satisfied, positive value indicating violation magnitude
         */
        float getViolation(State state);
    }
}