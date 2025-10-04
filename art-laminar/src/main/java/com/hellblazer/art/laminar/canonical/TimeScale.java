package com.hellblazer.art.laminar.canonical;

/**
 * Time scale enumeration for multi-scale temporal dynamics.
 * Based on Grossberg's canonical laminar circuit with distinct time scales
 * for different cognitive processes.
 *
 * Time scale separation is a fundamental principle in neural dynamics,
 * ensuring that fast processes settle before slow processes change significantly.
 *
 * @author Hal Hildebrand
 */
public enum TimeScale {

    /**
     * Fast time scale (10-100ms).
     * Neural activation dynamics, sensory processing, bottom-up propagation.
     * Example: Rapid feature detection, competitive selection.
     */
    FAST(0.01, 0.1, "Neural activation"),

    /**
     * Medium time scale (50-500ms).
     * Attention shifts, gain modulation, top-down expectations.
     * Example: Attentional blink, attentional shifts between features.
     */
    MEDIUM(0.05, 0.5, "Attention dynamics"),

    /**
     * Slow time scale (500-5000ms).
     * Learning updates, weight changes, memory encoding.
     * Example: Category learning, prototype formation.
     */
    SLOW(0.5, 5.0, "Learning dynamics"),

    /**
     * Very slow time scale (1000-10000ms).
     * Memory consolidation, long-term adaptation, structural changes.
     * Example: Memory consolidation, schema formation.
     */
    VERY_SLOW(1.0, 10.0, "Memory consolidation");

    private final double minTime;  // Minimum characteristic time (seconds)
    private final double maxTime;  // Maximum characteristic time (seconds)
    private final String description;

    TimeScale(double minTime, double maxTime, String description) {
        this.minTime = minTime;
        this.maxTime = maxTime;
        this.description = description;
    }

    /**
     * Get the minimum characteristic time for this scale.
     *
     * @return minimum time in seconds
     */
    public double getMinTime() {
        return minTime;
    }

    /**
     * Get the maximum characteristic time for this scale.
     *
     * @return maximum time in seconds
     */
    public double getMaxTime() {
        return maxTime;
    }

    /**
     * Get the typical time step for numerical integration at this scale.
     * Returns minimum time / 10 for stable integration.
     *
     * @return recommended time step in seconds
     */
    public double getTypicalTimeStep() {
        return minTime / 10.0;
    }

    /**
     * Get human-readable description of this time scale.
     *
     * @return description string
     */
    public String getDescription() {
        return description;
    }

    /**
     * Check if this time scale is faster than another.
     *
     * @param other the time scale to compare against
     * @return true if this scale is faster
     */
    public boolean isFasterThan(TimeScale other) {
        return this.minTime < other.minTime;
    }

    /**
     * Get the separation factor between this and another time scale.
     * Used to verify time scale separation principle:
     * faster dynamics should change at least separationFactor times
     * more quickly than slower dynamics.
     *
     * @param other the time scale to compare against
     * @return the ratio of characteristic times
     */
    public double getSeparationFactor(TimeScale other) {
        return other.minTime / this.minTime;
    }

    /**
     * Get the number of iterations needed for slower dynamics to
     * complete one characteristic time period.
     *
     * @param slower the slower time scale
     * @return number of iterations
     */
    public int getIterationsFor(TimeScale slower) {
        if (!this.isFasterThan(slower)) {
            throw new IllegalArgumentException(
                "Cannot compute iterations: " + this + " is not faster than " + slower);
        }
        return (int) Math.ceil(slower.minTime / this.getTypicalTimeStep());
    }
}