package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.laminar.canonical.TimeScale;

/**
 * Coordinates multi-scale temporal processing across different time scales.
 *
 * Integrates three temporal scales:
 * - FAST (10-100ms): Shunting dynamics, neural activation
 * - MEDIUM (50-500ms): Temporal chunking, LIST PARSE
 * - SLOW (500-5000ms): Transmitter habituation, learning
 *
 * Based on Grossberg & Kazerounian (2016) multi-scale temporal processing.
 *
 * @author Hal Hildebrand
 */
public class MultiScaleCoordinator {

    private final TimeScale fastTimeScale;
    private final TimeScale mediumTimeScale;
    private final TimeScale slowTimeScale;

    private double currentTime;
    private double lastChunkingUpdate;
    private double lastSlowUpdate;

    private final double chunkingUpdateInterval;
    private final double slowUpdateInterval;

    /**
     * Create coordinator with specified time scales.
     */
    public MultiScaleCoordinator(TimeScale fastTimeScale,
                                 TimeScale mediumTimeScale,
                                 TimeScale slowTimeScale) {
        this.fastTimeScale = fastTimeScale;
        this.mediumTimeScale = mediumTimeScale;
        this.slowTimeScale = slowTimeScale;

        this.currentTime = 0.0;
        this.lastChunkingUpdate = 0.0;
        this.lastSlowUpdate = 0.0;

        // Set update intervals based on time scale ratios
        this.chunkingUpdateInterval = mediumTimeScale.getTypicalTimeStep();
        this.slowUpdateInterval = slowTimeScale.getTypicalTimeStep();
    }

    /**
     * Create coordinator with standard time scales.
     */
    public static MultiScaleCoordinator standard() {
        return new MultiScaleCoordinator(
            TimeScale.FAST,
            TimeScale.MEDIUM,
            TimeScale.SLOW
        );
    }

    /**
     * Create coordinator for real-time processing (faster updates).
     */
    public static MultiScaleCoordinator realTime() {
        return new MultiScaleCoordinator(
            TimeScale.FAST,
            TimeScale.FAST,
            TimeScale.MEDIUM
        );
    }

    /**
     * Advance time by the fast time step.
     * @return time step used
     */
    public double advanceFastTimeStep() {
        double dt = fastTimeScale.getTypicalTimeStep();
        currentTime += dt;
        return dt;
    }

    /**
     * Check if chunking dynamics should be updated.
     */
    public boolean shouldUpdateChunking() {
        return (currentTime - lastChunkingUpdate) >= chunkingUpdateInterval;
    }

    /**
     * Check if slow dynamics should be updated.
     */
    public boolean shouldUpdateSlowDynamics() {
        return (currentTime - lastSlowUpdate) >= slowUpdateInterval;
    }

    /**
     * Get time step for chunking update.
     */
    public double getChunkingTimeStep() {
        double elapsed = currentTime - lastChunkingUpdate;
        lastChunkingUpdate = currentTime;
        return elapsed;
    }

    /**
     * Get time step for slow dynamics update.
     */
    public double getSlowTimeStep() {
        double elapsed = currentTime - lastSlowUpdate;
        lastSlowUpdate = currentTime;
        return elapsed;
    }

    /**
     * Get the separation ratio between medium and fast scales.
     */
    public double getChunkingToFastRatio() {
        return chunkingUpdateInterval / fastTimeScale.getTypicalTimeStep();
    }

    /**
     * Get the separation ratio between slow and fast scales.
     */
    public double getSlowToFastRatio() {
        return slowUpdateInterval / fastTimeScale.getTypicalTimeStep();
    }

    /**
     * Get the separation ratio between slow and medium scales.
     */
    public double getSlowToChunkingRatio() {
        return slowUpdateInterval / chunkingUpdateInterval;
    }

    /**
     * Reset all timing.
     */
    public void reset() {
        currentTime = 0.0;
        lastChunkingUpdate = 0.0;
        lastSlowUpdate = 0.0;
    }

    // Getters

    public double getCurrentTime() {
        return currentTime;
    }

    public TimeScale getFastTimeScale() {
        return fastTimeScale;
    }

    public TimeScale getMediumTimeScale() {
        return mediumTimeScale;
    }

    public TimeScale getSlowTimeScale() {
        return slowTimeScale;
    }

    public double getChunkingUpdateInterval() {
        return chunkingUpdateInterval;
    }

    public double getSlowUpdateInterval() {
        return slowUpdateInterval;
    }

    /**
     * Statistics about multi-scale timing.
     */
    public record TimingStatistics(
        double currentTime,
        double timeSinceLastChunkingUpdate,
        double timeSinceLastSlowUpdate,
        double chunkingToFastRatio,
        double slowToFastRatio,
        double slowToChunkingRatio
    ) {
        public static TimingStatistics from(MultiScaleCoordinator coordinator) {
            return new TimingStatistics(
                coordinator.currentTime,
                coordinator.currentTime - coordinator.lastChunkingUpdate,
                coordinator.currentTime - coordinator.lastSlowUpdate,
                coordinator.getChunkingToFastRatio(),
                coordinator.getSlowToFastRatio(),
                coordinator.getSlowToChunkingRatio()
            );
        }
    }

    public TimingStatistics getStatistics() {
        return TimingStatistics.from(this);
    }
}