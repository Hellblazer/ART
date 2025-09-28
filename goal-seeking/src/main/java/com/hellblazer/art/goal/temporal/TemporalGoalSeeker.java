package com.hellblazer.art.goal.temporal;

import com.hellblazer.art.temporal.integration.TemporalART;
import com.hellblazer.art.temporal.integration.TemporalARTParameters;
import com.hellblazer.art.temporal.memory.WorkingMemory;
import com.hellblazer.art.temporal.memory.TemporalPattern;
import com.hellblazer.art.temporal.dynamics.MultiScaleDynamics;
import com.hellblazer.art.temporal.dynamics.TimeScaleOrchestrator;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.goal.State;
import com.hellblazer.art.goal.StateTrajectoryPlanner;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * TemporalGoalSeeker - Integration of ART-Temporal sequence learning with goal-seeking trajectory planning.
 *
 * This class demonstrates how ART-Temporal's sequence learning and generation capabilities
 * can enhance goal-seeking by:
 * 1. Learning successful state transition sequences
 * 2. Generating new trajectories based on learned patterns
 * 3. Adapting sequences based on feedback
 * 4. Managing multi-scale temporal dynamics for different planning layers
 */
public class TemporalGoalSeeker {
    private static final Logger log = LoggerFactory.getLogger(TemporalGoalSeeker.class);

    // Temporal ART for sequence learning and generation
    private final TemporalART temporalART;

    // Multi-scale dynamics matching goal-seeking layers
    private final TimeScaleOrchestrator timeScaleOrchestrator;

    // Working memory for maintaining trajectory context
    private final WorkingMemory workingMemory;

    // Parameters
    private final TemporalARTParameters parameters;

    // Learned trajectory patterns
    private final List<TrajectoryPattern> learnedPatterns;

    public TemporalGoalSeeker() {
        this(TemporalARTParameters.builder()
            .vigilance(0.8f)
            .learningRate(0.1f)
            .build());
    }

    public TemporalGoalSeeker(TemporalARTParameters parameters) {
        this.parameters = parameters;
        this.temporalART = new TemporalART(parameters);
        this.workingMemory = new WorkingMemory(parameters.getWorkingMemoryParameters());
        this.timeScaleOrchestrator = new TimeScaleOrchestrator();
        this.learnedPatterns = new ArrayList<>();

        // Initialize multi-scale layers matching goal-seeking architecture
        initializeTimeScales();
    }

    /**
     * Initialize time scales corresponding to goal-seeking layers:
     * - Goal layer: slowest dynamics (1.0)
     * - Strategic layer: medium-slow (5.0)
     * - Tactical layer: medium-fast (10.0)
     * - Execution layer: fastest (40.0)
     */
    private void initializeTimeScales() {
        timeScaleOrchestrator.addTimeScale("goal", 1.0f);
        timeScaleOrchestrator.addTimeScale("strategic", 5.0f);
        timeScaleOrchestrator.addTimeScale("tactical", 10.0f);
        timeScaleOrchestrator.addTimeScale("execution", 40.0f);
    }

    /**
     * Learn from a successful trajectory.
     * The temporal ART network learns the sequence of states
     * and can later generate similar sequences.
     */
    public void learnTrajectory(List<State> trajectory, float success) {
        if (success < 0.5f) {
            log.debug("Skipping learning from unsuccessful trajectory (success={})", success);
            return;
        }

        // Convert trajectory to temporal patterns
        var patterns = trajectoryToPatterns(trajectory);

        // Learn each pattern in sequence
        for (var pattern : patterns) {
            temporalART.learn(pattern);
        }

        // Store as learned trajectory pattern
        var trajectoryPattern = new TrajectoryPattern(
            trajectory.get(0),  // start state
            trajectory.get(trajectory.size() - 1),  // goal state
            patterns,
            success
        );
        learnedPatterns.add(trajectoryPattern);

        log.info("Learned trajectory with {} states, success={}", trajectory.size(), success);
    }

    /**
     * Generate a new trajectory from current state to goal state
     * using learned temporal patterns.
     */
    public List<State> generateTrajectory(State current, State goal) {
        log.info("Generating trajectory from current to goal using temporal patterns");

        // Find similar learned patterns
        var similarPattern = findSimilarPattern(current, goal);

        if (similarPattern != null) {
            // Adapt existing pattern to current situation
            return adaptPattern(similarPattern, current, goal);
        }

        // Generate new sequence using temporal ART
        return generateNovelSequence(current, goal);
    }

    /**
     * Generate a novel sequence when no similar pattern exists.
     * Uses temporal ART's sequence generation capabilities.
     */
    private List<State> generateNovelSequence(State current, State goal) {
        var trajectory = new ArrayList<State>();
        trajectory.add(current);

        // Initialize working memory with current state
        workingMemory.reset();
        workingMemory.addItem(stateToPattern(current));

        // Generate sequence toward goal
        var currentState = current;
        int maxSteps = 100;

        for (int step = 0; step < maxSteps; step++) {
            // Use temporal ART to predict next state
            var currentPattern = stateToPattern(currentState);
            var prediction = temporalART.predict(currentPattern);

            if (prediction == null) {
                break;
            }

            // Convert prediction to state
            var nextState = patternToState(prediction);
            trajectory.add(nextState);

            // Update working memory
            workingMemory.addItem(prediction);

            // Check if goal reached
            if (isGoalReached(nextState, goal)) {
                log.info("Goal reached in {} steps", step + 1);
                break;
            }

            currentState = nextState;
        }

        return trajectory;
    }

    /**
     * Adapt a learned pattern to new start/goal states.
     */
    private List<State> adaptPattern(TrajectoryPattern pattern, State newStart, State newGoal) {
        log.info("Adapting learned pattern to new situation");

        var adaptedTrajectory = new ArrayList<State>();
        adaptedTrajectory.add(newStart);

        // Use temporal dynamics to interpolate between states
        var dynamics = new MultiScaleDynamics(parameters.getMultiScaleParameters());

        // Apply learned temporal structure with new endpoints
        for (int i = 1; i < pattern.patterns.size() - 1; i++) {
            var templatePattern = pattern.patterns.get(i);

            // Blend template with current context
            var progress = (float) i / pattern.patterns.size();
            var blended = blendStates(newStart, newGoal, progress, templatePattern);

            adaptedTrajectory.add(blended);
        }

        adaptedTrajectory.add(newGoal);
        return adaptedTrajectory;
    }

    /**
     * Find a similar learned pattern based on start and goal states.
     */
    private TrajectoryPattern findSimilarPattern(State current, State goal) {
        TrajectoryPattern bestMatch = null;
        float bestSimilarity = 0.0f;

        for (var pattern : learnedPatterns) {
            float startSimilarity = computeSimilarity(current, pattern.startState);
            float goalSimilarity = computeSimilarity(goal, pattern.goalState);
            float similarity = (startSimilarity + goalSimilarity) / 2.0f;

            if (similarity > bestSimilarity && similarity > 0.7f) {
                bestMatch = pattern;
                bestSimilarity = similarity;
            }
        }

        return bestMatch;
    }

    /**
     * Convert a trajectory to a list of patterns for temporal learning.
     */
    private List<Pattern> trajectoryToPatterns(List<State> trajectory) {
        var patterns = new ArrayList<Pattern>();
        for (var state : trajectory) {
            patterns.add(stateToPattern(state));
        }
        return patterns;
    }

    /**
     * Convert a state to a pattern for ART processing.
     */
    private Pattern stateToPattern(State state) {
        return Pattern.of(state.toArray());
    }

    /**
     * Convert a pattern back to a state.
     */
    private State patternToState(Pattern pattern) {
        return new State(pattern.toArray());
    }

    /**
     * Compute similarity between two states.
     */
    private float computeSimilarity(State s1, State s2) {
        var arr1 = s1.toArray();
        var arr2 = s2.toArray();

        if (arr1.length != arr2.length) {
            return 0.0f;
        }

        float dotProduct = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (int i = 0; i < arr1.length; i++) {
            dotProduct += arr1[i] * arr2[i];
            norm1 += arr1[i] * arr1[i];
            norm2 += arr2[i] * arr2[i];
        }

        if (norm1 == 0 || norm2 == 0) {
            return 0.0f;
        }

        return dotProduct / (float)(Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Check if goal state is reached.
     */
    private boolean isGoalReached(State current, State goal) {
        return computeSimilarity(current, goal) > 0.95f;
    }

    /**
     * Blend states based on progress and template pattern.
     */
    private State blendStates(State start, State goal, float progress, Pattern template) {
        var startArr = start.toArray();
        var goalArr = goal.toArray();
        var templateArr = template.toArray();
        var blended = new double[startArr.length];

        for (int i = 0; i < blended.length; i++) {
            // Linear interpolation with template influence
            var interpolated = startArr[i] * (1 - progress) + goalArr[i] * progress;

            // Blend with template (if available)
            if (i < templateArr.length) {
                blended[i] = interpolated * 0.7 + templateArr[i] * 0.3;
            } else {
                blended[i] = interpolated;
            }
        }

        return new State(blended);
    }

    /**
     * Get performance statistics from temporal ART.
     */
    public TemporalARTStatistics getStatistics() {
        return temporalART.getStatistics();
    }

    /**
     * Reset the temporal goal seeker.
     */
    public void reset() {
        temporalART.reset();
        workingMemory.reset();
        learnedPatterns.clear();
    }

    /**
     * Inner class representing a learned trajectory pattern.
     */
    private static class TrajectoryPattern {
        final State startState;
        final State goalState;
        final List<Pattern> patterns;
        final float successRate;

        TrajectoryPattern(State start, State goal, List<Pattern> patterns, float success) {
            this.startState = start;
            this.goalState = goal;
            this.patterns = patterns;
            this.successRate = success;
        }
    }

    /**
     * Inner class for temporal ART statistics.
     */
    private static class TemporalARTStatistics {
        final int categoriesLearned;
        final int sequencesGenerated;
        final float averageSuccess;

        TemporalARTStatistics(int categories, int sequences, float avgSuccess) {
            this.categoriesLearned = categories;
            this.sequencesGenerated = sequences;
            this.averageSuccess = avgSuccess;
        }
    }
}