package com.hellblazer.art.goal.temporal;

import com.hellblazer.art.goal.State;
import com.hellblazer.art.temporal.integration.TemporalARTParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test demonstrating ART-Temporal integration with goal-seeking.
 * Shows how temporal sequence learning enhances trajectory planning.
 */
public class TemporalGoalSeekerTest {

    private TemporalGoalSeeker goalSeeker;

    @BeforeEach
    void setUp() {
        var parameters = TemporalARTParameters.builder()
            .vigilance(0.85f)
            .learningRate(0.1f)
            .build();
        goalSeeker = new TemporalGoalSeeker(parameters);
    }

    @Test
    @DisplayName("Learn and reproduce successful trajectory")
    void testLearnSuccessfulTrajectory() {
        // Create a successful trajectory
        var trajectory = createSampleTrajectory();

        // Learn from the successful trajectory
        goalSeeker.learnTrajectory(trajectory, 0.95f);

        // Generate a new trajectory with similar start/goal
        var startState = trajectory.get(0);
        var goalState = trajectory.get(trajectory.size() - 1);

        var generatedTrajectory = goalSeeker.generateTrajectory(startState, goalState);

        // Verify trajectory was generated
        assertNotNull(generatedTrajectory);
        assertFalse(generatedTrajectory.isEmpty());

        // Should start at start state
        assertEquals(startState, generatedTrajectory.get(0));

        // Should reach goal or get close
        var lastState = generatedTrajectory.get(generatedTrajectory.size() - 1);
        assertTrue(isNearGoal(lastState, goalState));
    }

    @Test
    @DisplayName("Generate novel trajectory without prior learning")
    void testGenerateNovelTrajectory() {
        // Define start and goal states
        var start = new State(new double[]{0.0, 0.0, 0.0});
        var goal = new State(new double[]{1.0, 1.0, 1.0});

        // Generate trajectory without prior learning
        var trajectory = goalSeeker.generateTrajectory(start, goal);

        // Verify trajectory was generated
        assertNotNull(trajectory);
        assertFalse(trajectory.isEmpty());

        // Should start at start state
        assertEquals(start, trajectory.get(0));
    }

    @Test
    @DisplayName("Learn multiple trajectories and generalize")
    void testLearnMultipleTrajectories() {
        // Learn several successful trajectories
        for (int i = 0; i < 5; i++) {
            var trajectory = createVariedTrajectory(i);
            goalSeeker.learnTrajectory(trajectory, 0.8f + i * 0.04f);
        }

        // Test generalization to new start/goal pair
        var newStart = new State(new double[]{0.1, 0.1, 0.1});
        var newGoal = new State(new double[]{0.9, 0.9, 0.9});

        var generatedTrajectory = goalSeeker.generateTrajectory(newStart, newGoal);

        // Should generate reasonable trajectory
        assertNotNull(generatedTrajectory);
        assertTrue(generatedTrajectory.size() > 2, "Trajectory should have intermediate steps");
    }

    @Test
    @DisplayName("Adapt learned pattern to new situation")
    void testPatternAdaptation() {
        // Learn a curved trajectory
        var curvedTrajectory = createCurvedTrajectory();
        goalSeeker.learnTrajectory(curvedTrajectory, 0.9f);

        // Request similar trajectory with shifted endpoints
        var shiftedStart = new State(new double[]{0.2, 0.2, 0.0});
        var shiftedGoal = new State(new double[]{1.2, 1.2, 1.0});

        var adaptedTrajectory = goalSeeker.generateTrajectory(shiftedStart, shiftedGoal);

        // Should adapt the curved pattern to new endpoints
        assertNotNull(adaptedTrajectory);
        assertTrue(adaptedTrajectory.size() > 3, "Should preserve trajectory structure");

        // Verify adaptation worked
        assertEquals(shiftedStart, adaptedTrajectory.get(0));
    }

    @Test
    @DisplayName("Ignore unsuccessful trajectories during learning")
    void testIgnoreUnsuccessfulTrajectories() {
        var stats1 = goalSeeker.getStatistics();

        // Try to learn from unsuccessful trajectory
        var badTrajectory = createSampleTrajectory();
        goalSeeker.learnTrajectory(badTrajectory, 0.3f); // Low success rate

        var stats2 = goalSeeker.getStatistics();

        // Should not have learned from unsuccessful trajectory
        assertEquals(stats1.categoriesLearned, stats2.categoriesLearned);
    }

    // Helper methods

    private List<State> createSampleTrajectory() {
        var trajectory = new ArrayList<State>();
        for (int i = 0; i <= 10; i++) {
            double t = i / 10.0;
            trajectory.add(new State(new double[]{t, t, t}));
        }
        return trajectory;
    }

    private List<State> createVariedTrajectory(int variation) {
        var trajectory = new ArrayList<State>();
        for (int i = 0; i <= 10; i++) {
            double t = i / 10.0;
            double v = variation * 0.1;
            trajectory.add(new State(new double[]{t + v, t, t - v}));
        }
        return trajectory;
    }

    private List<State> createCurvedTrajectory() {
        var trajectory = new ArrayList<State>();
        for (int i = 0; i <= 20; i++) {
            double t = i / 20.0;
            double x = t;
            double y = t + 0.3 * Math.sin(t * Math.PI * 2);
            double z = t;
            trajectory.add(new State(new double[]{x, y, z}));
        }
        return trajectory;
    }

    private boolean isNearGoal(State current, State goal) {
        var curr = current.toArray();
        var g = goal.toArray();

        if (curr.length != g.length) {
            return false;
        }

        double distance = 0.0;
        for (int i = 0; i < curr.length; i++) {
            double diff = curr[i] - g[i];
            distance += diff * diff;
        }

        return Math.sqrt(distance) < 0.2; // Within threshold
    }
}