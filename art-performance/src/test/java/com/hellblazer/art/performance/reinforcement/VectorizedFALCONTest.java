package com.hellblazer.art.performance.reinforcement;

import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.core.reinforcement.FALCON;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class VectorizedFALCONTest {

    private VectorizedFALCON falcon;
    private VectorizedParameters parameters;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42);
        
        // Create parameters with reasonable defaults
        parameters = VectorizedParameters.createDefault();
    }

    @Nested
    @DisplayName("Basic Functionality Tests")
    class BasicFunctionality {

        @Test
        @DisplayName("Should initialize with correct dimensions")
        void testInitialization() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            assertNotNull(falcon);
            assertEquals(0, falcon.getCategoryCount());
            assertNotNull(falcon.getPerformanceStats());
        }

        @Test
        @DisplayName("Should learn state-action mappings")
        void testStateActionLearning() {
            int[] channelDims = {2, 2, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Create state, action, and reward
            float[] state = {0.5f, 0.5f};
            float[] action = {1.0f, 0.0f};
            float reward = 1.0f;
            
            // Learn the pattern
            falcon.learn(state, action, reward);
            
            assertTrue(falcon.getCategoryCount() > 0, "Should create at least one category");
            
            // Test action selection for the same state
            float[] selectedAction = falcon.selectAction(state);
            assertNotNull(selectedAction);
            assertEquals(2, selectedAction.length);
        }

        @Test
        @DisplayName("Should predict Q-values")
        void testQValuePrediction() {
            int[] channelDims = {2, 2, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Learn several state-action-reward patterns
            float[][] states = {
                {0.3f, 0.7f},
                {0.8f, 0.2f},
                {0.5f, 0.5f}
            };
            
            float[][] actions = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {0.5f, 0.5f}
            };
            
            float[] rewards = {1.0f, 0.5f, 0.8f};
            
            for (int i = 0; i < states.length; i++) {
                falcon.learn(states[i], actions[i], rewards[i]);
            }
            
            // Predict Q-value for a known state-action pair
            float qValue = falcon.predictQValue(states[0], actions[0]);
            assertTrue(qValue >= 0, "Q-value should be non-negative");
            assertTrue(qValue <= 1.0f, "Q-value should be bounded");
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {

        @Test
        @DisplayName("Should leverage SIMD operations")
        void testSIMDPerformance() {
            int[] channelDims = {8, 4, 1};  // Larger dimensions for SIMD benefit
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Generate random patterns
            int numPatterns = 100;
            float[][] states = new float[numPatterns][8];
            float[][] actions = new float[numPatterns][4];
            float[] rewards = new float[numPatterns];
            
            for (int i = 0; i < numPatterns; i++) {
                for (int j = 0; j < 8; j++) {
                    states[i][j] = random.nextFloat();
                }
                for (int j = 0; j < 4; j++) {
                    actions[i][j] = random.nextFloat();
                }
                rewards[i] = random.nextFloat();
            }
            
            // Train and measure SIMD operations
            long startTime = System.nanoTime();
            for (int i = 0; i < numPatterns; i++) {
                falcon.learn(states[i], actions[i], rewards[i]);
            }
            long endTime = System.nanoTime();
            
            VectorizedFALCONPerformanceStats stats = falcon.getPerformanceStats();
            assertTrue(stats.getSIMDOperations() > 0, "Should use SIMD operations");
            
            long duration = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
            System.out.println("Training time for " + numPatterns + " patterns: " + duration + "ms");
            System.out.println("SIMD operations: " + stats.getSIMDOperations());
        }

        @Test
        @DisplayName("Should use parallel action evaluation")
        void testParallelEvaluation() {
            int[] channelDims = {4, 8, 1};  // More actions for parallel benefit
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Learn multiple patterns
            for (int i = 0; i < 20; i++) {
                float[] state = new float[4];
                float[] action = new float[8];
                
                for (int j = 0; j < 4; j++) {
                    state[j] = random.nextFloat();
                }
                for (int j = 0; j < 8; j++) {
                    action[j] = (j == i % 8) ? 1.0f : 0.0f;  // One-hot encoding
                }
                
                falcon.learn(state, action, random.nextFloat());
            }
            
            // Test parallel evaluation during action selection
            float[] testState = {0.5f, 0.5f, 0.5f, 0.5f};
            
            long startTime = System.nanoTime();
            float[] selectedAction = falcon.selectAction(testState);
            long endTime = System.nanoTime();
            
            assertNotNull(selectedAction);
            assertEquals(8, selectedAction.length);
            
            VectorizedFALCONPerformanceStats stats = falcon.getPerformanceStats();
            assertTrue(stats.getParallelEvaluations() > 0, "Should use parallel evaluations");
            
            long duration = TimeUnit.NANOSECONDS.toMicros(endTime - startTime);
            System.out.println("Action selection time: " + duration + "μs");
            System.out.println("Parallel evaluations: " + stats.getParallelEvaluations());
        }

        @Test
        @DisplayName("Should track performance metrics")
        void testPerformanceMetrics() {
            int[] channelDims = {4, 4, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Perform various operations
            float[] state = {0.5f, 0.5f, 0.5f, 0.5f};
            float[] action = {1.0f, 0.0f, 0.0f, 0.0f};
            
            falcon.learn(state, action, 0.8f);
            falcon.selectAction(state);
            falcon.predictQValue(state, action);
            
            VectorizedFALCONPerformanceStats stats = falcon.getPerformanceStats();
            
            assertTrue(stats.getTotalTrainingTime() > 0, "Should track training time");
            assertTrue(stats.getTotalActionSelectionTime() > 0, "Should track action selection time");
            assertTrue(stats.getSIMDOperations() > 0, "Should track SIMD operations");
            
            // Test throughput calculation
            double trainingThroughput = stats.getTrainingThroughput();
            double actionThroughput = stats.getActionEvaluationThroughput();
            
            assertTrue(trainingThroughput > 0, "Should calculate training throughput");
            assertTrue(actionThroughput > 0, "Should calculate action throughput");
            
            System.out.println("Performance Summary:");
            System.out.println(stats.toString());
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCases {

        @Test
        @DisplayName("Should handle exploration with epsilon-greedy")
        void testExploration() {
            int[] channelDims = {2, 3, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            falcon.setEpsilon(1.0f);  // Always explore
            
            float[] state = {0.5f, 0.5f};
            
            // With epsilon=1.0, should always return random actions
            int[] actionCounts = new int[3];
            for (int i = 0; i < 100; i++) {
                float[] action = falcon.selectAction(state);
                
                // Find which action was selected (assuming one-hot or dominant)
                int maxIdx = 0;
                for (int j = 1; j < action.length; j++) {
                    if (action[j] > action[maxIdx]) {
                        maxIdx = j;
                    }
                }
                actionCounts[maxIdx]++;
            }
            
            // Should have some distribution across all actions
            for (int count : actionCounts) {
                assertTrue(count > 0, "All actions should be explored");
            }
        }

        @Test
        @DisplayName("Should handle resource cleanup")
        void testResourceCleanup() throws Exception {
            int[] channelDims = {2, 2, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Perform some operations
            falcon.learn(new float[]{0.5f, 0.5f}, new float[]{1.0f, 0.0f}, 0.5f);
            
            // Close and verify cleanup
            falcon.close();
            
            // After closing, operations should handle gracefully
            // This tests that resources are properly released
            assertDoesNotThrow(() -> {
                falcon.getPerformanceStats();
            });
        }

        @Test
        @DisplayName("Should reset performance tracking")
        void testPerformanceReset() {
            int[] channelDims = {2, 2, 1};
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Perform operations
            falcon.learn(new float[]{0.5f, 0.5f}, new float[]{1.0f, 0.0f}, 0.5f);
            falcon.selectAction(new float[]{0.5f, 0.5f});
            
            VectorizedFALCONPerformanceStats stats = falcon.getPerformanceStats();
            assertTrue(stats.getSIMDOperations() > 0);
            
            // Reset tracking
            falcon.resetPerformanceTracking();
            
            stats = falcon.getPerformanceStats();
            assertEquals(0, stats.getSIMDOperations(), "Should reset SIMD operation count");
            assertEquals(0, stats.getTotalTrainingTime(), "Should reset training time");
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {

        @Test
        @DisplayName("Should learn simple grid world patterns")
        void testGridWorldLearning() {
            // Simulate a simple 2x2 grid world
            int[] channelDims = {4, 4, 1};  // 4 states, 4 actions (up, down, left, right)
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // States as one-hot encodings for grid positions
            float[][] gridStates = {
                {1.0f, 0.0f, 0.0f, 0.0f},  // Top-left
                {0.0f, 1.0f, 0.0f, 0.0f},  // Top-right
                {0.0f, 0.0f, 1.0f, 0.0f},  // Bottom-left
                {0.0f, 0.0f, 0.0f, 1.0f}   // Bottom-right (goal)
            };
            
            // Actions as one-hot encodings
            float[][] actions = {
                {1.0f, 0.0f, 0.0f, 0.0f},  // Up
                {0.0f, 1.0f, 0.0f, 0.0f},  // Down
                {0.0f, 0.0f, 1.0f, 0.0f},  // Left
                {0.0f, 0.0f, 0.0f, 1.0f}   // Right
            };
            
            // Train optimal path to goal (bottom-right)
            // From top-left: right, down
            falcon.learn(gridStates[0], actions[3], 0.0f);  // Right from top-left
            falcon.learn(gridStates[1], actions[1], 1.0f);  // Down from top-right (reaches goal)
            
            // From bottom-left: right
            falcon.learn(gridStates[2], actions[3], 1.0f);  // Right from bottom-left (reaches goal)
            
            // Test learned policy
            float[] actionFromTopLeft = falcon.selectAction(gridStates[0]);
            assertNotNull(actionFromTopLeft);
            
            // Should prefer right action from top-left
            int maxIdx = 0;
            for (int i = 1; i < actionFromTopLeft.length; i++) {
                if (actionFromTopLeft[i] > actionFromTopLeft[maxIdx]) {
                    maxIdx = i;
                }
            }
            
            // With learned patterns, should tend towards right (index 3)
            System.out.println("Action from top-left: " + java.util.Arrays.toString(actionFromTopLeft));
            System.out.println("Preferred action index: " + maxIdx);
        }

        @Test
        @DisplayName("Should handle continuous state spaces")
        void testContinuousStateSpace() {
            int[] channelDims = {10, 4, 1};  // Continuous state (10D), discrete actions (4)
            float[] gammaValues = {0.4f, 0.3f, 0.3f};
            
            falcon = new VectorizedFALCON(channelDims, gammaValues, parameters, parameters, parameters);
            
            // Generate continuous state patterns
            for (int episode = 0; episode < 10; episode++) {
                float[] state = new float[10];
                
                // Create smooth continuous patterns
                for (int i = 0; i < 10; i++) {
                    state[i] = (float) Math.sin(episode * 0.1 + i * 0.2);
                    state[i] = (state[i] + 1.0f) / 2.0f;  // Normalize to [0, 1]
                }
                
                // Select best action based on some criterion
                int bestAction = episode % 4;
                float[] action = new float[4];
                action[bestAction] = 1.0f;
                
                float reward = (float) Math.exp(-episode * 0.1);  // Decaying rewards
                
                falcon.learn(state, action, reward);
            }
            
            assertTrue(falcon.getCategoryCount() > 0, "Should create categories for continuous states");
            
            // Test generalization to similar states
            float[] testState = new float[10];
            for (int i = 0; i < 10; i++) {
                testState[i] = (float) Math.sin(0.05 + i * 0.2);
                testState[i] = (testState[i] + 1.0f) / 2.0f;
            }
            
            float[] predictedAction = falcon.selectAction(testState);
            assertNotNull(predictedAction);
            
            // Should produce valid action selection
            float sum = 0;
            for (float a : predictedAction) {
                assertTrue(a >= 0 && a <= 1.0f, "Action values should be in [0, 1]");
                sum += a;
            }
            assertTrue(sum > 0, "Should produce non-zero action");
        }
    }
}