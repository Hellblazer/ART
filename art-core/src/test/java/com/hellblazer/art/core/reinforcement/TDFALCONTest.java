package com.hellblazer.art.core.reinforcement;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for TD-FALCON (Temporal Difference FALCON) implementation.
 * 
 * TD-FALCON extends FALCON with SARSA (State-Action-Reward-State-Action) learning,
 * enabling temporal difference learning for reinforcement learning applications.
 */
public class TDFALCONTest {
    
    private TD_FALCON tdFalcon;
    private Random random;
    
    @BeforeEach
    void setUp() {
        random = new Random(42);
        
        // Create TD-FALCON with FuzzyART modules
        var stateParams = FuzzyParameters.of(0.7, 0.01, 1.0);
        var actionParams = FuzzyParameters.of(0.8, 0.01, 1.0);
        var rewardParams = FuzzyParameters.of(0.9, 0.01, 1.0);
        
        var stateModule = new FuzzyART();
        var actionModule = new FuzzyART();
        var rewardModule = new FuzzyART();
        
        float[] gammaValues = {0.3f, 0.3f, 0.4f};  // Must sum to 1.0
        int[] channelDims = {2, 2, 1}; // Actual dimensions used in tests
        
        // TD-specific parameters
        float tdAlpha = 0.1f;  // TD learning rate
        float tdGamma = 0.9f;  // Discount factor
        
        tdFalcon = new TD_FALCON(
            stateModule, actionModule, rewardModule, 
            gammaValues, channelDims, tdAlpha, tdGamma
        );
    }
    
    @Test
    @DisplayName("Should create TD-FALCON with valid parameters")
    void testTDFALCONCreation() {
        assertNotNull(tdFalcon);
        assertEquals(0.1f, tdFalcon.getTDAlpha());
        assertEquals(0.9f, tdFalcon.getTDGamma());
    }
    
    @Test
    @DisplayName("Should calculate SARSA rewards correctly")
    void testSARSACalculation() {
        // First, train the system so Q-values exist
        float[][] initStates = {{0.1f, 0.1f}, {0.2f, 0.2f}};
        float[][] initActions = {{0.5f, 0.5f}, {0.6f, 0.4f}};
        float[][] initRewards = {{0.3f}, {0.4f}};
        tdFalcon.calculateSARSA(initStates, initActions, initRewards);
        
        // Simple state-action-reward sequence
        float[][] states = {
            {0.2f, 0.3f},  // s
            {0.4f, 0.5f}   // s'
        };
        
        float[][] actions = {
            {1.0f, 0.0f},  // a
            {0.0f, 1.0f}   // a'
        };
        
        float[][] rewards = {
            {0.5f},  // immediate reward
            {0.8f}   // next reward (for Q(s',a') estimation)
        };
        
        // Calculate TD rewards - second call should show TD effects
        float[][] tdRewards = tdFalcon.calculateSARSA(states, actions, rewards);
        
        assertNotNull(tdRewards);
        assertEquals(rewards.length, tdRewards.length);
        
        // For new state-action pairs with gamma=0.9, TD reward incorporates future value
        // tdReward[0] = reward[0] + gamma * Q(s',a') where Q(s',a') starts at 0
        // So initially tdReward[0] ≈ 0.5 + 0.9 * 0 = 0.5
        // But after updates, Q-values change
        assertTrue(tdRewards[0][0] >= 0, "TD reward should be non-negative");
        
        // Run again to see TD learning effects
        float[][] tdRewards2 = tdFalcon.calculateSARSA(states, actions, rewards);
        
        // After learning, TD rewards should reflect Q-value updates
        assertNotEquals(tdRewards[0][0], tdRewards2[0][0], 
            "TD rewards should change as Q-values are learned");
    }
    
    @Test
    @DisplayName("Should update Q-values using temporal difference")
    void testQValueUpdate() {
        // Train on a simple sequence
        float[] state1 = {0.3f, 0.7f};
        float[] state2 = {0.6f, 0.4f};
        float[] action1 = {1.0f, 0.0f};
        float[] action2 = {0.0f, 1.0f};
        float reward1 = 0.2f;
        float reward2 = 1.0f;
        
        // First transition: (s1, a1, r1, s2, a2)
        tdFalcon.updateSARSA(state1, action1, reward1, state2, action2);
        
        // Get Q-value for state1, action1
        float q1 = tdFalcon.getQValue(state1, action1);
        
        // Second transition updates Q-value
        tdFalcon.updateSARSA(state1, action1, reward1, state2, action2);
        float q2 = tdFalcon.getQValue(state1, action1);
        
        // Q-value should change due to TD learning
        assertNotEquals(q1, q2, 0.001);
    }
    
    @Test
    @DisplayName("Should learn optimal policy through TD learning")
    void testOptimalPolicyLearning() {
        // Simple grid world: 2 states, 2 actions
        // State 0 -> Action 0 -> State 1 (reward 1.0)
        // State 0 -> Action 1 -> State 0 (reward 0.0)
        // State 1 -> Any action -> State 1 (reward 0.0)
        
        float[][] stateSpace = {
            {0.0f, 0.0f},  // State 0
            {1.0f, 1.0f}   // State 1
        };
        
        float[][] actionSpace = {
            {1.0f, 0.0f},  // Action 0
            {0.0f, 1.0f}   // Action 1
        };
        
        // Training episodes
        for (int episode = 0; episode < 100; episode++) {
            float[] state = stateSpace[0]; // Start at state 0
            
            for (int step = 0; step < 10; step++) {
                // Get action from policy
                float[] action = tdFalcon.getAction(state, actionSpace, FALCON.OptimalityMode.MAX);
                
                // Simulate environment dynamics
                float reward;
                float[] nextState;
                
                if (Arrays.equals(state, stateSpace[0]) && Arrays.equals(action, actionSpace[0])) {
                    // Good transition: State 0 -> Action 0 -> State 1
                    reward = 1.0f;
                    nextState = stateSpace[1];
                } else if (Arrays.equals(state, stateSpace[0]) && Arrays.equals(action, actionSpace[1])) {
                    // Bad transition: State 0 -> Action 1 -> State 0
                    reward = -0.1f;
                    nextState = stateSpace[0];
                } else {
                    // Terminal state
                    reward = 0.0f;
                    nextState = stateSpace[1];
                }
                
                // Get next action
                float[] nextAction = tdFalcon.getAction(nextState, actionSpace, FALCON.OptimalityMode.MAX);
                
                // Update Q-values with SARSA
                tdFalcon.updateSARSA(state, action, reward, nextState, nextAction);
                
                state = nextState;
                
                // Terminal condition
                if (Arrays.equals(state, stateSpace[1])) {
                    break;
                }
            }
        }
        
        // Test learned policy
        float[] testState = stateSpace[0];
        float[] optimalAction = tdFalcon.getAction(testState, actionSpace, FALCON.OptimalityMode.MAX);
        
        // Should choose action 0 (leads to reward)
        assertArrayEquals(actionSpace[0], optimalAction);
    }
    
    @Test
    @DisplayName("Should handle exploration vs exploitation tradeoff")
    void testEpsilonGreedyExploration() {
        float epsilon = 0.1f; // 10% exploration
        tdFalcon.setEpsilon(epsilon);
        
        float[] state = {0.5f, 0.5f};
        float[][] actionSpace = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {0.5f, 0.5f}
        };
        
        // Track action selection over many trials
        int[] actionCounts = new int[3];
        for (int i = 0; i < 1000; i++) {
            float[] action = tdFalcon.getEpsilonGreedyAction(state, actionSpace);
            
            for (int j = 0; j < actionSpace.length; j++) {
                if (Arrays.equals(action, actionSpace[j])) {
                    actionCounts[j]++;
                    break;
                }
            }
        }
        
        // All actions should be selected at least sometimes due to exploration
        for (int count : actionCounts) {
            assertTrue(count > 0, "All actions should be explored");
        }
        
        // Greedy action should be selected most often
        int maxCount = Math.max(Math.max(actionCounts[0], actionCounts[1]), actionCounts[2]);
        assertTrue(maxCount > 800, "Greedy action should dominate");
    }
    
    @Test
    @DisplayName("Should discount future rewards appropriately")
    void testDiscountFactor() {
        float gamma1 = 0.0f; // No future reward consideration
        float gamma2 = 0.5f; // Moderate discounting
        float gamma3 = 1.0f; // Full future reward consideration
        
        var tdFalcon1 = createTDFALCONWithGamma(gamma1);
        var tdFalcon2 = createTDFALCONWithGamma(gamma2);
        var tdFalcon3 = createTDFALCONWithGamma(gamma3);
        
        // First establish some Q-values for the next state
        float[] initState = {0.8f, 0.2f};
        float[] initAction = {0.0f, 1.0f};
        float initReward = 1.0f;
        float[] terminalState = {0.9f, 0.1f};
        float[] terminalAction = {0.5f, 0.5f};
        
        // Train all systems with the same initial transition to establish Q(s',a')
        tdFalcon1.updateSARSA(initState, initAction, initReward, terminalState, terminalAction);
        tdFalcon2.updateSARSA(initState, initAction, initReward, terminalState, terminalAction);
        tdFalcon3.updateSARSA(initState, initAction, initReward, terminalState, terminalAction);
        
        // Now test with transitions that depend on gamma
        float[] state = {0.3f, 0.7f};
        float[] action = {1.0f, 0.0f};
        float reward = 0.5f;
        float[] nextState = {0.8f, 0.2f};  // This state has Q-value now
        float[] nextAction = {0.0f, 1.0f};
        
        tdFalcon1.updateSARSA(state, action, reward, nextState, nextAction);
        tdFalcon2.updateSARSA(state, action, reward, nextState, nextAction);
        tdFalcon3.updateSARSA(state, action, reward, nextState, nextAction);
        
        float q1 = tdFalcon1.getQValue(state, action);
        float q2 = tdFalcon2.getQValue(state, action);
        float q3 = tdFalcon3.getQValue(state, action);
        
        // With gamma=0, Q-value ignores future: Q ≈ 0.05 (0.1 * 0.5)
        // With gamma=0.5, Q-value partially considers future
        // With gamma=1.0, Q-value fully considers future
        // These should all be different
        assertTrue(q1 < q2, "Higher gamma should lead to higher Q-value when future is positive");
        assertTrue(q2 < q3, "Higher gamma should lead to higher Q-value when future is positive");
        
        // Verify they are actually different values
        assertNotEquals(q1, q2, 0.001, "Different gammas should produce different Q-values");
        assertNotEquals(q2, q3, 0.001, "Different gammas should produce different Q-values");
    }
    
    @Test
    @DisplayName("Should converge Q-values with sufficient training")
    void testQValueConvergence() {
        // Fixed state-action pair
        float[] state = {0.4f, 0.6f};
        float[] action = {1.0f, 0.0f};
        float[] nextState = {0.7f, 0.3f};
        float[] nextAction = {0.0f, 1.0f};
        float reward = 0.8f;
        
        float[] qValues = new float[50];
        
        // Track Q-value over iterations
        for (int i = 0; i < 50; i++) {
            tdFalcon.updateSARSA(state, action, reward, nextState, nextAction);
            qValues[i] = tdFalcon.getQValue(state, action);
        }
        
        // Q-values should stabilize
        float convergenceThreshold = 0.001f;
        boolean converged = false;
        
        for (int i = 10; i < qValues.length - 1; i++) {
            if (Math.abs(qValues[i] - qValues[i + 1]) < convergenceThreshold) {
                converged = true;
                break;
            }
        }
        
        assertTrue(converged, "Q-values should converge");
    }
    
    @Test
    @DisplayName("Should support batch TD learning")
    void testBatchTDLearning() {
        // Batch of experiences
        float[][] states = {
            {0.1f, 0.2f},
            {0.3f, 0.4f},
            {0.5f, 0.6f}
        };
        
        float[][] actions = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f}
        };
        
        float[] rewards = {0.2f, 0.5f, 1.0f};
        
        float[][] nextStates = {
            {0.3f, 0.4f},
            {0.5f, 0.6f},
            {0.7f, 0.8f}
        };
        
        float[][] nextActions = {
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };
        
        // Batch update
        tdFalcon.batchUpdateSARSA(states, actions, rewards, nextStates, nextActions);
        
        // Verify all transitions were processed
        for (int i = 0; i < states.length; i++) {
            float qValue = tdFalcon.getQValue(states[i], actions[i]);
            assertTrue(qValue > 0, "Q-value should be updated for all transitions");
        }
    }
    
    @Test
    @DisplayName("Should handle terminal states correctly")
    void testTerminalStateHandling() {
        float[] state = {0.5f, 0.5f};
        float[] action = {1.0f, 0.0f};
        float reward = 10.0f; // High terminal reward
        
        // Terminal state has no next state/action
        tdFalcon.updateTerminalSARSA(state, action, reward);
        
        float qValue = tdFalcon.getQValue(state, action);
        
        // Q-value should reflect terminal reward without future discounting
        assertTrue(qValue > 0, "Terminal state should update Q-value");
    }
    
    @Test
    @DisplayName("Should support eligibility traces for TD(λ)")
    void testEligibilityTraces() {
        float lambda = 0.9f; // Eligibility trace decay
        tdFalcon.setLambda(lambda);
        
        // Episode with multiple steps
        float[][] trajectory = {
            // state, action, reward
            {0.1f, 0.1f, 1.0f, 0.0f, 0.1f},
            {0.2f, 0.2f, 0.0f, 1.0f, 0.3f},
            {0.3f, 0.3f, 1.0f, 0.0f, 0.5f},
            {0.4f, 0.4f, 0.0f, 1.0f, 1.0f}
        };
        
        // Process trajectory with eligibility traces
        for (int i = 0; i < trajectory.length - 1; i++) {
            float[] state = {trajectory[i][0], trajectory[i][1]};
            float[] action = {trajectory[i][2], trajectory[i][3]};
            float reward = trajectory[i][4];
            float[] nextState = {trajectory[i + 1][0], trajectory[i + 1][1]};
            float[] nextAction = {trajectory[i + 1][2], trajectory[i + 1][3]};
            
            tdFalcon.updateSARSAWithTraces(state, action, reward, nextState, nextAction);
        }
        
        // Earlier states should be updated due to eligibility traces
        float q1 = tdFalcon.getQValue(new float[]{0.1f, 0.1f}, new float[]{1.0f, 0.0f});
        assertTrue(q1 > 0, "Eligibility traces should propagate updates");
    }
    
    @Test
    @DisplayName("Should save and restore learned Q-function")
    void testQFunctionPersistence() {
        // Train on some data
        float[] state = {0.3f, 0.7f};
        float[] action = {1.0f, 0.0f};
        float reward = 0.8f;
        float[] nextState = {0.6f, 0.4f};
        float[] nextAction = {0.0f, 1.0f};
        
        tdFalcon.updateSARSA(state, action, reward, nextState, nextAction);
        float originalQ = tdFalcon.getQValue(state, action);
        
        // Save Q-function
        var qFunction = tdFalcon.saveQFunction();
        
        // Create new TD-FALCON and restore
        var newTDFalcon = createDefaultTDFALCON();
        newTDFalcon.restoreQFunction(qFunction);
        
        float restoredQ = newTDFalcon.getQValue(state, action);
        
        assertEquals(originalQ, restoredQ, 0.001, "Q-function should be preserved");
    }
    
    // Helper methods
    
    private TD_FALCON createTDFALCONWithGamma(float gamma) {
        var stateModule = new FuzzyART();
        var actionModule = new FuzzyART();
        var rewardModule = new FuzzyART();
        float[] gammaValues = {0.3f, 0.3f, 0.4f};  // Must sum to 1.0
        int[] channelDims = {2, 2, 1};
        
        return new TD_FALCON(
            stateModule, actionModule, rewardModule,
            gammaValues, channelDims, 0.1f, gamma
        );
    }
    
    private TD_FALCON createDefaultTDFALCON() {
        var stateModule = new FuzzyART();
        var actionModule = new FuzzyART();
        var rewardModule = new FuzzyART();
        float[] gammaValues = {0.3f, 0.3f, 0.4f};  // Must sum to 1.0
        int[] channelDims = {2, 2, 1};
        
        return new TD_FALCON(
            stateModule, actionModule, rewardModule,
            gammaValues, channelDims, 0.1f, 0.9f
        );
    }
    
    private static class Arrays {
        static boolean equals(float[] a, float[] b) {
            if (a.length != b.length) return false;
            for (int i = 0; i < a.length; i++) {
                if (Math.abs(a[i] - b[i]) > 0.001) return false;
            }
            return true;
        }
    }
}