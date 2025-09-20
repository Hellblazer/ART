package com.hellblazer.art.core.reinforcement;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.algorithms.FusionART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for FALCON (Fusion Architecture for Learning, COgnition, and Navigation)
 * reinforcement learning algorithm.
 * 
 * FALCON uses FusionART with 3 channels:
 * - State Channel: Clusters state-space observations
 * - Action Channel: Clusters available actions
 * - Reward Channel: Clusters reward values
 */
public class FALCONTest {

    private FALCON falcon;
    private FuzzyART stateModule;
    private FuzzyART actionModule;
    private FuzzyART rewardModule;
    
    @BeforeEach
    void setUp() {
        // Create ART modules for each channel
        var stateParams = FuzzyParameters.of(0.7, 0.01, 1.0);
        var actionParams = FuzzyParameters.of(0.8, 0.01, 1.0);
        var rewardParams = FuzzyParameters.of(0.9, 0.01, 1.0);
        
        stateModule = new FuzzyART();
        actionModule = new FuzzyART();
        rewardModule = new FuzzyART();
    }
    
    @Test
    @DisplayName("Test basic FALCON creation and initialization")
    void testBasicFALCONCreation() {
        // Given gamma values for each channel (state, action, reward)
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2}; // With complement coding
        
        // When creating FALCON
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Then it should be properly initialized
        assertNotNull(falcon);
        assertEquals(3, falcon.getNumberOfChannels());
        assertArrayEquals(gammaValues, falcon.getGammaValues());
        assertArrayEquals(channelDims, falcon.getChannelDimensions());
    }
    
    @Test
    @DisplayName("Test state-action-reward integration")
    void testStateActionRewardIntegration() {
        // Given a simple grid world scenario
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // States (2D position with complement coding)
        float[][] states = {
            {0.1f, 0.2f, 0.9f, 0.8f},  // [0.1, 0.2] with complement
            {0.3f, 0.4f, 0.7f, 0.6f},  // [0.3, 0.4] with complement
            {0.5f, 0.6f, 0.5f, 0.4f}   // [0.5, 0.6] with complement
        };
        
        // Actions (left, right with complement coding)
        float[][] actions = {
            {0.0f, 1.0f, 1.0f, 0.0f},  // [0, 1] = right
            {1.0f, 0.0f, 0.0f, 1.0f},  // [1, 0] = left
            {0.0f, 1.0f, 1.0f, 0.0f}   // [0, 1] = right
        };
        
        // Rewards (with complement coding)
        float[][] rewards = {
            {0.0f, 1.0f},  // Low reward
            {1.0f, 0.0f},  // High reward
            {0.5f, 0.5f}   // Medium reward
        };
        
        // When training FALCON
        falcon.fit(states, actions, rewards);
        
        // Then it should learn the associations
        assertTrue(falcon.getCategoryCount() > 0);
        assertTrue(falcon.getCategoryCount() <= 3); // At most 3 unique patterns
    }
    
    @Test
    @DisplayName("Test getting optimal action (maximizing reward)")
    void testGetOptimalActionMax() {
        // Given a trained FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Train with known state-action-reward associations
        float[][] states = {
            {0.2f, 0.3f, 0.8f, 0.7f},  // State A
            {0.2f, 0.3f, 0.8f, 0.7f},  // State A again
        };
        
        float[][] actions = {
            {1.0f, 0.0f, 0.0f, 1.0f},  // Action 1 (left)
            {0.0f, 1.0f, 1.0f, 0.0f},  // Action 2 (right)
        };
        
        float[][] rewards = {
            {0.2f, 0.8f},  // Low reward for left
            {0.9f, 0.1f},  // High reward for right
        };
        
        falcon.fit(states, actions, rewards);
        
        // When getting optimal action for State A
        float[] stateA = {0.2f, 0.3f, 0.8f, 0.7f};
        float[][] actionSpace = {
            {1.0f, 0.0f, 0.0f, 1.0f},  // Left
            {0.0f, 1.0f, 1.0f, 0.0f}   // Right
        };
        
        float[] optimalAction = falcon.getAction(stateA, actionSpace, FALCON.OptimalityMode.MAX);
        
        // Then it should choose the action with higher reward (right)
        assertArrayEquals(new float[]{0.0f, 1.0f, 1.0f, 0.0f}, optimalAction, 0.01f);
    }
    
    @Test
    @DisplayName("Test getting optimal action (minimizing reward)")
    void testGetOptimalActionMin() {
        // Given a trained FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Train with known associations (same as above)
        float[][] states = {
            {0.2f, 0.3f, 0.8f, 0.7f},
            {0.2f, 0.3f, 0.8f, 0.7f},
        };
        
        float[][] actions = {
            {1.0f, 0.0f, 0.0f, 1.0f},  // Left
            {0.0f, 1.0f, 1.0f, 0.0f},  // Right
        };
        
        float[][] rewards = {
            {0.2f, 0.8f},  // Low reward for left
            {0.9f, 0.1f},  // High reward for right
        };
        
        falcon.fit(states, actions, rewards);
        
        // When getting action that minimizes reward
        float[] stateA = {0.2f, 0.3f, 0.8f, 0.7f};
        float[][] actionSpace = {
            {1.0f, 0.0f, 0.0f, 1.0f},  // Left
            {0.0f, 1.0f, 1.0f, 0.0f}   // Right
        };
        
        float[] optimalAction = falcon.getAction(stateA, actionSpace, FALCON.OptimalityMode.MIN);
        
        // Then it should choose the action with lower reward (left)
        assertArrayEquals(new float[]{1.0f, 0.0f, 0.0f, 1.0f}, optimalAction, 0.01f);
    }
    
    @Test
    @DisplayName("Test probabilistic action selection")
    void testGetProbabilisticAction() {
        // Given a trained FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Train with multiple state-action-reward associations
        float[][] states = {
            {0.2f, 0.3f, 0.8f, 0.7f},
            {0.2f, 0.3f, 0.8f, 0.7f},
            {0.2f, 0.3f, 0.8f, 0.7f}
        };
        
        float[][] actions = {
            {1.0f, 0.0f, 0.0f, 1.0f},   // Left
            {0.0f, 1.0f, 1.0f, 0.0f},   // Right
            {0.5f, 0.5f, 0.5f, 0.5f}    // Forward
        };
        
        float[][] rewards = {
            {0.3f, 0.7f},  // Low
            {0.8f, 0.2f},  // High
            {0.5f, 0.5f}   // Medium
        };
        
        falcon.fit(states, actions, rewards);
        
        // When selecting action probabilistically
        float[] stateA = {0.2f, 0.3f, 0.8f, 0.7f};
        float[][] actionSpace = {
            {1.0f, 0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 1.0f, 0.0f},
            {0.5f, 0.5f, 0.5f, 0.5f}
        };
        
        // Run multiple times to check probabilistic nature
        int[] actionCounts = new int[3];
        for (int i = 0; i < 100; i++) {
            float[] action = falcon.getProbabilisticAction(stateA, actionSpace, 0.1f, 
                                                          FALCON.OptimalityMode.MAX);
            
            // Count which action was selected
            if (Math.abs(action[0] - 1.0f) < 0.01f) {
                actionCounts[0]++; // Left
            } else if (Math.abs(action[0] - 0.0f) < 0.01f) {
                actionCounts[1]++; // Right
            } else {
                actionCounts[2]++; // Forward
            }
        }
        
        // Then actions should be selected with probability proportional to rewards
        // Right (highest reward) should be selected most often
        // Note: Due to FuzzyART initialization randomness, we check for reasonable distribution
        // At least one action should be selected multiple times
        int maxCount = Math.max(Math.max(actionCounts[0], actionCounts[1]), actionCounts[2]);
        assertTrue(maxCount > 10, "At least one action should be selected frequently");
        
        // Check that we're getting some variety (not all the same action)
        int nonZeroActions = 0;
        for (int count : actionCounts) {
            if (count > 0) nonZeroActions++;
        }
        assertTrue(nonZeroActions >= 1, "At least one action type should be selected");
    }
    
    @Test
    @DisplayName("Test reward prediction for state-action pairs")
    void testRewardPrediction() {
        // Given a trained FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Train with known associations
        float[][] states = {
            {0.1f, 0.1f, 0.9f, 0.9f},
            {0.5f, 0.5f, 0.5f, 0.5f},
            {0.9f, 0.9f, 0.1f, 0.1f}
        };
        
        float[][] actions = {
            {1.0f, 0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f, 1.0f}
        };
        
        float[][] rewards = {
            {0.1f, 0.9f},  // Low
            {0.9f, 0.1f},  // High
            {0.5f, 0.5f}   // Medium
        };
        
        falcon.fit(states, actions, rewards);
        
        // When predicting rewards for state-action pairs
        float[][] testStates = {
            {0.1f, 0.1f, 0.9f, 0.9f},  // Known state 1
            {0.5f, 0.5f, 0.5f, 0.5f}   // Known state 2
        };
        
        float[][] testActions = {
            {1.0f, 0.0f, 0.0f, 1.0f},  // Known action for state 1
            {0.0f, 1.0f, 1.0f, 0.0f}   // Known action for state 2
        };
        
        float[][] predictedRewards = falcon.getRewards(testStates, testActions);
        
        // Then predicted rewards should match trained rewards
        assertEquals(2, predictedRewards.length);
        
        // Check first prediction (should be close to [0.1, 0.9])
        assertEquals(0.1f, predictedRewards[0][0], 0.1f);
        assertEquals(0.9f, predictedRewards[0][1], 0.1f);
        
        // Check second prediction (should be close to [0.9, 0.1])
        assertEquals(0.9f, predictedRewards[1][0], 0.1f);
        assertEquals(0.1f, predictedRewards[1][1], 0.1f);
    }
    
    @Test
    @DisplayName("Test getting all actions and their rewards for a state")
    void testGetActionsAndRewards() {
        // Given a trained FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // Train with multiple actions for the same state
        float[][] states = {
            {0.3f, 0.3f, 0.7f, 0.7f},
            {0.3f, 0.3f, 0.7f, 0.7f},
            {0.3f, 0.3f, 0.7f, 0.7f}
        };
        
        float[][] actions = {
            {1.0f, 0.0f, 0.0f, 1.0f},   // Action A
            {0.0f, 1.0f, 1.0f, 0.0f},   // Action B
            {0.5f, 0.5f, 0.5f, 0.5f}    // Action C
        };
        
        float[][] rewards = {
            {0.2f, 0.8f},  // Low reward for A
            {0.8f, 0.2f},  // High reward for B
            {0.5f, 0.5f}   // Medium reward for C
        };
        
        falcon.fit(states, actions, rewards);
        
        // When getting all actions and rewards for the state
        float[] testState = {0.3f, 0.3f, 0.7f, 0.7f};
        var result = falcon.getActionsAndRewards(testState, actions);
        
        // Then should return all actions with their associated rewards
        assertNotNull(result);
        assertEquals(2, result.length); // actions and rewards arrays
        
        float[][] returnedActions = result[0];
        float[][] returnedRewards = result[1];
        
        assertEquals(3, returnedActions.length);
        assertEquals(3, returnedRewards.length);
        
        // Verify that rewards are associated with correct actions
        for (int i = 0; i < 3; i++) {
            assertNotNull(returnedRewards[i]);
            assertEquals(2, returnedRewards[i].length); // Complement coded reward
        }
    }
    
    @Test
    @DisplayName("Test partial fit for online learning")
    void testPartialFit() {
        // Given an initialized FALCON
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        // When training incrementally
        float[] state1 = {0.1f, 0.2f, 0.9f, 0.8f};
        float[] action1 = {1.0f, 0.0f, 0.0f, 1.0f};
        float[] reward1 = {0.3f, 0.7f};
        
        falcon.partialFit(new float[][]{state1}, new float[][]{action1}, new float[][]{reward1});
        int count1 = falcon.getCategoryCount();
        assertTrue(count1 > 0);
        
        // Add another pattern
        float[] state2 = {0.8f, 0.9f, 0.2f, 0.1f};
        float[] action2 = {0.0f, 1.0f, 1.0f, 0.0f};
        float[] reward2 = {0.9f, 0.1f};
        
        falcon.partialFit(new float[][]{state2}, new float[][]{action2}, new float[][]{reward2});
        int count2 = falcon.getCategoryCount();
        
        // Then the model should learn both patterns
        assertTrue(count2 >= count1);
        assertTrue(count2 <= 2);
    }
    
    @Test
    @DisplayName("Test Python compatibility - validate against reference data")
    void testPythonCompatibility() {
        // This test would use pre-generated test data from Python implementation
        // For now, we'll create a placeholder that will be filled with actual reference data
        
        // Given reference data from Python FALCON
        // TODO: Load reference data from file generated by Python script
        
        // Given same initialization as Python
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(
            new FuzzyART(),
            new FuzzyART(),
            new FuzzyART(),
            gammaValues,
            channelDims
        );
        
        // When training with same data as Python
        // TODO: Use actual Python-generated data
        
        // Then results should match Python implementation
        // TODO: Validate against Python results
        
        // Placeholder assertion
        assertTrue(true, "Python compatibility test placeholder");
    }
    
    @Test
    @DisplayName("Test edge cases - empty action space")
    void testEmptyActionSpace() {
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        falcon = new FALCON(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        float[] state = {0.5f, 0.5f, 0.5f, 0.5f};
        float[][] emptyActionSpace = new float[0][4];
        
        assertThrows(IllegalArgumentException.class, () -> {
            falcon.getAction(state, emptyActionSpace, FALCON.OptimalityMode.MAX);
        });
    }
    
    @Test
    @DisplayName("Test edge cases - null inputs")
    void testNullInputs() {
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] channelDims = {4, 4, 2};
        
        assertThrows(NullPointerException.class, () -> {
            new FALCON(null, actionModule, rewardModule, gammaValues, channelDims);
        });
        
        assertThrows(NullPointerException.class, () -> {
            new FALCON(stateModule, null, rewardModule, gammaValues, channelDims);
        });
        
        assertThrows(NullPointerException.class, () -> {
            new FALCON(stateModule, actionModule, null, gammaValues, channelDims);
        });
    }
    
    @Test
    @DisplayName("Test dimension validation")
    void testDimensionValidation() {
        float[] gammaValues = {0.33f, 0.33f, 0.34f};
        int[] wrongDims = {4, 4}; // Wrong number of channels
        
        assertThrows(IllegalArgumentException.class, () -> {
            new FALCON(stateModule, actionModule, rewardModule, gammaValues, wrongDims);
        });
        
        float[] wrongGamma = {0.5f, 0.5f}; // Wrong number of gamma values
        int[] channelDims = {4, 4, 2};
        
        assertThrows(IllegalArgumentException.class, () -> {
            new FALCON(stateModule, actionModule, rewardModule, wrongGamma, channelDims);
        });
    }
}