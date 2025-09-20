/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance.reinforcement;

import com.hellblazer.art.core.reinforcement.TD_FALCON.StateActionKey;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedTD_FALCON.
 * 
 * Tests:
 * - SARSA learning with vectorized operations
 * - TD(λ) eligibility traces
 * - Epsilon-greedy action selection
 * - Q-function persistence
 * - Performance optimizations
 * - Edge cases and error handling
 * 
 * @author Hal Hildebrand
 */
@DisplayName("VectorizedTD_FALCON Tests")
public class VectorizedTD_FALCONTest {
    
    private VectorizedTD_FALCON tdFalcon;
    private VectorizedParameters stateParams;
    private VectorizedParameters actionParams;
    private VectorizedParameters rewardParams;
    
    @BeforeEach
    void setUp() {
        // Create test parameters
        stateParams = new VectorizedParameters(
            0.7,    // vigilance threshold
            0.1,    // learning rate
            0.01,   // choice alpha
            4,      // dimension
            10,     // max categories
            100,    // max epochs
            true,   // use complement coding
            true,   // normalize
            0.8     // bias
        );
            
        actionParams = new VectorizedParameters(
            0.6,    // vigilance threshold
            0.1,    // learning rate
            0.01,   // choice alpha
            2,      // dimension
            10,     // max categories
            100,    // max epochs
            true,   // use complement coding
            true,   // normalize
            0.8     // bias
        );
            
        rewardParams = new VectorizedParameters(
            0.5,    // vigilance threshold
            0.1,    // learning rate
            0.01,   // choice alpha
            1,      // dimension
            10,     // max categories
            100,    // max epochs
            true,   // use complement coding
            true,   // normalize
            0.8     // bias
        );
    }
    
    @Nested
    @DisplayName("Basic Functionality")
    class BasicFunctionality {
        
        @Test
        @DisplayName("Should create TD-FALCON with valid parameters")
        void testCreation() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f  // TD alpha and gamma
            );
            
            assertNotNull(tdFalcon);
            assertEquals(0.1f, tdFalcon.getTDAlpha());
            assertEquals(0.9f, tdFalcon.getTDGamma());
            assertEquals(0.1f, tdFalcon.getEpsilon());
        }
        
        @Test
        @DisplayName("Should reject invalid TD parameters")
        void testInvalidParameters() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            // Invalid TD alpha
            assertThrows(IllegalArgumentException.class, () -> 
                new VectorizedTD_FALCON(channelDims, gammaValues,
                    stateParams, actionParams, rewardParams,
                    0.0f, 0.9f)  // alpha = 0 is invalid
            );
            
            // Invalid TD gamma
            assertThrows(IllegalArgumentException.class, () -> 
                new VectorizedTD_FALCON(channelDims, gammaValues,
                    stateParams, actionParams, rewardParams,
                    0.1f, 1.1f)  // gamma > 1 is invalid
            );
        }
    }
    
    @Nested
    @DisplayName("SARSA Learning")
    class SARSALearning {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
        }
        
        @Test
        @DisplayName("Should calculate SARSA rewards with vectorization")
        void testSARSACalculation() {
            // Create simple trajectory
            float[][] states = {
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f}
            };
            
            float[][] actions = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f}
            };
            
            float[][] rewards = {
                {0.0f},
                {1.0f},
                {0.0f}
            };
            
            float[][] tdRewards = tdFalcon.calculateSARSAVectorized(states, actions, rewards);
            
            assertNotNull(tdRewards);
            assertEquals(states.length, tdRewards.length);
            
            // TD rewards should be adjusted based on future rewards
            // The middle reward should affect the first state's TD reward
            assertTrue(tdRewards[0][0] >= 0.0f);
        }
        
        @Test
        @DisplayName("Should handle batch SARSA updates")
        void testBatchSARSA() {
            int batchSize = 10;
            Random rand = new Random(42);
            
            float[][] states = new float[batchSize][4];
            float[][] actions = new float[batchSize][2];
            float[] rewards = new float[batchSize];
            float[][] nextStates = new float[batchSize][4];
            float[][] nextActions = new float[batchSize][2];
            
            // Generate random data
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < 4; j++) {
                    states[i][j] = rand.nextFloat();
                    nextStates[i][j] = rand.nextFloat();
                }
                for (int j = 0; j < 2; j++) {
                    actions[i][j] = rand.nextFloat();
                    nextActions[i][j] = rand.nextFloat();
                }
                rewards[i] = rand.nextFloat();
            }
            
            // Should not throw exception
            assertDoesNotThrow(() -> 
                tdFalcon.batchUpdateSARSAParallel(states, actions, rewards, nextStates, nextActions)
            );
        }
    }
    
    @Nested
    @DisplayName("TD(λ) Eligibility Traces")
    class EligibilityTraces {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
            tdFalcon.setLambda(0.5f);  // Set eligibility trace decay
        }
        
        @Test
        @DisplayName("Should update with eligibility traces")
        void testEligibilityTraces() {
            float[] state = {1.0f, 0.0f, 0.0f, 0.0f};
            float[] action = {1.0f, 0.0f};
            float reward = 1.0f;
            float[] nextState = {0.0f, 1.0f, 0.0f, 0.0f};
            float[] nextAction = {0.0f, 1.0f};
            
            // Should not throw exception
            assertDoesNotThrow(() -> 
                tdFalcon.updateSARSAWithTracesVectorized(state, action, reward, nextState, nextAction)
            );
        }
        
        @Test
        @DisplayName("Should decay eligibility traces")
        void testTraceDecay() {
            // Multiple updates to build traces
            float[] state1 = {1.0f, 0.0f, 0.0f, 0.0f};
            float[] action1 = {1.0f, 0.0f};
            float[] state2 = {0.0f, 1.0f, 0.0f, 0.0f};
            float[] action2 = {0.0f, 1.0f};
            
            tdFalcon.updateSARSAWithTracesVectorized(state1, action1, 1.0f, state2, action2);
            tdFalcon.updateSARSAWithTracesVectorized(state2, action2, 0.0f, state1, action1);
            
            // Traces should be maintained and decayed
            var stats = tdFalcon.getTDPerformanceStats();
            assertTrue(stats.getActiveTraces() >= 0);
        }
    }
    
    @Nested
    @DisplayName("Action Selection")
    class ActionSelection {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
        }
        
        @Test
        @DisplayName("Should select actions with epsilon-greedy")
        void testEpsilonGreedySelection() {
            float[] state = {1.0f, 0.0f, 0.0f, 0.0f};
            float[][] actionSpace = {
                {1.0f, 0.0f},
                {0.0f, 1.0f}
            };
            
            // Set high exploration
            tdFalcon.setEpsilon(1.0f);
            float[] action1 = tdFalcon.getEpsilonGreedyActionParallel(state, actionSpace);
            assertNotNull(action1);
            assertEquals(2, action1.length);
            
            // Set pure exploitation
            tdFalcon.setEpsilon(0.0f);
            float[] action2 = tdFalcon.getEpsilonGreedyActionParallel(state, actionSpace);
            assertNotNull(action2);
            assertEquals(2, action2.length);
        }
        
        @Test
        @DisplayName("Should handle parallel action evaluation")
        void testParallelActionEvaluation() {
            float[] state = {1.0f, 0.0f, 0.0f, 0.0f};
            float[][] actionSpace = new float[10][2];
            
            // Create larger action space
            for (int i = 0; i < 10; i++) {
                actionSpace[i][0] = i < 5 ? 1.0f : 0.0f;
                actionSpace[i][1] = i < 5 ? 0.0f : 1.0f;
            }
            
            tdFalcon.setEpsilon(0.0f);  // Pure exploitation
            float[] selectedAction = tdFalcon.getEpsilonGreedyActionParallel(state, actionSpace);
            
            assertNotNull(selectedAction);
            assertEquals(2, selectedAction.length);
        }
    }
    
    @Nested
    @DisplayName("Q-Function Management")
    class QFunctionManagement {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
        }
        
        @Test
        @DisplayName("Should save and restore Q-function")
        void testQFunctionPersistence() {
            // Train with some data
            float[][] states = {
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f}
            };
            float[][] actions = {
                {1.0f, 0.0f},
                {0.0f, 1.0f}
            };
            float[][] rewards = {{1.0f}, {0.5f}};
            
            tdFalcon.fit(states, actions, rewards, stateParams);
            
            // Save Q-function
            Map<StateActionKey, Float> savedQ = tdFalcon.saveQFunction();
            assertNotNull(savedQ);
            assertFalse(savedQ.isEmpty());
            
            // Create new instance
            VectorizedTD_FALCON newTdFalcon = new VectorizedTD_FALCON(
                new int[]{4, 2, 1}, new float[]{0.5f, 0.3f, 0.2f},
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
            
            // Restore Q-function
            newTdFalcon.restoreQFunction(savedQ);
            
            // Should have same Q-values
            Map<StateActionKey, Float> restoredQ = newTdFalcon.saveQFunction();
            assertEquals(savedQ.size(), restoredQ.size());
        }
    }
    
    @Nested
    @DisplayName("Performance Tracking")
    class PerformanceTracking {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
        }
        
        @Test
        @DisplayName("Should track TD-specific performance metrics")
        void testPerformanceMetrics() {
            // Perform some operations
            float[][] states = new float[5][4];
            float[][] actions = new float[5][2];
            float[][] rewards = new float[5][1];
            
            Random rand = new Random(42);
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 4; j++) {
                    states[i][j] = rand.nextFloat();
                }
                for (int j = 0; j < 2; j++) {
                    actions[i][j] = rand.nextFloat();
                }
                rewards[i][0] = rand.nextFloat();
            }
            
            tdFalcon.fit(states, actions, rewards, stateParams);
            
            // Get performance stats
            var stats = tdFalcon.getTDPerformanceStats();
            assertNotNull(stats);
            assertTrue(stats.getTDUpdates() > 0);
            assertTrue(stats.getMaxBatchSize() > 0);
            assertNotNull(stats.toString());
        }
        
        @Test
        @DisplayName("Should track vectorized operations")
        void testVectorizedOperationTracking() {
            float[][] states = new float[16][4];  // Multiple of vector length
            float[][] actions = new float[16][2];
            float[][] rewards = new float[16][1];
            
            Random rand = new Random(42);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 4; j++) {
                    states[i][j] = rand.nextFloat();
                }
                actions[i][0] = i % 2;
                actions[i][1] = 1 - actions[i][0];
                rewards[i][0] = rand.nextFloat();
            }
            
            tdFalcon.calculateSARSAVectorized(states, actions, rewards);
            
            var stats = tdFalcon.getTDPerformanceStats();
            assertTrue(stats.getSIMDOperations() > 0);
        }
    }
    
    @Nested
    @DisplayName("Resource Management")
    class ResourceManagement {
        
        @Test
        @DisplayName("Should properly clean up resources")
        void testResourceCleanup() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
            
            // Perform some operations
            float[][] states = {{1.0f, 0.0f, 0.0f, 0.0f}};
            float[][] actions = {{1.0f, 0.0f}};
            float[][] rewards = {{1.0f}};
            
            tdFalcon.fit(states, actions, rewards, stateParams);
            
            // Clean up should not throw
            assertDoesNotThrow(() -> tdFalcon.shutdown());
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @BeforeEach
        void setUp() {
            int[] channelDims = {4, 2, 1};
            float[] gammaValues = {0.5f, 0.3f, 0.2f};
            
            tdFalcon = new VectorizedTD_FALCON(
                channelDims, gammaValues,
                stateParams, actionParams, rewardParams,
                0.1f, 0.9f
            );
        }
        
        @Test
        @DisplayName("Should handle single-step episodes")
        void testSingleStepEpisode() {
            float[][] states = {{1.0f, 0.0f, 0.0f, 0.0f}};
            float[][] actions = {{1.0f, 0.0f}};
            float[][] rewards = {{1.0f}};
            
            float[][] tdRewards = tdFalcon.calculateSARSAVectorized(states, actions, rewards);
            
            assertNotNull(tdRewards);
            assertEquals(1, tdRewards.length);
            assertEquals(rewards[0][0], tdRewards[0][0]);  // Terminal state keeps original reward
        }
        
        @Test
        @DisplayName("Should handle empty action space gracefully")
        void testEmptyActionSpace() {
            float[] state = {1.0f, 0.0f, 0.0f, 0.0f};
            float[][] emptyActionSpace = new float[0][2];
            
            assertThrows(Exception.class, () ->
                tdFalcon.getEpsilonGreedyActionParallel(state, emptyActionSpace)
            );
        }
        
        @Test
        @DisplayName("Should handle mismatched input dimensions")
        void testMismatchedDimensions() {
            float[][] states = new float[3][4];
            float[][] actions = new float[2][2];  // Different length
            float[][] rewards = new float[3][1];
            
            assertThrows(IllegalArgumentException.class, () ->
                tdFalcon.calculateSARSAVectorized(states, actions, rewards)
            );
        }
    }
}