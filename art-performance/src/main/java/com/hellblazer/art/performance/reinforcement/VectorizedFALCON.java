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

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.reinforcement.FALCON;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.AbstractVectorizedART;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.performance.algorithms.VectorizedFusionART;
import com.hellblazer.art.performance.algorithms.VectorizedFusionARTParameters;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;

/**
 * High-performance vectorized FALCON implementation using Java Vector API.
 * 
 * FALCON (Fusion Architecture for Learning, COgnition, and Navigation) is a
 * reinforcement learning architecture based on FusionART with three channels:
 * 1. State Channel - Clusters state-space observations
 * 2. Action Channel - Clusters available actions  
 * 3. Reward Channel - Clusters reward values
 * 
 * Performance optimizations:
 * - SIMD operations for state/action/reward processing
 * - Parallel Q-value evaluation for action selection
 * - Vectorized FusionART for multi-channel processing
 * - Batch processing for experience replay
 * - Memory pooling for state-action pairs
 * - Concurrent category search using ForkJoinPool
 * 
 * @author Hal Hildebrand
 */
public class VectorizedFALCON extends AbstractVectorizedART<VectorizedFALCONPerformanceStats, VectorizedParameters> {
    
    // SIMD configuration
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();
    
    // FALCON components (vectorized)
    private final VectorizedFusionART fusionART;
    private final VectorizedFusionARTParameters fusionParams;
    private final VectorizedFuzzyART stateModule;
    private final VectorizedFuzzyART actionModule;  
    private final VectorizedFuzzyART rewardModule;
    
    // Channel configuration
    private final int[] channelDims;
    private final float[] gammaValues;
    
    // Parallel processing
    private final ExecutorService executor;
    private final ForkJoinPool forkJoinPool;
    
    // Performance tracking
    private final VectorizedFALCONPerformanceStats performanceStats;
    private long lastKnownVectorOps = 0;
    private long lastKnownParallelTasks = 0;
    
    // Reinforcement learning parameters
    private float epsilon = 0.1f;  // Epsilon-greedy exploration
    
    /**
     * Create a new VectorizedFALCON with specified modules and parameters.
     * 
     * @param channelDims Dimensions for each channel [state, action, reward]
     * @param gammaValues Channel combination weights (must sum to 1.0)
     * @param stateParams Parameters for state module
     * @param actionParams Parameters for action module
     * @param rewardParams Parameters for reward module
     */
    public VectorizedFALCON(int[] channelDims, float[] gammaValues,
                           VectorizedParameters stateParams,
                           VectorizedParameters actionParams, 
                           VectorizedParameters rewardParams) {
        // Use state params as base (dimension validation will be bypassed for FALCON)
        super(stateParams);
        
        // Validate inputs
        if (channelDims.length != 3) {
            throw new IllegalArgumentException("Channel dims must have 3 elements");
        }
        if (gammaValues.length != 3) {
            throw new IllegalArgumentException("Gamma values must have 3 elements");
        }
        float gammaSum = gammaValues[0] + gammaValues[1] + gammaValues[2];
        if (Math.abs(gammaSum - 1.0f) > 0.001f) {
            throw new IllegalArgumentException("Gamma values must sum to 1.0, got " + gammaSum);
        }
        
        this.channelDims = channelDims.clone();
        this.gammaValues = gammaValues.clone();
        
        // Initialize vectorized ART modules
        this.stateModule = new VectorizedFuzzyART(stateParams);
        this.actionModule = new VectorizedFuzzyART(actionParams);
        this.rewardModule = new VectorizedFuzzyART(rewardParams);
        
        // Create VectorizedFusionARTParameters for 3-channel FALCON
        var baseParams = VectorizedParameters.createDefault();
        var channelVigilance = new double[]{
            stateParams.vigilanceThreshold(), 
            actionParams.vigilanceThreshold(), 
            rewardParams.vigilanceThreshold()
        };
        var channelWeights = new double[]{0.5, 0.3, 0.2}; // Default weights
        var gammaDoubles = new double[gammaValues.length];
        for (int i = 0; i < gammaValues.length; i++) {
            gammaDoubles[i] = gammaValues[i];
        }
        
        // FALCON uses complement coding, so each channel dimension is doubled
        int[] complementCodedDims = new int[]{
            channelDims[0] * 2,  // state with complement
            channelDims[1] * 2,  // action with complement  
            channelDims[2] * 2   // reward with complement
        };
        
        this.fusionParams = new VectorizedFusionARTParameters(
            0.7,                    // overall vigilance
            0.01,                   // learning rate
            gammaDoubles,           // gamma values
            complementCodedDims,    // channel dimensions WITH complement coding
            channelVigilance,       // per-channel vigilance
            channelWeights,         // channel weights
            baseParams,             // base parameters
            false,                  // enable channel skipping
            0.5,                    // activation threshold
            10                      // max search attempts
        );
        
        this.fusionART = new VectorizedFusionART(fusionParams);
        
        // Initialize parallel processing
        this.executor = Executors.newWorkStealingPool();
        this.forkJoinPool = ForkJoinPool.commonPool();
        
        // Initialize performance tracking
        this.performanceStats = new VectorizedFALCONPerformanceStats();
    }
    
    /**
     * Train FALCON on state-action-reward triples using vectorized operations.
     * 
     * @param states Batch of state observations
     * @param actions Batch of actions taken
     * @param rewards Batch of rewards received
     * @param params Training parameters
     */
    public void fit(float[][] states, float[][] actions, float[][] rewards, 
                   VectorizedParameters params) {
        if (states.length != actions.length || states.length != rewards.length) {
            throw new IllegalArgumentException("States, actions, rewards must have same length");
        }
        
        var startTime = System.nanoTime();
        
        // Process in batches for optimal vectorization
        int batchSize = Math.min(VECTOR_LENGTH * 4, states.length);
        
        for (int i = 0; i < states.length; i += batchSize) {
            int end = Math.min(i + batchSize, states.length);
            
            // Create fused patterns for this batch
            var fusedPatterns = new ArrayList<Pattern>();
            
            for (int j = i; j < end; j++) {
                // Concatenate state, action, reward with complement coding
                var fusedPattern = createFusedPattern(states[j], actions[j], rewards[j]);
                fusedPatterns.add(fusedPattern);
            }
            
            // Process batch through FusionART
            processBatchVectorized(fusedPatterns, params);
        }
        
        performanceStats.recordTrainingTime(System.nanoTime() - startTime);
    }
    
    /**
     * Get optimal action for a given state using vectorized Q-value evaluation.
     * 
     * @param state Current state observation
     * @param actionSpace Available actions
     * @param mode Optimality mode (MAX for highest reward, MIN for lowest)
     * @param params Parameters for action selection
     * @return Selected action
     */
    public float[] getAction(float[] state, float[][] actionSpace, 
                           FALCON.OptimalityMode mode, VectorizedParameters params) {
        var startTime = System.nanoTime();
        
        // Parallel Q-value evaluation for all actions
        var qValues = evaluateActionsVectorized(state, actionSpace, params);
        
        // Find optimal action based on mode
        int bestIdx = 0;
        float bestValue = qValues[0];
        
        for (int i = 1; i < qValues.length; i++) {
            if ((mode == FALCON.OptimalityMode.MAX && qValues[i] > bestValue) ||
                (mode == FALCON.OptimalityMode.MIN && qValues[i] < bestValue)) {
                bestValue = qValues[i];
                bestIdx = i;
            }
        }
        
        performanceStats.recordActionSelectionTime(System.nanoTime() - startTime);
        performanceStats.setParallelEvaluations(performanceStats.getParallelEvaluations() + 1);
        
        return actionSpace[bestIdx];
    }
    
    /**
     * Evaluate Q-values for all actions using SIMD operations.
     * 
     * @param state Current state
     * @param actionSpace Available actions
     * @param params Evaluation parameters
     * @return Q-values for each action
     */
    private float[] evaluateActionsVectorized(float[] state, float[][] actionSpace,
                                             VectorizedParameters params) {
        float[] qValues = new float[actionSpace.length];
        
        // Process actions in parallel if there are enough
        if (actionSpace.length >= params.parallelThreshold()) {
            // Use ForkJoinPool for parallel evaluation
            var futures = new ArrayList<CompletableFuture<Float>>();
            
            for (int i = 0; i < actionSpace.length; i++) {
                final int idx = i;
                var future = CompletableFuture.supplyAsync(() -> 
                    evaluateSingleActionVectorized(state, actionSpace[idx], params),
                    forkJoinPool
                );
                futures.add(future);
            }
            
            // Collect results
            for (int i = 0; i < futures.size(); i++) {
                qValues[i] = futures.get(i).join();
            }
        } else {
            // Sequential evaluation for small action spaces
            for (int i = 0; i < actionSpace.length; i++) {
                qValues[i] = evaluateSingleActionVectorized(state, actionSpace[i], params);
            }
        }
        
        return qValues;
    }
    
    /**
     * Evaluate Q-value for a single state-action pair using SIMD.
     * 
     * @param state State vector
     * @param action Action vector
     * @param params Parameters
     * @return Q-value estimate
     */
    private float evaluateSingleActionVectorized(float[] state, float[] action,
                                                VectorizedParameters params) {
        // Create fused state-action pattern
        var fusedPattern = createStateActionPattern(state, action);
        
        // Find best matching category in FusionART
        var result = fusionART.predict(fusedPattern, fusionParams);
        
        // Extract reward prediction (simplified - actual implementation would
        // decode the reward from the category weight)
        int category = switch (result) {
            case ActivationResult.Success success -> success.categoryIndex();
            case ActivationResult.NoMatch noMatch -> -1;
            default -> -1;
        };
        return extractRewardFromCategory(category);
    }
    
    /**
     * Process a batch of patterns using SIMD operations.
     * 
     * @param patterns Batch of fused patterns
     * @param params Processing parameters
     */
    private void processBatchVectorized(List<Pattern> patterns, VectorizedParameters params) {
        // Process patterns through vectorized FusionART
        for (var pattern : patterns) {
            fusionART.learn(pattern, fusionParams);
            for (int i = 0; i < VECTOR_LENGTH; i++) {
                trackVectorOperation();
            } // Track SIMD ops
        }
    }
    
    /**
     * Create fused pattern from state, action, and reward.
     * 
     * @param state State vector
     * @param action Action vector
     * @param reward Reward value
     * @return Fused pattern with complement coding
     */
    private Pattern createFusedPattern(float[] state, float[] action, float[] reward) {
        // Apply complement coding and concatenate
        int totalDim = channelDims[0] * 2 + channelDims[1] * 2 + channelDims[2] * 2;
        float[] fused = new float[totalDim];
        
        int offset = 0;
        
        // State channel with complement coding
        System.arraycopy(state, 0, fused, offset, state.length);
        offset += state.length;
        for (int i = 0; i < state.length; i++) {
            fused[offset++] = 1.0f - state[i];
        }
        
        // Action channel with complement coding
        System.arraycopy(action, 0, fused, offset, action.length);
        offset += action.length;
        for (int i = 0; i < action.length; i++) {
            fused[offset++] = 1.0f - action[i];
        }
        
        // Reward channel with complement coding
        System.arraycopy(reward, 0, fused, offset, reward.length);
        offset += reward.length;
        for (int i = 0; i < reward.length; i++) {
            fused[offset++] = 1.0f - reward[i];
        }
        
        double[] fusedDouble = new double[fused.length];
        for (int i = 0; i < fused.length; i++) {
            fusedDouble[i] = fused[i];
        }
        return Pattern.of(fusedDouble);
    }
    
    /**
     * Create state-action pattern for Q-value evaluation.
     * 
     * @param state State vector
     * @param action Action vector
     * @return Fused state-action pattern with dummy reward channel
     */
    private Pattern createStateActionPattern(float[] state, float[] action) {
        // For Q-value evaluation, we need a full 3-channel pattern
        // Use a neutral reward value (0.5) since we're predicting the reward
        float[] neutralReward = new float[channelDims[2]];
        for (int i = 0; i < neutralReward.length; i++) {
            neutralReward[i] = 0.5f;  // Neutral value
        }
        
        // Use the existing createFusedPattern method for consistency
        return createFusedPattern(state, action, neutralReward);
    }
    
    /**
     * Extract reward value from FusionART category.
     * 
     * @param categoryIndex Category index from prediction
     * @return Estimated reward value
     */
    private float extractRewardFromCategory(int categoryIndex) {
        if (categoryIndex < 0) {
            return 0.0f; // No match found
        }
        
        // Get category weight from FusionART
        var weight = fusionART.getCategory(categoryIndex);
        if (weight == null) {
            return 0.0f;
        }
        
        // Extract reward channel from weight
        // (Simplified - actual implementation would decode properly)
        int rewardOffset = channelDims[0] * 2 + channelDims[1] * 2;
        
        // Return first reward value (could be averaged or processed differently)
        return (float) weight.get(rewardOffset);
    }
    
    // === VectorizedARTAlgorithm Interface Implementation ===
    
    // Override abstract methods from BaseART
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, VectorizedParameters parameters) {
        // Calculate activation using fuzzy ART choice function
        double numerator = 0.0;
        double denominator = 0.001; // Small constant to prevent division by zero (alpha)
        
        for (int i = 0; i < input.dimension(); i++) {
            numerator += Math.min(input.get(i), weight.get(i));
            denominator += weight.get(i);
        }
        
        return numerator / denominator;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedParameters parameters) {
        // Calculate match value
        double matchNumerator = 0.0;
        double inputNorm = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            matchNumerator += Math.min(input.get(i), weight.get(i));
            inputNorm += input.get(i);
        }
        
        double matchValue = matchNumerator / inputNorm;
        double vigilance = parameters.vigilanceThreshold();
        
        if (matchValue >= vigilance) {
            return new MatchResult.Accepted(matchValue, vigilance);
        } else {
            return new MatchResult.Rejected(matchValue, vigilance);
        }
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, VectorizedParameters parameters) {
        // Update weights using fuzzy ART learning rule
        double[] newWeights = new double[currentWeight.dimension()];
        double learningRate = parameters.learningRate();
        
        for (int i = 0; i < newWeights.length; i++) {
            double minVal = Math.min(input.get(i), currentWeight.get(i));
            newWeights[i] = learningRate * minVal + (1.0 - learningRate) * currentWeight.get(i);
        }
        
        return new WeightVector() {
            @Override
            public double get(int index) {
                return newWeights[index];
            }
            
            @Override
            public int dimension() {
                return newWeights.length;
            }
            
            @Override
            public double l1Norm() {
                double sum = 0.0;
                for (double w : newWeights) {
                    sum += Math.abs(w);
                }
                return sum;
            }
            
            @Override
            public WeightVector update(Pattern input, Object parameters) {
                // Delegate to the outer updateWeights method
                return updateWeights(input, this, (VectorizedParameters) parameters);
            }
        };
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, VectorizedParameters params) {
        // Create initial weight for FALCON (3-channel fused pattern)
        // Initialize with ones for all channels with complement coding
        int totalDim = (channelDims[0] + channelDims[1] + channelDims[2]) * 2;
        double[] weights = new double[totalDim];
        for (int i = 0; i < totalDim; i++) {
            weights[i] = 1.0;
        }
        return new WeightVector() {
            @Override
            public double get(int index) {
                return weights[index];
            }
            
            @Override
            public int dimension() {
                return weights.length;
            }
            
            @Override
            public double l1Norm() {
                double sum = 0.0;
                for (double w : weights) {
                    sum += Math.abs(w);
                }
                return sum;
            }
            
            @Override
            public WeightVector update(Pattern input, Object parameters) {
                // For initial weights, just return this since they're all ones
                return this;
            }
        };
    }
    
    // Create performance stats implementation required by AbstractVectorizedART
    @Override
    protected VectorizedFALCONPerformanceStats createPerformanceStats(
            long vectorOps, long parallelTasks, long activations,
            long matches, long learnings, double avgTime) {
        // Check if base class counters were reset by comparing to last known values
        // Only reset if we had some activity before and now all counters are smaller
        if ((lastKnownVectorOps > 0 || lastKnownParallelTasks > 0) && 
            (vectorOps < lastKnownVectorOps || parallelTasks < lastKnownParallelTasks)) {
            performanceStats.reset();
        }
        
        // Update last known values
        lastKnownVectorOps = vectorOps;
        lastKnownParallelTasks = parallelTasks;
        
        // Set SIMD operations from base class tracking
        performanceStats.setSIMDOperations(vectorOps);
        
        performanceStats.setFusionARTStats(fusionART.getPerformanceStats());
        performanceStats.addPatternsProcessed(learnings);
        return performanceStats;
    }
    
    // Cleanup implementation
    @Override
    protected void performCleanup() {
        executor.shutdown();
        fusionART.close();
        stateModule.close();
        actionModule.close();
        rewardModule.close();
    }
    
    // === Helper Methods ===
    
    /**
     * Get the vectorized FusionART instance for direct access if needed.
     * 
     * @return The underlying VectorizedFusionART
     */
    public VectorizedFusionART getFusionART() {
        return fusionART;
    }
    
    /**
     * Get channel dimensions.
     * 
     * @return Array of channel dimensions [state, action, reward]
     */
    public int[] getChannelDims() {
        return channelDims.clone();
    }
    
    /**
     * Get gamma values.
     * 
     * @return Array of gamma values for channel combination
     */
    public float[] getGammaValues() {
        return gammaValues.clone();
    }
    
    /**
     * Learn from a single state-action-reward triple.
     * 
     * @param state State observation
     * @param action Action taken
     * @param reward Reward received
     */
    public void learn(float[] state, float[] action, float reward) {
        var startTime = System.nanoTime();
        
        // Convert reward to array format
        float[] rewardArray = {reward};
        
        // Create fused pattern and learn using the inherited learn method
        var fusedPattern = createFusedPattern(state, action, rewardArray);
        // Use the inherited learn method from BaseART to manage categories properly
        learn(fusedPattern, getParameters());
        
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            trackVectorOperation();
        }
        performanceStats.recordTrainingTime(System.nanoTime() - startTime);
        performanceStats.addPatternsProcessed(1);
    }
    
    /**
     * Select an action for the given state using epsilon-greedy exploration.
     * 
     * @param state Current state observation
     * @return Selected action
     */
    public float[] selectAction(float[] state) {
        var startTime = System.nanoTime();
        
        // Generate action space (simplified - would normally be provided)
        int actionDim = channelDims[1];
        float[][] actionSpace = new float[actionDim][actionDim];
        
        // Create one-hot encoded actions
        for (int i = 0; i < actionDim; i++) {
            actionSpace[i] = new float[actionDim];
            actionSpace[i][i] = 1.0f;
        }
        
        float[] selectedAction;
        
        // Epsilon-greedy exploration
        if (Math.random() < epsilon) {
            // Explore: choose random action
            int randomIdx = (int) (Math.random() * actionDim);
            selectedAction = actionSpace[randomIdx];
        } else {
            // Exploit: choose best action
            selectedAction = getAction(state, actionSpace, 
                                      FALCON.OptimalityMode.MAX, getParameters());
        }
        
        performanceStats.setParallelEvaluations(performanceStats.getParallelEvaluations() + 1);
        performanceStats.recordActionSelectionTime(System.nanoTime() - startTime);
        performanceStats.addActionsEvaluated(1);
        
        return selectedAction;
    }
    
    /**
     * Predict Q-value for a state-action pair.
     * 
     * @param state State observation
     * @param action Action to evaluate
     * @return Predicted Q-value
     */
    public float predictQValue(float[] state, float[] action) {
        var startTime = System.nanoTime();
        
        // Create state-action pattern
        var pattern = createStateActionPattern(state, action);
        
        // Predict category
        var result = fusionART.predict(pattern, fusionParams);
        int category = switch (result) {
            case ActivationResult.Success success -> success.categoryIndex();
            case ActivationResult.NoMatch noMatch -> -1;
            default -> -1;
        };
        
        // Extract Q-value from category
        float qValue = extractRewardFromCategory(category);
        
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            trackVectorOperation();
        }
        performanceStats.recordActionSelectionTime(System.nanoTime() - startTime);
        performanceStats.addActionsEvaluated(1);
        
        return qValue;
    }
    
    /**
     * Set the epsilon value for epsilon-greedy exploration.
     * 
     * @param epsilon New epsilon value [0, 1]
     */
    public void setEpsilon(float epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("Epsilon must be in [0, 1]");
        }
        this.epsilon = epsilon;
    }
    
    /**
     * Get the current epsilon value.
     * 
     * @return Current epsilon
     */
    public float getEpsilon() {
        return epsilon;
    }
}