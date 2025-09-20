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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

/**
 * High-performance vectorized TD-FALCON implementation using Java Vector API.
 * 
 * TD-FALCON (Temporal Difference FALCON) extends VectorizedFALCON with SARSA 
 * learning for temporal difference reinforcement learning. This implementation
 * provides SIMD-optimized Q-value updates and parallel batch processing.
 * 
 * Performance optimizations:
 * - SIMD Q-value calculations for batch updates
 * - Parallel state-action pair processing
 * - Vectorized TD error computation
 * - Concurrent eligibility trace updates
 * - Lock-free Q-value map updates using ConcurrentHashMap
 * - Memory-mapped Q-function persistence
 * 
 * Based on:
 * - Tan, A.-H., Lu, N., & Xiao, D. (2008). Integrating temporal difference methods 
 *   and self-organizing neural networks for reinforcement learning with delayed 
 *   evaluative feedback. IEEE Transactions on Neural Networks, 19(2), 230-244.
 * 
 * @author Hal Hildebrand
 */
public class VectorizedTD_FALCON extends VectorizedFALCON {
    
    // SIMD configuration
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();
    
    // TD learning parameters
    private float tdAlpha;  // TD learning rate (0 < α ≤ 1)
    private float tdGamma;  // Discount factor (0 ≤ γ ≤ 1)
    private float lambda = 0.0f;  // Eligibility trace decay (0 ≤ λ ≤ 1)
    
    // Random for exploration
    private final Random random = new Random();
    
    // Q-value storage with concurrent access
    private final ConcurrentHashMap<StateActionKey, Float> qValues = new ConcurrentHashMap<>();
    
    // Eligibility traces for TD(λ)
    private final ConcurrentHashMap<StateActionKey, Float> eligibilityTraces = new ConcurrentHashMap<>();
    
    // Performance tracking
    private final LongAdder tdUpdates = new LongAdder();
    private final LongAdder batchProcessingTime = new LongAdder();
    private final AtomicInteger maxBatchSize = new AtomicInteger(0);
    
    // Thread pool for parallel batch processing
    private final ForkJoinPool tdProcessingPool;
    
    /**
     * Create a new VectorizedTD_FALCON with specified parameters.
     * 
     * @param channelDims Dimensions for each channel [state, action, reward]
     * @param gammaValues Channel combination weights (must sum to 1.0)
     * @param stateParams Parameters for state module
     * @param actionParams Parameters for action module
     * @param rewardParams Parameters for reward module
     * @param tdAlpha TD learning rate
     * @param tdGamma Discount factor for future rewards
     */
    public VectorizedTD_FALCON(int[] channelDims, float[] gammaValues,
                              VectorizedParameters stateParams,
                              VectorizedParameters actionParams,
                              VectorizedParameters rewardParams,
                              float tdAlpha, float tdGamma) {
        super(channelDims, gammaValues, stateParams, actionParams, rewardParams);
        
        if (tdAlpha <= 0 || tdAlpha > 1) {
            throw new IllegalArgumentException("TD alpha must be in (0, 1], got: " + tdAlpha);
        }
        if (tdGamma < 0 || tdGamma > 1) {
            throw new IllegalArgumentException("TD gamma must be in [0, 1], got: " + tdGamma);
        }
        
        this.tdAlpha = tdAlpha;
        this.tdGamma = tdGamma;
        this.tdProcessingPool = new ForkJoinPool(ForkJoinPool.getCommonPoolParallelism());
    }
    
    /**
     * Calculate SARSA rewards using vectorized operations.
     * 
     * @param states Sequence of states
     * @param actions Sequence of actions
     * @param rewards Immediate rewards
     * @return TD-adjusted rewards for FusionART training
     */
    public float[][] calculateSARSAVectorized(float[][] states, float[][] actions, float[][] rewards) {
        if (states.length != actions.length || states.length != rewards.length) {
            throw new IllegalArgumentException("States, actions, and rewards must have same length");
        }
        
        int batchSize = states.length;
        float[][] tdRewards = new float[batchSize][rewards[0].length];
        
        // Update max batch size for performance tracking
        maxBatchSize.updateAndGet(current -> Math.max(current, batchSize));
        
        long startTime = System.nanoTime();
        
        // Process in vectorized chunks
        int chunks = (batchSize - 1 + VECTOR_LENGTH - 1) / VECTOR_LENGTH;
        
        tdProcessingPool.submit(() -> {
            IntStream.range(0, chunks).parallel().forEach(chunk -> {
                int start = chunk * VECTOR_LENGTH;
                int end = Math.min(start + VECTOR_LENGTH, batchSize - 1);
                
                // Process chunk with SIMD
                processSARSAChunk(states, actions, rewards, tdRewards, start, end);
            });
        }).join();
        
        // Handle terminal state
        if (batchSize > 0) {
            int lastIdx = batchSize - 1;
            var lastKey = new StateActionKey(states[lastIdx], actions[lastIdx]);
            float lastQ = qValues.getOrDefault(lastKey, 0.0f);
            float newQ = lastQ + tdAlpha * (rewards[lastIdx][0] - lastQ);
            qValues.put(lastKey, newQ);
            tdRewards[lastIdx][0] = rewards[lastIdx][0];
        }
        
        long elapsed = System.nanoTime() - startTime;
        batchProcessingTime.add(elapsed);
        tdUpdates.add(batchSize);
        
        return tdRewards;
    }
    
    /**
     * Process a chunk of SARSA updates using SIMD operations.
     */
    private void processSARSAChunk(float[][] states, float[][] actions, float[][] rewards,
                                   float[][] tdRewards, int start, int end) {
        // Load Q-values for current and next state-actions
        float[] currentQs = new float[VECTOR_LENGTH];
        float[] nextQs = new float[VECTOR_LENGTH];
        float[] immediateRewards = new float[VECTOR_LENGTH];
        
        // Gather Q-values
        for (int i = start, j = 0; i < end && i < states.length - 1; i++, j++) {
            var currentKey = new StateActionKey(states[i], actions[i]);
            var nextKey = new StateActionKey(states[i + 1], actions[i + 1]);
            
            currentQs[j] = qValues.getOrDefault(currentKey, 0.0f);
            nextQs[j] = qValues.getOrDefault(nextKey, 0.0f);
            immediateRewards[j] = rewards[i][0];
            
            // Track vector operation
            trackVectorOperation();
        }
        
        // Vectorized TD calculation
        var currentQVector = FloatVector.fromArray(SPECIES, currentQs, 0);
        var nextQVector = FloatVector.fromArray(SPECIES, nextQs, 0);
        var rewardVector = FloatVector.fromArray(SPECIES, immediateRewards, 0);
        var gammaVector = FloatVector.broadcast(SPECIES, tdGamma);
        var alphaVector = FloatVector.broadcast(SPECIES, tdAlpha);
        
        // TD target = reward + gamma * nextQ
        var tdTargetVector = rewardVector.add(gammaVector.mul(nextQVector));
        
        // TD error = tdTarget - currentQ
        var tdErrorVector = tdTargetVector.sub(currentQVector);
        
        // New Q = currentQ + alpha * tdError
        var newQVector = currentQVector.add(alphaVector.mul(tdErrorVector));
        
        // Store results
        float[] newQs = new float[VECTOR_LENGTH];
        float[] tdTargets = new float[VECTOR_LENGTH];
        newQVector.intoArray(newQs, 0);
        tdTargetVector.intoArray(tdTargets, 0);
        
        // Update Q-values and TD rewards
        for (int i = start, j = 0; i < end && i < states.length - 1; i++, j++) {
            var key = new StateActionKey(states[i], actions[i]);
            qValues.put(key, newQs[j]);
            tdRewards[i][0] = tdTargets[j];
        }
    }
    
    /**
     * Batch update SARSA using parallel processing.
     * 
     * @param states Batch of states
     * @param actions Batch of actions
     * @param rewards Batch of rewards
     * @param nextStates Batch of next states
     * @param nextActions Batch of next actions
     */
    public void batchUpdateSARSAParallel(float[][] states, float[][] actions, float[] rewards,
                                         float[][] nextStates, float[][] nextActions) {
        int batchSize = states.length;
        
        // Process in parallel batches
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        int chunkSize = Math.max(1, batchSize / ForkJoinPool.getCommonPoolParallelism());
        
        for (int i = 0; i < batchSize; i += chunkSize) {
            final int start = i;
            final int end = Math.min(i + chunkSize, batchSize);
            
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                for (int j = start; j < end; j++) {
                    updateSARSAOptimized(states[j], actions[j], rewards[j], 
                                        nextStates[j], nextActions[j]);
                }
            }, tdProcessingPool);
            
            futures.add(future);
        }
        
        // Wait for all updates to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }
    
    /**
     * Optimized single SARSA update.
     */
    private void updateSARSAOptimized(float[] state, float[] action, float reward,
                                      float[] nextState, float[] nextAction) {
        var currentKey = new StateActionKey(state, action);
        var nextKey = new StateActionKey(nextState, nextAction);
        
        // Compute update atomically
        qValues.compute(currentKey, (key, currentQ) -> {
            float q = currentQ != null ? currentQ : 0.0f;
            float nextQ = qValues.getOrDefault(nextKey, 0.0f);
            float tdTarget = reward + tdGamma * nextQ;
            return q + tdAlpha * (tdTarget - q);
        });
        
        tdUpdates.increment();
    }
    
    /**
     * Update SARSA with vectorized eligibility traces for TD(λ).
     * 
     * @param state Current state
     * @param action Current action
     * @param reward Immediate reward
     * @param nextState Next state
     * @param nextAction Next action
     */
    public void updateSARSAWithTracesVectorized(float[] state, float[] action, float reward,
                                                float[] nextState, float[] nextAction) {
        var currentKey = new StateActionKey(state, action);
        var nextKey = new StateActionKey(nextState, nextAction);
        
        float currentQ = qValues.getOrDefault(currentKey, 0.0f);
        float nextQ = qValues.getOrDefault(nextKey, 0.0f);
        float tdError = reward + tdGamma * nextQ - currentQ;
        
        // Update eligibility trace
        eligibilityTraces.merge(currentKey, 1.0f, Float::sum);
        
        // Process traces in vectorized batches
        List<Map.Entry<StateActionKey, Float>> traceList = new ArrayList<>(eligibilityTraces.entrySet());
        int numTraces = traceList.size();
        
        for (int i = 0; i < numTraces; i += VECTOR_LENGTH) {
            int batchEnd = Math.min(i + VECTOR_LENGTH, numTraces);
            processTracesBatch(traceList, i, batchEnd, tdError);
        }
    }
    
    /**
     * Process a batch of eligibility traces using SIMD.
     */
    private void processTracesBatch(List<Map.Entry<StateActionKey, Float>> traces,
                                    int start, int end, float tdError) {
        float[] traceValues = new float[VECTOR_LENGTH];
        float[] qValuesArray = new float[VECTOR_LENGTH];
        
        // Gather values
        for (int i = start, j = 0; i < end; i++, j++) {
            var entry = traces.get(i);
            traceValues[j] = entry.getValue();
            qValuesArray[j] = qValues.getOrDefault(entry.getKey(), 0.0f);
        }
        
        // Vectorized update
        var traceVector = FloatVector.fromArray(SPECIES, traceValues, 0);
        var qVector = FloatVector.fromArray(SPECIES, qValuesArray, 0);
        var alphaVector = FloatVector.broadcast(SPECIES, tdAlpha);
        var errorVector = FloatVector.broadcast(SPECIES, tdError);
        var gammaLambdaVector = FloatVector.broadcast(SPECIES, tdGamma * lambda);
        
        // New Q = Q + alpha * tdError * trace
        var updateVector = alphaVector.mul(errorVector).mul(traceVector);
        var newQVector = qVector.add(updateVector);
        
        // Decay traces
        var newTraceVector = traceVector.mul(gammaLambdaVector);
        
        // Store results
        float[] newQs = new float[VECTOR_LENGTH];
        float[] newTraces = new float[VECTOR_LENGTH];
        newQVector.intoArray(newQs, 0);
        newTraceVector.intoArray(newTraces, 0);
        
        // Update maps
        for (int i = start, j = 0; i < end; i++, j++) {
            var key = traces.get(i).getKey();
            if (newTraces[j] > 0.001f) {
                qValues.put(key, newQs[j]);
                eligibilityTraces.put(key, newTraces[j]);
            } else {
                eligibilityTraces.remove(key);
            }
        }
        
        // Track vector operations
        for (int i = 0; i < (end - start); i++) {
            trackVectorOperation();
        }
    }
    
    /**
     * Get epsilon-greedy action using parallel Q-value evaluation.
     * 
     * @param state Current state
     * @param actionSpace Available actions
     * @return Selected action
     */
    public float[] getEpsilonGreedyActionParallel(float[] state, float[][] actionSpace) {
        if (random.nextFloat() < getEpsilon()) {
            // Exploration: random action
            return actionSpace[random.nextInt(actionSpace.length)];
        } else {
            // Exploitation: parallel greedy action selection
            return getActionParallel(state, actionSpace);
        }
    }
    
    /**
     * Select optimal action using parallel Q-value computation.
     */
    private float[] getActionParallel(float[] state, float[][] actionSpace) {
        int numActions = actionSpace.length;
        Float[] qValues = new Float[numActions];
        
        // Parallel Q-value computation
        CompletableFuture<?>[] futures = new CompletableFuture[numActions];
        
        for (int i = 0; i < numActions; i++) {
            final int idx = i;
            futures[i] = CompletableFuture.runAsync(() -> {
                var key = new StateActionKey(state, actionSpace[idx]);
                qValues[idx] = this.qValues.getOrDefault(key, 0.0f);
            }, tdProcessingPool);
        }
        
        CompletableFuture.allOf(futures).join();
        
        // Find max Q-value action
        int bestAction = 0;
        float maxQ = qValues[0];
        
        for (int i = 1; i < numActions; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        
        return actionSpace[bestAction];
    }
    
    public void fit(float[][] states, float[][] actions, float[][] rewards, VectorizedParameters params) {
        // Apply vectorized SARSA before training FusionART
        float[][] tdRewards = calculateSARSAVectorized(states, actions, rewards);
        super.fit(states, actions, tdRewards, params);
    }
    
    public void partialFit(float[][] states, float[][] actions, float[][] rewards, VectorizedParameters params) {
        // Apply vectorized SARSA for online learning
        float[][] tdRewards = calculateSARSAVectorized(states, actions, rewards);
        // Note: VectorizedFALCON doesn't have partialFit, just use fit
        super.fit(states, actions, tdRewards, params);
    }
    
    /**
     * Get performance statistics including TD-specific metrics.
     * 
     * @return Extended performance statistics
     */
    public VectorizedTD_FALCONPerformanceStats getTDPerformanceStats() {
        var baseStats = getPerformanceStats();
        var tdStats = new VectorizedTD_FALCONPerformanceStats(baseStats);
        
        tdStats.setTDUpdates(tdUpdates.sum());
        tdStats.setBatchProcessingTime(batchProcessingTime.sum());
        tdStats.setMaxBatchSize(maxBatchSize.get());
        tdStats.setQTableSize(qValues.size());
        tdStats.setActiveTraces(eligibilityTraces.size());
        
        return tdStats;
    }
    
    /**
     * Save the learned Q-function.
     * 
     * @return Map of state-action pairs to Q-values
     */
    public Map<StateActionKey, Float> saveQFunction() {
        return new HashMap<>(qValues);
    }
    
    /**
     * Restore a previously learned Q-function.
     * 
     * @param qFunction Map of state-action pairs to Q-values
     */
    public void restoreQFunction(Map<StateActionKey, Float> qFunction) {
        qValues.clear();
        qValues.putAll(qFunction);
    }
    
    public void shutdown() {
        tdProcessingPool.shutdown();
        try {
            if (!tdProcessingPool.awaitTermination(60, TimeUnit.SECONDS)) {
                tdProcessingPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            tdProcessingPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    @Override
    protected void performCleanup() {
        shutdown();
        super.performCleanup();
    }
    
    // Getters and setters
    
    public float getTDAlpha() {
        return tdAlpha;
    }
    
    public void setTDAlpha(float alpha) {
        if (alpha <= 0 || alpha > 1) {
            throw new IllegalArgumentException("TD alpha must be in (0, 1]");
        }
        this.tdAlpha = alpha;
    }
    
    public float getTDGamma() {
        return tdGamma;
    }
    
    public void setTDGamma(float gamma) {
        if (gamma < 0 || gamma > 1) {
            throw new IllegalArgumentException("TD gamma must be in [0, 1]");
        }
        this.tdGamma = gamma;
    }
    
    @Override
    public float getEpsilon() {
        return super.getEpsilon();
    }
    
    @Override
    public void setEpsilon(float epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("Epsilon must be in [0, 1]");
        }
        super.setEpsilon(epsilon);
    }
    
    public float getLambda() {
        return lambda;
    }
    
    public void setLambda(float lambda) {
        if (lambda < 0 || lambda > 1) {
            throw new IllegalArgumentException("Lambda must be in [0, 1]");
        }
        this.lambda = lambda;
    }
}

/**
 * Extended performance statistics for TD-FALCON.
 */
class VectorizedTD_FALCONPerformanceStats extends VectorizedFALCONPerformanceStats {
    private long tdUpdates;
    private long batchProcessingTime;
    private int maxBatchSize;
    private int qTableSize;
    private int activeTraces;
    
    public VectorizedTD_FALCONPerformanceStats(VectorizedFALCONPerformanceStats baseStats) {
        super();
        // Copy base stats
        this.recordTrainingTime(baseStats.getTotalTrainingTime());
        this.recordPredictionTime(baseStats.getTotalPredictionTime());
        this.setSIMDOperations(baseStats.getSIMDOperations());
        this.setParallelEvaluations(baseStats.getParallelEvaluations());
    }
    
    public long getTDUpdates() {
        return tdUpdates;
    }
    
    public void setTDUpdates(long tdUpdates) {
        this.tdUpdates = tdUpdates;
    }
    
    public long getBatchProcessingTime() {
        return batchProcessingTime;
    }
    
    public void setBatchProcessingTime(long batchProcessingTime) {
        this.batchProcessingTime = batchProcessingTime;
    }
    
    public int getMaxBatchSize() {
        return maxBatchSize;
    }
    
    public void setMaxBatchSize(int maxBatchSize) {
        this.maxBatchSize = maxBatchSize;
    }
    
    public int getQTableSize() {
        return qTableSize;
    }
    
    public void setQTableSize(int qTableSize) {
        this.qTableSize = qTableSize;
    }
    
    public int getActiveTraces() {
        return activeTraces;
    }
    
    public void setActiveTraces(int activeTraces) {
        this.activeTraces = activeTraces;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedTD_FALCONPerformanceStats{tdUpdates=%d, batchProcessingTime=%d, " +
                           "maxBatchSize=%d, qTableSize=%d, activeTraces=%d, simdOps=%d, parallelEvals=%d}",
                           tdUpdates, batchProcessingTime, maxBatchSize, qTableSize, activeTraces,
                           getSIMDOperations(), getParallelEvaluations());
    }
}