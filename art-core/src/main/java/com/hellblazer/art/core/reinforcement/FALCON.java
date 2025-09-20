package com.hellblazer.art.core.reinforcement;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.algorithms.FusionART;
import com.hellblazer.art.core.algorithms.FusionParameters;
import com.hellblazer.art.core.results.ActivationResult;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * FALCON (Fusion Architecture for Learning, COgnition, and Navigation) for Reinforcement Learning.
 * 
 * This implementation is based on:
 * Tan, A.-H. (2004). FALCON: a fusion architecture for learning, cognition, and navigation.
 * In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
 * (pp. 3297–3302). volume 4. doi:10.1109/IJCNN.2004.1381208.
 * 
 * FALCON is based on a FusionART backbone with exactly 3 channels:
 * - State Channel: Clusters state-space observations
 * - Action Channel: Clusters available actions  
 * - Reward Channel: Clusters reward values
 * 
 * The algorithm provides functions for getting optimal reward and action predictions
 * for reinforcement learning applications.
 */
public class FALCON {
    
    /**
     * Optimality mode for action selection
     */
    public enum OptimalityMode {
        MAX, // Choose action with maximum reward
        MIN  // Choose action with minimum reward
    }
    
    protected final FusionART fusionART;
    protected final BaseART<?> stateModule;
    protected final BaseART<?> actionModule;
    protected final BaseART<?> rewardModule;
    protected final float[] gammaValues;
    protected final int[] channelDims;
    protected final Random random = new Random();
    
    private static final int STATE_CHANNEL = 0;
    private static final int ACTION_CHANNEL = 1;
    private static final int REWARD_CHANNEL = 2;
    private static final int NUM_CHANNELS = 3;
    
    /**
     * Initialize the FALCON model.
     * 
     * @param stateModule The ART module that will cluster the state-space
     * @param actionModule The ART module that will cluster the action-space
     * @param rewardModule The ART module that will cluster the reward-space
     * @param gammaValues The activation ratio for each channel (must be length 3)
     * @param channelDims The dimension of each channel (must be length 3)
     */
    public FALCON(BaseART<?> stateModule, BaseART<?> actionModule, BaseART<?> rewardModule,
                  float[] gammaValues, int[] channelDims) {
        
        // Validate inputs
        Objects.requireNonNull(stateModule, "State module cannot be null");
        Objects.requireNonNull(actionModule, "Action module cannot be null");
        Objects.requireNonNull(rewardModule, "Reward module cannot be null");
        Objects.requireNonNull(gammaValues, "Gamma values cannot be null");
        Objects.requireNonNull(channelDims, "Channel dimensions cannot be null");
        
        if (gammaValues.length != NUM_CHANNELS) {
            throw new IllegalArgumentException(
                "Gamma values must have exactly 3 elements (state, action, reward), got " + gammaValues.length);
        }
        
        if (channelDims.length != NUM_CHANNELS) {
            throw new IllegalArgumentException(
                "Channel dimensions must have exactly 3 elements (state, action, reward), got " + channelDims.length);
        }
        
        // Validate gamma values sum to approximately 1.0
        float gammaSum = 0;
        for (float gamma : gammaValues) {
            if (gamma < 0 || gamma > 1) {
                throw new IllegalArgumentException("Gamma values must be between 0 and 1");
            }
            gammaSum += gamma;
        }
        if (Math.abs(gammaSum - 1.0f) > 0.01f) {
            throw new IllegalArgumentException("Gamma values must sum to approximately 1.0, got " + gammaSum);
        }
        
        this.stateModule = stateModule;
        this.actionModule = actionModule;
        this.rewardModule = rewardModule;
        this.gammaValues = gammaValues.clone();
        this.channelDims = channelDims.clone();
        
        // Create FusionART with the three modules
        @SuppressWarnings("rawtypes")
        List<BaseART> modules = Arrays.asList((BaseART) stateModule, (BaseART) actionModule, (BaseART) rewardModule);
        // Convert float[] to double[]
        var doubleGammaValues = new double[gammaValues.length];
        for (int i = 0; i < gammaValues.length; i++) {
            doubleGammaValues[i] = gammaValues[i];
        }
        this.fusionART = new FusionART(modules, doubleGammaValues, channelDims);
    }
    
    /**
     * Get the number of channels (always 3 for FALCON).
     */
    public int getNumberOfChannels() {
        return NUM_CHANNELS;
    }
    
    /**
     * Get the gamma values for each channel.
     */
    public float[] getGammaValues() {
        return gammaValues.clone();
    }
    
    /**
     * Get the dimensions of each channel.
     */
    public int[] getChannelDimensions() {
        return channelDims.clone();
    }
    
    /**
     * Get the number of learned categories.
     */
    public int getCategoryCount() {
        return fusionART.getCategoryCount();
    }
    
    /**
     * Fit the FALCON model to the data.
     * 
     * @param states The state data
     * @param actions The action data
     * @param rewards The reward data
     */
    public void fit(float[][] states, float[][] actions, float[][] rewards) {
        Objects.requireNonNull(states, "States cannot be null");
        Objects.requireNonNull(actions, "Actions cannot be null");
        Objects.requireNonNull(rewards, "Rewards cannot be null");
        
        if (states.length != actions.length || states.length != rewards.length) {
            throw new IllegalArgumentException(
                "States, actions, and rewards must have the same number of samples");
        }
        
        // Validate dimensions
        validateDimensions(states, STATE_CHANNEL);
        validateDimensions(actions, ACTION_CHANNEL);
        validateDimensions(rewards, REWARD_CHANNEL);
        
        // Join the channel data and train FusionART
        for (int i = 0; i < states.length; i++) {
            float[] joinedData = joinChannelData(states[i], actions[i], rewards[i]);
            var pattern = createPattern(joinedData);
            var params = FusionParameters.builder().build();
            fusionART.learn(pattern, params);
        }
    }
    
    /**
     * Partially fit the FALCON model to the data (online learning).
     * 
     * @param states The state data
     * @param actions The action data
     * @param rewards The reward data
     */
    public void partialFit(float[][] states, float[][] actions, float[][] rewards) {
        // For base FALCON, partial fit is the same as fit
        fit(states, actions, rewards);
    }
    
    /**
     * Get the best action for a given state based on optimality.
     * 
     * @param state The current state
     * @param actionSpace The available action space (null to use learned actions)
     * @param mode Whether to choose the action with minimum or maximum reward
     * @return The optimal action
     */
    public float[] getAction(float[] state, float[][] actionSpace, OptimalityMode mode) {
        Objects.requireNonNull(state, "State cannot be null");
        Objects.requireNonNull(mode, "Optimality mode cannot be null");
        
        if (actionSpace == null || actionSpace.length == 0) {
            // Use learned action centers if no action space provided
            actionSpace = getLearnedActionCenters();
        }
        
        if (actionSpace.length == 0) {
            throw new IllegalArgumentException("No actions available");
        }
        
        validateDimensions(new float[][]{state}, STATE_CHANNEL);
        validateDimensions(actionSpace, ACTION_CHANNEL);
        
        // Get rewards for each action
        float[][][] actionsAndRewards = getActionsAndRewards(state, actionSpace);
        float[][] rewards = actionsAndRewards[1];
        
        // Find the best action based on optimality mode
        int bestIdx = 0;
        float bestRewardValue = getRewardValue(rewards[0]);
        
        for (int i = 1; i < rewards.length; i++) {
            float rewardValue = getRewardValue(rewards[i]);
            
            if (mode == OptimalityMode.MAX) {
                if (rewardValue > bestRewardValue) {
                    bestRewardValue = rewardValue;
                    bestIdx = i;
                }
            } else { // MIN
                if (rewardValue < bestRewardValue) {
                    bestRewardValue = rewardValue;
                    bestIdx = i;
                }
            }
        }
        
        return actionSpace[bestIdx].clone();
    }
    
    /**
     * Get a probabilistic action for a given state based on reward distribution.
     * 
     * @param state The current state
     * @param actionSpace The available action space
     * @param offset The reward offset to adjust probability distribution
     * @param mode Whether to prefer minimum or maximum rewards
     * @return The chosen action based on probability
     */
    public float[] getProbabilisticAction(float[] state, float[][] actionSpace, 
                                         float temperature, OptimalityMode mode) {
        Objects.requireNonNull(state, "State cannot be null");
        Objects.requireNonNull(mode, "Optimality mode cannot be null");
        
        if (actionSpace == null || actionSpace.length == 0) {
            actionSpace = getLearnedActionCenters();
        }
        
        if (actionSpace.length == 0) {
            throw new IllegalArgumentException("No actions available");
        }
        
        // Get rewards for each action
        float[][][] actionsAndRewards = getActionsAndRewards(state, actionSpace);
        float[][] rewards = actionsAndRewards[1];
        
        // Calculate reward distribution
        float[] rewardDist = new float[rewards.length];
        float sum = 0;
        
        for (int i = 0; i < rewards.length; i++) {
            rewardDist[i] = getRewardValue(rewards[i]);
            sum += rewardDist[i];
        }
        
        // Normalize
        if (sum > 0) {
            for (int i = 0; i < rewardDist.length; i++) {
                rewardDist[i] /= sum;
            }
        } else {
            // Uniform distribution if all rewards are zero
            Arrays.fill(rewardDist, 1.0f / rewardDist.length);
        }
        
        // Invert distribution if minimizing
        if (mode == OptimalityMode.MIN) {
            for (int i = 0; i < rewardDist.length; i++) {
                rewardDist[i] = 1.0f - rewardDist[i];
            }
        }
        
        // Apply offset and re-normalize
        sum = 0;
        for (int i = 0; i < rewardDist.length; i++) {
            rewardDist[i] = Math.max(rewardDist[i] * temperature, 0.0001f);
            sum += rewardDist[i];
        }
        
        for (int i = 0; i < rewardDist.length; i++) {
            rewardDist[i] /= sum;
        }
        
        // Sample action based on probability distribution
        float rand = random.nextFloat();
        float cumSum = 0;
        
        for (int i = 0; i < rewardDist.length; i++) {
            cumSum += rewardDist[i];
            if (rand <= cumSum) {
                return actionSpace[i].clone();
            }
        }
        
        // Fallback to last action (shouldn't happen)
        return actionSpace[actionSpace.length - 1].clone();
    }
    
    /**
     * Get possible actions and their associated rewards for a given state.
     * 
     * @param state The current state
     * @param actionSpace The available action space (null to use learned actions)
     * @return Array containing [actions, rewards]
     */
    public float[][][] getActionsAndRewards(float[] state, float[][] actionSpace) {
        Objects.requireNonNull(state, "State cannot be null");
        
        if (actionSpace == null || actionSpace.length == 0) {
            actionSpace = getLearnedActionCenters();
        }
        
        if (actionSpace.length == 0) {
            return new float[][][]{new float[0][], new float[0][]};
        }
        
        float[][] rewards = new float[actionSpace.length][];
        
        // For each action, predict the reward
        for (int i = 0; i < actionSpace.length; i++) {
            // Create partial pattern with state and action (no reward)
            float[] partialPattern = joinChannelData(state, actionSpace[i], null);
            
            // Predict which category this state-action pair belongs to
            var pattern = createPattern(partialPattern);
            var params = FusionParameters.builder().build();
            var result = fusionART.predict(pattern, params);
            int category = result instanceof ActivationResult.Success success ? 
                           success.categoryIndex() : -1;
            
            if (category >= 0) {
                // Get the reward part of the weight vector for this category
                rewards[i] = getRewardFromCategory(category);
            } else {
                // No matching category, use default reward
                rewards[i] = getDefaultReward();
            }
        }
        
        return new float[][][]{actionSpace, rewards};
    }
    
    /**
     * Get the rewards for given states and actions.
     * 
     * @param states The state data
     * @param actions The action data
     * @return The rewards corresponding to the given state-action pairs
     */
    public float[][] getRewards(float[][] states, float[][] actions) {
        Objects.requireNonNull(states, "States cannot be null");
        Objects.requireNonNull(actions, "Actions cannot be null");
        
        if (states.length != actions.length) {
            throw new IllegalArgumentException("States and actions must have same length");
        }
        
        float[][] rewards = new float[states.length][];
        
        for (int i = 0; i < states.length; i++) {
            float[] partialPattern = joinChannelData(states[i], actions[i], null);
            var pattern = createPattern(partialPattern);
            var params = FusionParameters.builder().build();
            var result = fusionART.predict(pattern, params);
            int category = result instanceof ActivationResult.Success success ? 
                           success.categoryIndex() : -1;
            
            if (category >= 0) {
                rewards[i] = getRewardFromCategory(category);
            } else {
                rewards[i] = getDefaultReward();
            }
        }
        
        return rewards;
    }
    
    // Helper methods
    
    private void validateDimensions(float[][] data, int channelIndex) {
        if (data.length > 0 && data[0].length != channelDims[channelIndex]) {
            throw new IllegalArgumentException(
                String.format("Channel %d expects dimension %d, got %d",
                            channelIndex, channelDims[channelIndex], data[0].length));
        }
    }
    
    private float[] joinChannelData(float[] state, float[] action, float[] reward) {
        int totalDim = channelDims[STATE_CHANNEL] + channelDims[ACTION_CHANNEL] + 
                      channelDims[REWARD_CHANNEL];
        float[] joined = new float[totalDim];
        
        // Copy state data
        System.arraycopy(state, 0, joined, 0, channelDims[STATE_CHANNEL]);
        
        // Copy action data
        System.arraycopy(action, 0, joined, channelDims[STATE_CHANNEL], 
                        channelDims[ACTION_CHANNEL]);
        
        // Copy reward data if provided, otherwise leave as zeros
        if (reward != null) {
            System.arraycopy(reward, 0, joined, 
                           channelDims[STATE_CHANNEL] + channelDims[ACTION_CHANNEL],
                           channelDims[REWARD_CHANNEL]);
        }
        
        return joined;
    }
    
    private float[][] getLearnedActionCenters() {
        // Get action centers from the action module
        int categoryCount = actionModule.getCategoryCount();
        if (categoryCount == 0) {
            return new float[0][];
        }
        
        var categories = actionModule.getCategories();
        float[][] centers = new float[categoryCount][];
        for (int i = 0; i < categoryCount; i++) {
            if (i < categories.size()) {
                WeightVector weight = categories.get(i);
                centers[i] = new float[channelDims[ACTION_CHANNEL]];
                for (int j = 0; j < channelDims[ACTION_CHANNEL]; j++) {
                    centers[i][j] = (float) weight.get(j);
                }
            }
        }
        
        return centers;
    }
    
    private float[] getRewardFromCategory(int category) {
        if (category < 0 || category >= fusionART.getCategoryCount()) {
            return getDefaultReward();
        }
        
        // Get the full weight vector and extract reward part
        var categories = fusionART.getCategories();
        if (category >= categories.size()) {
            return getDefaultReward();
        }
        var weightVector = categories.get(category);
        float[] reward = new float[channelDims[REWARD_CHANNEL]];
        
        int rewardStart = channelDims[STATE_CHANNEL] + channelDims[ACTION_CHANNEL];
        for (int i = 0; i < channelDims[REWARD_CHANNEL]; i++) {
            reward[i] = (float) weightVector.get(rewardStart + i);
        }
        
        return reward;
    }
    
    private float[] getDefaultReward() {
        // Return neutral reward (0.5 for complement coded)
        float[] reward = new float[channelDims[REWARD_CHANNEL]];
        Arrays.fill(reward, 0.5f);
        return reward;
    }
    
    private float getRewardValue(float[] complementCodedReward) {
        // For complement coded reward [r, 1-r], return r
        if (complementCodedReward.length >= 2) {
            return complementCodedReward[0];
        }
        return complementCodedReward[0];
    }
    
    private Pattern createPattern(float[] data) {
        // Convert float array to double array for Pattern
        double[] doubleData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            doubleData[i] = data[i];
        }
        return new DenseVector(doubleData);
    }
}