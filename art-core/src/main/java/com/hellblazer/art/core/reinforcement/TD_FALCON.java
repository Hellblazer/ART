package com.hellblazer.art.core.reinforcement;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import java.util.*;

/**
 * TD-FALCON (Temporal Difference FALCON) for reinforcement learning with SARSA.
 * 
 * Extends FALCON with temporal difference learning capabilities, enabling
 * the system to learn from delayed rewards through SARSA (State-Action-Reward-State-Action)
 * updates. This allows for more efficient learning in sequential decision-making tasks.
 * 
 * Based on:
 * - Tan, A.-H., Lu, N., & Xiao, D. (2008). Integrating temporal difference methods 
 *   and self-organizing neural networks for reinforcement learning with delayed 
 *   evaluative feedback. IEEE Transactions on Neural Networks, 19(2), 230-244.
 * 
 * Key features:
 * - SARSA learning for Q-value updates
 * - Epsilon-greedy exploration
 * - Eligibility traces for TD(λ)
 * - Discount factor for future rewards
 * 
 * @author Hal Hildebrand
 */
public class TD_FALCON extends FALCON {
    
    // TD learning parameters
    private float tdAlpha;  // TD learning rate (0 < α ≤ 1)
    private float tdGamma;  // Discount factor (0 ≤ γ ≤ 1)
    private float epsilon = 0.1f;  // Epsilon for ε-greedy exploration
    private float lambda = 0.0f;   // Eligibility trace decay (0 ≤ λ ≤ 1)
    
    // Q-value storage
    private final Map<StateActionKey, Float> qValues = new HashMap<>();
    
    // Eligibility traces for TD(λ)
    private final Map<StateActionKey, Float> eligibilityTraces = new HashMap<>();
    
    /**
     * Create a new TD-FALCON with specified modules and TD parameters.
     * 
     * @param stateModule The ART module for state clustering
     * @param actionModule The ART module for action clustering
     * @param rewardModule The ART module for reward clustering
     * @param gammaValues Channel combination weights for FusionART
     * @param channelDims The dimension of each channel (state, action, reward)
     * @param tdAlpha TD learning rate
     * @param tdGamma Discount factor for future rewards
     */
    public TD_FALCON(BaseART<?> stateModule, BaseART<?> actionModule, 
                     BaseART<?> rewardModule, float[] gammaValues, int[] channelDims,
                     float tdAlpha, float tdGamma) {
        super(stateModule, actionModule, rewardModule, gammaValues, channelDims);
        
        if (tdAlpha <= 0 || tdAlpha > 1) {
            throw new IllegalArgumentException("TD alpha must be in (0, 1], got: " + tdAlpha);
        }
        if (tdGamma < 0 || tdGamma > 1) {
            throw new IllegalArgumentException("TD gamma must be in [0, 1], got: " + tdGamma);
        }
        
        this.tdAlpha = tdAlpha;
        this.tdGamma = tdGamma;
    }
    
    /**
     * Calculate SARSA rewards by applying temporal difference to immediate rewards.
     * 
     * SARSA update rule:
     * Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
     * 
     * @param states Sequence of states
     * @param actions Sequence of actions
     * @param rewards Immediate rewards
     * @return TD-adjusted rewards for FusionART training
     */
    public float[][] calculateSARSA(float[][] states, float[][] actions, float[][] rewards) {
        if (states.length != actions.length || states.length != rewards.length) {
            throw new IllegalArgumentException("States, actions, and rewards must have same length");
        }
        
        float[][] tdRewards = new float[rewards.length][rewards[0].length];
        
        for (int i = 0; i < states.length - 1; i++) {
            // Current state-action
            var currentKey = new StateActionKey(states[i], actions[i]);
            float currentQ = qValues.getOrDefault(currentKey, 0.0f);
            
            // Next state-action
            var nextKey = new StateActionKey(states[i + 1], actions[i + 1]);
            float nextQ = qValues.getOrDefault(nextKey, 0.0f);
            
            // TD target
            float tdTarget = rewards[i][0] + tdGamma * nextQ;
            
            // TD error
            float tdError = tdTarget - currentQ;
            
            // Update Q-value
            float newQ = currentQ + tdAlpha * tdError;
            qValues.put(currentKey, newQ);
            
            // Adjusted reward for FusionART
            tdRewards[i][0] = tdTarget;
        }
        
        // Last reward (terminal state)
        if (states.length > 0) {
            int lastIdx = states.length - 1;
            var lastKey = new StateActionKey(states[lastIdx], actions[lastIdx]);
            float lastQ = qValues.getOrDefault(lastKey, 0.0f);
            float newQ = lastQ + tdAlpha * (rewards[lastIdx][0] - lastQ);
            qValues.put(lastKey, newQ);
            tdRewards[lastIdx][0] = rewards[lastIdx][0];
        }
        
        return tdRewards;
    }
    
    /**
     * Update Q-value for a single SARSA transition.
     * 
     * @param state Current state
     * @param action Current action
     * @param reward Immediate reward
     * @param nextState Next state
     * @param nextAction Next action
     */
    public void updateSARSA(float[] state, float[] action, float reward, 
                           float[] nextState, float[] nextAction) {
        var currentKey = new StateActionKey(state, action);
        var nextKey = new StateActionKey(nextState, nextAction);
        
        float currentQ = qValues.getOrDefault(currentKey, 0.0f);
        float nextQ = qValues.getOrDefault(nextKey, 0.0f);
        
        float tdTarget = reward + tdGamma * nextQ;
        float tdError = tdTarget - currentQ;
        
        float newQ = currentQ + tdAlpha * tdError;
        qValues.put(currentKey, newQ);
        
        // Also update FusionART with the experience
        float[][] states = {state};
        float[][] actions = {action};
        float[][] rewards = {{tdTarget}};
        super.fit(states, actions, rewards);
    }
    
    /**
     * Update Q-value for a terminal state (no next state/action).
     * 
     * @param state Terminal state
     * @param action Action taken
     * @param reward Terminal reward
     */
    public void updateTerminalSARSA(float[] state, float[] action, float reward) {
        var key = new StateActionKey(state, action);
        float currentQ = qValues.getOrDefault(key, 0.0f);
        float newQ = currentQ + tdAlpha * (reward - currentQ);
        qValues.put(key, newQ);
        
        // Update FusionART
        float[][] states = {state};
        float[][] actions = {action};
        float[][] rewards = {{reward}};
        super.fit(states, actions, rewards);
    }
    
    /**
     * Batch update SARSA for multiple transitions.
     * 
     * @param states Batch of states
     * @param actions Batch of actions
     * @param rewards Batch of rewards
     * @param nextStates Batch of next states
     * @param nextActions Batch of next actions
     */
    public void batchUpdateSARSA(float[][] states, float[][] actions, float[] rewards,
                                 float[][] nextStates, float[][] nextActions) {
        for (int i = 0; i < states.length; i++) {
            updateSARSA(states[i], actions[i], rewards[i], nextStates[i], nextActions[i]);
        }
    }
    
    /**
     * Get Q-value for a state-action pair.
     * 
     * @param state State vector
     * @param action Action vector
     * @return Q-value
     */
    public float getQValue(float[] state, float[] action) {
        var key = new StateActionKey(state, action);
        return qValues.getOrDefault(key, 0.0f);
    }
    
    /**
     * Get epsilon-greedy action (exploration vs exploitation).
     * 
     * @param state Current state
     * @param actionSpace Available actions
     * @return Selected action
     */
    public float[] getEpsilonGreedyAction(float[] state, float[][] actionSpace) {
        if (random.nextFloat() < epsilon) {
            // Exploration: random action
            return actionSpace[random.nextInt(actionSpace.length)];
        } else {
            // Exploitation: greedy action
            return getAction(state, actionSpace, OptimalityMode.MAX);
        }
    }
    
    /**
     * Update SARSA with eligibility traces for TD(λ).
     * 
     * @param state Current state
     * @param action Current action
     * @param reward Immediate reward
     * @param nextState Next state
     * @param nextAction Next action
     */
    public void updateSARSAWithTraces(float[] state, float[] action, float reward,
                                      float[] nextState, float[] nextAction) {
        var currentKey = new StateActionKey(state, action);
        var nextKey = new StateActionKey(nextState, nextAction);
        
        float currentQ = qValues.getOrDefault(currentKey, 0.0f);
        float nextQ = qValues.getOrDefault(nextKey, 0.0f);
        
        float tdError = reward + tdGamma * nextQ - currentQ;
        
        // Update eligibility trace for current state-action
        eligibilityTraces.put(currentKey, 
            eligibilityTraces.getOrDefault(currentKey, 0.0f) + 1.0f);
        
        // Update all Q-values using eligibility traces
        for (var entry : eligibilityTraces.entrySet()) {
            var key = entry.getKey();
            float trace = entry.getValue();
            
            if (trace > 0.001f) {  // Only update if trace is significant
                float oldQ = qValues.getOrDefault(key, 0.0f);
                float newQ = oldQ + tdAlpha * tdError * trace;
                qValues.put(key, newQ);
                
                // Decay trace
                eligibilityTraces.put(key, trace * tdGamma * lambda);
            } else {
                // Remove negligible traces
                eligibilityTraces.remove(key);
            }
        }
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
    
    @Override
    public void fit(float[][] states, float[][] actions, float[][] rewards) {
        // Apply SARSA before training FusionART
        float[][] tdRewards = calculateSARSA(states, actions, rewards);
        super.fit(states, actions, tdRewards);
    }
    
    @Override
    public void partialFit(float[][] states, float[][] actions, float[][] rewards) {
        // Apply SARSA for online learning
        float[][] tdRewards = calculateSARSA(states, actions, rewards);
        super.partialFit(states, actions, tdRewards);
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
    
    public float getEpsilon() {
        return epsilon;
    }
    
    public void setEpsilon(float epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("Epsilon must be in [0, 1]");
        }
        this.epsilon = epsilon;
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
    
    /**
     * Key for storing state-action pairs in Q-value map.
     */
    public static class StateActionKey {
        private final float[] state;
        private final float[] action;
        private final int hashCode;
        
        public StateActionKey(float[] state, float[] action) {
            this.state = state.clone();
            this.action = action.clone();
            this.hashCode = computeHashCode();
        }
        
        private int computeHashCode() {
            int result = 1;
            for (float s : state) {
                result = 31 * result + Float.floatToIntBits(s);
            }
            for (float a : action) {
                result = 31 * result + Float.floatToIntBits(a);
            }
            return result;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof StateActionKey)) return false;
            
            StateActionKey other = (StateActionKey) obj;
            return Arrays.equals(state, other.state) && Arrays.equals(action, other.action);
        }
        
        @Override
        public int hashCode() {
            return hashCode;
        }
    }
}