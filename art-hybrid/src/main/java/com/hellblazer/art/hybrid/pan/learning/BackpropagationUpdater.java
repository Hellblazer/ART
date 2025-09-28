package com.hellblazer.art.hybrid.pan.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.util.Arrays;

/**
 * Implements correct backpropagation according to PAN paper Equation 7.
 *
 * Paper Equation 7: w_i(k+1) = w_i(k) - η(-{O_j* + λ_j})(1-O_j)x_i
 * Which simplifies to: w_i(k+1) = w_i(k) + η(O_j* + λ_j)(1-O_j)x_i
 *
 * This fixes the double negative error in the original implementation.
 */
public class BackpropagationUpdater {

    private final double momentumFactor;
    private double[] momentum;

    public BackpropagationUpdater(double momentumFactor) {
        this.momentumFactor = momentumFactor;
    }

    /**
     * Apply backpropagation to update weights according to PAN paper.
     *
     * @param weight The current weight to update
     * @param input The input pattern
     * @param output The output activation (O_j in the paper)
     * @param lightInduction The light induction factor (λ_j in the paper)
     * @param learningRate The learning rate (η in the paper)
     * @return Updated weight vector
     */
    public BPARTWeight applyBackpropagation(BPARTWeight weight, Pattern input,
                                           double output, double lightInduction,
                                           double learningRate) {
        double[] forwardWeights = weight.forwardWeights();
        double[] newForward = Arrays.copyOf(forwardWeights, forwardWeights.length);

        // Initialize momentum if needed
        if (momentum == null || momentum.length != forwardWeights.length) {
            momentum = new double[forwardWeights.length];
        }

        // Paper Equation 7: w_i(k+1) = w_i(k) + η(O_j* + λ_j)(1-O_j)x_i
        // Where O_j* is the target output (for unsupervised, we use output itself)
        double factor = learningRate * (output + lightInduction) * (1.0 - output);

        for (int i = 0; i < Math.min(input.dimension(), newForward.length); i++) {
            // Calculate weight update
            double delta = factor * input.get(i);

            // Apply momentum if configured
            if (momentumFactor > 0) {
                momentum[i] = momentumFactor * momentum[i] + delta;
                newForward[i] += momentum[i];
            } else {
                newForward[i] += delta;
            }
        }

        // Normalize weights to prevent unbounded growth
        normalizeWeights(newForward);

        // Return new weight with updated forward weights
        return new BPARTWeight(
            newForward,
            weight.backwardWeights(),
            weight.hiddenBias(),
            weight.outputBias(),
            weight.lastHiddenState(),
            weight.lastOutput(),
            weight.lastError(),
            weight.updateCount() + 1
        );
    }

    /**
     * Apply supervised backpropagation with target pattern.
     */
    public BPARTWeight applySupervisedBackpropagation(BPARTWeight weight, Pattern input,
                                                     Pattern target, double lightInduction,
                                                     double learningRate) {
        double[] forwardWeights = weight.forwardWeights();
        double[] backwardWeights = weight.backwardWeights();
        double[] newForward = Arrays.copyOf(forwardWeights, forwardWeights.length);
        double[] newBackward = Arrays.copyOf(backwardWeights, backwardWeights.length);

        // Compute current output
        double output = weight.calculateActivation(input);

        // Extract target value
        double targetValue = extractTargetValue(target);

        // Compute error for supervised learning
        double error = targetValue - output;

        // Apply gradient with light induction
        // w_i(k+1) = w_i(k) + η * error * (1-output) * input_i + λ
        double factor = learningRate * (error + lightInduction) * (1.0 - output);

        // Update forward weights
        for (int i = 0; i < Math.min(input.dimension(), newForward.length); i++) {
            newForward[i] += factor * input.get(i);
        }

        // Update backward weights for LTM
        for (int i = 0; i < newBackward.length && i < target.dimension(); i++) {
            newBackward[i] += learningRate * error * target.get(i);
        }

        // Normalize both weight sets
        normalizeWeights(newForward);
        normalizeWeights(newBackward);

        return new BPARTWeight(
            newForward,
            newBackward,
            weight.hiddenBias(),
            weight.outputBias() + learningRate * error, // Update output bias
            weight.lastHiddenState(),
            output,
            error,
            weight.updateCount() + 1
        );
    }

    private void normalizeWeights(double[] weights) {
        double norm = 0.0;
        for (double w : weights) {
            norm += w * w;
        }
        if (norm > 0) {
            norm = Math.sqrt(norm);
            for (int i = 0; i < weights.length; i++) {
                weights[i] /= norm;
            }
        }
    }

    private double extractTargetValue(Pattern target) {
        if (target == null || target.dimension() == 0) {
            return 1.0; // Default target for unsupervised
        }

        // For one-hot encoding, find the max value
        double maxVal = 0.0;
        for (int i = 0; i < target.dimension(); i++) {
            maxVal = Math.max(maxVal, target.get(i));
        }
        return maxVal;
    }

    /**
     * Reset momentum for new training session.
     */
    public void resetMomentum() {
        if (momentum != null) {
            Arrays.fill(momentum, 0.0);
        }
    }
}