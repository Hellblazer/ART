package com.hellblazer.art.hybrid.pan.weight;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * BPART (Backpropagation ART) Weight - The revolutionary weight vector that
 * combines ART's weight vectors with neural network backpropagation.
 *
 * This is an immutable record that implements WeightVector, allowing negative
 * weights (key PAN innovation) and gradient-based updates.
 *
 * Architecture:
 * - Forward weights: Input → Hidden layer connections
 * - Backward weights: Hidden → Output layer connections
 * - Supports both forward activation and backward gradient flow
 * - Immutable: all updates return new instances
 */
public record BPARTWeight(
    double[] forwardWeights,    // Input → Hidden connections
    double[] backwardWeights,   // Hidden → Output connections
    double[] hiddenBias,        // Hidden layer bias
    double outputBias,          // Output layer bias
    double[] lastHiddenState,   // Cached hidden activations
    double lastOutput,          // Cached output activation
    double lastError,           // Cached error for backprop
    long updateCount            // Number of updates (for learning rate decay)
) implements WeightVector {

    // Estimated memory footprint per node
    public static final long ESTIMATED_SIZE_BYTES = 8192; // ~8KB per node

    /**
     * Compact constructor with validation.
     */
    public BPARTWeight {
        // Defensive copying for immutability
        forwardWeights = Arrays.copyOf(forwardWeights, forwardWeights.length);
        backwardWeights = Arrays.copyOf(backwardWeights, backwardWeights.length);
        hiddenBias = Arrays.copyOf(hiddenBias, hiddenBias.length);
        lastHiddenState = Arrays.copyOf(lastHiddenState, lastHiddenState.length);

        // Validation
        if (forwardWeights.length == 0 || backwardWeights.length == 0) {
            throw new IllegalArgumentException("Weight arrays cannot be empty");
        }
        if (hiddenBias.length != backwardWeights.length) {
            throw new IllegalArgumentException("Hidden bias size must match backward weights");
        }
        if (lastHiddenState.length != backwardWeights.length) {
            throw new IllegalArgumentException("Hidden state size must match layer size");
        }
    }

    /**
     * Create a new BPART weight from an input pattern.
     */
    public static BPARTWeight createFromPattern(Pattern input, PANParameters parameters) {
        int inputSize = input.dimension();
        int hiddenSize = parameters.hiddenUnits();

        // Initialize weights using He initialization for ReLU
        var rand = ThreadLocalRandom.current();
        double scale = Math.sqrt(2.0 / inputSize);

        double[] forward = new double[inputSize * hiddenSize];
        double[] backward = new double[hiddenSize];
        double[] hBias = new double[hiddenSize];

        for (int i = 0; i < forward.length; i++) {
            forward[i] = rand.nextGaussian() * scale;
            if (!parameters.allowNegativeWeights() && forward[i] < 0) {
                forward[i] = Math.abs(forward[i]);
            }
        }

        scale = Math.sqrt(2.0 / hiddenSize);
        for (int i = 0; i < backward.length; i++) {
            backward[i] = rand.nextGaussian() * scale;
            if (!parameters.allowNegativeWeights() && backward[i] < 0) {
                backward[i] = Math.abs(backward[i]);
            }
            hBias[i] = 0.01; // Small positive bias for ReLU
        }

        // Initialize from input pattern (ART-style)
        for (int i = 0; i < Math.min(inputSize, forward.length); i++) {
            forward[i] += input.get(i % inputSize) * 0.1;
        }

        return new BPARTWeight(
            forward, backward, hBias, 0.0,
            new double[hiddenSize], 0.0, 0.0, 0L
        );
    }

    /**
     * Create with supervised target.
     */
    public static BPARTWeight createFromPatternWithTarget(Pattern input, Pattern target,
                                                          PANParameters parameters) {
        var weight = createFromPattern(input, parameters);

        // Bias initialization towards target
        if (target != null && target.dimension() > 0) {
            double targetMean = 0;
            for (int i = 0; i < target.dimension(); i++) {
                targetMean += target.get(i);
            }
            targetMean /= target.dimension();

            return new BPARTWeight(
                weight.forwardWeights, weight.backwardWeights,
                weight.hiddenBias, targetMean, // Initialize output bias to target mean
                weight.lastHiddenState, 0.0, 0.0, 0L
            );
        }

        return weight;
    }

    /**
     * Calculate forward activation for a pattern.
     */
    public double calculateActivation(Pattern input) {
        int inputSize = input.dimension();
        int hiddenSize = backwardWeights.length;

        // Forward pass: Input → Hidden
        double[] hidden = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++) {
            double sum = hiddenBias[h];
            for (int i = 0; i < inputSize; i++) {
                int idx = h * inputSize + i;
                if (idx < forwardWeights.length) {
                    sum += input.get(i) * forwardWeights[idx];
                }
            }
            // ReLU activation
            hidden[h] = Math.max(0, sum);
        }

        // Hidden → Output
        double output = outputBias;
        for (int h = 0; h < hiddenSize; h++) {
            output += hidden[h] * backwardWeights[h];
        }

        // Sigmoid activation for bounded output
        return 1.0 / (1.0 + Math.exp(-output));
    }

    /**
     * Compute gradients for backpropagation.
     */
    public double[] computeGradients(Pattern input, Pattern target) {
        int inputSize = input.dimension();
        int hiddenSize = backwardWeights.length;

        // Forward pass with state tracking
        double[] hidden = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++) {
            double sum = hiddenBias[h];
            for (int i = 0; i < inputSize; i++) {
                int idx = h * inputSize + i;
                if (idx < forwardWeights.length) {
                    sum += input.get(i) * forwardWeights[idx];
                }
            }
            hidden[h] = Math.max(0, sum); // ReLU
        }

        double output = outputBias;
        for (int h = 0; h < hiddenSize; h++) {
            output += hidden[h] * backwardWeights[h];
        }
        output = 1.0 / (1.0 + Math.exp(-output)); // Sigmoid

        // Compute error
        double error;
        if (target != null) {
            // Supervised: MSE loss gradient
            double targetValue = target.dimension() > 0 ? target.get(0) : 0.0;
            error = output - targetValue;
        } else {
            // Unsupervised: maximize activation (negative gradient)
            error = -output * (1 - output);
        }

        // Backpropagation
        double outputGrad = error * output * (1 - output); // Sigmoid derivative

        // Gradients for backward weights
        double[] gradients = new double[forwardWeights.length + backwardWeights.length + hiddenSize + 1];
        int idx = 0;

        // Hidden → Output gradients
        for (int h = 0; h < hiddenSize; h++) {
            gradients[idx++] = outputGrad * hidden[h];
        }

        // Input → Hidden gradients
        for (int h = 0; h < hiddenSize; h++) {
            double hiddenGrad = outputGrad * backwardWeights[h];
            if (hidden[h] > 0) { // ReLU derivative
                for (int i = 0; i < inputSize; i++) {
                    int widx = h * inputSize + i;
                    if (widx < forwardWeights.length && idx < gradients.length) {
                        gradients[idx++] = hiddenGrad * input.get(i);
                    }
                }
            } else {
                idx += inputSize; // Skip gradients for inactive neurons
            }
        }

        return gradients;
    }

    // WeightVector interface implementation

    @Override
    public double get(int index) {
        if (index < forwardWeights.length) {
            return forwardWeights[index];
        } else if (index < forwardWeights.length + backwardWeights.length) {
            return backwardWeights[index - forwardWeights.length];
        } else {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds");
        }
    }

    @Override
    public int dimension() {
        return forwardWeights.length + backwardWeights.length;
    }

    @Override
    public double l1Norm() {
        double sum = 0;
        for (double w : forwardWeights) sum += Math.abs(w);
        for (double w : backwardWeights) sum += Math.abs(w);
        for (double b : hiddenBias) sum += Math.abs(b);
        sum += Math.abs(outputBias);
        return sum;
    }

    @Override
    public WeightVector update(Pattern input, Object parameters) {
        if (!(parameters instanceof PANParameters params)) {
            throw new IllegalArgumentException("Parameters must be PANParameters");
        }

        // Handle gradient pattern for backprop updates
        double[] gradients = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            gradients[i] = input.get(i);
        }

        // Apply gradient descent with momentum
        double lr = params.learningRate() / (1.0 + updateCount * 0.0001); // Learning rate decay
        double momentum = params.momentum();
        double decay = params.weightDecay();

        // Update forward weights
        double[] newForward = Arrays.copyOf(forwardWeights, forwardWeights.length);
        for (int i = 0; i < newForward.length && i < gradients.length; i++) {
            newForward[i] -= lr * (gradients[i] + decay * newForward[i]);
            if (!params.allowNegativeWeights() && newForward[i] < 0) {
                newForward[i] = 0;
            }
        }

        // Update backward weights
        double[] newBackward = Arrays.copyOf(backwardWeights, backwardWeights.length);
        int offset = forwardWeights.length;
        for (int i = 0; i < newBackward.length && offset + i < gradients.length; i++) {
            newBackward[i] -= lr * (gradients[offset + i] + decay * newBackward[i]);
            if (!params.allowNegativeWeights() && newBackward[i] < 0) {
                newBackward[i] = 0;
            }
        }

        // Update biases
        double[] newHiddenBias = Arrays.copyOf(hiddenBias, hiddenBias.length);
        double newOutputBias = outputBias;

        // Add light induction bias (ε from paper Equation 9)
        if (params.biasFactor() > 0) {
            for (int i = 0; i < newHiddenBias.length; i++) {
                newHiddenBias[i] += params.biasFactor() * lr;
            }
            newOutputBias += params.biasFactor() * lr;
        }

        return new BPARTWeight(
            newForward, newBackward, newHiddenBias, newOutputBias,
            lastHiddenState, lastOutput, lastError, updateCount + 1
        );
    }

    /**
     * Check if this weight should be consolidated to LTM.
     */
    public boolean shouldConsolidateToLTM(double threshold) {
        // Consolidate if weight has been updated frequently
        return updateCount > 100 && Math.abs(lastError) < threshold;
    }

    /**
     * Compute similarity to another BPART weight.
     */
    public double similarity(BPARTWeight other) {
        double sim = 0;
        double norm1 = 0, norm2 = 0;

        // Cosine similarity of forward weights
        for (int i = 0; i < Math.min(forwardWeights.length, other.forwardWeights.length); i++) {
            sim += forwardWeights[i] * other.forwardWeights[i];
            norm1 += forwardWeights[i] * forwardWeights[i];
            norm2 += other.forwardWeights[i] * other.forwardWeights[i];
        }

        if (norm1 > 0 && norm2 > 0) {
            return sim / (Math.sqrt(norm1) * Math.sqrt(norm2));
        }
        return 0;
    }
}