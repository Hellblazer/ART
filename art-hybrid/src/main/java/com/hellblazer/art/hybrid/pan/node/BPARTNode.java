package com.hellblazer.art.hybrid.pan.node;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * BPART (Backpropagation ART) Node - the key innovation in PAN.
 *
 * Unlike traditional ART nodes with non-negative weight constraints,
 * BPART nodes use local backpropagation allowing negative weights.
 * This enables more flexible pattern representation and better generalization.
 */
public class BPARTNode {

    private final int inputSize;
    private final int hiddenSize;
    private final boolean allowNegativeWeights;

    // Two-layer network: input -> hidden -> output
    private float[][] weightsInputHidden;  // [inputSize][hiddenSize]
    private float[] weightsHiddenOutput;   // [hiddenSize]
    private float[] biasHidden;            // [hiddenSize]
    private float biasOutput;

    // Momentum for SGD
    private float[][] momentumInputHidden;
    private float[] momentumHiddenOutput;
    private float[] momentumBiasHidden;
    private float momentumBiasOutput;

    // Cached activations for backprop
    private float[] lastInput;
    private float[] hiddenActivations;
    private float lastOutput;

    // Node metadata
    private int label = -1;  // Supervised label if applicable
    private long lastAccessTime = System.currentTimeMillis();
    private int activationCount = 0;
    private float resonance = 0;

    public BPARTNode(int inputSize, int hiddenSize, boolean allowNegativeWeights) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.allowNegativeWeights = allowNegativeWeights;

        // Initialize weights
        initializeWeightsRandom();

        // Initialize momentum
        this.momentumInputHidden = new float[inputSize][hiddenSize];
        this.momentumHiddenOutput = new float[hiddenSize];
        this.momentumBiasHidden = new float[hiddenSize];
        this.momentumBiasOutput = 0;

        // Initialize activations
        this.hiddenActivations = new float[hiddenSize];
    }

    /**
     * Initialize weights randomly (Xavier/He initialization)
     */
    private void initializeWeightsRandom() {
        Random rand = ThreadLocalRandom.current();
        float scale = (float) Math.sqrt(2.0 / inputSize);

        // Input to hidden weights
        weightsInputHidden = new float[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = (rand.nextFloat() - 0.5f) * 2 * scale;
                if (!allowNegativeWeights && weightsInputHidden[i][j] < 0) {
                    weightsInputHidden[i][j] = Math.abs(weightsInputHidden[i][j]);
                }
            }
        }

        // Hidden to output weights
        weightsHiddenOutput = new float[hiddenSize];
        scale = (float) Math.sqrt(2.0 / hiddenSize);
        for (int i = 0; i < hiddenSize; i++) {
            weightsHiddenOutput[i] = (rand.nextFloat() - 0.5f) * 2 * scale;
            if (!allowNegativeWeights && weightsHiddenOutput[i] < 0) {
                weightsHiddenOutput[i] = Math.abs(weightsHiddenOutput[i]);
            }
        }

        // Biases
        biasHidden = new float[hiddenSize];
        biasOutput = 0;
    }

    /**
     * Initialize weights from a pattern (ART-style)
     */
    public void initializeWeights(float[] pattern) {
        lastInput = pattern.clone();

        // Set first layer weights to encode the pattern
        float scale = 1.0f / pattern.length;
        for (int i = 0; i < Math.min(pattern.length, inputSize); i++) {
            for (int j = 0; j < hiddenSize; j++) {
                // Distribute pattern across hidden units
                weightsInputHidden[i][j] = pattern[i] * scale * (1 + 0.1f * (j % 3));
                if (!allowNegativeWeights && weightsInputHidden[i][j] < 0) {
                    weightsInputHidden[i][j] = Math.abs(weightsInputHidden[i][j]);
                }
            }
        }

        lastAccessTime = System.currentTimeMillis();
        activationCount++;
    }

    /**
     * Calculate activation (forward pass)
     */
    public float calculateActivation(float[] input) {
        lastInput = input;
        lastAccessTime = System.currentTimeMillis();
        activationCount++;

        // Input -> Hidden layer
        for (int j = 0; j < hiddenSize; j++) {
            float sum = biasHidden[j];
            for (int i = 0; i < Math.min(input.length, inputSize); i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            // ReLU activation
            hiddenActivations[j] = Math.max(0, sum);
        }

        // Hidden -> Output layer
        float output = biasOutput;
        for (int i = 0; i < hiddenSize; i++) {
            output += hiddenActivations[i] * weightsHiddenOutput[i];
        }

        // Sigmoid for output (bounded activation)
        lastOutput = 1.0f / (1.0f + (float) Math.exp(-output));
        resonance = lastOutput;  // Store as resonance

        return lastOutput;
    }

    /**
     * Calculate match score (for vigilance check)
     */
    public float calculateMatch(float[] input) {
        // Use cosine similarity between input and encoded pattern
        float dot = 0, normA = 0, normB = 0;

        // Compare with first layer weights (pattern encoding)
        for (int i = 0; i < Math.min(input.length, inputSize); i++) {
            float weightSum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                weightSum += weightsInputHidden[i][j];
            }
            weightSum /= hiddenSize;

            dot += input[i] * weightSum;
            normA += input[i] * input[i];
            normB += weightSum * weightSum;
        }

        if (normA < 1e-8 || normB < 1e-8) {
            return 0;
        }

        return dot / (float)(Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Backpropagate error and update weights
     */
    public void backpropagate(float[] target, float learningRate,
                             float momentum, float weightDecay) {
        // For unsupervised, target is the input itself (autoencoder-style)
        float targetOutput = calculateTargetSimilarity(target);

        // Output layer error
        float outputError = targetOutput - lastOutput;
        float outputGrad = outputError * lastOutput * (1 - lastOutput);  // Sigmoid derivative

        // Update hidden->output weights
        for (int i = 0; i < hiddenSize; i++) {
            float grad = outputGrad * hiddenActivations[i] - weightDecay * weightsHiddenOutput[i];
            momentumHiddenOutput[i] = momentum * momentumHiddenOutput[i] + learningRate * grad;
            weightsHiddenOutput[i] += momentumHiddenOutput[i];

            if (!allowNegativeWeights && weightsHiddenOutput[i] < 0) {
                weightsHiddenOutput[i] = 0;
            }
        }

        // Update output bias
        momentumBiasOutput = momentum * momentumBiasOutput + learningRate * outputGrad;
        biasOutput += momentumBiasOutput;

        // Hidden layer errors
        float[] hiddenErrors = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            hiddenErrors[j] = outputGrad * weightsHiddenOutput[j];
            // ReLU derivative
            if (hiddenActivations[j] <= 0) {
                hiddenErrors[j] = 0;
            }
        }

        // Update input->hidden weights
        for (int i = 0; i < Math.min(lastInput.length, inputSize); i++) {
            for (int j = 0; j < hiddenSize; j++) {
                float grad = hiddenErrors[j] * lastInput[i] - weightDecay * weightsInputHidden[i][j];
                momentumInputHidden[i][j] = momentum * momentumInputHidden[i][j] + learningRate * grad;
                weightsInputHidden[i][j] += momentumInputHidden[i][j];

                if (!allowNegativeWeights && weightsInputHidden[i][j] < 0) {
                    weightsInputHidden[i][j] = 0;
                }
            }
        }

        // Update hidden biases
        for (int j = 0; j < hiddenSize; j++) {
            momentumBiasHidden[j] = momentum * momentumBiasHidden[j] + learningRate * hiddenErrors[j];
            biasHidden[j] += momentumBiasHidden[j];
        }
    }

    /**
     * Calculate target similarity for unsupervised learning
     */
    private float calculateTargetSimilarity(float[] target) {
        // Compute similarity between target and encoded pattern
        float similarity = 0;
        float count = 0;

        for (int i = 0; i < Math.min(target.length, inputSize); i++) {
            float weightAvg = 0;
            for (int j = 0; j < hiddenSize; j++) {
                weightAvg += weightsInputHidden[i][j];
            }
            weightAvg /= hiddenSize;

            similarity += target[i] * weightAvg;
            count++;
        }

        return count > 0 ? similarity / count : 0;
    }

    /**
     * Strengthen weights (for LTM transfer)
     */
    public void strengthenWeights(float factor) {
        // Increase weight magnitudes
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] *= factor;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            weightsHiddenOutput[i] *= factor;
        }
    }

    /**
     * Get flattened weights for compatibility
     */
    public float[] getWeights() {
        // Flatten all weights into single array
        int totalSize = inputSize * hiddenSize + hiddenSize + hiddenSize + 1;
        float[] weights = new float[totalSize];

        int idx = 0;
        // Input->hidden weights
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights[idx++] = weightsInputHidden[i][j];
            }
        }

        // Hidden->output weights
        for (int i = 0; i < hiddenSize; i++) {
            weights[idx++] = weightsHiddenOutput[i];
        }

        // Biases
        for (int i = 0; i < hiddenSize; i++) {
            weights[idx++] = biasHidden[i];
        }
        weights[idx++] = biasOutput;

        return weights;
    }

    /**
     * Estimate memory usage
     */
    public long getMemoryUsage() {
        // Weights + momentum + activations
        long size = 0;
        size += (long) inputSize * hiddenSize * 4 * 2;  // weights + momentum
        size += hiddenSize * 4 * 4;  // hidden->output + biases
        size += hiddenSize * 4 * 2;  // activations
        size += inputSize * 4;  // cached input
        return size;
    }

    // Getters and setters

    public int getLabel() { return label; }
    public void setLabel(int label) { this.label = label; }

    public long getLastAccessTime() { return lastAccessTime; }
    public int getActivationCount() { return activationCount; }
    public float getResonance() { return resonance; }

    public boolean isAllowingNegativeWeights() { return allowNegativeWeights; }
}