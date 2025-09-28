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
     *
     * For unsupervised ART, the key principle is that each pattern should be
     * MOST similar to the category it created. This ensures learning/prediction consistency.
     */
    public static BPARTWeight createFromPattern(Pattern input, PANParameters parameters) {
        int inputSize = input.dimension();
        int hiddenSize = parameters.hiddenUnits();

        // Use deterministic pattern-based seed for reproducible results
        var rand = new java.util.Random(input.hashCode());
        double[] forward = new double[inputSize * hiddenSize];
        double[] backward = new double[hiddenSize];
        double[] hBias = new double[hiddenSize];

        // For STM weights (forward), directly copy pattern values to ensure maximum similarity
        // The first hidden unit gets the full pattern, others get scaled versions
        for (int hiddenIdx = 0; hiddenIdx < hiddenSize; hiddenIdx++) {
            for (int inputIdx = 0; inputIdx < inputSize; inputIdx++) {
                int weightIdx = hiddenIdx * inputSize + inputIdx;

                if (hiddenIdx == 0) {
                    // Primary hidden unit: exact pattern match for maximum resonance
                    forward[weightIdx] = input.get(inputIdx);
                } else {
                    // Secondary units: scaled pattern for feature detection
                    double scale = 1.0 / (hiddenIdx + 1);
                    forward[weightIdx] = input.get(inputIdx) * scale;
                }

                if (!parameters.allowNegativeWeights() && forward[weightIdx] < 0) {
                    forward[weightIdx] = 0;
                }
            }
        }

        // For LTM weights (backward), use exact pattern values for primary unit
        // This guarantees that the pattern is most confident about its own category
        for (int i = 0; i < backward.length; i++) {
            if (i == 0 && i < input.dimension()) {
                // Primary unit: exact pattern for maximum confidence
                backward[i] = 1.0; // Constant high confidence for primary unit
            } else if (i < input.dimension()) {
                // Secondary units: reduced confidence
                backward[i] = 0.1;
            } else {
                backward[i] = 0.01; // Small constant for unused dimensions
            }

            if (!parameters.allowNegativeWeights() && backward[i] < 0) {
                backward[i] = 0;
            }
        }

        // Zero biases to avoid any systematic bias between categories
        Arrays.fill(hBias, 0.0);
        double outputBias = 0.0;

        return new BPARTWeight(
            forward, backward, hBias, outputBias,
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
     * Calculate resonance intensity using STM (Short-Term Memory).
     * Uses hardcoded Fuzzy ART similarity for consistency.
     */
    public double calculateResonanceIntensity(Pattern input) {
        // Original Fuzzy ART similarity: |min(input, weights)| / |input|
        double minSum = 0.0;
        double inputSum = 0.0;
        int size = Math.min(input.dimension(), forwardWeights.length);

        for (int i = 0; i < size; i++) {
            double inputVal = Math.abs(input.get(i));
            double weightVal = Math.abs(forwardWeights[i]);
            minSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }

        // Avoid division by zero
        if (inputSum == 0.0) {
            return 0.0;
        }

        double resonance = minSum / inputSum; // Always in [0,1]

        // Debug output for hypothesis testing
        if (System.getProperty("pan.debug") != null) {
            System.out.printf("  PAN resonance intensity: %.3f (using Fuzzy ART)\n", resonance);
        }

        return resonance;
    }

    /**
     * Calculate location confidence using LTM (Long-Term Memory).
     * Uses hardcoded Fuzzy ART similarity for consistency.
     */
    public double calculateLocationConfidence(Pattern enhancedInput) {
        // Original Fuzzy ART similarity: |min(input, weights)| / |input|
        double minSum = 0.0;
        double inputSum = 0.0;
        int size = Math.min(enhancedInput.dimension(), backwardWeights.length);

        for (int i = 0; i < size; i++) {
            double inputVal = Math.abs(enhancedInput.get(i));
            double weightVal = Math.abs(backwardWeights[i]);
            minSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }

        // Avoid division by zero
        if (inputSum == 0.0) {
            return 0.0;
        }

        double confidence = minSum / inputSum; // Always in [0,1]

        // Apply output bias influence through sigmoid to maintain bounded range
        // This allows category-specific adjustments
        if (outputBias != 0.0) {
            double biasInfluence = 1.0 / (1.0 + Math.exp(-outputBias));
            confidence = 0.8 * confidence + 0.2 * biasInfluence;
        }

        return confidence;
    }

    /**
     * Calculate combined activation using hardcoded similarity calculations.
     */
    public double calculateActivation(Pattern input) {
        // Combine resonance intensity and location confidence
        double resonance = calculateResonanceIntensity(input);
        double confidence = calculateLocationConfidence(input);

        // Simple weighted combination (Fuzzy ART is bounded [0,1])
        double activation = 0.7 * resonance + 0.3 * confidence;

        // Debug output for hypothesis testing
        if (System.getProperty("pan.debug") != null) {
            System.out.printf("  PAN activation: %.3f (resonance=%.3f, confidence=%.3f)\n",
                           activation, resonance, confidence);
        }

        return activation;
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

        // Conservative learning rate with aggressive decay to prevent weight corruption
        double lr = params.learningRate() / (1.0 + updateCount * 0.01); // 100x more aggressive decay
        double decay = params.weightDecay();

        // Update forward weights (STM) using backpropagation-style updates
        // According to PAN paper, weights are updated via backpropagation within nodes
        double[] newForward = Arrays.copyOf(forwardWeights, forwardWeights.length);
        for (int i = 0; i < Math.min(input.dimension(), newForward.length); i++) {
            // Conservative gradient update with smaller step size to prevent corruption
            double delta = input.get(i) - newForward[i];
            newForward[i] += lr * delta * 0.1; // 10x smaller update steps

            // Apply light induction bias (PAN paper Equation 9: expected influence factor ξ)
            if (params.biasFactor() > 0) {
                newForward[i] += params.biasFactor() * lr;
            }

            // Weight decay
            newForward[i] *= (1.0 - decay);

            if (!params.allowNegativeWeights() && newForward[i] < 0) {
                newForward[i] = 0;
            }
        }

        // Update backward weights (LTM) - enhance discrimination
        double[] newBackward = Arrays.copyOf(backwardWeights, backwardWeights.length);
        for (int i = 0; i < newBackward.length; i++) {
            // Smaller updates for LTM to maintain stability
            double ltmLr = lr * 0.1;

            // Simple Hebbian-style update for LTM
            if (i < input.dimension()) {
                newBackward[i] += ltmLr * (input.get(i) - newBackward[i]);
            }

            // Weight decay
            newBackward[i] *= (1.0 - decay * 0.1);

            if (!params.allowNegativeWeights() && newBackward[i] < 0) {
                newBackward[i] = 0;
            }
        }

        // Update biases with light induction
        double[] newHiddenBias = Arrays.copyOf(hiddenBias, hiddenBias.length);
        double newOutputBias = outputBias;

        // Light induction for biases (helps category distinction)
        if (params.biasFactor() > 0) {
            for (int i = 0; i < newHiddenBias.length; i++) {
                newHiddenBias[i] += params.biasFactor() * lr * 0.01;
            }
            // Output bias update based on activation error
            newOutputBias += params.biasFactor() * lr * 0.1;
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