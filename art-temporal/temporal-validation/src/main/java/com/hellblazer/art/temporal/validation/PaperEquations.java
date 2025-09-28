package com.hellblazer.art.temporal.validation;

/**
 * Reference implementation of equations from Kazerounian & Grossberg (2014).
 * "Neural dynamics of speech and language coding: Developmental programs, perceptual grouping, and competition for short-term memory"
 *
 * All equation numbers refer to the paper.
 */
public class PaperEquations {

    /**
     * Equation 1: Shunting on-center off-surround network dynamics.
     * dx_i/dt = -A_i * x_i + (B - x_i) * f(x_i) * [I_i + Σ_k∈N_i^+ C_ik * g(x_k)]
     *          - x_i * [Σ_j∈N_i^- D_ij * h(x_j)]
     */
    public static double shuntingDynamics(
        double x_i,           // Current activation
        double A_i,           // Decay rate
        double B,             // Upper bound (ceiling)
        double I_i,           // External input
        double[] excitation,  // Excitatory inputs from neighbors
        double[] inhibition,  // Inhibitory inputs from neighbors
        boolean selfExcite    // Whether to include self-excitation
    ) {
        // Activation function (typically sigmoid or linear threshold)
        double f_xi = activationFunction(x_i);

        // Excitatory term
        double exciteSum = I_i;
        for (double e : excitation) {
            exciteSum += e;
        }
        if (selfExcite) {
            exciteSum *= f_xi;
        }

        // Inhibitory term
        double inhibitSum = 0.0;
        for (double inh : inhibition) {
            inhibitSum += inh;
        }

        // Shunting equation
        return -A_i * x_i + (B - x_i) * exciteSum - x_i * inhibitSum;
    }

    /**
     * Equation 2: Transmitter habituation dynamics.
     * dz_i/dt = ε(1 - z_i) - z_i * y_i * (λ + μ * y_i)
     *
     * Where y_i is the habituative transmitter gate that multiplies signal S_i.
     */
    public static double transmitterDynamics(
        double z_i,      // Current transmitter level
        double epsilon,  // Recovery rate
        double y_i,      // Habituative signal (activation)
        double lambda,   // Linear depletion rate
        double mu        // Quadratic depletion rate
    ) {
        // Recovery term
        double recovery = epsilon * (1.0 - z_i);

        // Depletion term (linear + quadratic)
        double depletion = z_i * y_i * (lambda + mu * y_i);

        return recovery - depletion;
    }

    /**
     * Equation 3: Item node activation in working memory.
     * dx_i/dt = -x_i + (1 - x_i) * [bottom_up_i + top_down_i] - x_i * reset_i
     */
    public static double itemNodeDynamics(
        double x_i,         // Current item activation
        double bottomUp,    // Bottom-up input
        double topDown,     // Top-down priming
        double reset        // Reset signal
    ) {
        // Simplified shunting equation for item nodes
        return -x_i + (1.0 - x_i) * (bottomUp + topDown) - x_i * reset;
    }

    /**
     * Equation 4: List chunk activation.
     * dy_j/dt = -y_j + (1 - y_j) * F_j * [Σ_i∈chunk_j w_ij * x_i]
     */
    public static double listChunkDynamics(
        double y_j,              // Current chunk activation
        double F_j,              // Chunk gate/readiness
        double[] itemActivations, // Activations of items in chunk
        double[] weights         // Weights from items to chunk
    ) {
        // Weighted sum of item activations
        double weightedSum = 0.0;
        for (int i = 0; i < itemActivations.length; i++) {
            weightedSum += weights[i] * itemActivations[i];
        }

        // Chunk dynamics with gating
        return -y_j + (1.0 - y_j) * F_j * weightedSum;
    }

    /**
     * Equation 5: Competitive queuing for item selection.
     * dx_i/dt = h(-x_i + I_i - f(Σ_j≠i x_j))
     *
     * Where h is a faster-than-linear signal function and f is the inhibitory kernel.
     */
    public static double competitiveQueuingDynamics(
        double x_i,              // Current activation
        double I_i,              // Input strength (priority)
        double[] otherActivations // Activations of competing items
    ) {
        // Sum of competing activations
        double competitionSum = 0.0;
        for (double other : otherActivations) {
            competitionSum += other;
        }

        // Apply inhibitory kernel (e.g., sigmoid)
        double inhibition = sigmoidFunction(competitionSum, 1.0, 2.0);

        // Competitive dynamics with faster-than-linear function
        double signal = -x_i + I_i - inhibition;
        return fasterThanLinear(signal);
    }

    /**
     * Equation 6: Masking field reset dynamics.
     * dr_i/dt = -α * r_i + β * (1 - r_i) * chunk_signal_i
     */
    public static double resetDynamics(
        double r_i,         // Current reset level
        double alpha,       // Decay rate
        double beta,        // Growth rate
        double chunkSignal  // Signal from completed chunk
    ) {
        return -alpha * r_i + beta * (1.0 - r_i) * chunkSignal;
    }

    /**
     * Equation 7: Spectral timing for adaptive resonance.
     * dT_k/dt = -T_k/τ_k + G_k(t)
     *
     * Where G_k(t) is a Gaussian centered at preferred interval τ_k.
     */
    public static double spectralTimingDynamics(
        double T_k,    // Current timing component
        double tau_k,  // Preferred interval
        double t,      // Current time
        double sigma   // Width of Gaussian
    ) {
        // Gaussian timing function
        double diff = t - tau_k;
        double G_k = Math.exp(-diff * diff / (2.0 * sigma * sigma));

        // Timing dynamics
        return -T_k / tau_k + G_k;
    }

    /**
     * Equation 8: Primacy gradient in working memory.
     * P_i = exp(-γ * i) * (1 + δ * recency_i)
     *
     * Combines exponential primacy with recency boost.
     */
    public static double primacyGradient(
        int position,      // Item position in list
        double gamma,      // Primacy decay rate
        double delta,      // Recency boost factor
        double recency     // Recency signal (0-1)
    ) {
        // Exponential primacy
        double primacy = Math.exp(-gamma * position);

        // Add recency boost
        return primacy * (1.0 + delta * recency);
    }

    /**
     * Equation 9: Learning rate adaptation.
     * dw_ij/dt = η * x_i * (x_j - w_ij)
     *
     * Instar learning rule for adaptive weights.
     */
    public static double instarLearning(
        double w_ij,   // Current weight
        double x_i,    // Presynaptic activation
        double x_j,    // Postsynaptic activation
        double eta     // Learning rate
    ) {
        return eta * x_i * (x_j - w_ij);
    }

    /**
     * Equation 10: Outstar learning for motor sequences.
     * dw_ij/dt = η * x_j * (x_i - w_ij)
     */
    public static double outstarLearning(
        double w_ij,   // Current weight
        double x_i,    // Presynaptic activation
        double x_j,    // Postsynaptic activation
        double eta     // Learning rate
    ) {
        return eta * x_j * (x_i - w_ij);
    }

    // Utility functions

    private static double activationFunction(double x) {
        // Linear threshold
        return Math.max(0, x);
    }

    private static double sigmoidFunction(double x, double gain, double threshold) {
        return 1.0 / (1.0 + Math.exp(-gain * (x - threshold)));
    }

    private static double fasterThanLinear(double x) {
        // Power function with exponent > 1
        if (x > 0) {
            return x * x;  // Quadratic
        }
        return 0;
    }

    /**
     * Verify conservation laws for shunting dynamics.
     * Total activity should be bounded: 0 ≤ x_i ≤ B
     */
    public static boolean verifyShuntingBounds(double x_i, double B) {
        return x_i >= 0 && x_i <= B;
    }

    /**
     * Verify transmitter conservation.
     * Transmitter levels should be bounded: 0 ≤ z_i ≤ 1
     */
    public static boolean verifyTransmitterBounds(double z_i) {
        return z_i >= 0.0 && z_i <= 1.0;
    }

    /**
     * Verify energy function decreases (Lyapunov stability).
     * E = Σ_i (x_i^2)/2 + interaction terms
     */
    public static double computeLyapunovEnergy(double[] activations, double[][] weights) {
        double energy = 0.0;

        // Self-energy terms
        for (double x : activations) {
            energy += 0.5 * x * x;
        }

        // Interaction energy
        for (int i = 0; i < activations.length; i++) {
            for (int j = i + 1; j < activations.length; j++) {
                energy -= weights[i][j] * activations[i] * activations[j];
            }
        }

        return energy;
    }
}