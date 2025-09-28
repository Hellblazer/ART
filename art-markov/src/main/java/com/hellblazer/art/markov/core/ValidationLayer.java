package com.hellblazer.art.markov.core;

import java.util.Arrays;

/**
 * Validation layer ensuring mathematical correctness of the hybrid ART-Markov system.
 *
 * This class provides static methods to validate:
 * - Stochastic matrix properties (row sums = 1, non-negativity)
 * - Markov property compliance
 * - Convergence detection
 * - Probability distribution validity
 */
public final class ValidationLayer {

    private ValidationLayer() {
        // Utility class - no instantiation
    }

    /**
     * Tolerance for floating-point comparisons.
     */
    private static final double TOLERANCE = 1e-10;

    /**
     * Validates that a matrix is stochastic (row sums = 1, all entries ≥ 0).
     *
     * @param matrix The transition matrix to validate
     * @throws IllegalArgumentException if the matrix is not stochastic
     */
    public static void validateStochasticMatrix(double[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }

        if (matrix.length == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }

        int n = matrix.length;

        for (int i = 0; i < n; i++) {
            if (matrix[i] == null) {
                throw new IllegalArgumentException("Matrix row " + i + " cannot be null");
            }

            if (matrix[i].length != n) {
                throw new IllegalArgumentException(
                    "Matrix must be square, but row " + i + " has length " + matrix[i].length +
                    " instead of " + n
                );
            }

            double rowSum = 0.0;
            for (int j = 0; j < n; j++) {
                double value = matrix[i][j];

                // Check non-negativity
                if (value < 0.0) {
                    throw new IllegalArgumentException(
                        String.format("Matrix entry [%d,%d] = %f is negative", i, j, value)
                    );
                }

                // Check for NaN/Infinity
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        String.format("Matrix entry [%d,%d] = %f is not finite", i, j, value)
                    );
                }

                rowSum += value;
            }

            // Check row sum equals 1
            if (Math.abs(rowSum - 1.0) > TOLERANCE) {
                throw new IllegalArgumentException(
                    String.format("Row %d sum = %f does not equal 1.0 (tolerance: %e)",
                        i, rowSum, TOLERANCE)
                );
            }
        }
    }

    /**
     * Normalizes a row to ensure it sums to 1.0.
     *
     * @param row The probability row to normalize
     * @return A new normalized array
     */
    public static double[] normalizeRow(double[] row) {
        if (row == null) {
            throw new IllegalArgumentException("Row cannot be null");
        }

        double sum = 0.0;
        var result = new double[row.length];

        // Check for non-negativity and calculate sum
        for (int i = 0; i < row.length; i++) {
            if (row[i] < 0.0) {
                throw new IllegalArgumentException(
                    String.format("Row entry [%d] = %f is negative", i, row[i])
                );
            }
            if (!Double.isFinite(row[i])) {
                throw new IllegalArgumentException(
                    String.format("Row entry [%d] = %f is not finite", i, row[i])
                );
            }
            sum += row[i];
        }

        // If sum is zero, return uniform distribution
        if (sum <= TOLERANCE) {
            Arrays.fill(result, 1.0 / row.length);
            return result;
        }

        // Normalize
        for (int i = 0; i < row.length; i++) {
            result[i] = row[i] / sum;
        }

        return result;
    }

    /**
     * Validates that a probability distribution is valid.
     *
     * @param distribution The probability distribution to validate
     * @throws IllegalArgumentException if the distribution is invalid
     */
    public static void validateProbabilityDistribution(double[] distribution) {
        if (distribution == null) {
            throw new IllegalArgumentException("Distribution cannot be null");
        }

        if (distribution.length == 0) {
            throw new IllegalArgumentException("Distribution cannot be empty");
        }

        double sum = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            double value = distribution[i];

            if (value < 0.0) {
                throw new IllegalArgumentException(
                    String.format("Distribution entry [%d] = %f is negative", i, value)
                );
            }

            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    String.format("Distribution entry [%d] = %f is not finite", i, value)
                );
            }

            sum += value;
        }

        if (Math.abs(sum - 1.0) > TOLERANCE) {
            throw new IllegalArgumentException(
                String.format("Distribution sum = %f does not equal 1.0 (tolerance: %e)",
                    sum, TOLERANCE)
            );
        }
    }

    /**
     * Checks if a transition matrix has converged to steady state.
     * Uses the total variation distance between successive powers of the matrix.
     *
     * @param matrix The transition matrix
     * @param threshold The convergence threshold
     * @return true if the matrix has converged to steady state
     */
    public static boolean hasConverged(double[][] matrix, double threshold) {
        validateStochasticMatrix(matrix);

        int n = matrix.length;
        var matrixSquared = multiplyMatrices(matrix, matrix);

        // Calculate total variation distance between P and P²
        double totalVariation = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                totalVariation += Math.abs(matrix[i][j] - matrixSquared[i][j]);
            }
        }
        totalVariation /= 2.0; // Total variation distance formula

        return totalVariation < threshold;
    }

    /**
     * Computes the steady-state distribution of a transition matrix.
     * Uses power iteration to find the eigenvector corresponding to eigenvalue 1.
     *
     * @param matrix The transition matrix
     * @param maxIterations Maximum number of iterations
     * @param tolerance Convergence tolerance
     * @return The steady-state distribution
     */
    public static double[] computeSteadyState(double[][] matrix, int maxIterations, double tolerance) {
        validateStochasticMatrix(matrix);

        int n = matrix.length;
        var distribution = new double[n];
        var nextDistribution = new double[n];

        // Initialize with uniform distribution
        Arrays.fill(distribution, 1.0 / n);

        for (int iter = 0; iter < maxIterations; iter++) {
            // Multiply distribution by matrix
            Arrays.fill(nextDistribution, 0.0);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    nextDistribution[j] += distribution[i] * matrix[i][j];
                }
            }

            // Check for convergence
            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(nextDistribution[i] - distribution[i]));
            }

            if (maxDiff < tolerance) {
                return nextDistribution;
            }

            // Swap arrays
            var temp = distribution;
            distribution = nextDistribution;
            nextDistribution = temp;
        }

        return distribution;
    }

    /**
     * Tests the Markov property by checking if P(X_t+1 | X_t, X_t-1) = P(X_t+1 | X_t).
     * This is a simplified test using conditional probability estimation.
     *
     * @param stateSequence Observed sequence of states
     * @param tolerance Tolerance for probability comparison
     * @return true if the Markov property appears to hold
     */
    public static boolean testMarkovProperty(int[] stateSequence, double tolerance) {
        if (stateSequence == null || stateSequence.length < 3) {
            return false; // Need at least 3 observations
        }

        // Find maximum state value to determine state space size
        int maxState = Arrays.stream(stateSequence).max().orElse(0);
        int numStates = maxState + 1;

        // Count transitions for first-order and second-order dependencies
        var firstOrderCounts = new int[numStates][numStates];
        var secondOrderCounts = new int[numStates][numStates][numStates];

        for (int t = 2; t < stateSequence.length; t++) {
            int currentState = stateSequence[t];
            int prevState = stateSequence[t - 1];
            int prevPrevState = stateSequence[t - 2];

            firstOrderCounts[prevState][currentState]++;
            secondOrderCounts[prevPrevState][prevState][currentState]++;
        }

        // Test if second-order transitions are close to first-order transitions
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numStates; j++) {
                int firstOrderTotal = Arrays.stream(firstOrderCounts[j]).sum();
                if (firstOrderTotal == 0) continue;

                double firstOrderProb = (double) firstOrderCounts[j][i] / firstOrderTotal;

                for (int k = 0; k < numStates; k++) {
                    int secondOrderTotal = Arrays.stream(secondOrderCounts[k][j]).sum();
                    if (secondOrderTotal == 0) continue;

                    double secondOrderProb = (double) secondOrderCounts[k][j][i] / secondOrderTotal;

                    if (Math.abs(firstOrderProb - secondOrderProb) > tolerance) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    /**
     * Helper method to multiply two matrices.
     */
    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        int n = a.length;
        var result = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }
}