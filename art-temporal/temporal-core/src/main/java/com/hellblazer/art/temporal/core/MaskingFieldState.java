package com.hellblazer.art.temporal.core;

import java.util.Arrays;

/**
 * State representation for multi-scale masking field.
 * Based on Equation 5 from Kazerounian & Grossberg (2014):
 * dY_jk/dt = -α * Y_jk + (β - Y_jk) * [f(Y_jk) + I_jk] - Y_jk * Σ g(Y_lm)
 */
public class MaskingFieldState extends State {
    private final double[][] cellActivations;  // Y_jk: [scale][position]
    private final double[][] bottomUpInputs;   // I_jk: inputs from working memory
    private final int[] preferredLengths;      // Preferred chunk length for each scale
    private final double[] scaleActivations;   // Total activation per scale

    public MaskingFieldState(double[][] cellActivations, double[][] bottomUpInputs, int[] preferredLengths) {
        this.cellActivations = deepClone(cellActivations);
        this.bottomUpInputs = deepClone(bottomUpInputs);
        this.preferredLengths = preferredLengths.clone();
        this.scaleActivations = computeScaleActivations();
    }

    public MaskingFieldState(int numScales, int fieldSize) {
        this.cellActivations = new double[numScales][fieldSize];
        this.bottomUpInputs = new double[numScales][fieldSize];
        this.preferredLengths = new int[numScales];
        this.scaleActivations = new double[numScales];

        // Initialize preferred lengths (1, 3, 5, 7, ...)
        for (int i = 0; i < numScales; i++) {
            preferredLengths[i] = 2 * i + 1;
        }
    }

    private double[][] deepClone(double[][] array) {
        var result = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i].clone();
        }
        return result;
    }

    private double[] computeScaleActivations() {
        var activations = new double[cellActivations.length];
        for (int scale = 0; scale < cellActivations.length; scale++) {
            double sum = 0.0;
            for (double activation : cellActivations[scale]) {
                sum += activation;
            }
            activations[scale] = sum;
        }
        return activations;
    }

    public double[][] getCellActivations() {
        return deepClone(cellActivations);
    }

    public double[] getScaleActivations() {
        return scaleActivations.clone();
    }

    public int[] getPreferredLengths() {
        return preferredLengths.clone();
    }

    /**
     * Get activation at specific scale and position.
     */
    public double getActivation(int scale, int position) {
        return cellActivations[scale][position];
    }

    /**
     * Set bottom-up input from working memory.
     */
    public void setBottomUpInput(int scale, int position, double input) {
        bottomUpInputs[scale][position] = input;
    }

    /**
     * Find the winner-take-all cell across all scales.
     */
    public WinnerCell findGlobalWinner() {
        int winnerScale = -1;
        int winnerPosition = -1;
        double maxActivation = 0.0;

        for (int scale = 0; scale < cellActivations.length; scale++) {
            for (int pos = 0; pos < cellActivations[scale].length; pos++) {
                if (cellActivations[scale][pos] > maxActivation) {
                    maxActivation = cellActivations[scale][pos];
                    winnerScale = scale;
                    winnerPosition = pos;
                }
            }
        }

        return new WinnerCell(winnerScale, winnerPosition, maxActivation);
    }

    /**
     * Find winner within a specific scale.
     */
    public int findScaleWinner(int scale) {
        int winner = -1;
        double maxActivation = 0.0;

        for (int pos = 0; pos < cellActivations[scale].length; pos++) {
            if (cellActivations[scale][pos] > maxActivation) {
                maxActivation = cellActivations[scale][pos];
                winner = pos;
            }
        }

        return winner;
    }

    /**
     * Apply asymmetric lateral inhibition between scales.
     * Larger scales inhibit smaller scales more strongly.
     */
    public double computeAsymmetricInhibition(int fromScale, int toScale) {
        if (fromScale == toScale) {
            return 0.0;
        }

        // Larger scales have stronger inhibition
        double baseFactor = scaleActivations[fromScale];
        double scaleFactor = (fromScale > toScale) ? 2.0 : 0.5;

        return baseFactor * scaleFactor;
    }

    /**
     * Check if masking field has converged.
     */
    public boolean hasConverged(MaskingFieldState previousState, double threshold) {
        if (previousState == null) {
            return false;
        }

        double totalChange = 0.0;
        for (int scale = 0; scale < cellActivations.length; scale++) {
            for (int pos = 0; pos < cellActivations[scale].length; pos++) {
                double diff = Math.abs(cellActivations[scale][pos] -
                                      previousState.cellActivations[scale][pos]);
                totalChange += diff;
            }
        }

        return totalChange < threshold;
    }

    @Override
    public State add(State other) {
        if (!(other instanceof MaskingFieldState m)) {
            throw new IllegalArgumentException("Can only add MaskingFieldState to MaskingFieldState");
        }

        var result = new double[cellActivations.length][];
        for (int scale = 0; scale < cellActivations.length; scale++) {
            result[scale] = vectorizedOperation(cellActivations[scale], m.cellActivations[scale],
                (a, b) -> a.add(b));
        }

        return new MaskingFieldState(result, bottomUpInputs, preferredLengths);
    }

    @Override
    public State scale(double scalar) {
        var result = new double[cellActivations.length][];
        for (int scale = 0; scale < cellActivations.length; scale++) {
            result[scale] = new double[cellActivations[scale].length];
            for (int pos = 0; pos < cellActivations[scale].length; pos++) {
                result[scale][pos] = Math.max(0.0, cellActivations[scale][pos] * scalar);
            }
        }
        return new MaskingFieldState(result, bottomUpInputs, preferredLengths);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof MaskingFieldState m)) {
            throw new IllegalArgumentException("Can only compute distance to MaskingFieldState");
        }

        double sum = 0.0;
        for (int scale = 0; scale < cellActivations.length; scale++) {
            for (int pos = 0; pos < cellActivations[scale].length; pos++) {
                double diff = cellActivations[scale][pos] - m.cellActivations[scale][pos];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }

    @Override
    public int dimension() {
        int total = 0;
        for (double[] scale : cellActivations) {
            total += scale.length;
        }
        return total;
    }

    @Override
    public State copy() {
        return new MaskingFieldState(cellActivations, bottomUpInputs, preferredLengths);
    }

    @Override
    public double[] toArray() {
        // Flatten all scales into single array
        var result = new double[dimension()];
        int index = 0;
        for (double[] scale : cellActivations) {
            System.arraycopy(scale, 0, result, index, scale.length);
            index += scale.length;
        }
        return result;
    }

    @Override
    public State fromArray(double[] values) {
        // Reconstruct multi-scale structure from flat array
        var result = new double[cellActivations.length][];
        int index = 0;
        for (int scale = 0; scale < cellActivations.length; scale++) {
            int length = cellActivations[scale].length;
            result[scale] = Arrays.copyOfRange(values, index, index + length);
            index += length;
        }
        return new MaskingFieldState(result, bottomUpInputs, preferredLengths);
    }

    /**
     * Winner cell representation.
     */
    public record WinnerCell(int scale, int position, double activation) {
        public boolean isValid() {
            return scale >= 0 && position >= 0 && activation > 0.0;
        }

        public int getPreferredLength(int[] lengths) {
            return scale >= 0 && scale < lengths.length ? lengths[scale] : 0;
        }
    }
}