package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.parameters.ILayerParameters;

/**
 * Category layer implementation (F2) for category representation and selection.
 * Implements winner-take-all dynamics and category prototype storage.
 *
 * @author Hal Hildebrand
 */
public class CategoryLayer extends AbstractLayer {

    private int activeCategory = -1;

    public CategoryLayer(int size) {
        super(LayerType.CATEGORY, size);
    }

    public CategoryLayer(String id, int size) {
        super(id, LayerType.CATEGORY, size);
    }

    @Override
    public Pattern processBottomUp(Pattern input, ILayerParameters parameters) {
        // Category selection based on bottom-up input
        var activations = new double[size];

        // Calculate choice function for each category
        for (int i = 0; i < size; i++) {
            double numerator = 0.0;
            double denominator = 0.0;

            for (int j = 0; j < input.dimension(); j++) {
                var weight = weights.get(i, j);
                var inputValue = input.get(j);
                numerator += Math.min(inputValue, weight);
                denominator += weight;
            }

            // ART choice function: |I ∧ w| / (α + |w|)
            var alpha = 0.001; // Small choice parameter
            activations[i] = numerator / (alpha + denominator);
        }

        // Winner-take-all: select category with highest activation
        var winnerIndex = findWinner(activations);
        activeCategory = winnerIndex;

        // Create winner-take-all output
        var result = new double[size];
        if (winnerIndex >= 0) {
            result[winnerIndex] = activations[winnerIndex];
        }

        return new DenseVector(result);
    }

    @Override
    public Pattern processTopDown(Pattern feedback, ILayerParameters parameters) {
        // Generate top-down expectation from active category
        if (activeCategory >= 0 && activeCategory < size) {
            var expectation = new double[weights.getCols()];

            // Read out weights of active category as expectation
            for (int j = 0; j < weights.getCols(); j++) {
                expectation[j] = weights.get(activeCategory, j);
            }

            return new DenseVector(expectation);
        } else {
            // No active category - return zero expectation
            return new DenseVector(new double[weights.getCols()]);
        }
    }

    @Override
    public Pattern processLateral(Pattern lateral, ILayerParameters parameters) {
        // Strong lateral inhibition in category layer (winner-take-all)
        return enforceWinnerTakeAll(lateral);
    }

    private int findWinner(double[] activations) {
        var maxActivation = Double.NEGATIVE_INFINITY;
        var winner = -1;

        for (int i = 0; i < activations.length; i++) {
            if (activations[i] > maxActivation) {
                maxActivation = activations[i];
                winner = i;
            }
        }

        return winner;
    }

    private Pattern enforceWinnerTakeAll(Pattern input) {
        var activations = new double[input.dimension()];
        var winnerIndex = -1;
        var maxValue = Double.NEGATIVE_INFINITY;

        // Find winner
        for (int i = 0; i < input.dimension(); i++) {
            var value = input.get(i);
            if (value > maxValue) {
                maxValue = value;
                winnerIndex = i;
            }
        }

        // Set only winner to be active
        if (winnerIndex >= 0) {
            activations[winnerIndex] = maxValue;
        }

        return new DenseVector(activations);
    }

    public int getActiveCategory() {
        return activeCategory;
    }

    public void setActiveCategory(int category) {
        this.activeCategory = category;
    }
}