package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.parameters.ILayerParameters;

/**
 * Feature layer implementation (F1) for feature representation and competition.
 * Implements competitive dynamics and feature selection.
 *
 * @author Hal Hildebrand
 */
public class FeatureLayer extends AbstractLayer {

    public FeatureLayer(int size) {
        super(LayerType.FEATURE, size);
    }

    public FeatureLayer(String id, int size) {
        super(id, LayerType.FEATURE, size);
    }

    @Override
    public Pattern processBottomUp(Pattern input, ILayerParameters parameters) {
        // Apply bottom-up weights and competition
        var result = new double[size];

        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < input.dimension(); j++) {
                sum += weights.get(i, j) * input.get(j);
            }
            result[i] = sum;
        }

        // Apply competition (winner-take-all or k-winners-take-all)
        return applyCompetition(new DenseVector(result), parameters);
    }

    @Override
    public Pattern processTopDown(Pattern feedback, ILayerParameters parameters) {
        // Process top-down expectations
        var result = new double[size];

        for (int i = 0; i < size; i++) {
            if (i < feedback.dimension()) {
                result[i] = feedback.get(i) * activation.get(i);
            } else {
                result[i] = activation.get(i);
            }
        }

        return new DenseVector(result);
    }

    @Override
    public Pattern processLateral(Pattern lateral, ILayerParameters parameters) {
        // Lateral inhibition for competitive dynamics
        return applyLateralInhibition(lateral, parameters);
    }

    private Pattern applyCompetition(Pattern input, ILayerParameters parameters) {
        var threshold = parameters.getActivationThreshold();
        var result = new double[input.dimension()];

        // Find maximum activation
        var maxActivation = calculateMax(input);

        // Apply threshold and competition
        for (int i = 0; i < input.dimension(); i++) {
            var value = input.get(i);
            if (value >= threshold && value >= maxActivation * 0.7) {
                result[i] = value;
            } else {
                result[i] = 0.0;
            }
        }

        return new DenseVector(result);
    }

    private Pattern applyLateralInhibition(Pattern input, ILayerParameters parameters) {
        var inhibitionStrength = 0.1; // Could be parameterized
        var result = new double[input.dimension()];

        var meanActivation = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            meanActivation += input.get(i);
        }
        meanActivation /= input.dimension();

        for (int i = 0; i < input.dimension(); i++) {
            var inhibition = meanActivation * inhibitionStrength;
            result[i] = Math.max(0.0, input.get(i) - inhibition);
        }

        return new DenseVector(result);
    }
}