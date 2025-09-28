package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Lateral pathway implementation for horizontal connections within a layer.
 * Implements competitive dynamics and lateral inhibition.
 *
 * @author Hal Hildebrand
 */
public class LateralPathway extends AbstractPathway {

    private final double inhibitionRadius;
    private final boolean useCenterSurround;

    public LateralPathway(String layerId, int layerSize) {
        this(layerId, layerSize, 2.0, true);
    }

    public LateralPathway(String layerId, int layerSize,
                         double inhibitionRadius, boolean useCenterSurround) {
        super(PathwayType.HORIZONTAL, layerId, layerId, layerSize, layerSize);
        this.inhibitionRadius = inhibitionRadius;
        this.useCenterSurround = useCenterSurround;
    }

    @Override
    protected Pattern processSignal(Pattern signal, IPathwayParameters parameters) {
        if (useCenterSurround) {
            return applyCenterSurroundInhibition(signal);
        } else {
            return applyGlobalInhibition(signal);
        }
    }

    /**
     * Apply center-surround lateral inhibition.
     * Nearby neurons inhibit each other more strongly.
     */
    private Pattern applyCenterSurroundInhibition(Pattern signal) {
        var result = new double[signal.dimension()];
        var dimension = signal.dimension();

        for (int i = 0; i < dimension; i++) {
            var activation = signal.get(i);
            var inhibition = 0.0;

            // Calculate inhibition from neighboring neurons
            for (int j = 0; j < dimension; j++) {
                if (i != j) {
                    var distance = Math.abs(i - j);
                    if (distance <= inhibitionRadius) {
                        // Stronger inhibition for closer neighbors
                        var inhibitionStrength = 1.0 / (1.0 + distance);
                        inhibition += signal.get(j) * inhibitionStrength * 0.1;
                    }
                }
            }

            // Apply inhibition
            result[i] = Math.max(0.0, activation - inhibition);
        }

        return new DenseVector(result);
    }

    /**
     * Apply global lateral inhibition.
     * All neurons inhibit each other equally.
     */
    private Pattern applyGlobalInhibition(Pattern signal) {
        var result = new double[signal.dimension()];

        // Calculate global inhibition
        var totalActivation = 0.0;
        var maxActivation = 0.0;
        for (int i = 0; i < signal.dimension(); i++) {
            var value = signal.get(i);
            totalActivation += value;
            maxActivation = Math.max(maxActivation, value);
        }

        var meanActivation = totalActivation / signal.dimension();
        var inhibitionFactor = 0.2; // Could be parameterized

        // Apply inhibition based on mean
        for (int i = 0; i < signal.dimension(); i++) {
            var value = signal.get(i);

            // Preserve strong activations, suppress weak ones
            if (value >= maxActivation * 0.8) {
                result[i] = value; // Winner preservation
            } else {
                result[i] = Math.max(0.0, value - meanActivation * inhibitionFactor);
            }
        }

        return new DenseVector(result);
    }

    @Override
    protected void initializeWeights() {
        // Lateral weights form an inhibitory matrix
        var size = weights.getRows();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    // Self-excitation (recurrent connection)
                    weights.set(i, j, 0.5);
                } else {
                    // Lateral inhibition
                    var distance = Math.abs(i - j);
                    if (distance <= inhibitionRadius) {
                        weights.set(i, j, -0.1 / (1.0 + distance));
                    } else {
                        weights.set(i, j, 0.0);
                    }
                }
            }
        }
    }

    @Override
    public void updateWeights(Pattern source, Pattern target, double learningRate) {
        if (!isAdaptive()) return;

        // Lateral weights typically don't learn in standard ART
        // But can implement competitive Hebbian learning if needed
        if (learningRate > 0) {
            for (int i = 0; i < weights.getRows(); i++) {
                for (int j = 0; j < weights.getCols(); j++) {
                    if (i != j) {
                        var srcValue = j < source.dimension() ? source.get(j) : 0.0;
                        var tgtValue = i < target.dimension() ? target.get(i) : 0.0;

                        // Anti-Hebbian learning for competition
                        var deltaW = -learningRate * tgtValue * srcValue * 0.01;
                        var newWeight = weights.get(i, j) + deltaW;

                        // Keep inhibitory
                        weights.set(i, j, Math.max(-1.0, Math.min(0.0, newWeight)));
                    }
                }
            }
        }
    }

    public double getInhibitionRadius() {
        return inhibitionRadius;
    }

    public boolean usesCenter() {
        return useCenterSurround;
    }
}