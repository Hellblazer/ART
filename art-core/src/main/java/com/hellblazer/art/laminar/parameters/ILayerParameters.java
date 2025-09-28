package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for layer dynamics and processing.
 *
 * @author Hal Hildebrand
 */
public interface ILayerParameters {

    // Shunting equation parameters
    double getDecayRate();           // A in dx/dt = -Ax + ...
    double getUpperBound();           // B in (B-x)E
    double getLowerBound();           // C in (x+C)I

    // Processing parameters
    double getActivationThreshold();
    double getSaturationLevel();
    boolean useNormalization();
    NormalizationType getNormalizationType();

    // Temporal parameters
    double getTimeConstant();
    int getIntegrationSteps();

    // Noise parameters
    double getNoiseLevel();
    NoiseType getNoiseType();

    enum NormalizationType {
        L1, L2, MAX, NONE
    }

    enum NoiseType {
        GAUSSIAN, UNIFORM, NONE
    }
}