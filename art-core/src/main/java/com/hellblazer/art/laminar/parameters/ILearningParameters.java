package com.hellblazer.art.laminar.parameters;

/**
 * Learning parameters.
 *
 * @author Hal Hildebrand
 */
public interface ILearningParameters {
    double getLearningRate();
    double getMomentum();
    boolean useAdaptiveLearning();
    double getWeightDecay();
}