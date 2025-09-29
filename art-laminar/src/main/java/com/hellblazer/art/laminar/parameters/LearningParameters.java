package com.hellblazer.art.laminar.parameters;

/**
 * Learning parameters for laminar circuits.
 *
 * @author Hal Hildebrand
 */
public interface LearningParameters {
    double getLearningRate();
    double getMomentum();
    boolean isFastLearning();
    double getWeightDecay();
}