/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 */
package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.parameters.QuadraticNeuronARTParameters;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.QuadraticNeuronARTWeight;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Quadratic Neuron ART for Clustering.
 * 
 * This module implements Quadratic Neuron ART as first published in:
 * Su, M.-C., & Liu, T.-K. (2001). Application of neural networks using quadratic 
 * junctions in cluster analysis. Neurocomputing, 37, 165–175.
 * 
 * Su, M.-C., & Liu, Y.-C. (2005). A new approach to clustering data with arbitrary 
 * shapes. Pattern Recognition, 38, 1887–1901.
 * 
 * Quadratic Neuron ART clusters data in hyper-ellipsoids by utilizing a quadratic
 * neural network for activation and resonance.
 * 
 * @author Hal Hildebrand
 */
public final class QuadraticNeuronART extends BaseART<QuadraticNeuronARTParameters> {
    
    /**
     * Create a new QuadraticNeuronART instance
     */
    public QuadraticNeuronART() {
        super();
    }
    
    /**
     * Calculate the activation value for a category using the quadratic neuron function.
     * 
     * Activation function: T = exp(-s^2 * ||W*x - b||^2)
     * Where:
     * - x is the input vector
     * - W is the transformation matrix
     * - b is the centroid/bias vector
     * - s is the quadratic term
     * 
     * @param input the input vector
     * @param weight the category weight vector (must be QuadraticNeuronARTWeight)
     * @param parameters the algorithm parameters (must be QuadraticNeuronARTParameters)
     * @return the activation value for this category
     */
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, QuadraticNeuronARTParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var params = parameters;
        
        if (!(weight instanceof QuadraticNeuronARTWeight qnWeight)) {
            throw new IllegalArgumentException("Weight must be QuadraticNeuronARTWeight");
        }
        
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        var matrix = qnWeight.getMatrix();
        var centroid = qnWeight.getCentroid();
        var s = qnWeight.getS();
        
        // Calculate z = W * input
        var z = matrixVectorMultiply(matrix, inputArray);
        
        // Calculate ||z - b||^2
        double l2norm2 = 0.0;
        for (int i = 0; i < z.length; i++) {
            double diff = z[i] - centroid[i];
            l2norm2 += diff * diff;
        }
        
        // Calculate activation = exp(-s^2 * ||z - b||^2)
        return Math.exp(-s * s * l2norm2);
    }
    
    /**
     * Test whether the input matches the category according to the vigilance criterion.
     * 
     * In QuadraticNeuronART, the match criterion is the same as the activation.
     * The pattern is accepted if the activation exceeds the vigilance parameter.
     * 
     * @param input the input vector
     * @param weight the category weight vector
     * @param parameters the algorithm parameters
     * @return MatchResult.Accepted if vigilance test passes, MatchResult.Rejected otherwise
     */
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, QuadraticNeuronARTParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var params = parameters;
        
        // In QuadraticNeuronART, match criterion is the same as activation
        double activation = calculateActivation(input, weight, parameters);
        
        if (activation >= params.vigilance()) {
            return new MatchResult.Accepted(activation, params.vigilance());
        } else {
            return new MatchResult.Rejected(activation, params.vigilance());
        }
    }
    
    /**
     * Update the weight vector using the quadratic neuron learning rule.
     * 
     * @param input the input pattern
     * @param weight the weight vector to update
     * @param parameters the algorithm parameters
     * @return the updated weight vector
     */
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector weight, QuadraticNeuronARTParameters parameters) {
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var params = parameters;
        
        if (!(weight instanceof QuadraticNeuronARTWeight qnWeight)) {
            throw new IllegalArgumentException("Weight must be QuadraticNeuronARTWeight");
        }
        
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        var matrix = qnWeight.getMatrix();
        var centroid = qnWeight.getCentroid();
        var s = qnWeight.getS();
        int dim = inputArray.length;
        
        // Calculate z = W * input
        var z = matrixVectorMultiply(matrix, inputArray);
        
        // Calculate ||z - b||^2
        double l2norm2 = 0.0;
        var zMinusB = new double[dim];
        for (int i = 0; i < dim; i++) {
            zMinusB[i] = z[i] - centroid[i];
            l2norm2 += zMinusB[i] * zMinusB[i];
        }
        
        // Calculate activation T
        double T = Math.exp(-s * s * l2norm2);
        
        // Calculate 2 * s^2 * T
        double sst2 = 2 * s * s * T;
        
        // Update centroid: b_new = b + lr_b * (2 * s^2 * T * (z - b))
        var newCentroid = new double[dim];
        for (int i = 0; i < dim; i++) {
            newCentroid[i] = centroid[i] + params.getLearningRateB() * (sst2 * zMinusB[i]);
        }
        
        // Update matrix: W_new = W + lr_w * (-2 * s^2 * T * ((z - b) * input^T))
        var newMatrix = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                newMatrix[i][j] = matrix[i][j] + 
                    params.getLearningRateW() * (-sst2 * zMinusB[i] * inputArray[j]);
            }
        }
        
        // Update s: s_new = s + lr_s * (-2 * s * T * ||z - b||^2)
        double newS = s + params.getLearningRateS() * (-2 * s * T * l2norm2);
        
        // Create updated weight
        return QuadraticNeuronARTWeight.fromComponents(newMatrix, newCentroid, newS);
    }
    
    /**
     * Create a new weight vector for a new category.
     * 
     * @param input the input pattern that will form the new category
     * @param parameters the algorithm parameters
     * @return a new weight vector initialized for the input
     */
    @Override
    protected WeightVector createInitialWeight(Pattern input, QuadraticNeuronARTParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        var params = parameters;
        
        int dim = input.dimension();
        int dim2 = dim * dim;
        var data = new double[dim2 + dim + 1];
        
        // Create identity matrix
        for (int i = 0; i < dim; i++) {
            data[i * dim + i] = 1.0;
        }
        
        // Set centroid to input pattern
        for (int i = 0; i < dim; i++) {
            data[dim2 + i] = input.get(i);
        }
        
        // Set initial s parameter
        data[dim2 + dim] = params.getSInit();
        
        return new QuadraticNeuronARTWeight(data, dim);
    }
    
    
    /**
     * Get the cluster centers (centroids) for all categories
     * 
     * @return List of cluster centers
     */
    public List<double[]> getClusterCenters() {
        var centers = new ArrayList<double[]>();
        for (var weight : categories) {
            if (weight instanceof QuadraticNeuronARTWeight qnWeight) {
                centers.add(qnWeight.getCentroid());
            }
        }
        return centers;
    }
    
    /**
     * Get all weights as QuadraticNeuronARTWeight objects
     * 
     * @return List of weights
     */
    public List<QuadraticNeuronARTWeight> getWeights() {
        var weights = new ArrayList<QuadraticNeuronARTWeight>();
        for (var weight : categories) {
            if (weight instanceof QuadraticNeuronARTWeight qnWeight) {
                weights.add(qnWeight);
            }
        }
        return weights;
    }
    
    
    /**
     * Helper method to multiply matrix by vector
     */
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = vector.length;
        var result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        
        return result;
    }

    @Override
    public void close() throws Exception {
        // No-op for vanilla implementation
    }
}