package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.parameters.QuadraticNeuronARTParameters;
import com.hellblazer.art.core.utils.Matrix;

import java.util.Objects;

/**
 * Weight vector for QuadraticNeuronART algorithm.
 * 
 * Represents the hyper-ellipsoid parameters: transformation matrix W, bias vector b, and quadratic term s.
 * The weight structure is: [w11, w12, ..., w1n, w21, ..., w2n, ..., wnn, b1, b2, ..., bn, s]
 * where W is an n×n transformation matrix, b is an n-dimensional bias vector, and s is a scalar.
 */
public final class QuadraticNeuronARTWeight implements WeightVector {
    
    private final Matrix transformationMatrix;  // W matrix (n×n)
    private final double[] biasVector;          // b vector (n×1)  
    private final double quadraticTerm;         // s scalar
    private final int dimension;

    /**
     * Creates a new QuadraticNeuronARTWeight.
     * 
     * @param transformationMatrix the transformation matrix W (n×n)
     * @param biasVector          the bias vector b (n×1)
     * @param quadraticTerm       the quadratic term s
     * @throws NullPointerException     if transformationMatrix or biasVector is null
     * @throws IllegalArgumentException if dimensions don't match or are invalid
     */
    public QuadraticNeuronARTWeight(Matrix transformationMatrix, double[] biasVector, double quadraticTerm) {
        this.transformationMatrix = Objects.requireNonNull(transformationMatrix, "Transformation matrix cannot be null");
        this.biasVector = Objects.requireNonNull(biasVector, "Bias vector cannot be null").clone();
        this.quadraticTerm = quadraticTerm;
        
        if (transformationMatrix.getRowCount() != transformationMatrix.getColumnCount()) {
            throw new IllegalArgumentException("Transformation matrix must be square");
        }
        if (transformationMatrix.getRowCount() != biasVector.length) {
            throw new IllegalArgumentException("Matrix dimension must match bias vector length");
        }
        if (transformationMatrix.getRowCount() <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        
        this.dimension = transformationMatrix.getRowCount();
    }

    /**
     * Creates a new weight with identity matrix and input as bias (for new categories).
     * 
     * @param input the input pattern to use as initial bias
     * @param sInit the initial quadratic term value
     * @return a new QuadraticNeuronARTWeight initialized for a new category
     * @throws NullPointerException if input is null
     */
    public static QuadraticNeuronARTWeight createNew(Pattern input, double sInit) {
        Objects.requireNonNull(input, "Input pattern cannot be null");
        
        var dimension = input.dimension();
        var identityMatrix = Matrix.eye(dimension);
        var biasVector = new double[dimension];
        
        for (int i = 0; i < dimension; i++) {
            biasVector[i] = input.get(i);
        }
        
        return new QuadraticNeuronARTWeight(identityMatrix, biasVector, sInit);
    }

    @Override
    public double get(int index) {
        var totalSize = dimension * dimension + dimension + 1;
        if (index < 0 || index >= totalSize) {
            throw new IndexOutOfBoundsException("Index " + index + " is out of bounds [0, " + totalSize + ")");
        }
        
        var matrixSize = dimension * dimension;
        if (index < matrixSize) {
            // Matrix elements: row-major order
            var row = index / dimension;
            var col = index % dimension;
            return transformationMatrix.get(row, col);
        } else if (index < matrixSize + dimension) {
            // Bias vector elements
            return biasVector[index - matrixSize];
        } else {
            // Quadratic term
            return quadraticTerm;
        }
    }

    @Override
    public int dimension() {
        return dimension * dimension + dimension + 1;  // W matrix + b vector + s scalar
    }

    @Override
    public double l1Norm() {
        var sum = 0.0;
        
        // Add matrix elements
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                sum += Math.abs(transformationMatrix.get(i, j));
            }
        }
        
        // Add bias vector elements
        for (var bias : biasVector) {
            sum += Math.abs(bias);
        }
        
        // Add quadratic term
        sum += Math.abs(quadraticTerm);
        
        return sum;
    }

    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input pattern cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof QuadraticNeuronARTParameters params)) {
            throw new IllegalArgumentException("Parameters must be QuadraticNeuronARTParameters");
        }
        
        if (input.dimension() != dimension) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                                             " doesn't match weight dimension " + dimension);
        }

        // Calculate z = W * input
        var z = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            z[i] = 0.0;
            for (int j = 0; j < dimension; j++) {
                z[i] += transformationMatrix.get(i, j) * input.get(j);
            }
        }

        // Calculate z - b
        var zMinusB = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            zMinusB[i] = z[i] - biasVector[i];
        }

        // Calculate ||z - b||²
        var l2norm2ZB = 0.0;
        for (var diff : zMinusB) {
            l2norm2ZB += diff * diff;
        }

        // Calculate activation T = exp(-s² * ||z - b||²)
        var activation = Math.exp(-quadraticTerm * quadraticTerm * l2norm2ZB);
        
        // Common factor for updates: 2 * s² * T
        var sst2 = 2.0 * quadraticTerm * quadraticTerm * activation;

        // Update bias: b_new = b + lr_b * (2 * s² * T * (z - b))
        var newBias = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            newBias[i] = biasVector[i] + params.lrB() * sst2 * zMinusB[i];
        }

        // Update transformation matrix: W_new = W + lr_w * (-2 * s² * T * (z - b) ⊗ input)
        var newMatrix = new Matrix(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                var update = -params.lrW() * sst2 * zMinusB[i] * input.get(j);
                newMatrix.set(i, j, transformationMatrix.get(i, j) + update);
            }
        }

        // Update quadratic term: s_new = s + lr_s * (-2 * s * T * ||z - b||²)
        var newQuadraticTerm = quadraticTerm + params.lrS() * (-2.0 * quadraticTerm * activation * l2norm2ZB);

        return new QuadraticNeuronARTWeight(newMatrix, newBias, newQuadraticTerm);
    }

    /**
     * Calculate the activation value for this weight given an input pattern.
     * 
     * @param input the input pattern
     * @return the activation value T = exp(-s² * ||W*input - b||²)
     * @throws NullPointerException     if input is null
     * @throws IllegalArgumentException if input dimension doesn't match
     */
    public double calculateActivation(Pattern input) {
        Objects.requireNonNull(input, "Input pattern cannot be null");
        
        if (input.dimension() != dimension) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                                             " doesn't match weight dimension " + dimension);
        }

        // Calculate z = W * input  
        var z = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            z[i] = 0.0;
            for (int j = 0; j < dimension; j++) {
                z[i] += transformationMatrix.get(i, j) * input.get(j);
            }
        }

        // Calculate ||z - b||²
        var l2norm2ZB = 0.0;
        for (int i = 0; i < dimension; i++) {
            var diff = z[i] - biasVector[i];
            l2norm2ZB += diff * diff;
        }

        // Return activation T = exp(-s² * ||z - b||²)
        return Math.exp(-quadraticTerm * quadraticTerm * l2norm2ZB);
    }

    /**
     * Get the transformation matrix W.
     * 
     * @return a copy of the transformation matrix
     */
    public Matrix getTransformationMatrix() {
        // Create a manual copy since Matrix doesn't have a copy() method
        var copy = new Matrix(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                copy.set(i, j, transformationMatrix.get(i, j));
            }
        }
        return copy;
    }

    /**
     * Get the bias vector b.
     * 
     * @return a copy of the bias vector
     */
    public double[] getBiasVector() {
        return biasVector.clone();
    }

    /**
     * Get the quadratic term s.
     * 
     * @return the quadratic term value
     */
    public double getQuadraticTerm() {
        return quadraticTerm;
    }

    /**
     * Get the input dimension (not the total weight dimension).
     * 
     * @return the input space dimension
     */
    public int getInputDimension() {
        return dimension;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof QuadraticNeuronARTWeight other)) return false;
        
        return Double.compare(quadraticTerm, other.quadraticTerm) == 0 &&
               transformationMatrix.equals(other.transformationMatrix) &&
               java.util.Arrays.equals(biasVector, other.biasVector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(transformationMatrix, java.util.Arrays.hashCode(biasVector), quadraticTerm);
    }

    @Override
    public String toString() {
        return String.format("QuadraticNeuronARTWeight{dim=%d, s=%.3f, ||b||=%.3f}", 
                           dimension, quadraticTerm, java.util.Arrays.stream(biasVector).map(Math::abs).sum());
    }
}