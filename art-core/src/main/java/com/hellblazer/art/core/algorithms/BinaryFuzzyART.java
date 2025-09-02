package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;

import java.util.Arrays;
import java.util.List;

/**
 * Binary Fuzzy ART implementation.
 * 
 * This algorithm is designed for binary input patterns and uses complement coding
 * to create a symmetric representation. It combines fuzzy set operations with
 * binary constraints for robust pattern recognition.
 */
public class BinaryFuzzyART extends BaseART {
    
    public static class BinaryFuzzyARTParameters {
        private final double rho;    // vigilance parameter
        private final double alpha;  // choice parameter
        private final double beta;   // learning rate
        private final double gamma;  // contribution parameter
        private final int maxCategories;
        
        private BinaryFuzzyARTParameters(Builder builder) {
            this.rho = builder.rho;
            this.alpha = builder.alpha;
            this.beta = builder.beta;
            this.gamma = builder.gamma;
            this.maxCategories = builder.maxCategories;
        }
        
        public static Builder builder() {
            return new Builder();
        }
        
        public static class Builder {
            private double rho = 0.9;
            private double alpha = 0.01;
            private double beta = 1.0;
            private double gamma = 3.0;
            private int maxCategories = 100;
            
            public Builder rho(double rho) {
                this.rho = rho;
                return this;
            }
            
            public Builder alpha(double alpha) {
                this.alpha = alpha;
                return this;
            }
            
            public Builder beta(double beta) {
                this.beta = beta;
                return this;
            }
            
            public Builder gamma(double gamma) {
                this.gamma = gamma;
                return this;
            }
            
            public Builder maxCategories(int maxCategories) {
                this.maxCategories = maxCategories;
                return this;
            }
            
            public BinaryFuzzyARTParameters build() {
                return new BinaryFuzzyARTParameters(this);
            }
        }
    }
    
    public static class BinaryFuzzyARTWeight implements WeightVector {
        private final double[] values;
        
        public BinaryFuzzyARTWeight(int dimension) {
            // Double dimension for complement coding
            this.values = new double[dimension * 2];
            Arrays.fill(values, 1.0); // Initialize to all ones
        }
        
        private BinaryFuzzyARTWeight(double[] values) {
            this.values = Arrays.copyOf(values, values.length);
        }
        
        @Override
        public int dimension() {
            return values.length;
        }
        
        @Override
        public double get(int index) {
            return values[index];
        }
        
        @Override
        public double l1Norm() {
            var sum = 0.0;
            for (var v : values) {
                sum += Math.abs(v);
            }
            return sum;
        }
        
        @Override
        public WeightVector update(Pattern input, Object parameters) {
            var params = (BinaryFuzzyARTParameters) parameters;
            var complemented = complementCode(input);
            var newValues = new double[values.length];
            
            // Update weights using fuzzy AND with learning rate
            for (int i = 0; i < values.length; i++) {
                newValues[i] = params.beta * Math.min(complemented.get(i), values[i]) + 
                              (1 - params.beta) * values[i];
            }
            
            return new BinaryFuzzyART.BinaryFuzzyARTWeight(newValues);
        }
        
        private Pattern complementCode(Pattern input) {
            var complemented = new double[input.dimension() * 2];
            for (int i = 0; i < input.dimension(); i++) {
                complemented[i] = input.get(i);
                complemented[i + input.dimension()] = 1.0 - input.get(i);
            }
            return Pattern.of(complemented);
        }
    }
    
    private final int inputSize;
    
    public BinaryFuzzyART(int inputSize, BinaryFuzzyARTParameters params) {
        super();
        this.inputSize = inputSize;
        validateParameters(params);
    }
    
    public int getInputSize() {
        return inputSize;
    }
    
    public List<WeightVector> getWeights() {
        return getCategories();
    }
    
    private void validateParameters(BinaryFuzzyARTParameters params) {
        if (params.rho < 0 || params.rho > 1) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1]");
        }
        if (params.alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        if (params.beta < 0 || params.beta > 1) {
            throw new IllegalArgumentException("Beta must be in [0, 1]");
        }
        if (params.gamma < 0) {
            throw new IllegalArgumentException("Gamma must be non-negative");
        }
    }
    
    private void validateBinaryInput(Pattern input) {
        for (int i = 0; i < input.dimension(); i++) {
            var value = input.get(i);
            if (value != 0.0 && value != 1.0) {
                throw new IllegalArgumentException("BinaryFuzzyART requires binary input (0 or 1)");
            }
        }
    }
    
    private Pattern complementCode(Pattern input) {
        var complemented = new double[input.dimension() * 2];
        for (int i = 0; i < input.dimension(); i++) {
            complemented[i] = input.get(i);
            complemented[i + input.dimension()] = 1.0 - input.get(i);
        }
        return Pattern.of(complemented);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        var params = (BinaryFuzzyARTParameters) parameters;
        
        // Initialize with complement coded input
        var complemented = complementCode(input);
        var values = new double[complemented.dimension()];
        for (int i = 0; i < complemented.dimension(); i++) {
            values[i] = complemented.get(i);
        }
        
        return new BinaryFuzzyARTWeight(values);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        var params = (BinaryFuzzyARTParameters) parameters;
        var complemented = complementCode(input);
        
        // Calculate fuzzy AND (minimum) between input and weight
        var minSum = 0.0;
        for (int i = 0; i < complemented.dimension(); i++) {
            minSum += Math.min(complemented.get(i), weight.get(i));
        }
        
        // Calculate norm of weight
        var weightNorm = 0.0;
        for (int i = 0; i < weight.dimension(); i++) {
            weightNorm += weight.get(i);
        }
        
        // Activation function with choice parameter
        return minSum / (params.alpha + weightNorm);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        var params = (BinaryFuzzyARTParameters) parameters;
        var complemented = complementCode(input);
        
        // Calculate fuzzy AND (minimum) between input and weight
        var minSum = 0.0;
        for (int i = 0; i < complemented.dimension(); i++) {
            minSum += Math.min(complemented.get(i), weight.get(i));
        }
        
        // Calculate norm of input
        var inputNorm = 0.0;
        for (int i = 0; i < complemented.dimension(); i++) {
            inputNorm += complemented.get(i);
        }
        
        // Match function
        var match = minSum / inputNorm;
        
        if (match >= params.rho) {
            return new MatchResult.Accepted(match, params.rho);
        } else {
            return new MatchResult.Rejected(match, params.rho);
        }
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector weight, Object parameters) {
        var params = (BinaryFuzzyARTParameters) parameters;
        var complemented = complementCode(input);
        var newValues = new double[weight.dimension()];
        
        // Update weights using fuzzy AND with learning rate
        for (int i = 0; i < weight.dimension(); i++) {
            var oldWeight = weight.get(i);
            newValues[i] = params.beta * Math.min(complemented.get(i), oldWeight) + 
                          (1 - params.beta) * oldWeight;
        }
        
        return new BinaryFuzzyART.BinaryFuzzyARTWeight(newValues);
    }
    
    public ActivationResult stepFit(Pattern input, BinaryFuzzyARTParameters parameters) {
        validateBinaryInput(input);
        return super.stepFit(input, parameters);
    }
}