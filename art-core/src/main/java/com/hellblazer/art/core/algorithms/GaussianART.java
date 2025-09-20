package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.parameters.GaussianParameters;
import com.hellblazer.art.core.AbstractStatisticalART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.GaussianWeight;
import java.util.Objects;

/**
 * GaussianART implementation using the AbstractStatisticalART framework.
 * 
 * GaussianART is a neural network architecture based on Adaptive Resonance Theory (ART)
 * that performs unsupervised learning using Gaussian probability distributions to model categories.
 * Each category is represented as a multivariate Gaussian with mean, covariance, and sample count.
 * 
 * Key Features:
 * - Probabilistic activation based on Gaussian likelihood
 * - Vigilance test using probability density threshold
 * - Incremental learning with online mean/variance updates
 * - No complement coding (works directly with input dimensions)
 * 
 * Mathematical Foundation:
 * - Activation: A_j = p(x | μ_j, Σ_j) (Gaussian probability density)
 * - Vigilance: p(x | μ_j, Σ_j) ≥ ρ (probability threshold)
 * - Learning: Online updates to mean and covariance statistics
 * 
 * @see AbstractStatisticalART for the statistical template framework
 * @see GaussianWeight for Gaussian cluster representation
 * @see GaussianParameters for algorithm parameters (ρ, σ_init)
 */
public final class GaussianART extends AbstractStatisticalART<GaussianParameters> {
    
    // Constant for normalization in multivariate Gaussian PDF calculation
    private static final double TWO_PI = 2.0 * Math.PI;
    
    /**
     * Create a new GaussianART network with no initial categories.
     */
    public GaussianART() {
        super();
    }

    @Override
    protected Class<GaussianParameters> getParameterClass() {
        return GaussianParameters.class;
    }
    
    /**
     * Compute statistical activation using Gaussian probability density function.
     * 
     * For a multivariate Gaussian with diagonal covariance matrix:
     * p(x | μ, Σ) = (2π)^(-k/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^(-1) (x-μ))
     * 
     * Where:
     * - x is the input vector
     * - μ is the mean vector (stored in GaussianWeight)
     * - Σ is the diagonal covariance matrix (diagonal elements in sigma array)
     * - k is the dimensionality
     * 
     * @param input the input vector
     * @param weight the category weight vector (must be GaussianWeight)
     * @param parameters the algorithm parameters
     * @return the probability density value for this category
     * @throws IllegalArgumentException if weight is not GaussianWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected double computeStatisticalActivation(Pattern input, WeightVector weight, GaussianParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(weight instanceof GaussianWeight gaussianWeight)) {
            throw new IllegalArgumentException("Weight vector must be GaussianWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        if (input.dimension() != gaussianWeight.dimension()) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match weight dimension " + gaussianWeight.dimension());
        }
        
        // Calculate (x - μ)^T Σ^(-1) (x - μ) for diagonal covariance
        double quadraticForm = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            var diff = input.get(i) - gaussianWeight.mean()[i];
            quadraticForm += (diff * diff) * gaussianWeight.invSigma()[i];
        }
        
        // Calculate multivariate Gaussian PDF
        var k = input.dimension();
        var normalization = Math.pow(TWO_PI, -k / 2.0) * (1.0 / gaussianWeight.sqrtDetSigma());
        var probability = normalization * Math.exp(-0.5 * quadraticForm);
        
        return probability;
    }
    
    /**
     * Compute statistical vigilance using Gaussian probability density.
     * 
     * For GaussianART, the vigilance test checks if the probability density
     * exceeds the vigilance threshold:
     * p(x | μ_j, Σ_j) ≥ ρ
     * 
     * This ensures that only inputs with sufficiently high probability
     * under the Gaussian model are accepted by the category.
     * 
     * @param input the input vector
     * @param weight the category weight vector (must be GaussianWeight)
     * @param parameters the algorithm parameters
     * @return MatchResult.Accepted if vigilance test passes, MatchResult.Rejected otherwise
     * @throws IllegalArgumentException if weight is not GaussianWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected MatchResult computeStatisticalVigilance(Pattern input, WeightVector weight, GaussianParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Calculate probability density (same as activation)
        var probability = computeStatisticalActivation(input, weight, parameters);
        
        // Test against vigilance parameter
        if (probability >= parameters.vigilance()) {
            return new MatchResult.Accepted(probability, parameters.vigilance());
        } else {
            return new MatchResult.Rejected(probability, parameters.vigilance());
        }
    }
    
    /**
     * Compute statistical weight update using incremental Gaussian learning.
     * 
     * This method delegates to GaussianWeight.update() which implements
     * online algorithms for updating mean and variance:
     * - Incremental mean: μ_new = μ_old + (x - μ_old) / n_new
     * - Incremental variance: updates using Welford's online algorithm
     * 
     * The learning automatically maintains statistical consistency
     * and prevents numerical instability.
     * 
     * @param input the input vector
     * @param currentWeight the current category weight (must be GaussianWeight)
     * @param parameters the algorithm parameters
     * @return the updated weight vector with new statistics
     * @throws IllegalArgumentException if weight is not GaussianWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected WeightVector computeStatisticalWeightUpdate(Pattern input, WeightVector currentWeight, GaussianParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Delegate to GaussianWeight.update() which implements incremental statistics
        return currentWeight.update(input, parameters);
    }
    
    /**
     * Create an initial weight vector for a new category based on the input.
     *
     * For GaussianART, the initial weight is a Gaussian centered at the input point
     * with initial variance taken from the GaussianParameters.sigmaInit values.
     * The sample count starts at 1.
     *
     * This ensures that each new category begins as a tight Gaussian cluster
     * around the input that caused its creation.
     *
     * @param input the input vector that will become the center of this category
     * @param parameters the algorithm parameters containing initial sigma values
     * @return the initial GaussianWeight with mean=input, sigma=sigmaInit, count=1
     * @throws NullPointerException if input or parameters are null
     */
    @Override
    protected WeightVector createStatisticalWeightVector(Pattern input, GaussianParameters parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");

        if (input.dimension() != parameters.dimension()) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() +
                " must match parameters dimension " + parameters.dimension());
        }

        // Create mean vector from input
        var mean = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            mean[i] = input.get(i);
        }

        // Use initial sigma values from parameters
        var sigma = parameters.sigmaInit().clone();

        // Create GaussianWeight with sample count = 1
        return GaussianWeight.of(mean, sigma, 1L);
    }
    
    /**
     * Get a string representation of this GaussianART network.
     * @return string showing the class name and number of categories
     */
    @Override
    public String toString() {
        return "GaussianART{categories=" + getCategoryCount() + "}";
    }

    @Override
    public void close() throws Exception {
        // No-op for vanilla implementation
    }
}