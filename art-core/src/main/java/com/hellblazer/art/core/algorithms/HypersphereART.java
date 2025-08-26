package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.parameters.HypersphereParameters;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.HypersphereWeight;
import java.util.Objects;

/**
 * HypersphereART implementation using the BaseART template framework.
 * 
 * HypersphereART is a neural network architecture based on Adaptive Resonance Theory (ART)
 * that performs unsupervised learning using hyperspheres to model categories.
 * Each category is represented as a hypersphere with a center point and radius.
 * 
 * Key Features:
 * - Distance-based activation using Euclidean distance
 * - Vigilance test based on hypersphere inclusion
 * - Learning by expanding hypersphere radius when necessary
 * - Geometric interpretation: categories as regions in feature space
 * 
 * Mathematical Foundation:
 * - Activation: A_j = 1 / (1 + d(x, c_j)) where d is Euclidean distance
 * - Vigilance: d(x, c_j) ≤ r_j (point within hypersphere)
 * - Learning: r_j = max(r_j, d(x, c_j)) (expand radius if needed)
 * 
 * @see BaseART for the template method framework
 * @see HypersphereWeight for hypersphere representation
 * @see HypersphereParameters for algorithm parameters (ρ, defaultRadius, adaptiveRadius)
 */
public final class HypersphereART extends BaseART {
    
    /**
     * Create a new HypersphereART network with no initial categories.
     */
    public HypersphereART() {
        super();
    }
    
    /**
     * Calculate the activation value using distance-based function.
     * 
     * The activation is computed as:
     * A_j = 1 / (1 + d(x, c_j))
     * 
     * Where:
     * - x is the input vector
     * - c_j is the center of hypersphere j
     * - d(x, c_j) is the Euclidean distance between x and c_j
     * 
     * This gives higher activation for points closer to the hypersphere center,
     * with activation = 1.0 for perfect matches and approaching 0 for distant points.
     * 
     * @param input the input vector
     * @param weight the category weight vector (must be HypersphereWeight)
     * @param parameters the algorithm parameters (must be HypersphereParameters)
     * @return the activation value for this category (in range (0, 1])
     * @throws IllegalArgumentException if parameters are not HypersphereParameters or weight is not HypersphereWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof HypersphereParameters hypersphereParams)) {
            throw new IllegalArgumentException("Parameters must be HypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof HypersphereWeight hypersphereWeight)) {
            throw new IllegalArgumentException("Weight vector must be HypersphereWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        if (input.dimension() != hypersphereWeight.dimension()) {
            throw new IllegalArgumentException("Input dimension " + input.dimension() + 
                " must match weight dimension " + hypersphereWeight.dimension());
        }
        
        // Calculate Euclidean distance from input to hypersphere center
        var distance = calculateEuclideanDistance(input, hypersphereWeight);
        
        // Activation: 1 / (1 + distance)
        // This gives activation = 1.0 for distance = 0 (perfect match)
        // and approaches 0 as distance increases
        return 1.0 / (1.0 + distance);
    }
    
    /**
     * Test whether the input falls within the hypersphere according to vigilance.
     * 
     * For HypersphereART, the vigilance test checks if the point is within
     * the hypersphere radius. We normalize the distance by radius to create
     * a match ratio in [0,1] range:
     * match_ratio = max(0, 1 - d(x, c_j)/r_j)
     * 
     * Accept if match_ratio >= vigilance parameter.
     * 
     * @param input the input vector
     * @param weight the category weight vector (must be HypersphereWeight)
     * @param parameters the algorithm parameters (must be HypersphereParameters)
     * @return MatchResult.Accepted if point satisfies vigilance, MatchResult.Rejected otherwise
     * @throws IllegalArgumentException if parameters are not HypersphereParameters or weight is not HypersphereWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof HypersphereParameters hypersphereParams)) {
            throw new IllegalArgumentException("Parameters must be HypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof HypersphereWeight hypersphereWeight)) {
            throw new IllegalArgumentException("Weight vector must be HypersphereWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        // Calculate distance from input to hypersphere center
        var distance = calculateEuclideanDistance(input, hypersphereWeight);
        var radius = hypersphereWeight.radius();
        
        // Calculate match ratio: closer points have higher match values
        // For zero radius, use distance-based acceptance with vigilance as threshold
        // For non-zero radius, use standard ratio: max(0, 1 - distance/radius)
        double matchRatio;
        if (radius == 0.0) {
            // For zero radius, use vigilance as expansion threshold
            // Accept points within reasonable distance based on vigilance 
            // Lower vigilance = more restrictive, higher vigilance = more permissive
            var maxAcceptableDistance = 1.0 - hypersphereParams.vigilance(); // vigilance 0.5 -> max distance 0.5
            matchRatio = (distance <= maxAcceptableDistance) ? 1.0 : 0.0;
        } else {
            matchRatio = Math.max(0.0, 1.0 - distance / radius);
        }
        
        // Test against vigilance parameter
        if (matchRatio >= hypersphereParams.vigilance()) {
            return new MatchResult.Accepted(matchRatio, hypersphereParams.vigilance());
        } else {
            return new MatchResult.Rejected(matchRatio, hypersphereParams.vigilance());
        }
    }
    
    /**
     * Update the hypersphere by expanding its radius if necessary.
     * 
     * HypersphereART learning rule:
     * - If the input point is already within the hypersphere (d ≤ r), no change
     * - If the input point is outside the hypersphere (d > r), expand radius to r = d
     * 
     * This ensures the hypersphere grows to include all points assigned to it
     * while maintaining the original center.
     * 
     * This method delegates to HypersphereWeight.update() which implements
     * the radius expansion logic.
     * 
     * @param input the input vector
     * @param currentWeight the current category weight (must be HypersphereWeight)
     * @param parameters the algorithm parameters (must be HypersphereParameters)
     * @return the updated weight vector with potentially expanded radius
     * @throws IllegalArgumentException if parameters are not HypersphereParameters or weight is not HypersphereWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof HypersphereParameters)) {
            throw new IllegalArgumentException("Parameters must be HypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(currentWeight instanceof HypersphereWeight)) {
            throw new IllegalArgumentException("Weight vector must be HypersphereWeight, got: " + 
                currentWeight.getClass().getSimpleName());
        }
        
        // Delegate to HypersphereWeight.update() which implements radius expansion
        return currentWeight.update(input, parameters);
    }
    
    /**
     * Create an initial hypersphere for a new category centered at the input point.
     * 
     * For HypersphereART, the initial weight is a hypersphere centered at the input
     * with radius determined by the HypersphereParameters:
     * - Center = input vector
     * - Radius = defaultRadius from parameters
     * 
     * This ensures that each new category begins as a hypersphere of standard size
     * around the input that caused its creation.
     * 
     * @param input the input vector that will become the center of this hypersphere
     * @param parameters the algorithm parameters containing default radius
     * @return the initial HypersphereWeight with center=input, radius=defaultRadius
     * @throws IllegalArgumentException if parameters are not HypersphereParameters
     * @throws NullPointerException if input or parameters are null
     */
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof HypersphereParameters hypersphereParams)) {
            throw new IllegalArgumentException("Parameters must be HypersphereParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        // Create center vector from input
        var center = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            center[i] = input.get(i);
        }
        
        // Use default radius from parameters
        return HypersphereWeight.of(center, hypersphereParams.defaultRadius());
    }
    
    /**
     * Calculate the Euclidean distance between an input vector and hypersphere center.
     * 
     * d(x, c) = √(Σ(x_i - c_i)²)
     * 
     * @param input the input vector
     * @param hypersphereWeight the hypersphere weight containing the center
     * @return the Euclidean distance
     */
    private double calculateEuclideanDistance(Pattern input, HypersphereWeight hypersphereWeight) {
        double sumSquares = 0.0;
        for (int i = 0; i < input.dimension(); i++) {
            var diff = input.get(i) - hypersphereWeight.center()[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Get a string representation of this HypersphereART network.
     * @return string showing the class name and number of categories
     */
    @Override
    public String toString() {
        return "HypersphereART{categories=" + getCategoryCount() + "}";
    }
}