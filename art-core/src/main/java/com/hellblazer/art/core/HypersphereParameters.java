package com.hellblazer.art.core;

/**
 * Immutable parameters for HypersphereART algorithm.
 * 
 * @param vigilance the vigilance parameter (ρ) in range [0, 1]
 * @param defaultRadius the default radius for new hyperspheres (must be positive)
 * @param adaptiveRadius whether to use adaptive radius adjustment
 */
public record HypersphereParameters(double vigilance, double defaultRadius, boolean adaptiveRadius) {
    
    /**
     * Constructor with validation.
     */
    public HypersphereParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (defaultRadius < 0.0) {
            throw new IllegalArgumentException("Default radius must be non-negative, got: " + defaultRadius);
        }
    }
    
    /**
     * Create HypersphereParameters with specified values.
     * @param vigilance the vigilance parameter ρ ∈ [0, 1]
     * @param defaultRadius the default radius for new hyperspheres (> 0)
     * @param adaptiveRadius whether to use adaptive radius adjustment
     * @return new HypersphereParameters instance
     */
    public static HypersphereParameters of(double vigilance, double defaultRadius, boolean adaptiveRadius) {
        return new HypersphereParameters(vigilance, defaultRadius, adaptiveRadius);
    }
    
    /**
     * Create HypersphereParameters with default values.
     * Default: vigilance=0.5, defaultRadius=1.0, adaptiveRadius=false
     * @return default HypersphereParameters
     */
    public static HypersphereParameters defaults() {
        return new HypersphereParameters(0.5, 1.0, false);
    }
    
    /**
     * Create a new HypersphereParameters with different vigilance value.
     * @param newVigilance the new vigilance value
     * @return new HypersphereParameters instance
     */
    public HypersphereParameters withVigilance(double newVigilance) {
        return new HypersphereParameters(newVigilance, defaultRadius, adaptiveRadius);
    }
    
    /**
     * Create a new HypersphereParameters with different default radius.
     * @param newDefaultRadius the new default radius value
     * @return new HypersphereParameters instance
     */
    public HypersphereParameters withDefaultRadius(double newDefaultRadius) {
        return new HypersphereParameters(vigilance, newDefaultRadius, adaptiveRadius);
    }
    
    /**
     * Create a new HypersphereParameters with different adaptive radius setting.
     * @param newAdaptiveRadius the new adaptive radius setting
     * @return new HypersphereParameters instance
     */
    public HypersphereParameters withAdaptiveRadius(boolean newAdaptiveRadius) {
        return new HypersphereParameters(vigilance, defaultRadius, newAdaptiveRadius);
    }
    
    @Override
    public String toString() {
        return String.format("HypersphereParameters{ρ=%.3f, r_default=%.3f, adaptive=%s}", 
                           vigilance, defaultRadius, adaptiveRadius);
    }
}