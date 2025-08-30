package com.hellblazer.art.core;

/**
 * Enum representing different match tracking methods for ARTMAP algorithms.
 * Match tracking is a mechanism for adjusting the vigilance parameter when 
 * a prediction mismatch occurs during supervised learning.
 * 
 * <p>These methods provide different strategies for handling conflicts between
 * the ART module's categorization and the desired output classification.</p>
 */
public enum MatchTrackingMethod {
    
    /**
     * MT+ (Positive Match Tracking) - Traditional ARTMAP approach.
     * Increases vigilance by a small epsilon amount when mismatch occurs.
     * This forces the system to search for a better matching category.
     * 
     * <p>Formula: ρ_new = min(ρ_old + ε, 1.0)</p>
     * <p>Best for: Standard supervised learning with stable categories</p>
     */
    MT_PLUS("MT+", "Positive Match Tracking"),
    
    /**
     * MT- (Negative Match Tracking) - Decreases vigilance.
     * Reduces vigilance to allow more general categories when appropriate.
     * Useful for noisy data where exact matches are rare.
     * 
     * <p>Formula: ρ_new = max(ρ_old - ε, ρ_min)</p>
     * <p>Best for: Noisy data, promoting generalization</p>
     */
    MT_MINUS("MT-", "Negative Match Tracking"),
    
    /**
     * MT0 (Zero Match Tracking) - Sets vigilance to exact match value.
     * Directly sets vigilance to the match value plus epsilon.
     * Provides precise control over category boundaries.
     * 
     * <p>Formula: ρ_new = match_value + ε</p>
     * <p>Best for: Fine-grained category control</p>
     */
    MT_ZERO("MT0", "Zero Match Tracking"),
    
    /**
     * MT~ (Approximate Match Tracking) - Interpolated adjustment.
     * Uses weighted interpolation between current and target vigilance.
     * Provides smooth transitions and avoids abrupt changes.
     * 
     * <p>Formula: ρ_new = ρ_old + α(ρ_target - ρ_old), where α ∈ [0,1]</p>
     * <p>Best for: Gradual adaptation, avoiding oscillations</p>
     */
    MT_APPROXIMATE("MT~", "Approximate Match Tracking"),
    
    /**
     * MT1 (Unity Match Tracking) - Sets vigilance to maximum.
     * Immediately sets vigilance to 1.0, effectively disabling the category.
     * Used to permanently exclude a category from future matches.
     * 
     * <p>Formula: ρ_new = 1.0</p>
     * <p>Best for: Permanent category exclusion, outlier handling</p>
     */
    MT_ONE("MT1", "Unity Match Tracking");
    
    private final String symbol;
    private final String description;
    
    MatchTrackingMethod(String symbol, String description) {
        this.symbol = symbol;
        this.description = description;
    }
    
    /**
     * Get the mathematical symbol for this match tracking method.
     * @return the symbol (e.g., "MT+", "MT-")
     */
    public String getSymbol() {
        return symbol;
    }
    
    /**
     * Get a human-readable description of this method.
     * @return the description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Apply this match tracking method to adjust vigilance.
     * 
     * @param currentVigilance the current vigilance value
     * @param matchValue the match value that triggered tracking
     * @param epsilon the adjustment increment
     * @param minVigilance minimum allowed vigilance
     * @param interpolationFactor factor for MT_APPROXIMATE (ignored for other methods)
     * @return the new vigilance value
     */
    public double adjustVigilance(double currentVigilance, 
                                  double matchValue, 
                                  double epsilon,
                                  double minVigilance,
                                  double interpolationFactor) {
        return switch (this) {
            case MT_PLUS -> Math.min(1.0, currentVigilance + epsilon);
            case MT_MINUS -> Math.max(minVigilance, currentVigilance - epsilon);
            case MT_ZERO -> Math.min(1.0, matchValue + epsilon);
            case MT_APPROXIMATE -> {
                double target = matchValue + epsilon;
                yield currentVigilance + interpolationFactor * (target - currentVigilance);
            }
            case MT_ONE -> 1.0;
        };
    }
    
    /**
     * Check if this method increases vigilance.
     * @return true if this method typically increases vigilance
     */
    public boolean increasesVigilance() {
        return this == MT_PLUS || this == MT_ONE;
    }
    
    /**
     * Check if this method decreases vigilance.
     * @return true if this method typically decreases vigilance
     */
    public boolean decreasesVigilance() {
        return this == MT_MINUS;
    }
    
    /**
     * Check if this method uses interpolation.
     * @return true if this method uses interpolation
     */
    public boolean usesInterpolation() {
        return this == MT_APPROXIMATE;
    }
    
    /**
     * Get the default epsilon value for this method.
     * @return suggested epsilon value
     */
    public double getDefaultEpsilon() {
        return switch (this) {
            case MT_PLUS, MT_MINUS -> 0.001;
            case MT_ZERO -> 0.0001;  // Smaller for direct setting
            case MT_APPROXIMATE -> 0.0;  // Not used directly
            case MT_ONE -> 0.0;  // Not used
        };
    }
    
    /**
     * Get the default interpolation factor for MT_APPROXIMATE.
     * @return default interpolation factor (0.5 for 50% blend)
     */
    public double getDefaultInterpolationFactor() {
        return 0.5;
    }
    
    @Override
    public String toString() {
        return String.format("%s (%s)", symbol, description);
    }
}