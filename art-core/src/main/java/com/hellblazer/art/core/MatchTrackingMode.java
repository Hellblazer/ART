package com.hellblazer.art.core;

/**
 * Match tracking modes for ART algorithms, based on the reference parity implementation.
 * 
 * Match tracking provides different strategies for handling vigilance failures and
 * parameter adjustments during the search for suitable categories.
 */
public enum MatchTrackingMode {
    
    /**
     * MT+ (Match Tracking Plus): Standard match tracking with positive parameter adjustment.
     * When vigilance fails but match reset succeeds, parameters are adjusted upward.
     */
    MT_PLUS("MT+"),
    
    /**
     * MT- (Match Tracking Minus): Match tracking with negative parameter adjustment.
     * When vigilance fails but match reset succeeds, parameters are adjusted downward.
     */
    MT_MINUS("MT-"),
    
    /**
     * MT0 (Match Tracking Zero): Match tracking with no parameter adjustment.
     * Parameters remain unchanged during match tracking.
     */
    MT_ZERO("MT0"),
    
    /**
     * MT1 (Match Tracking One): Match tracking with parameter set to 1.
     * When vigilance fails but match reset succeeds, relevant parameters are set to 1.
     */
    MT_ONE("MT1"),
    
    /**
     * MT~ (Match Tracking Complement): Special match tracking mode.
     * Uses complement-based parameter adjustment strategies.
     */
    MT_COMPLEMENT("MT~");
    
    private final String name;
    
    MatchTrackingMode(String name) {
        this.name = name;
    }
    
    /**
     * Get the Reference-compatible name for this match tracking mode.
     */
    public String getName() {
        return name;
    }
    
    /**
     * Parse a match tracking mode from its Python name.
     */
    public static MatchTrackingMode fromName(String name) {
        for (MatchTrackingMode mode : values()) {
            if (mode.name.equals(name)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("Unknown match tracking mode: " + name);
    }
    
    /**
     * Get the operator used for match tracking comparisons.
     */
    public java.util.function.BinaryOperator<Double> getOperator() {
        return switch (this) {
            case MT_PLUS -> Double::sum;
            case MT_MINUS -> (a, b) -> a - b;
            case MT_ZERO -> (a, b) -> a;  // No change
            case MT_ONE -> (a, b) -> 1.0;
            case MT_COMPLEMENT -> (a, b) -> 1.0 - a;
        };
    }
}