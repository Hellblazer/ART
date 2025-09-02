package com.hellblazer.art.core.parameters;

/**
 * Parameters for ART1 binary pattern recognition algorithm.
 * 
 * ART1 is designed exclusively for clustering binary data using:
 * - Vigilance parameter (rho): Controls cluster selectivity
 * - Uncommitted node bias (L): Affects new cluster creation
 */
public record ART1Parameters(
    double vigilance,     // Vigilance parameter [0, 1] 
    double L             // Uncommitted node bias >= 1.0
) {
    
    public ART1Parameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (L < 1.0) {
            throw new IllegalArgumentException("L must be >= 1.0, got: " + L);
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.7;
        private double L = 2.0;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder L(double L) {
            this.L = L;
            return this;
        }
        
        public ART1Parameters build() {
            return new ART1Parameters(vigilance, L);
        }
    }
}
