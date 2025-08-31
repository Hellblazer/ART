package com.hellblazer.art.core.parameters;

/**
 * Parameters for QuadraticNeuronART algorithm.
 * 
 * QuadraticNeuronART clusters data in hyper-ellipsoids by utilizing a quadratic
 * neural network for activation and resonance.
 *
 * @param vigilance   Vigilance parameter rho [0, 1]
 * @param sInit       Initial quadratic term s_init 
 * @param lrB         Learning rate for cluster mean (bias) lr_b (0, 1]
 * @param lrW         Learning rate for cluster weights lr_w [0, 1]
 * @param lrS         Learning rate for the quadratic term lr_s [0, 1]
 */
public record QuadraticNeuronARTParameters(
    double vigilance,
    double sInit,
    double lrB,
    double lrW,
    double lrS
) {
    
    public QuadraticNeuronARTParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (lrB <= 0.0 || lrB > 1.0) {
            throw new IllegalArgumentException("Learning rate for bias (lrB) must be in range (0, 1], got: " + lrB);
        }
        if (lrW < 0.0 || lrW > 1.0) {
            throw new IllegalArgumentException("Learning rate for weights (lrW) must be in range [0, 1], got: " + lrW);
        }
        if (lrS < 0.0 || lrS > 1.0) {
            throw new IllegalArgumentException("Learning rate for quadratic term (lrS) must be in range [0, 1], got: " + lrS);
        }
    }

    /**
     * Builder for QuadraticNeuronARTParameters.
     */
    public static class Builder {
        private double vigilance = 0.7;
        private double sInit = 0.5;
        private double lrB = 0.1;
        private double lrW = 0.1;
        private double lrS = 0.05;

        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }

        public Builder sInit(double sInit) {
            this.sInit = sInit;
            return this;
        }

        public Builder lrB(double lrB) {
            this.lrB = lrB;
            return this;
        }

        public Builder lrW(double lrW) {
            this.lrW = lrW;
            return this;
        }

        public Builder lrS(double lrS) {
            this.lrS = lrS;
            return this;
        }

        public QuadraticNeuronARTParameters build() {
            return new QuadraticNeuronARTParameters(vigilance, sInit, lrB, lrW, lrS);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}