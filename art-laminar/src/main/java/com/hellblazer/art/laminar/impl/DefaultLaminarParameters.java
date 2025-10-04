package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.*;

/**
 * Default implementation of laminar parameters.
 *
 * @author Hal Hildebrand
 */
public class DefaultLaminarParameters implements LaminarParameters {
    private final LearningParameters learningParameters;
    private final LaminarShuntingParameters shuntingParameters;
    private final ResonanceParameters resonanceParameters;
    private final double vigilance;
    private final boolean complementCoding;

    private DefaultLaminarParameters(Builder builder) {
        this.learningParameters = builder.learningParameters != null ?
            builder.learningParameters : new DefaultLearningParameters(0.5, 0.0, false, 0.0);
        this.shuntingParameters = builder.shuntingParameters != null ?
            builder.shuntingParameters : new DefaultShuntingParameters(0.1, 1.0, 0.0, 0.5);
        this.resonanceParameters = builder.resonanceParameters != null ?
            builder.resonanceParameters : new DefaultResonanceParameters(0.9, 0.01, 0.99, false);
        this.vigilance = builder.vigilance;
        this.complementCoding = builder.complementCoding;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public LearningParameters getLearningParameters() {
        return learningParameters;
    }

    @Override
    public LaminarShuntingParameters getShuntingParameters() {
        return shuntingParameters;
    }

    @Override
    public ResonanceParameters getResonanceParameters() {
        return resonanceParameters;
    }

    @Override
    public LayerParameters getLayerParameters(String layerId) {
        return new DefaultLayerParameters(0.1, 1.0, 0.0, 0.5, 0.1);
    }

    @Override
    public PathwayParameters getPathwayParameters(String pathwayId) {
        return new DefaultPathwayParameters(1.0, 0.5, true);
    }

    @Override
    public double getVigilance() {
        return vigilance;
    }

    @Override
    public boolean isComplementCoding() {
        return complementCoding;
    }

    public static class Builder {
        private LearningParameters learningParameters;
        private LaminarShuntingParameters shuntingParameters;
        private ResonanceParameters resonanceParameters;
        private double vigilance = 0.9;
        private boolean complementCoding = false;

        public Builder withLearningParameters(LearningParameters params) {
            this.learningParameters = params;
            return this;
        }

        public Builder withShuntingParameters(LaminarShuntingParameters params) {
            this.shuntingParameters = params;
            return this;
        }

        public Builder withResonanceParameters(ResonanceParameters params) {
            this.resonanceParameters = params;
            return this;
        }

        public Builder withVigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }

        public Builder withComplementCoding(boolean complementCoding) {
            this.complementCoding = complementCoding;
            return this;
        }

        public DefaultLaminarParameters build() {
            return new DefaultLaminarParameters(this);
        }
    }
}