package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Default implementation of laminar circuit parameters.
 *
 * @author Hal Hildebrand
 */
public class DefaultLaminarParameters implements ILaminarParameters {
    private static final long serialVersionUID = 1L;

    private final Map<String, ILayerParameters> layerParameters = new HashMap<>();
    private final Map<String, IPathwayParameters> pathwayParameters = new HashMap<>();
    private final IResonanceParameters resonanceParameters;
    private final IShuntingParameters shuntingParameters;
    private final ILearningParameters learningParameters;

    private DefaultLaminarParameters(Builder builder) {
        this.layerParameters.putAll(builder.layerParameters);
        this.pathwayParameters.putAll(builder.pathwayParameters);
        this.resonanceParameters = builder.resonanceParameters;
        this.shuntingParameters = builder.shuntingParameters;
        this.learningParameters = builder.learningParameters;
    }

    @Override
    public ILayerParameters getLayerParameters(String layerId) {
        return layerParameters.getOrDefault(layerId, DefaultLayerParameters.DEFAULT);
    }

    @Override
    public IPathwayParameters getPathwayParameters(String pathwayId) {
        return pathwayParameters.getOrDefault(pathwayId, DefaultPathwayParameters.DEFAULT);
    }

    @Override
    public IResonanceParameters getResonanceParameters() {
        return resonanceParameters;
    }

    @Override
    public IShuntingParameters getShuntingParameters() {
        return shuntingParameters;
    }

    @Override
    public ILearningParameters getLearningParameters() {
        return learningParameters;
    }

    @Override
    public boolean validate() {
        return resonanceParameters != null &&
               shuntingParameters != null &&
               learningParameters != null &&
               resonanceParameters.getVigilance() >= 0.0 &&
               resonanceParameters.getVigilance() <= 1.0 &&
               shuntingParameters.getDecayRate() > 0.0 &&
               learningParameters.getLearningRate() > 0.0;
    }

    @Override
    public ILaminarParameters copy() {
        return new Builder()
                .withResonanceParameters(resonanceParameters)
                .withShuntingParameters(shuntingParameters)
                .withLearningParameters(learningParameters)
                .copyLayerParameters(layerParameters)
                .copyPathwayParameters(pathwayParameters)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final Map<String, ILayerParameters> layerParameters = new HashMap<>();
        private final Map<String, IPathwayParameters> pathwayParameters = new HashMap<>();
        private IResonanceParameters resonanceParameters = DefaultResonanceParameters.DEFAULT;
        private IShuntingParameters shuntingParameters = DefaultShuntingParameters.DEFAULT;
        private ILearningParameters learningParameters = DefaultLearningParameters.DEFAULT;

        public Builder withLayerParameters(String layerId, ILayerParameters params) {
            layerParameters.put(layerId, params);
            return this;
        }

        public Builder withPathwayParameters(String pathwayId, IPathwayParameters params) {
            pathwayParameters.put(pathwayId, params);
            return this;
        }

        public Builder withResonanceParameters(IResonanceParameters params) {
            this.resonanceParameters = params;
            return this;
        }

        public Builder withShuntingParameters(IShuntingParameters params) {
            this.shuntingParameters = params;
            return this;
        }

        public Builder withLearningParameters(ILearningParameters params) {
            this.learningParameters = params;
            return this;
        }

        public Builder copyLayerParameters(Map<String, ILayerParameters> params) {
            layerParameters.putAll(params);
            return this;
        }

        public Builder copyPathwayParameters(Map<String, IPathwayParameters> params) {
            pathwayParameters.putAll(params);
            return this;
        }

        public DefaultLaminarParameters build() {
            return new DefaultLaminarParameters(this);
        }
    }
}