package com.hellblazer.art.laminar.builders;

import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.impl.*;
import com.hellblazer.art.laminar.parameters.LaminarParameters;
import com.hellblazer.art.laminar.events.CircuitEventListener;

/**
 * Builder for creating laminar circuits.
 *
 * @author Hal Hildebrand
 */
public class LaminarCircuitBuilder<P extends LaminarParameters> {
    private LaminarCircuitImpl<P> circuit;
    private P parameters;

    public LaminarCircuitBuilder() {
        // Will be initialized when parameters are set
    }

    public LaminarCircuitBuilder<P> withParameters(P parameters) {
        this.parameters = parameters;
        this.circuit = new LaminarCircuitImpl<>(parameters);
        return this;
    }

    public LaminarCircuitBuilder<P> withInputLayer(int size, boolean complementCoding) {
        if (circuit == null) {
            throw new IllegalStateException("Parameters must be set first");
        }
        var layer = new InputLayer("input", size, complementCoding);
        circuit.addLayer(layer, 0);
        return this;
    }

    public LaminarCircuitBuilder<P> withFeatureLayer(int size) {
        if (circuit == null) {
            throw new IllegalStateException("Parameters must be set first");
        }
        var layer = new FeatureLayer("feature", size);
        circuit.addLayer(layer, 1);
        return this;
    }

    public LaminarCircuitBuilder<P> withCategoryLayer(int maxCategories) {
        if (circuit == null) {
            throw new IllegalStateException("Parameters must be set first");
        }
        var layer = new CategoryLayer("category", maxCategories);
        circuit.addLayer(layer, 2);
        return this;
    }

    public LaminarCircuitBuilder<P> withStandardConnections() {
        if (circuit == null) {
            throw new IllegalStateException("Parameters must be set first");
        }
        // Add standard pathways
        circuit.connectLayers(new BottomUpPathway("bu-input-feature", "input", "feature"));
        circuit.connectLayers(new BottomUpPathway("bu-feature-category", "feature", "category"));
        circuit.connectLayers(new TopDownPathway("td-category-feature", "category", "feature"));
        circuit.connectLayers(new LateralPathway("lat-feature", "feature", "feature"));
        return this;
    }

    public LaminarCircuitBuilder<P> withVigilance(double vigilance) {
        if (circuit != null) {
            circuit.getResonanceController().setVigilance(vigilance);
        }
        return this;
    }

    public LaminarCircuitBuilder<P> withListener(CircuitEventListener listener) {
        if (circuit != null) {
            circuit.addListener(listener);
        }
        return this;
    }

    public LaminarCircuit<P> build() {
        if (circuit == null) {
            throw new IllegalStateException("Parameters must be set before building");
        }
        return circuit;
    }

    // Concrete layer implementations
    static class InputLayer extends AbstractLayer {
        private final boolean complementCoding;

        public InputLayer(String id, int size, boolean complementCoding) {
            super(id, complementCoding ? size * 2 : size, LayerType.INPUT);
            this.complementCoding = complementCoding;
        }
    }

    static class FeatureLayer extends AbstractLayer {
        public FeatureLayer(String id, int size) {
            super(id, size, LayerType.FEATURE);
        }
    }

    static class CategoryLayer extends AbstractLayer {
        public CategoryLayer(String id, int size) {
            super(id, size, LayerType.CATEGORY);
        }
    }

    // Concrete pathway implementations
    static class BottomUpPathway extends AbstractPathway {
        public BottomUpPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.BOTTOM_UP);
        }
    }

    static class TopDownPathway extends AbstractPathway {
        public TopDownPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.TOP_DOWN);
        }
    }

    static class LateralPathway extends AbstractPathway {
        public LateralPathway(String id, String sourceId, String targetId) {
            super(id, sourceId, targetId, PathwayType.LATERAL);
        }
    }
}