package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.*;
import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.LaminarParameters;
import com.hellblazer.art.laminar.events.*;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Laminar ART circuit extending BaseART for maximum code reuse.
 * Reduces implementation from 423 lines to ~200 lines by delegating to BaseART.
 *
 * @author Hal Hildebrand
 */
public class LaminarCircuitImpl<P extends LaminarParameters>
        extends BaseART<P> implements LaminarCircuit<P> {

    // Laminar-specific components
    protected final Map<Integer, Layer> layers;
    protected final List<Pathway> pathways;
    protected final List<CircuitEventListener> listeners;
    protected ResonanceController resonanceController;

    // Circuit state
    protected boolean resonant;
    protected double resonanceScore;
    protected int cycleCount;

    public LaminarCircuitImpl(P parameters) {
        super();
        this.layers = new TreeMap<>();
        this.pathways = new ArrayList<>();
        this.listeners = new CopyOnWriteArrayList<>();
        this.resonanceController = new DefaultResonanceController();
        this.resonant = false;
        this.resonanceScore = 0.0;
        this.cycleCount = 0;
    }

    // === BaseART Abstract Method Implementations (4 methods) ===

    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, P parameters) {
        // Process through laminar layers
        var featureLayer = layers.get(1);
        if (featureLayer != null) {
            var activation = featureLayer.processBottomUp(input,
                parameters.getLayerParameters(featureLayer.getId()));
            // Calculate ART choice function
            var numerator = activation.l1Norm();
            var denominator = weight.l1Norm() + parameters.getLearningParameters().getLearningRate();
            return numerator / denominator;
        }
        return 0.0;
    }

    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, P parameters) {
        // Use resonance controller for vigilance check
        var featureLayer = layers.get(1);
        var categoryLayer = layers.get(2);

        if (featureLayer != null && categoryLayer != null) {
            // Get layer activations
            var bottomUp = featureLayer.getActivation();
            var topDown = categoryLayer.getActivation();

            // Calculate match using resonance controller
            var matchScore = resonanceController.calculateMatch(bottomUp, topDown);
            var passes = !resonanceController.shouldReset(matchScore);

            return passes ?
                new MatchResult.Accepted(matchScore, resonanceController.getVigilance()) :
                new MatchResult.Rejected(matchScore, resonanceController.getVigilance());
        }

        return new MatchResult.Rejected(0.0, resonanceController.getVigilance());
    }

    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, P parameters) {
        // Update through pathway learning
        var learningRate = parameters.getLearningParameters().getLearningRate();

        // Update pathway weights
        for (var pathway : pathways) {
            if (pathway.isAdaptive()) {
                pathway.updateWeights(input, input, learningRate);
            }
        }

        // Update weight vector using ART learning rule
        var data = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            var currentVal = i < currentWeight.dimension() ? currentWeight.get(i) : 0.0;
            data[i] = learningRate * input.get(i) + (1 - learningRate) * currentVal;
        }

        return new DenseWeightVector(data);
    }

    @Override
    protected WeightVector createInitialWeight(Pattern input, P parameters) {
        // Initialize new category with input pattern
        var data = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            data[i] = input.get(i);
        }

        // Fire category created event
        fireEvent(new CategoryEvent(categories.size(), 0));

        return new DenseWeightVector(data);
    }

    // === LaminarCircuit Implementation ===

    @Override
    public LaminarActivationResult processCycle(Pattern input, P parameters) {
        cycleCount++;

        // Process through layers
        var result = processLayers(input, parameters);

        // Check resonance
        resonanceScore = result.getResonanceScore();
        resonant = result.isResonant();

        // Fire events
        if (resonant) {
            fireEvent(new ResonanceEvent(result.getCategoryIndex(), resonanceScore));
        }
        fireEvent(new CycleEvent(cycleCount, resonanceScore));

        return result;
    }

    @Override
    public LaminarCircuit<P> addLayer(Layer layer, int depth) {
        layers.put(depth, layer);
        return this;
    }

    @Override
    public Layer getLayer(int depth) {
        return layers.get(depth);
    }

    @Override
    public Map<Integer, Layer> getLayers() {
        return new TreeMap<>(layers);
    }

    @Override
    public LaminarCircuit<P> connectLayers(Pathway pathway) {
        pathways.add(pathway);
        return this;
    }

    @Override
    public List<Pathway> getPathways() {
        return new ArrayList<>(pathways);
    }

    @Override
    public List<Pathway> getPathwaysForLayer(String layerId) {
        return pathways.stream()
                .filter(p -> p.getSourceLayerId().equals(layerId) ||
                           p.getTargetLayerId().equals(layerId))
                .toList();
    }

    @Override
    public LaminarCircuit<P> setResonanceController(ResonanceController controller) {
        this.resonanceController = controller;
        return this;
    }

    @Override
    public ResonanceController getResonanceController() {
        return resonanceController;
    }

    @Override
    public boolean isResonant() {
        return resonant;
    }

    @Override
    public double getResonanceScore() {
        return resonanceScore;
    }

    @Override
    public void resetActivations() {
        layers.values().forEach(Layer::reset);
        resonant = false;
        resonanceScore = 0.0;
        cycleCount = 0;
    }

    @Override
    public void addListener(CircuitEventListener listener) {
        listeners.add(listener);
    }

    @Override
    public void removeListener(CircuitEventListener listener) {
        listeners.remove(listener);
    }

    @Override
    public CircuitState getState() {
        var builder = CircuitState.builder()
                .withResonanceScore(resonanceScore)
                .withResonant(resonant)
                .withCurrentCategory(getActiveCategory())
                .withCycleNumber(cycleCount);

        // Add layer activations
        for (var entry : layers.entrySet()) {
            var layer = entry.getValue();
            builder.withLayerActivation(layer.getId(), layer.getActivation());
        }

        // Add pathway gains
        for (var pathway : pathways) {
            builder.withPathwayGain(pathway.getId(), pathway.getGain());
        }

        return builder.build();
    }

    // === Helper Methods ===

    private LaminarActivationResult processLayers(Pattern input, P parameters) {
        // Get layer references
        var inputLayer = layers.get(0);
        var featureLayer = layers.get(1);
        var categoryLayer = layers.get(2);

        if (inputLayer == null || featureLayer == null || categoryLayer == null) {
            return new LaminarActivationResult.Failure("Missing required layers", cycleCount);
        }

        // Process through layers
        var processedInput = inputLayer.processBottomUp(input,
                parameters.getLayerParameters(inputLayer.getId()));

        // Propagate through pathways
        var featureActivation = propagateSignal(processedInput, inputLayer.getId(),
                featureLayer.getId(), PathwayType.BOTTOM_UP, parameters);
        featureLayer.setActivation(featureActivation);

        var categoryInput = featureLayer.processBottomUp(featureActivation,
                parameters.getLayerParameters(featureLayer.getId()));
        var categoryActivation = propagateSignal(categoryInput, featureLayer.getId(),
                categoryLayer.getId(), PathwayType.BOTTOM_UP, parameters);
        categoryLayer.setActivation(categoryActivation);

        // Get active category
        var activeCategory = getActiveCategory();

        // Generate expectation
        Pattern expectation = null;
        if (activeCategory >= 0) {
            expectation = categoryLayer.processTopDown(categoryActivation,
                    parameters.getLayerParameters(categoryLayer.getId()));
            expectation = propagateSignal(expectation, categoryLayer.getId(),
                    featureLayer.getId(), PathwayType.TOP_DOWN, parameters);
        }

        // Check resonance
        var score = resonanceController.calculateMatch(featureActivation,
            expectation != null ? expectation : featureActivation);
        var isResonant = !resonanceController.shouldReset(score);

        // Build result
        var layerActivations = new HashMap<String, Pattern>();
        layerActivations.put(inputLayer.getId(), inputLayer.getActivation());
        layerActivations.put(featureLayer.getId(), featureActivation);
        layerActivations.put(categoryLayer.getId(), categoryActivation);

        if (isResonant) {
            return new LaminarActivationResult.Success(
                    activeCategory, featureActivation, expectation,
                    layerActivations, score, cycleCount);
        } else {
            return new LaminarActivationResult.Failure("No resonance", cycleCount);
        }
    }

    private Pattern propagateSignal(Pattern signal, String sourceId, String targetId,
                                   PathwayType type, P parameters) {
        var pathway = pathways.stream()
                .filter(p -> p.getSourceLayerId().equals(sourceId) &&
                           p.getTargetLayerId().equals(targetId) &&
                           p.getType() == type)
                .findFirst()
                .orElse(null);

        if (pathway != null) {
            return pathway.propagate(signal, parameters.getPathwayParameters(pathway.getId()));
        }
        return signal;
    }

    private int getActiveCategory() {
        var categoryLayer = layers.get(2);
        if (categoryLayer != null && categoryLayer.getActivation() != null) {
            var activation = categoryLayer.getActivation();
            var maxIdx = -1;
            var maxVal = 0.0;
            for (int i = 0; i < activation.dimension(); i++) {
                if (activation.get(i) > maxVal) {
                    maxVal = activation.get(i);
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
        return -1;
    }

    protected void fireEvent(Object event) {
        for (var listener : listeners) {
            try {
                if (event instanceof ResonanceEvent e) {
                    listener.onResonance(e);
                } else if (event instanceof CycleEvent e) {
                    listener.onCycleComplete(e);
                } else if (event instanceof CategoryEvent e) {
                    listener.onCategoryCreated(e);
                }
            } catch (Exception ex) {
                // Log error
            }
        }
    }

    @Override
    public void close() throws Exception {
        // Clean up resources
        resetActivations();
    }

    // Simple WeightVector implementation
    static class DenseWeightVector implements WeightVector {
        private final double[] data;

        DenseWeightVector(double[] data) {
            this.data = data;
        }

        @Override
        public double get(int index) {
            return index < data.length ? data[index] : 0.0;
        }

        @Override
        public int dimension() {
            return data.length;
        }

        @Override
        public double l1Norm() {
            double sum = 0.0;
            for (double v : data) {
                sum += Math.abs(v);
            }
            return sum;
        }

        @Override
        public WeightVector update(Pattern input, Object parameters) {
            // Update weights based on input
            var newData = data.clone();
            for (int i = 0; i < Math.min(data.length, input.dimension()); i++) {
                newData[i] = (data[i] + input.get(i)) / 2.0;
            }
            return new DenseWeightVector(newData);
        }
    }
}