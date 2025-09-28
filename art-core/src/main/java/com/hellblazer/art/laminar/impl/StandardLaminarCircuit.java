package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.weights.FuzzyWeight;
import com.hellblazer.art.core.results.LaminarActivationResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.laminar.core.*;
import com.hellblazer.art.laminar.parameters.ILaminarParameters;
import com.hellblazer.art.laminar.events.*;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Standard implementation of laminar ART circuit.
 * Orchestrates layers, pathways, and resonance dynamics.
 *
 * @author Hal Hildebrand
 */
public class StandardLaminarCircuit<P extends ILaminarParameters>
        implements ILaminarCircuit<P> {

    private final Map<Integer, ILayer> layers;
    private final List<IPathway> pathways;
    private final List<ICircuitEventListener> listeners;
    private IResonanceController resonanceController;
    private P parameters;

    // Circuit state
    private boolean resonant;
    private double resonanceScore;
    private int currentCategory;
    private int cycleCount;
    private final int maxCycles = 100;

    public StandardLaminarCircuit(P parameters) {
        this.parameters = parameters;
        this.layers = new TreeMap<>();
        this.pathways = new ArrayList<>();
        this.listeners = new CopyOnWriteArrayList<>();
        this.resonanceController = new DefaultResonanceController();
        this.resonant = false;
        this.resonanceScore = 0.0;
        this.currentCategory = -1;
        this.cycleCount = 0;
    }

    @Override
    public ILaminarCircuit<P> addLayer(ILayer layer, int depth) {
        layers.put(depth, layer);
        return this;
    }

    @Override
    public ILayer getLayer(int depth) {
        return layers.get(depth);
    }

    @Override
    public Map<Integer, ILayer> getLayers() {
        return new TreeMap<>(layers);
    }

    @Override
    public ILaminarCircuit<P> connectLayers(IPathway pathway) {
        pathways.add(pathway);
        return this;
    }

    @Override
    public List<IPathway> getPathways() {
        return new ArrayList<>(pathways);
    }

    @Override
    public List<IPathway> getPathwaysForLayer(String layerId) {
        return pathways.stream()
                .filter(p -> p.getSourceLayerId().equals(layerId) ||
                           p.getTargetLayerId().equals(layerId))
                .toList();
    }

    @Override
    public ILaminarCircuit<P> setResonanceController(IResonanceController controller) {
        this.resonanceController = controller;
        return this;
    }

    @Override
    public IResonanceController getResonanceController() {
        return resonanceController;
    }

    @Override
    public LaminarActivationResult processCycle(Pattern input, P parameters) {
        cycleCount++;

        // Get layer references (assume standard 3-layer architecture)
        var inputLayer = layers.get(0);
        var featureLayer = layers.get(1);
        var categoryLayer = layers.get(2);

        if (inputLayer == null || featureLayer == null || categoryLayer == null) {
            throw new IllegalStateException("Circuit requires layers at depths 0, 1, and 2");
        }

        // Process input through layers
        var processedInput = inputLayer.processBottomUp(input,
                parameters.getLayerParameters(inputLayer.getId()));

        // Bottom-up propagation to feature layer
        var featureActivation = propagateBottomUp(processedInput, inputLayer.getId(),
                featureLayer.getId(), parameters);
        featureLayer.setActivation(featureActivation);

        // Bottom-up to category layer
        var categoryInput = featureLayer.processBottomUp(featureActivation,
                parameters.getLayerParameters(featureLayer.getId()));
        var categoryActivation = propagateBottomUp(categoryInput, featureLayer.getId(),
                categoryLayer.getId(), parameters);
        categoryLayer.setActivation(categoryActivation);

        // Find active category
        currentCategory = findActiveCategory(categoryActivation);

        // Top-down expectation
        Pattern expectation = new DenseVector(new double[featureActivation.dimension()]);
        if (currentCategory >= 0) {
            expectation = categoryLayer.processTopDown(categoryActivation,
                    parameters.getLayerParameters(categoryLayer.getId()));
            expectation = propagateTopDown(expectation, categoryLayer.getId(),
                    featureLayer.getId(), parameters);
        }

        // Check resonance
        resonanceScore = resonanceController.calculateMatch(featureActivation, expectation);
        resonant = !resonanceController.shouldReset(resonanceScore);

        // Fire events
        fireEvents(resonant, currentCategory, resonanceScore);

        // Build result
        var layerActivations = new HashMap<String, Pattern>();
        layerActivations.put(inputLayer.getId(), inputLayer.getActivation());
        layerActivations.put(featureLayer.getId(), featureActivation);
        layerActivations.put(categoryLayer.getId(), categoryActivation);

        if (resonant) {
            return new LaminarActivationResult.Success(
                    currentCategory, featureActivation, expectation,
                    layerActivations, resonanceScore, cycleCount);
        } else {
            return new LaminarActivationResult.Failure(
                    "No resonance achieved", cycleCount);
        }
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
        layers.values().forEach(ILayer::reset);
        resonant = false;
        resonanceScore = 0.0;
        currentCategory = -1;
        cycleCount = 0;
    }

    @Override
    public void addListener(ICircuitEventListener listener) {
        listeners.add(listener);
    }

    @Override
    public void removeListener(ICircuitEventListener listener) {
        listeners.remove(listener);
    }

    @Override
    public CircuitState getState() {
        var builder = CircuitState.builder()
                .withResonanceScore(resonanceScore)
                .withResonant(resonant)
                .withCurrentCategory(currentCategory)
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

    // === ARTAlgorithm Implementation ===

    @Override
    public ActivationResult learn(Pattern input, P parameters) {
        // Reset for new input
        resetActivations();

        // Search for resonance
        var excludedCategories = new HashSet<Integer>();
        var found = false;

        while (!found && cycleCount < maxCycles) {
            var result = processCycle(input, parameters);

            if (result.isResonant()) {
                found = true;

                // Update weights if resonant
                updateWeights(input);

                return new LaminarActivationResult.Success(currentCategory,
                    getLayer(1).getActivation(), getLayer(2).getActivation(),
                    getLayerActivations(), resonanceScore, cycleCount);
            } else {
                // Reset and try next category
                if (currentCategory >= 0) {
                    resonanceController.reset(currentCategory);
                    excludedCategories.add(currentCategory);
                }

                var nextCategory = resonanceController.getNextCategory(excludedCategories);
                if (nextCategory < 0) {
                    // No more categories - create new one
                    var newCategoryIdx = createNewCategory(input);
                    return new LaminarActivationResult.Success(newCategoryIdx,
                        getLayer(1).getActivation(), getLayer(2).getActivation(),
                        getLayerActivations(), 1.0, cycleCount);
                }
            }
        }

        return new LaminarActivationResult.Failure("Max cycles reached", cycleCount);
    }

    @Override
    public ActivationResult predict(Pattern input, P parameters) {
        resetActivations();
        var result = processCycle(input, parameters);

        if (result.isResonant()) {
            return new LaminarActivationResult.Success(currentCategory,
                getLayer(1).getActivation(), getLayer(2).getActivation(),
                getLayerActivations(), resonanceScore, cycleCount);
        } else {
            return new LaminarActivationResult.Failure("No match", cycleCount);
        }
    }

    @Override
    public int getCategoryCount() {
        // Assuming categories are in layer 2
        var categoryLayer = layers.get(2);
        return categoryLayer != null ? categoryLayer.size() : 0;
    }

    @Override
    public List<WeightVector> getCategories() {
        // Return category layer weights
        var categoryLayer = layers.get(2);
        if (categoryLayer != null) {
            var weights = categoryLayer.getWeights();
            var categories = new ArrayList<WeightVector>();
            for (int i = 0; i < weights.getRows(); i++) {
                var row = new double[weights.getCols()];
                for (int j = 0; j < weights.getCols(); j++) {
                    row[j] = weights.get(i, j);
                }
                categories.add(new FuzzyWeight(row, row.length / 2));
            }
            return categories;
        }
        return Collections.emptyList();
    }

    @Override
    public WeightVector getCategory(int index) {
        var categoryLayer = layers.get(2);
        if (categoryLayer != null && index >= 0 && index < categoryLayer.size()) {
            var weights = categoryLayer.getWeights();
            var row = new double[weights.getCols()];
            for (int j = 0; j < weights.getCols(); j++) {
                row[j] = weights.get(index, j);
            }
            return new FuzzyWeight(row, row.length / 2);
        }
        throw new IndexOutOfBoundsException("Category index out of bounds: " + index);
    }

    @Override
    public void clear() {
        // Clear all layers and pathways
        layers.values().forEach(ILayer::reset);
        pathways.clear();
        resetActivations();
    }

    @Override
    public void close() throws Exception {
        // Clean up any resources
        clear();
    }

    public P getParameters() {
        return parameters;
    }

    public void setParameters(P parameters) {
        this.parameters = parameters;
    }

    // === Helper Methods ===

    private Map<String, Pattern> getLayerActivations() {
        var activations = new HashMap<String, Pattern>();
        for (var layer : layers.values()) {
            activations.put(layer.getId(), layer.getActivation());
        }
        return activations;
    }

    private Pattern propagateBottomUp(Pattern signal, String sourceId, String targetId, P parameters) {
        var pathway = findPathway(sourceId, targetId, PathwayType.BOTTOM_UP);
        if (pathway != null) {
            return pathway.propagate(signal, parameters.getPathwayParameters(pathway.getId()));
        }
        return signal;
    }

    private Pattern propagateTopDown(Pattern signal, String sourceId, String targetId, P parameters) {
        var pathway = findPathway(sourceId, targetId, PathwayType.TOP_DOWN);
        if (pathway != null) {
            return pathway.propagate(signal, parameters.getPathwayParameters(pathway.getId()));
        }
        return signal;
    }

    private IPathway findPathway(String sourceId, String targetId, PathwayType type) {
        return pathways.stream()
                .filter(p -> p.getSourceLayerId().equals(sourceId) &&
                           p.getTargetLayerId().equals(targetId) &&
                           p.getType() == type)
                .findFirst()
                .orElse(null);
    }

    private int findActiveCategory(Pattern categoryActivation) {
        var maxActivation = 0.0;
        var activeIdx = -1;

        for (int i = 0; i < categoryActivation.dimension(); i++) {
            if (categoryActivation.get(i) > maxActivation) {
                maxActivation = categoryActivation.get(i);
                activeIdx = i;
            }
        }

        return activeIdx;
    }

    private int createNewCategory(Pattern input) {
        // Simple new category creation - would need more sophisticated implementation
        var categoryLayer = layers.get(2);
        if (categoryLayer instanceof CategoryLayer catLayer) {
            // Initialize new category weights with input pattern
            catLayer.updateWeights(input, 1.0); // Fast learning for new category
            return catLayer.size(); // Return index of new category
        }
        return -1;
    }

    private void updateWeights(Pattern input) {
        // Update pathway weights for learning
        for (var pathway : pathways) {
            if (pathway.isAdaptive()) {
                var learningRate = parameters.getLearningParameters().getLearningRate();
                // Simplified weight update - would need proper source/target patterns
                pathway.updateWeights(input, input, learningRate);
            }
        }
    }

    private void fireEvents(boolean resonant, int category, double score) {
        if (resonant) {
            var event = new ResonanceEvent(category, score);
            for (var listener : listeners) {
                try {
                    listener.onResonance(event);
                } catch (Exception e) {
                    System.err.println("Error in resonance listener: " + e.getMessage());
                }
            }
        }

        var cycleEvent = new CycleEvent(cycleCount, score);
        for (var listener : listeners) {
            try {
                listener.onCycleComplete(cycleEvent);
            } catch (Exception e) {
                System.err.println("Error in cycle listener: " + e.getMessage());
            }
        }
    }
}