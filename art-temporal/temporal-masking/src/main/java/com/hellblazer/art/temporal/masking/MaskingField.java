package com.hellblazer.art.temporal.masking;

import com.hellblazer.art.temporal.core.*;
import com.hellblazer.art.temporal.memory.WorkingMemory;
import com.hellblazer.art.temporal.memory.TemporalPattern;

import java.util.ArrayList;
import java.util.List;

/**
 * Masking field implementation for multi-scale temporal chunking.
 * Based on Grossberg & Kazerounian (2016) LIST PARSE model.
 *
 * The masking field operates at an intermediate time scale (50-500ms)
 * between working memory (10-100ms) and long-term memory (seconds).
 * It performs spatial competition to create item chunks and list chunks.
 */
public class MaskingField {

    private final MaskingFieldParameters parameters;
    private final List<ItemNode> itemNodes;
    private final List<ListChunk> listChunks;
    private final WorkingMemory workingMemory;

    // Dynamics
    private final ShuntingDynamics shuntingDynamics;

    // State tracking
    private MaskingFieldState currentState;
    private double currentTime;
    private int activeChunkIndex;

    public MaskingField(MaskingFieldParameters parameters, WorkingMemory workingMemory) {
        this.parameters = parameters;
        this.workingMemory = workingMemory;
        this.itemNodes = new ArrayList<>();
        this.listChunks = new ArrayList<>();
        this.shuntingDynamics = new ShuntingDynamics();

        // Initialize state
        int maxNodes = parameters.getMaxItemNodes();
        this.currentState = new MaskingFieldState(maxNodes);
        this.currentTime = 0.0;
        this.activeChunkIndex = -1;
    }

    /**
     * Process temporal pattern from working memory through masking field.
     * This performs multi-scale integration and chunking.
     */
    public void processTemporalPattern(TemporalPattern pattern) {
        // Extract pattern information
        var patterns = pattern.patterns();
        var weights = pattern.weights();

        // Create or update item nodes for each pattern
        for (int i = 0; i < patterns.size(); i++) {
            var itemPattern = patterns.get(i);
            var weight = weights.get(i);

            // Check if pattern matches existing item node
            int matchingNode = findMatchingItemNode(itemPattern);

            if (matchingNode >= 0) {
                // Strengthen existing node
                strengthenItemNode(matchingNode, weight);
            } else {
                // Create new item node
                createItemNode(itemPattern, weight, i);
            }
        }

        // Perform spatial competition among item nodes
        performSpatialCompetition();

        // Check for list chunk formation
        if (shouldFormListChunk()) {
            formListChunk();
        }

        // Update dynamics
        evolveDynamics(parameters.getIntegrationTimeStep());
    }

    /**
     * Find item node matching the given pattern.
     */
    private int findMatchingItemNode(double[] pattern) {
        double threshold = parameters.getMatchingThreshold();

        for (int i = 0; i < itemNodes.size(); i++) {
            var node = itemNodes.get(i);
            if (node.matches(pattern, threshold)) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Strengthen an existing item node.
     */
    private void strengthenItemNode(int index, double weight) {
        var node = itemNodes.get(index);
        node.strengthen(weight * parameters.getLearningRate());

        // Update state activation
        var activations = currentState.getItemActivations();
        activations[index] += weight * parameters.getActivationBoost();
        currentState.setItemActivations(activations);
    }

    /**
     * Create a new item node.
     */
    private void createItemNode(double[] pattern, double weight, int position) {
        if (itemNodes.size() >= parameters.getMaxItemNodes()) {
            // Prune weakest node if at capacity
            pruneWeakestNode();
        }

        var node = new ItemNode(pattern, weight, position, currentTime);
        itemNodes.add(node);

        // Initialize activation
        int index = itemNodes.size() - 1;
        var activations = currentState.getItemActivations();
        activations[index] = weight * parameters.getInitialActivation();
        currentState.setItemActivations(activations);

        // Update active item count
        currentState.incrementActiveItemCount();
    }

    /**
     * Perform spatial competition among item nodes.
     * This implements winner-take-all dynamics with Mexican hat connectivity.
     */
    private void performSpatialCompetition() {
        var activations = currentState.getItemActivations();
        int numNodes = Math.min(itemNodes.size(), activations.length);

        if (numNodes < 2) return;

        var newActivations = new double[activations.length];

        // Apply Mexican hat competition
        for (int i = 0; i < numNodes; i++) {
            double netInput = 0.0;

            for (int j = 0; j < numNodes; j++) {
                if (i == j) continue;

                double distance = Math.abs(i - j) / parameters.getSpatialScale();
                double connection = mexicanHat(distance,
                                              parameters.getExcitationRange(),
                                              parameters.getInhibitionRange());

                netInput += connection * activations[j];
            }

            // Update activation with competition
            double competitiveInput = parameters.getCompetitionStrength() * netInput;
            newActivations[i] = activations[i] + parameters.getIntegrationTimeStep() * competitiveInput;

            // Apply bounds
            newActivations[i] = Math.max(0.0, Math.min(1.0, newActivations[i]));
        }

        // Apply contrast enhancement
        enhanceContrast(newActivations, numNodes);

        // Update state
        currentState.setItemActivations(newActivations);

        // Identify winning nodes
        identifyWinners();
    }

    /**
     * Mexican hat connectivity function.
     */
    private double mexicanHat(double distance, double excitationRange, double inhibitionRange) {
        double excitation = Math.exp(-distance * distance / (2.0 * excitationRange * excitationRange));
        double inhibition = 0.5 * Math.exp(-distance * distance / (2.0 * inhibitionRange * inhibitionRange));
        return excitation - inhibition;
    }

    /**
     * Enhance contrast through normalization.
     */
    private void enhanceContrast(double[] activations, int numNodes) {
        if (numNodes == 0) return;

        double maxActivation = 0.0;
        for (int i = 0; i < numNodes; i++) {
            maxActivation = Math.max(maxActivation, activations[i]);
        }

        if (maxActivation == 0.0) return;

        for (int i = 0; i < numNodes; i++) {
            activations[i] = activations[i] / (maxActivation + 0.1);
        }
    }

    /**
     * Identify winning nodes after competition.
     */
    private void identifyWinners() {
        var activations = currentState.getItemActivations();
        double threshold = parameters.getWinnerThreshold();

        List<Integer> winners = new ArrayList<>();
        for (int i = 0; i < Math.min(itemNodes.size(), activations.length); i++) {
            if (activations[i] > threshold) {
                winners.add(i);
            }
        }

        currentState.setWinningNodes(winners);
    }

    /**
     * Check if conditions are met to form a list chunk.
     */
    private boolean shouldFormListChunk() {
        var winners = currentState.getWinningNodes();

        // Need at least minChunkSize winners
        if (winners.size() < parameters.getMinChunkSize()) {
            return false;
        }

        // Check if winners form a coherent sequence
        if (!isCoherentSequence(winners)) {
            return false;
        }

        // Check if sufficient time has passed since last chunk
        if (activeChunkIndex >= 0) {
            var lastChunk = listChunks.get(activeChunkIndex);
            double timeSinceLastChunk = currentTime - lastChunk.getFormationTime();
            if (timeSinceLastChunk < parameters.getMinChunkInterval()) {
                return false;
            }
        }

        return true;
    }

    /**
     * Check if winning nodes form a coherent sequence.
     */
    private boolean isCoherentSequence(List<Integer> winners) {
        if (winners.size() < 2) return true;

        // Check temporal coherence
        double maxGap = parameters.getMaxTemporalGap();
        for (int i = 1; i < winners.size(); i++) {
            var node1 = itemNodes.get(winners.get(i-1));
            var node2 = itemNodes.get(winners.get(i));

            if (Math.abs(node2.getPosition() - node1.getPosition()) > maxGap) {
                return false;
            }
        }

        return true;
    }

    /**
     * Form a new list chunk from winning nodes.
     */
    private void formListChunk() {
        var winners = currentState.getWinningNodes();
        List<ItemNode> chunkNodes = new ArrayList<>();

        for (int idx : winners) {
            chunkNodes.add(itemNodes.get(idx));
        }

        var chunk = new ListChunk(chunkNodes, currentTime, listChunks.size());
        listChunks.add(chunk);
        activeChunkIndex = listChunks.size() - 1;

        // Reset item activations after chunk formation
        if (parameters.isResetAfterChunk()) {
            resetItemActivations();
        }
    }

    /**
     * Reset item activations after chunk formation.
     */
    private void resetItemActivations() {
        var activations = currentState.getItemActivations();
        for (int i = 0; i < activations.length; i++) {
            activations[i] *= parameters.getResetDecayFactor();
        }
        currentState.setItemActivations(activations);
        currentState.clearWinners();
    }

    /**
     * Prune the weakest item node.
     */
    private void pruneWeakestNode() {
        if (itemNodes.isEmpty()) return;

        int weakestIndex = 0;
        double weakestStrength = itemNodes.get(0).getStrength();

        for (int i = 1; i < itemNodes.size(); i++) {
            double strength = itemNodes.get(i).getStrength();
            if (strength < weakestStrength) {
                weakestStrength = strength;
                weakestIndex = i;
            }
        }

        itemNodes.remove(weakestIndex);

        // Shift activations
        var activations = currentState.getItemActivations();
        for (int i = weakestIndex; i < activations.length - 1; i++) {
            activations[i] = activations[i + 1];
        }
        activations[activations.length - 1] = 0.0;
        currentState.setItemActivations(activations);

        // Update active item count
        currentState.decrementActiveItemCount();
    }

    /**
     * Evolve masking field dynamics.
     */
    private void evolveDynamics(double dt) {
        // Update time
        currentTime += dt;

        // Apply shunting dynamics to item activations
        var shuntingParams = ShuntingParameters.builder()
            .decayRate(parameters.getItemDecayRate())
            .upperBound(parameters.getMaxActivation())
            .lowerBound(0.0)
            .selfExcitation(parameters.getSelfExcitation())
            .lateralInhibition(0.0) // Competition handled separately
            .enableNormalization(parameters.isNormalizationEnabled())
            .build();

        var activations = currentState.getItemActivations();
        var shuntingState = new ShuntingState(activations, new double[activations.length]);

        shuntingState = shuntingDynamics.step(shuntingState, shuntingParams, currentTime, dt);
        currentState.setItemActivations(shuntingState.getActivations());

        // Update chunk activations
        updateChunkActivations(dt);
    }

    /**
     * Update list chunk activations.
     */
    private void updateChunkActivations(double dt) {
        var chunkActivations = currentState.getChunkActivations();
        double decayRate = parameters.getChunkDecayRate();

        for (int i = 0; i < listChunks.size() && i < chunkActivations.length; i++) {
            // Exponential decay
            chunkActivations[i] *= Math.exp(-decayRate * dt);

            // Boost active chunk
            if (i == activeChunkIndex) {
                chunkActivations[i] += parameters.getActiveChunkBoost() * dt;
            }

            // Clamp to bounds
            chunkActivations[i] = Math.max(0.0,
                Math.min(parameters.getMaxActivation(), chunkActivations[i]));
        }

        currentState.setChunkActivations(chunkActivations);
    }

    /**
     * Get the current masking field state.
     */
    public MaskingFieldState getState() {
        var state = currentState.copy();
        // Synchronize the active item count with the actual number of item nodes
        state.setActiveItemCount(itemNodes.size());
        return state;
    }

    /**
     * Get formed list chunks.
     */
    public List<ListChunk> getListChunks() {
        return new ArrayList<>(listChunks);
    }

    /**
     * Get active item nodes.
     */
    public List<ItemNode> getItemNodes() {
        return new ArrayList<>(itemNodes);
    }

    /**
     * Reset the masking field.
     */
    public void reset() {
        itemNodes.clear();
        listChunks.clear();
        currentState = new MaskingFieldState(parameters.getMaxItemNodes());
        currentTime = 0.0;
        activeChunkIndex = -1;
    }

    /**
     * Get the current active chunk.
     */
    public ListChunk getActiveChunk() {
        if (activeChunkIndex >= 0 && activeChunkIndex < listChunks.size()) {
            return listChunks.get(activeChunkIndex);
        }
        return null;
    }

    /**
     * Get chunking statistics.
     */
    public ChunkingStatistics getStatistics() {
        return new ChunkingStatistics(
            itemNodes.size(),
            listChunks.size(),
            activeChunkIndex,
            computeAverageChunkSize(),
            computeChunkingEfficiency()
        );
    }

    private double computeAverageChunkSize() {
        if (listChunks.isEmpty()) return 0.0;

        double totalSize = 0.0;
        for (var chunk : listChunks) {
            totalSize += chunk.size();
        }
        return totalSize / listChunks.size();
    }

    private double computeChunkingEfficiency() {
        if (itemNodes.isEmpty()) return 0.0;

        int chunkedItems = 0;
        for (var chunk : listChunks) {
            chunkedItems += chunk.size();
        }

        return (double) chunkedItems / itemNodes.size();
    }

    /**
     * Statistics about chunking performance.
     */
    public record ChunkingStatistics(
        int totalItemNodes,
        int totalChunks,
        int activeChunkIndex,
        double averageChunkSize,
        double chunkingEfficiency
    ) {}
}