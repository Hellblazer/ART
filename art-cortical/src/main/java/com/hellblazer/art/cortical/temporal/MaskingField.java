package com.hellblazer.art.cortical.temporal;

import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Masking field implementation for multi-scale temporal chunking.
 * Based on Grossberg & Kazerounian (2016) LIST PARSE model.
 *
 * The masking field operates at an intermediate time scale (50-500ms)
 * between working memory (10-100ms) and long-term memory (seconds).
 * It performs spatial competition to create item chunks and list chunks.
 *
 * Integrates with Phase 1 ShuntingDynamics for neural competition.
 *
 * @author Hal Hildebrand
 */
public class MaskingField {

    private final MaskingFieldParameters parameters;
    private final List<ItemNode> itemNodes;
    private final List<ListChunk> listChunks;
    private final ShuntingDynamics shuntingDynamics;

    // State tracking
    private MaskingFieldState currentState;
    private double currentTime;
    private int activeChunkIndex;

    /**
     * Create masking field with parameters.
     * Creates internal shunting dynamics for competitive processing.
     */
    public MaskingField(MaskingFieldParameters parameters) {
        this.parameters = parameters;
        this.itemNodes = new ArrayList<>();
        this.listChunks = new ArrayList<>();

        // Create shunting dynamics for competition
        var shuntingParams = createShuntingParameters(parameters);
        this.shuntingDynamics = new ShuntingDynamics(shuntingParams);

        // Initialize state
        this.currentState = MaskingFieldState.create(parameters.maxItemNodes());
        this.currentTime = 0.0;
        this.activeChunkIndex = -1;
    }

    /**
     * Create shunting parameters from masking field parameters.
     */
    private static ShuntingParameters createShuntingParameters(MaskingFieldParameters params) {
        return ShuntingParameters.builder(params.maxItemNodes())
            .uniformDecay(params.itemDecayRate())
            .ceiling(params.maxActivation())
            .floor(0.0)
            .selfExcitation(params.selfExcitation())
            .excitatoryRange(params.excitationRange())
            .inhibitoryRange(params.inhibitionRange())
            .excitatoryStrength(params.competitionStrength())
            .inhibitoryStrength(params.competitionStrength())
            .timeStep(params.integrationTimeStep())
            .build();
    }

    /**
     * Update masking field with input pattern.
     * Returns updated state after processing.
     */
    public MaskingFieldState update(double[] inputPattern, double dt) {
        // Process input as temporal pattern from working memory
        processInputPattern(inputPattern);

        // Perform spatial competition among item nodes
        performSpatialCompetition();

        // Check for list chunk formation
        if (shouldFormListChunk()) {
            formListChunk();
        }

        // Update dynamics
        evolveDynamics(dt);

        return currentState;
    }

    /**
     * Process input pattern, creating or strengthening item nodes.
     */
    private void processInputPattern(double[] pattern) {
        // Find matching item node
        var matchingNode = findMatchingItemNode(pattern);

        if (matchingNode >= 0) {
            // Strengthen existing node
            strengthenItemNode(matchingNode, 1.0);
        } else {
            // Create new item node
            createItemNode(pattern, 1.0, itemNodes.size());
        }
    }

    /**
     * Find item node matching the given pattern.
     */
    private int findMatchingItemNode(double[] pattern) {
        var threshold = parameters.matchingThreshold();

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
        node.strengthen(weight * parameters.learningRate());

        // Update state activation
        var activations = currentState.itemActivations();
        activations[index] += weight * parameters.activationBoost();
        currentState = currentState.withItemActivations(activations);
    }

    /**
     * Create a new item node.
     */
    private void createItemNode(double[] pattern, double weight, int position) {
        if (itemNodes.size() >= parameters.maxItemNodes()) {
            // Prune weakest node if at capacity
            pruneWeakestNode();
        }

        var node = new ItemNode(pattern, weight, position, currentTime);
        itemNodes.add(node);

        // Initialize activation
        var index = itemNodes.size() - 1;
        var activations = currentState.itemActivations();
        activations[index] = weight * parameters.initialActivation();
        currentState = currentState.withItemActivations(activations)
                                   .withIncrementedItemCount();
    }

    /**
     * Perform spatial competition among item nodes.
     * Uses ShuntingDynamics for winner-take-all dynamics with Mexican hat connectivity.
     */
    private void performSpatialCompetition() {
        var activations = currentState.itemActivations();
        var numNodes = Math.min(itemNodes.size(), activations.length);

        if (numNodes < 2) return;

        // Set activations as excitatory input to shunting dynamics
        shuntingDynamics.setExcitatoryInput(activations);

        // Run one step of competition
        var newActivations = shuntingDynamics.update(parameters.integrationTimeStep());

        // Copy only active node activations
        System.arraycopy(newActivations, 0, activations, 0, numNodes);

        // Apply contrast enhancement if enabled
        if (parameters.normalizationEnabled()) {
            enhanceContrast(activations, numNodes);
        }

        // Update state
        currentState = currentState.withItemActivations(activations);

        // Identify winning nodes
        identifyWinners();
    }

    /**
     * Enhance contrast through normalization.
     */
    private void enhanceContrast(double[] activations, int numNodes) {
        if (numNodes == 0) return;

        var maxActivation = 0.0;
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
        var activations = currentState.itemActivations();
        var threshold = parameters.winnerThreshold();

        List<Integer> winners = new ArrayList<>();
        for (int i = 0; i < Math.min(itemNodes.size(), activations.length); i++) {
            if (activations[i] > threshold) {
                winners.add(i);
            }
        }

        currentState = currentState.withWinningNodes(winners);
    }

    /**
     * Check if conditions are met to form a list chunk.
     */
    private boolean shouldFormListChunk() {
        var winners = currentState.winningNodes();

        // Need at least minChunkSize winners
        if (winners.size() < parameters.minChunkSize()) {
            return false;
        }

        // Check if winners form a coherent sequence
        if (!isCoherentSequence(winners)) {
            return false;
        }

        // Check if sufficient time has passed since last chunk
        if (activeChunkIndex >= 0 && activeChunkIndex < listChunks.size()) {
            var lastChunk = listChunks.get(activeChunkIndex);
            var timeSinceLastChunk = currentTime - lastChunk.getFormationTime();
            if (timeSinceLastChunk < parameters.minChunkInterval()) {
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
        var maxGap = parameters.maxTemporalGap();
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
        var winners = currentState.winningNodes();
        List<ItemNode> chunkNodes = new ArrayList<>();

        for (int idx : winners) {
            chunkNodes.add(itemNodes.get(idx));
        }

        var chunk = new ListChunk(chunkNodes, currentTime, listChunks.size());
        listChunks.add(chunk);
        activeChunkIndex = listChunks.size() - 1;

        // Reset item activations after chunk formation
        if (parameters.resetAfterChunk()) {
            resetItemActivations();
        }
    }

    /**
     * Reset item activations after chunk formation.
     */
    private void resetItemActivations() {
        var activations = currentState.itemActivations();
        for (int i = 0; i < activations.length; i++) {
            activations[i] *= parameters.resetDecayFactor();
        }
        currentState = currentState.withItemActivations(activations)
                                   .withClearedWinners();
    }

    /**
     * Prune the weakest item node.
     */
    private void pruneWeakestNode() {
        if (itemNodes.isEmpty()) return;

        var weakestIndex = 0;
        var weakestStrength = itemNodes.get(0).getStrength();

        for (int i = 1; i < itemNodes.size(); i++) {
            var strength = itemNodes.get(i).getStrength();
            if (strength < weakestStrength) {
                weakestStrength = strength;
                weakestIndex = i;
            }
        }

        itemNodes.remove(weakestIndex);

        // Shift activations
        var activations = currentState.itemActivations();
        for (int i = weakestIndex; i < activations.length - 1; i++) {
            activations[i] = activations[i + 1];
        }
        activations[activations.length - 1] = 0.0;
        currentState = currentState.withItemActivations(activations)
                                   .withDecrementedItemCount();
    }

    /**
     * Evolve masking field dynamics.
     */
    private void evolveDynamics(double dt) {
        // Update time
        currentTime += dt;

        // Item activations already updated by shunting dynamics in performSpatialCompetition()

        // Update chunk activations
        updateChunkActivations(dt);
    }

    /**
     * Update list chunk activations.
     */
    private void updateChunkActivations(double dt) {
        var chunkActivations = currentState.chunkActivations();
        var decayRate = parameters.chunkDecayRate();

        for (int i = 0; i < listChunks.size() && i < chunkActivations.length; i++) {
            // Exponential decay
            chunkActivations[i] *= Math.exp(-decayRate * dt);

            // Boost active chunk
            if (i == activeChunkIndex) {
                chunkActivations[i] += parameters.activeChunkBoost() * dt;
            }

            // Clamp to bounds
            chunkActivations[i] = Math.max(0.0,
                Math.min(parameters.maxActivation(), chunkActivations[i]));
        }

        currentState = currentState.withChunkActivations(chunkActivations);
    }

    /**
     * Get the current masking field state.
     */
    public MaskingFieldState getState() {
        return currentState.withActiveItemCount(itemNodes.size());
    }

    /**
     * Get active list chunks.
     */
    public List<ListChunk> getActiveChunks() {
        return new ArrayList<>(listChunks);
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
        currentState = MaskingFieldState.create(parameters.maxItemNodes());
        currentTime = 0.0;
        activeChunkIndex = -1;
        shuntingDynamics.reset();
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

        var totalSize = 0.0;
        for (var chunk : listChunks) {
            totalSize += chunk.size();
        }
        return totalSize / listChunks.size();
    }

    private double computeChunkingEfficiency() {
        if (itemNodes.isEmpty()) return 0.0;

        var chunkedItems = 0;
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
