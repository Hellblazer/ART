package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.temporal.masking.*;
import com.hellblazer.art.temporal.memory.TemporalPattern;
import jdk.incubator.vector.*;
import org.jctools.queues.MpscArrayQueue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * High-performance vectorized masking field implementation.
 * Uses SIMD and concurrent data structures for maximum throughput.
 */
public class VectorizedMaskingField {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final MaskingFieldParameters parameters;
    private final VectorizedWorkingMemory workingMemory;

    private final int maxNodes;
    private final int vectorLength;
    private final int loopBound;

    // Concurrent structures for multi-threaded access
    private final MpscArrayQueue<ItemNode> itemNodes;
    private final MpscArrayQueue<ListChunk> listChunks;
    private final AtomicInteger nodeCount;

    // Vectorized state arrays
    private double[] itemActivations;
    private double[] chunkActivations;
    private double[] lateralWeights;
    private double[] resetSignals;

    // Pre-computed Mexican hat kernel
    private double[] mexicanHatKernel;

    private double currentTime;

    public VectorizedMaskingField(MaskingFieldParameters parameters,
                                 VectorizedWorkingMemory workingMemory) {
        this.parameters = parameters;
        this.workingMemory = workingMemory;

        this.maxNodes = parameters.getMaxItemNodes();
        this.vectorLength = SPECIES.length();
        this.loopBound = SPECIES.loopBound(maxNodes);

        this.itemNodes = new MpscArrayQueue<>(maxNodes * 2);
        this.listChunks = new MpscArrayQueue<>(parameters.getMaxChunks() * 2);
        this.nodeCount = new AtomicInteger(0);

        this.itemActivations = new double[maxNodes];
        this.chunkActivations = new double[parameters.getMaxChunks()];
        this.lateralWeights = new double[maxNodes * maxNodes];
        this.resetSignals = new double[maxNodes];

        precomputeMexicanHat();
        this.currentTime = 0.0;
    }

    /**
     * Process temporal pattern with vectorized operations.
     */
    public void processTemporalPattern(TemporalPattern pattern) {
        if (!pattern.isValid()) return;

        var patterns = pattern.patterns();
        var weights = pattern.weights();

        // Create or match item nodes (parallel)
        processItemNodesParallel(patterns, weights);

        // Run competitive dynamics (vectorized)
        runCompetitiveDynamicsVectorized();

        // Detect and form chunks (vectorized)
        detectChunksVectorized();

        currentTime += parameters.getIntegrationTimeStep();
    }

    /**
     * Process item nodes with potential parallelization.
     */
    private void processItemNodesParallel(List<double[]> patterns, List<Double> weights) {
        int currentNodes = nodeCount.get();

        for (int i = 0; i < patterns.size() && currentNodes < maxNodes; i++) {
            var pattern = patterns.get(i);
            var weight = weights.get(i);

            // Check for existing match (vectorized similarity)
            ItemNode matched = findBestMatchVectorized(pattern);

            if (matched != null && computeSimilarity(matched.getPattern(), pattern)
                >= parameters.getMatchingThreshold()) {
                // Strengthen existing node
                matched.strengthen(weight * parameters.getLearningRate());
            } else if (currentNodes < maxNodes) {
                // Create new node
                var newNode = new ItemNode(pattern, weight, i, currentTime);
                if (itemNodes.offer(newNode)) {
                    nodeCount.incrementAndGet();
                    currentNodes++;
                }
            }
        }
    }

    /**
     * Vectorized competitive dynamics.
     */
    private void runCompetitiveDynamicsVectorized() {
        var nodes = new ArrayList<ItemNode>();
        itemNodes.forEach(nodes::add);

        int size = Math.min(nodes.size(), maxNodes);
        if (size == 0) return;

        // Initialize activations from node strengths
        for (int i = 0; i < size; i++) {
            itemActivations[i] = nodes.get(i).getStrength() *
                                 parameters.getInitialActivation();
        }

        // Run dynamics for several iterations
        int iterations = (int)(1.0 / parameters.getIntegrationTimeStep());
        for (int iter = 0; iter < iterations; iter++) {
            updateActivationsVectorized(size);
        }

        // Apply winner selection
        selectWinnersVectorized(size);
    }

    /**
     * Vectorized activation update with Mexican hat.
     */
    private void updateActivationsVectorized(int size) {
        double[] newActivations = new double[size];
        int bound = SPECIES.loopBound(size);

        int i = 0;
        for (; i < bound; i += vectorLength) {
            var vAct = DoubleVector.fromArray(SPECIES, itemActivations, i);

            // Compute lateral interactions (Mexican hat)
            var vLateral = computeLateralVectorized(i, size);

            // Apply reset signals
            var vReset = DoubleVector.fromArray(SPECIES, resetSignals, i);

            // Shunting equation
            var vDerivative = vAct.neg()
                .add(vLateral.mul(1.0 - vAct.reduceLanes(VectorOperators.MAX)))
                .sub(vAct.mul(vReset));

            // Update with time step
            var vNew = vAct.add(vDerivative.mul(parameters.getIntegrationTimeStep()));

            // Apply bounds and store
            vNew = vNew.max(0.0).min(parameters.getMaxActivation());
            vNew.intoArray(newActivations, i);
        }

        // Scalar tail
        for (; i < size; i++) {
            double lateral = computeLateral(i, size);
            double derivative = -itemActivations[i] +
                              lateral * (1.0 - itemActivations[i]) -
                              itemActivations[i] * resetSignals[i];
            newActivations[i] = Math.max(0, Math.min(parameters.getMaxActivation(),
                itemActivations[i] + parameters.getIntegrationTimeStep() * derivative));
        }

        System.arraycopy(newActivations, 0, itemActivations, 0, size);
    }

    /**
     * Vectorized lateral interaction computation.
     */
    private DoubleVector computeLateralVectorized(int start, int size) {
        var vSum = DoubleVector.zero(SPECIES);

        for (int j = 0; j < size; j++) {
            if (Math.abs(j - start) > vectorLength) continue;

            double actJ = itemActivations[j];
            if (actJ > 0.01) {
                // Use pre-computed Mexican hat kernel
                int kernelIdx = Math.abs(j - start - vectorLength/2);
                if (kernelIdx < mexicanHatKernel.length) {
                    vSum = vSum.add(DoubleVector.broadcast(SPECIES,
                        mexicanHatKernel[kernelIdx] * actJ));
                }
            }
        }

        return vSum;
    }

    /**
     * Scalar lateral computation for tail.
     */
    private double computeLateral(int i, int size) {
        double sum = 0.0;

        for (int j = 0; j < size; j++) {
            if (i == j) continue;

            int distance = Math.abs(i - j);
            if (distance < mexicanHatKernel.length) {
                sum += mexicanHatKernel[distance] * itemActivations[j];
            }
        }

        return sum;
    }

    /**
     * Vectorized winner selection.
     */
    private void selectWinnersVectorized(int size) {
        double threshold = parameters.getWinnerThreshold();

        // Find maximum activation (vectorized)
        double maxActivation = 0.0;
        int bound = SPECIES.loopBound(size);

        int i = 0;
        for (; i < bound; i += vectorLength) {
            var vAct = DoubleVector.fromArray(SPECIES, itemActivations, i);
            maxActivation = Math.max(maxActivation, vAct.reduceLanes(VectorOperators.MAX));
        }
        for (; i < size; i++) {
            maxActivation = Math.max(maxActivation, itemActivations[i]);
        }

        // Apply soft winner-take-all
        double winnerThreshold = maxActivation * threshold;

        i = 0;
        for (; i < bound; i += vectorLength) {
            var vAct = DoubleVector.fromArray(SPECIES, itemActivations, i);
            var vThresh = DoubleVector.broadcast(SPECIES, winnerThreshold);
            var vMask = vAct.lt(vThresh);
            vAct = vAct.blend(0.0, vMask);
            vAct.intoArray(itemActivations, i);
        }

        for (; i < size; i++) {
            if (itemActivations[i] < winnerThreshold) {
                itemActivations[i] = 0.0;
            }
        }
    }

    /**
     * Vectorized chunk detection.
     */
    private void detectChunksVectorized() {
        var nodes = new ArrayList<ItemNode>();
        itemNodes.forEach(nodes::add);

        if (nodes.size() < parameters.getMinChunkSize()) return;

        // Find sequences of active nodes
        List<List<ItemNode>> sequences = new ArrayList<>();
        List<ItemNode> currentSeq = new ArrayList<>();

        // Ensure we don't exceed bounds of either array/list
        var maxIndex = Math.min(nodes.size(), itemActivations.length);
        for (int i = 0; i < maxIndex; i++) {
            if (itemActivations[i] > parameters.getWinnerThreshold()) {
                currentSeq.add(nodes.get(i));

                // Check for gap
                if (i < maxIndex - 1) {
                    int gap = nodes.get(i + 1).getPosition() - nodes.get(i).getPosition();
                    if (gap > parameters.getMaxTemporalGap()) {
                        if (currentSeq.size() >= parameters.getMinChunkSize()) {
                            sequences.add(new ArrayList<>(currentSeq));
                        }
                        currentSeq.clear();
                    }
                }
            }
        }

        // Add final sequence
        if (currentSeq.size() >= parameters.getMinChunkSize()) {
            sequences.add(currentSeq);
        }

        // Create chunks
        for (var seq : sequences) {
            if (seq.size() <= parameters.getMaxChunkSize()) {
                var chunk = new ListChunk(seq, currentTime, listChunks.size());
                listChunks.offer(chunk);

                // Set reset signals for chunked items
                if (parameters.isResetAfterChunk()) {
                    applyResetSignalsVectorized(seq);
                }
            }
        }
    }

    /**
     * Vectorized reset signal application.
     */
    private void applyResetSignalsVectorized(List<ItemNode> chunkedItems) {
        double resetStrength = parameters.getResetDecayFactor();

        var nodes = new ArrayList<ItemNode>();
        itemNodes.forEach(nodes::add);

        for (int i = 0; i < nodes.size(); i++) {
            if (chunkedItems.contains(nodes.get(i))) {
                resetSignals[i] = resetStrength;
            }
        }

        // Decay reset signals (vectorized)
        int bound = SPECIES.loopBound(maxNodes);
        var vDecay = DoubleVector.broadcast(SPECIES, 0.95);

        int i = 0;
        for (; i < bound; i += vectorLength) {
            var vReset = DoubleVector.fromArray(SPECIES, resetSignals, i);
            vReset = vReset.mul(vDecay);
            vReset.intoArray(resetSignals, i);
        }

        for (; i < maxNodes; i++) {
            resetSignals[i] *= 0.95;
        }
    }

    /**
     * Find best match using vectorized similarity.
     */
    private ItemNode findBestMatchVectorized(double[] pattern) {
        ItemNode bestMatch = null;
        double bestSimilarity = 0.0;

        for (ItemNode node : itemNodes) {
            double similarity = computeSimilarity(node.getPattern(), pattern);
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = node;
            }
        }

        return bestMatch;
    }

    /**
     * Vectorized cosine similarity computation.
     */
    private double computeSimilarity(double[] a, double[] b) {
        if (a.length != b.length) return 0.0;

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        int bound = SPECIES.loopBound(a.length);
        int i = 0;

        for (; i < bound; i += vectorLength) {
            var vA = DoubleVector.fromArray(SPECIES, a, i);
            var vB = DoubleVector.fromArray(SPECIES, b, i);

            dotProduct += vA.mul(vB).reduceLanes(VectorOperators.ADD);
            normA += vA.mul(vA).reduceLanes(VectorOperators.ADD);
            normB += vB.mul(vB).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA > 0 && normB > 0) {
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        }
        return 0.0;
    }

    /**
     * Pre-compute Mexican hat kernel.
     */
    private void precomputeMexicanHat() {
        mexicanHatKernel = new double[maxNodes];
        double exciteRange = parameters.getExcitationRange();
        double inhibitRange = parameters.getInhibitionRange();
        double spatialScale = parameters.getSpatialScale();

        for (int d = 0; d < maxNodes; d++) {
            double scaledDist = d / spatialScale;
            double excitation = Math.exp(-scaledDist * scaledDist / (2 * exciteRange * exciteRange));
            double inhibition = 0.5 * Math.exp(-scaledDist * scaledDist / (2 * inhibitRange * inhibitRange));
            mexicanHatKernel[d] = parameters.getCompetitionStrength() * (excitation - inhibition);
        }
    }

    public void reset() {
        itemNodes.clear();
        listChunks.clear();
        nodeCount.set(0);

        // Reset arrays (vectorized)
        var vZero = DoubleVector.zero(SPECIES);

        for (int i = 0; i < loopBound; i += vectorLength) {
            vZero.intoArray(itemActivations, i);
            vZero.intoArray(resetSignals, i);
        }

        for (int i = loopBound; i < maxNodes; i++) {
            itemActivations[i] = 0.0;
            resetSignals[i] = 0.0;
        }

        currentTime = 0.0;
    }

    // Getters
    public List<ItemNode> getItemNodes() {
        var list = new ArrayList<ItemNode>();
        itemNodes.forEach(list::add);
        return list;
    }

    public List<ListChunk> getListChunks() {
        var list = new ArrayList<ListChunk>();
        listChunks.forEach(list::add);
        return list;
    }

    public MaskingFieldState getState() {
        var state = new MaskingFieldState(itemActivations.length);
        state.setItemActivations(itemActivations);
        state.setChunkActivations(chunkActivations);

        // Convert winning item nodes to indices
        var winningIndices = new java.util.ArrayList<Integer>();
        for (int i = 0; i < itemActivations.length; i++) {
            if (itemActivations[i] > 0.01) { // Threshold for being considered active
                winningIndices.add(i);
            }
        }
        state.setWinningNodes(winningIndices);

        return state;
    }
}