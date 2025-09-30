package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.Layer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.core.WeightMatrix;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.LayerParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Decorator that adds temporal chunking capabilities to any layer.
 * Non-invasive enhancement preserving existing layer functionality.
 *
 * Integrates MaskingField-style chunking into laminar layer processing
 * for multi-scale temporal grouping and working memory organization.
 *
 * @author Hal Hildebrand
 */
public class TemporalChunkingLayerDecorator implements TemporalChunkingLayer {

    private final Layer delegate;
    private final ChunkingState chunkingState;
    private ChunkingParameters chunkingParameters;
    private boolean chunkingEnabled;
    private int sequencePosition;
    private double contextWeight;

    public TemporalChunkingLayerDecorator(Layer delegate, ChunkingParameters parameters) {
        this.delegate = delegate;
        this.chunkingParameters = parameters;
        this.chunkingState = new ChunkingState(parameters.getMaxHistorySize());
        this.chunkingEnabled = true;
        this.sequencePosition = 0;
        this.contextWeight = 0.3;  // Default: 30% temporal context
    }

    // ============ TemporalChunkingLayer Methods ============

    @Override
    public List<TemporalChunk> getTemporalChunks() {
        return chunkingState.getActiveChunks();
    }

    @Override
    public Pattern processWithChunking(Pattern input, double timeStep) {
        if (!chunkingEnabled) {
            return input;
        }

        // Use input pattern for chunking (before processing through delegate)
        double activation = computeActivationMagnitude(input);

        // Add to activation history
        chunkingState.addActivation(input, activation, chunkingState.getCurrentTime() + timeStep);

        // Check for chunk formation
        if (shouldFormChunk()) {
            var chunk = formChunk();
            if (chunk != null) {
                chunkingState.addChunk(chunk);
            }
        }

        // Update chunking dynamics
        updateChunkingDynamics(timeStep);

        sequencePosition++;

        // Return delegate's current activation
        return delegate.getActivation();
    }

    @Override
    public ChunkingState getChunkingState() {
        return chunkingState;
    }

    @Override
    public void updateChunkingDynamics(double timeStep) {
        // Decay existing chunks
        chunkingState.decayChunks(chunkingParameters.getChunkDecayRate());

        // Prune inactive chunks
        chunkingState.pruneInactiveChunks(chunkingParameters.getActivityThreshold());
    }

    @Override
    public boolean shouldFormChunk() {
        // Need minimum history
        int minSize = chunkingParameters.getMinChunkSize();
        if (chunkingState.getHistorySize() < minSize) {
            return false;
        }

        // Get recent history - use min size for initial chunk formation
        var recent = chunkingState.getRecentHistory(Math.max(minSize,
                                                             Math.min(chunkingState.getHistorySize(),
                                                                     chunkingParameters.getMaxChunkSize())));
        if (recent.size() < minSize) {
            return false;
        }

        // Check if recent activations are strong enough
        double avgActivation = recent.stream()
            .mapToDouble(ChunkingState.ActivationSnapshot::activation)
            .average()
            .orElse(0.0);

        if (avgActivation < chunkingParameters.getChunkFormationThreshold()) {
            return false;
        }

        // Check temporal coherence (patterns should be similar)
        double coherence = computeCoherence(recent);
        return coherence >= chunkingParameters.getChunkCoherenceThreshold();
    }

    /**
     * Compute temporal coherence of a sequence.
     */
    private double computeCoherence(List<ChunkingState.ActivationSnapshot> sequence) {
        if (sequence.size() < 2) return 1.0;

        double totalSimilarity = 0.0;
        for (int i = 0; i < sequence.size() - 1; i++) {
            totalSimilarity += computeSimilarity(
                sequence.get(i).pattern(),
                sequence.get(i + 1).pattern()
            );
        }

        return totalSimilarity / (sequence.size() - 1);
    }

    /**
     * Compute cosine similarity between patterns.
     */
    private double computeSimilarity(Pattern p1, Pattern p2) {
        if (p1.dimension() != p2.dimension()) return 0.0;

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < p1.dimension(); i++) {
            dotProduct += p1.get(i) * p2.get(i);
            norm1 += p1.get(i) * p1.get(i);
            norm2 += p2.get(i) * p2.get(i);
        }

        if (norm1 == 0 || norm2 == 0) return 0.0;
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    @Override
    public TemporalChunk formChunk() {
        // Get recent history for chunk formation
        int chunkSize = Math.min(
            chunkingState.getHistorySize(),
            chunkingParameters.getMaxChunkSize()
        );

        if (chunkSize < chunkingParameters.getMinChunkSize()) {
            return null;
        }

        var recent = chunkingState.getRecentHistory(chunkSize);

        // Create chunk items
        List<TemporalChunk.ChunkItem> items = new ArrayList<>();
        int position = sequencePosition - recent.size();

        for (var snapshot : recent) {
            items.add(new TemporalChunk.ChunkItem(
                snapshot.pattern(),
                snapshot.activation(),
                snapshot.time(),
                position++
            ));
        }

        return new TemporalChunk(
            items,
            chunkingState.getCurrentTime(),
            chunkingState.getNextChunkId()
        );
    }

    @Override
    public void resetChunking() {
        chunkingState.reset();
        sequencePosition = 0;
    }

    @Override
    public Pattern getTemporalContext() {
        // Aggregate chunk representations into temporal context
        var chunks = chunkingState.getActiveChunks();
        if (chunks.isEmpty()) {
            return delegate.getActivation();  // Fall back to current activation
        }

        // Weight chunks by recency and strength
        int dimension = delegate.size();
        double[] context = new double[dimension];
        double totalWeight = 0.0;

        for (var chunk : chunks) {
            var repr = chunk.getRepresentativePattern();
            if (repr != null && repr.dimension() == dimension) {
                double weight = chunk.getStrength();
                totalWeight += weight;

                for (int i = 0; i < dimension; i++) {
                    context[i] += repr.get(i) * weight;
                }
            }
        }

        // Normalize
        if (totalWeight > 0) {
            for (int i = 0; i < dimension; i++) {
                context[i] /= totalWeight;
            }
        }

        return new com.hellblazer.art.core.DenseVector(context);
    }

    @Override
    public void setChunkingParameters(ChunkingParameters params) {
        this.chunkingParameters = params;
    }

    @Override
    public ChunkingStatistics getChunkingStatistics() {
        return ChunkingStatistics.from(chunkingState, chunkingState.getCurrentTime());
    }

    @Override
    public LayerState getLayerState() {
        var activation = delegate.getActivation();
        var context = getTemporalContext();
        return LayerState.withContext(activation, context, chunkingState.getCurrentTime());
    }

    @Override
    public void setContextWeight(double weight) {
        if (weight < 0.0 || weight > 1.0) {
            throw new IllegalArgumentException("Context weight must be in [0,1], got: " + weight);
        }
        this.contextWeight = weight;
    }

    @Override
    public double getContextWeight() {
        return contextWeight;
    }

    /**
     * Enable or disable chunking.
     */
    public void setChunkingEnabled(boolean enabled) {
        this.chunkingEnabled = enabled;
    }

    /**
     * Check if chunking is enabled.
     */
    public boolean isChunkingEnabled() {
        return chunkingEnabled;
    }

    // ============ Delegated Layer Methods ============

    @Override
    public String getId() {
        return delegate.getId();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public LayerType getType() {
        return delegate.getType();
    }

    @Override
    public Pattern getActivation() {
        return delegate.getActivation();
    }

    @Override
    public void setActivation(Pattern activation) {
        delegate.setActivation(activation);
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        var result = delegate.processBottomUp(input, parameters);

        // Integrate with chunking if enabled
        if (chunkingEnabled) {
            processWithChunking(result, 0.01);  // Default time step
        }

        return result;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        return delegate.processTopDown(expectation, parameters);
    }

    @Override
    public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
        return delegate.processLateral(lateral, parameters);
    }

    @Override
    public WeightMatrix getWeights() {
        return delegate.getWeights();
    }

    @Override
    public void setWeights(WeightMatrix weights) {
        delegate.setWeights(weights);
    }

    @Override
    public void updateWeights(Pattern input, double learningRate) {
        delegate.updateWeights(input, learningRate);
    }

    @Override
    public void reset() {
        delegate.reset();
        resetChunking();
    }

    @Override
    public void addActivationListener(LayerActivationListener listener) {
        delegate.addActivationListener(listener);
    }

    // ============ Helper Methods ============

    private double computeActivationMagnitude(Pattern pattern) {
        double sum = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            sum += pattern.get(i) * pattern.get(i);
        }
        return Math.sqrt(sum);
    }
}