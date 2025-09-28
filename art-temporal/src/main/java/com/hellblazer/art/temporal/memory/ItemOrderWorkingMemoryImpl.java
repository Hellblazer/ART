package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.parameters.WorkingMemoryParameters;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * STORE 2 model implementation of Item-and-Order Working Memory
 * Implements primacy gradient storage and temporal-to-spatial transformation
 * Based on Kazerounian & Grossberg (2014)
 */
public class ItemOrderWorkingMemoryImpl implements ItemOrderWorkingMemory {

    private final WorkingMemoryParameters parameters;
    private final int maxSequenceLength;
    private final int itemDimension;

    // Working memory state
    private float[][] itemActivations;  // [position][dimension]
    private float[] primacyGradient;    // Primacy weights for each position
    private float[] contrastState;      // Contrast-enhanced activations
    private int currentSequenceLength;

    // Shunting dynamics state
    private float[][] shuntingActivations;
    private float[] lateralInhibition;

    // LTM Invariance state
    private float totalActivation;
    private final float activationThreshold;

    public ItemOrderWorkingMemoryImpl(WorkingMemoryParameters parameters, int maxSequenceLength, int itemDimension) {
        this.parameters = parameters;
        this.maxSequenceLength = maxSequenceLength;
        this.itemDimension = itemDimension;
        this.activationThreshold = 0.1f;

        // Initialize state arrays
        this.itemActivations = new float[maxSequenceLength][itemDimension];
        this.primacyGradient = new float[maxSequenceLength];
        this.contrastState = new float[maxSequenceLength];
        this.shuntingActivations = new float[maxSequenceLength][itemDimension];
        this.lateralInhibition = new float[maxSequenceLength];

        this.currentSequenceLength = 0;
    }

    // Not from interface - internal method
    private void storeSequence(List<Pattern> sequence) {
        if (sequence.size() > maxSequenceLength) {
            throw new IllegalArgumentException("Sequence length exceeds maximum capacity");
        }

        reset();
        currentSequenceLength = sequence.size();

        // Store items with primacy gradient
        for (int i = 0; i < sequence.size(); i++) {
            var pattern = sequence.get(i);
            var features = extractPatternFeatures(pattern);

            // Copy features to item activations
            System.arraycopy(features, 0, itemActivations[i], 0, Math.min(features.length, itemDimension));

            // Apply primacy gradient: earlier items have higher activation
            primacyGradient[i] = computePrimacyWeight(i, sequence.size());

            // Apply primacy to activations
            for (int j = 0; j < itemDimension; j++) {
                itemActivations[i][j] *= primacyGradient[i];
            }
        }

        // Update shunting dynamics
        updateShuntingDynamics();

        // Compute total activation for LTM Invariance
        computeTotalActivation();
    }

    // Note: updateDynamics is not part of ItemOrderWorkingMemory interface
    public void updateDynamics(float deltaTime) {
        // Update shunting dynamics with on-center off-surround
        for (int i = 0; i < currentSequenceLength; i++) {
            // Compute lateral inhibition from other positions
            lateralInhibition[i] = 0;
            for (int j = 0; j < currentSequenceLength; j++) {
                if (i != j) {
                    lateralInhibition[i] += parameters.competitiveRate() *
                                           sumActivation(shuntingActivations[j]);
                }
            }

            // Update each dimension with shunting equation
            for (int d = 0; d < itemDimension; d++) {
                var x = shuntingActivations[i][d];
                var input = itemActivations[i][d];

                // Shunting equation: dx/dt = -Ax + (B-x)I - (x+C)L
                var dx = (float)(-parameters.decayRate() * x +
                        (parameters.maxActivation() - x) * input -
                        x * lateralInhibition[i]);

                shuntingActivations[i][d] = x + dx * deltaTime;

                // Bound activations
                shuntingActivations[i][d] = Math.max(0, Math.min((float)parameters.maxActivation(),
                                                                 shuntingActivations[i][d]));
            }
        }

        // Update contrast enhancement
        updateContrastEnhancement();
    }

    public float[] getTemporalPattern() {
        // Concatenate all position activations into single pattern
        var pattern = new float[currentSequenceLength * itemDimension];
        var index = 0;

        for (int i = 0; i < currentSequenceLength; i++) {
            for (int d = 0; d < itemDimension; d++) {
                pattern[index++] = shuntingActivations[i][d];
            }
        }

        return pattern;
    }

    public float[] getPrimacyGradient() {
        return Arrays.copyOf(primacyGradient, currentSequenceLength);
    }

    public void applyContrastEnhancement(int targetPosition) {
        // Reset contrast state
        Arrays.fill(contrastState, 0);

        if (targetPosition < 0 || targetPosition >= currentSequenceLength) {
            return;
        }

        // Enhance target position, suppress others
        for (int i = 0; i < currentSequenceLength; i++) {
            if (i == targetPosition) {
                // Enhance current item for execution
                contrastState[i] = 2.0f;  // Default enhancement factor
            } else if (i < targetPosition) {
                // Suppress already executed items
                contrastState[i] = 0.1f;  // Default suppression factor
            } else {
                // Maintain future items
                contrastState[i] = 1.0f;
            }

            // Apply contrast to activations
            for (int d = 0; d < itemDimension; d++) {
                shuntingActivations[i][d] *= contrastState[i];
            }
        }
    }

    public void reset() {
        // Clear all activations
        for (int i = 0; i < maxSequenceLength; i++) {
            Arrays.fill(itemActivations[i], 0);
            Arrays.fill(shuntingActivations[i], 0);
        }
        Arrays.fill(primacyGradient, 0);
        Arrays.fill(contrastState, 1.0f);
        Arrays.fill(lateralInhibition, 0);
        currentSequenceLength = 0;
        totalActivation = 0;
    }

    public boolean satisfiesLTMInvariance() {
        // Check if current activation pattern satisfies LTM Invariance Principle
        // Total activation should remain bounded to prevent catastrophic forgetting
        return totalActivation > activationThreshold &&
               totalActivation < parameters.maxActivation() * currentSequenceLength;
    }

    public TemporalPattern getCurrentState() {
        return new TemporalPatternImpl(
            getTemporalPattern(),
            currentSequenceLength,
            primacyGradient,
            System.currentTimeMillis()
        );
    }

    // Helper methods

    private float computePrimacyWeight(int position, int totalLength) {
        // Primacy gradient: exponential decay with position
        // First item has weight 1.0, decreasing for later items
        var decayFactor = parameters.decayRate();  // Use decay rate for primacy
        return (float) Math.exp(-position * decayFactor);
    }

    private void updateShuntingDynamics() {
        // Initialize shunting activations from item activations
        for (int i = 0; i < currentSequenceLength; i++) {
            System.arraycopy(itemActivations[i], 0, shuntingActivations[i], 0, itemDimension);
        }
    }

    private float sumActivation(float[] activation) {
        var sum = 0.0f;
        for (var a : activation) {
            sum += Math.abs(a);
        }
        return sum;
    }

    private void computeTotalActivation() {
        totalActivation = 0;
        for (int i = 0; i < currentSequenceLength; i++) {
            totalActivation += sumActivation(shuntingActivations[i]);
        }
    }

    private void updateContrastEnhancement() {
        // Apply stored contrast state
        for (int i = 0; i < currentSequenceLength; i++) {
            if (contrastState[i] != 1.0f) {
                for (int d = 0; d < itemDimension; d++) {
                    shuntingActivations[i][d] *= contrastState[i];
                }
            }
        }
    }

    // Inner class for temporal pattern
    private static class TemporalPatternImpl implements TemporalPattern {
        private final float[] features;
        private final int sequenceLength;
        private final float[] temporalWeights;
        private final long timestamp;

        public TemporalPatternImpl(float[] features, int sequenceLength,
                                  float[] temporalWeights, long timestamp) {
            this.features = features;
            this.sequenceLength = sequenceLength;
            this.temporalWeights = Arrays.copyOf(temporalWeights, sequenceLength);
            this.timestamp = timestamp;
        }

        @Override
        public List<Pattern> getSequence() {
            // Convert features back to patterns
            var patterns = new ArrayList<Pattern>();
            if (sequenceLength == 0 || features.length == 0) return patterns;

            var dimPerItem = features.length / sequenceLength;
            for (int i = 0; i < sequenceLength; i++) {
                var itemFeatures = new double[dimPerItem];
                for (int j = 0; j < dimPerItem; j++) {
                    itemFeatures[j] = features[i * dimPerItem + j];
                }
                patterns.add(Pattern.of(itemFeatures));
            }
            return patterns;
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime < 0 || endTime > sequenceLength || startTime >= endTime) {
                throw new IndexOutOfBoundsException("Invalid subsequence bounds");
            }
            var subLength = endTime - startTime;
            var dimPerItem = features.length / sequenceLength;
            var subFeatures = Arrays.copyOfRange(features,
                                                startTime * dimPerItem,
                                                endTime * dimPerItem);
            var subWeights = Arrays.copyOfRange(temporalWeights, startTime, endTime);
            return new TemporalPatternImpl(subFeatures, subLength, subWeights, timestamp);
        }

        @Override
        public boolean isEmpty() {
            return sequenceLength == 0;
        }
    }

    // Add missing helper methods
    private float[] extractPatternFeatures(Pattern pattern) {
        var features = new float[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            features[i] = (float) pattern.get(i);
        }
        return features;
    }

    private float sigmoid(double x, double slope) {
        return (float) (1.0 / (1.0 + Math.exp(-slope * x)));
    }

    // === Implement missing ItemOrderWorkingMemory interface methods ===

    @Override
    public void storeItem(Pattern item, double timestamp) {
        if (currentSequenceLength >= maxSequenceLength) {
            throw new IllegalStateException("Working memory is full");
        }

        var features = extractPatternFeatures(item);
        var position = currentSequenceLength;

        // Store item at current position
        System.arraycopy(features, 0, itemActivations[position], 0, Math.min(features.length, itemDimension));

        // Apply primacy gradient
        primacyGradient[position] = computePrimacyWeight(position, currentSequenceLength + 1);

        // Apply primacy to activations
        for (int j = 0; j < itemDimension; j++) {
            itemActivations[position][j] *= primacyGradient[position];
        }

        currentSequenceLength++;
        updateShuntingDynamics();
        computeTotalActivation();
    }

    @Override
    public void storeSequence(TemporalPattern sequence) {
        var items = sequence.getSequence();
        storeSequence(items);
    }

    @Override
    public TemporalPattern getCurrentContents() {
        return getCurrentState();
    }

    @Override
    public double[] getPrimacyValues() {
        var result = new double[currentSequenceLength];
        for (int i = 0; i < currentSequenceLength; i++) {
            result[i] = primacyGradient[i];
        }
        return result;
    }

    @Override
    public double[] getTemporalPositions() {
        var result = new double[currentSequenceLength];
        for (int i = 0; i < currentSequenceLength; i++) {
            result[i] = i;  // Simple sequential positions
        }
        return result;
    }

    @Override
    public void updateDynamics(double deltaTime) {
        updateDynamics((float)deltaTime);
    }

    @Override
    public void clear() {
        reset();
    }

    @Override
    public boolean isEmpty() {
        return currentSequenceLength == 0;
    }

    @Override
    public int getItemCount() {
        return currentSequenceLength;
    }

    @Override
    public int getCapacity() {
        return maxSequenceLength;
    }

    @Override
    public TemporalPattern getItemsAboveThreshold(double threshold) {
        var activeItems = new ArrayList<Pattern>();
        for (int i = 0; i < currentSequenceLength; i++) {
            if (primacyGradient[i] > threshold) {
                var itemFeatures = new double[itemDimension];
                for (int j = 0; j < itemDimension; j++) {
                    itemFeatures[j] = itemActivations[i][j];
                }
                activeItems.add(Pattern.of(itemFeatures));
            }
        }
        return new SimpleTemporalPattern(activeItems);
    }

    @Override
    public TemporalPattern getRecentItems(int count) {
        var start = Math.max(0, currentSequenceLength - count);
        var recentItems = new ArrayList<Pattern>();
        for (int i = start; i < currentSequenceLength; i++) {
            var itemFeatures = new double[itemDimension];
            for (int j = 0; j < itemDimension; j++) {
                itemFeatures[j] = itemActivations[i][j];
            }
            recentItems.add(Pattern.of(itemFeatures));
        }
        return new SimpleTemporalPattern(recentItems);
    }

    @Override
    public Pattern getMostSalientItem() {
        if (currentSequenceLength == 0) return null;

        int maxIndex = 0;
        double maxPrimacy = primacyGradient[0];
        for (int i = 1; i < currentSequenceLength; i++) {
            if (primacyGradient[i] > maxPrimacy) {
                maxPrimacy = primacyGradient[i];
                maxIndex = i;
            }
        }

        var itemFeatures = new double[itemDimension];
        for (int j = 0; j < itemDimension; j++) {
            itemFeatures[j] = itemActivations[maxIndex][j];
        }
        return Pattern.of(itemFeatures);
    }

    @Override
    public double getTotalActivation() {
        return totalActivation;
    }

    @Override
    public boolean containsItem(Pattern item) {
        // Simple containment check based on feature similarity
        var features = extractPatternFeatures(item);
        for (int i = 0; i < currentSequenceLength; i++) {
            boolean match = true;
            for (int j = 0; j < Math.min(features.length, itemDimension); j++) {
                if (Math.abs(features[j] - itemActivations[i][j]) > 0.01) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        return false;
    }

    @Override
    public double getItemPrimacy(Pattern item) {
        var features = extractPatternFeatures(item);
        for (int i = 0; i < currentSequenceLength; i++) {
            boolean match = true;
            for (int j = 0; j < Math.min(features.length, itemDimension); j++) {
                if (Math.abs(features[j] - itemActivations[i][j]) > 0.01) {
                    match = false;
                    break;
                }
            }
            if (match) return primacyGradient[i];
        }
        return 0.0;
    }

    @Override
    public WorkingMemoryPerformanceMetrics getPerformanceMetrics() {
        return new WorkingMemoryPerformanceMetrics() {
            @Override
            public long getStoreOperations() { return 0L; }
            @Override
            public long getUpdateOperations() { return 0L; }
            @Override
            public long getComputationTime() { return 0L; }
            @Override
            public long getMemoryUsage() { return 0L; }
            @Override
            public double getAverageAccessTime() { return 0.0; }
        };
    }

    @Override
    public void resetPerformanceTracking() {
        // Reset any performance counters if needed
    }

    @Override
    public void setParameters(WorkingMemoryParameters newParameters) {
        // Can't modify final field, so we'd need to reconstruct if needed
        // For now, log a warning or throw an exception
        throw new UnsupportedOperationException("Cannot change parameters after construction");
    }

    @Override
    public WorkingMemoryParameters getParameters() {
        return parameters;
    }

    @Override
    public WorkingMemorySnapshot createSnapshot() {
        return new WorkingMemorySnapshot() {
            private final TemporalPattern pattern = getCurrentState();
            private final double[] primacy = getPrimacyValues();
            private final double time = System.currentTimeMillis();
            private final double activation = getTotalActivation();

            @Override
            public TemporalPattern getStoredPattern() { return pattern; }
            @Override
            public double[] getPrimacyValues() { return primacy; }
            @Override
            public double getSnapshotTime() { return time; }
            @Override
            public double getTotalActivation() { return activation; }
        };
    }

    @Override
    public void restoreSnapshot(WorkingMemorySnapshot snapshot) {
        clear();
        var pattern = snapshot.getStoredPattern();
        if (pattern != null) {
            storeSequence(pattern.getSequence());
        }
    }

    @Override
    public void close() {
        // No resources to close in this implementation
    }

    // Simple temporal pattern implementation for return values
    private static class SimpleTemporalPattern implements TemporalPattern {
        private final List<Pattern> sequence;

        SimpleTemporalPattern(List<Pattern> sequence) {
            this.sequence = sequence;
        }

        @Override
        public List<Pattern> getSequence() {
            return sequence;
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            return new SimpleTemporalPattern(sequence.subList(startTime, endTime));
        }

        @Override
        public boolean isEmpty() {
            return sequence.isEmpty();
        }
    }
}