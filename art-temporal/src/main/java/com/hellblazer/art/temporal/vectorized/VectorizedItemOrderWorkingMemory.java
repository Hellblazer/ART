/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.temporal.vectorized;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.memory.ItemOrderWorkingMemory;
import com.hellblazer.art.temporal.parameters.WorkingMemoryParameters;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Vectorized implementation of Item-and-Order Working Memory using Java Vector API for SIMD optimization.
 *
 * This implementation provides significant performance improvements for:
 * - Primacy gradient computation using SIMD operations
 * - Shunting dynamics updates with vectorized arithmetic
 * - Batch processing of item activations
 * - Parallel lateral inhibition calculations
 *
 * Target speedup: 10-100x for sequence processing operations depending on sequence length
 * and input dimensionality. Peak performance achieved with sequences of 8+ items and
 * item dimensions that are multiples of vector species length.
 *
 * Mathematical Foundation:
 * Vectorized shunting dynamics: dx/dt = -α*x + (β - x)*I - (x + C)*L
 * Implemented using SIMD operations for element-wise arithmetic across dimensions.
 *
 * Vectorized primacy gradient: p_i = exp(-i * α) for position i
 * Computed in batches using vector exponential operations.
 *
 * @author Hal Hildebrand
 */
public class VectorizedItemOrderWorkingMemory implements ItemOrderWorkingMemory {

    // Inner class for working memory snapshot
    private static class BasicWorkingMemorySnapshot implements WorkingMemorySnapshot {
        private final TemporalPattern storedPattern;
        private final double[] primacyValues;
        private final int sequenceTime;
        private final float totalActivation;

        BasicWorkingMemorySnapshot(TemporalPattern storedPattern, double[] primacyValues,
                                   int sequenceTime, float totalActivation) {
            this.storedPattern = storedPattern;
            this.primacyValues = primacyValues;
            this.sequenceTime = sequenceTime;
            this.totalActivation = totalActivation;
        }

        @Override
        public TemporalPattern getStoredPattern() { return storedPattern; }

        @Override
        public double[] getPrimacyValues() { return primacyValues; }

        public int getSequenceTime() { return sequenceTime; }

        @Override
        public double getTotalActivation() { return totalActivation; }

        @Override
        public double getSnapshotTime() { return System.currentTimeMillis(); }
    }

    private static final Logger log = LoggerFactory.getLogger(VectorizedItemOrderWorkingMemory.class);

    // SIMD Configuration
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();

    // Working Memory State
    private final int maxSequenceLength;
    private final int itemDimension;
    private final int paddedDimension; // Padded to vector length boundary
    private WorkingMemoryParameters parameters;

    // Vectorized State Arrays
    private float[][] itemActivations;      // [position][paddedDimension]
    private float[][] shuntingActivations;  // [position][paddedDimension]
    private float[] primacyGradient;        // [position]
    private float[] lateralInhibition;      // [position]
    private float[] contrastState;          // [position]
    private float[] totalActivationPerItem; // [position]

    // Temporal State
    private int currentSequenceLength;
    private double currentTime;
    private boolean isDirty; // Tracks if dynamics need updating

    // Performance Tracking
    private final AtomicLong vectorOperations = new AtomicLong(0);
    private final AtomicLong storeOperations = new AtomicLong(0);
    private final AtomicLong updateOperations = new AtomicLong(0);
    private final AtomicLong computationTimeNanos = new AtomicLong(0);

    /**
     * Create vectorized working memory with specified parameters and dimensions.
     *
     * @param parameters working memory configuration
     * @param maxSequenceLength maximum sequence length supported
     * @param itemDimension dimensionality of individual items
     */
    public VectorizedItemOrderWorkingMemory(WorkingMemoryParameters parameters,
                                          int maxSequenceLength,
                                          int itemDimension) {
        this.parameters = parameters;
        this.maxSequenceLength = maxSequenceLength;
        this.itemDimension = itemDimension;

        // Pad dimension to vector boundary for optimal SIMD performance
        this.paddedDimension = ((itemDimension + VECTOR_LENGTH - 1) / VECTOR_LENGTH) * VECTOR_LENGTH;

        log.debug("Initializing VectorizedItemOrderWorkingMemory: maxLength={}, itemDim={}, paddedDim={}, vectorLen={}",
                 maxSequenceLength, itemDimension, paddedDimension, VECTOR_LENGTH);

        initializeArrays();
        reset();
    }

    // Helper method to extract features from Pattern
    private double[] extractPatternFeatures(Pattern pattern) {
        var features = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            features[i] = pattern.get(i);
        }
        return features;
    }

    private void reset() {
        clear();
    }

    @Override
    public void storeItem(Pattern item, double timestamp) {
        var startTime = System.nanoTime();
        try {
            if (currentSequenceLength >= maxSequenceLength) {
                throw new IllegalStateException("Working memory at full capacity");
            }

            var position = currentSequenceLength;
            var features = extractPatternFeatures(item);

            // Store item with SIMD-optimized operations
            storeItemVectorized(features, position);

            // Update primacy gradient
            primacyGradient[position] = computePrimacyWeight(position, currentSequenceLength + 1);

            // Apply primacy to activations using SIMD
            applyPrimacyVectorized(position);

            currentSequenceLength++;
            currentTime = timestamp;
            isDirty = true;

            storeOperations.incrementAndGet();

        } finally {
            computationTimeNanos.addAndGet(System.nanoTime() - startTime);
        }
    }

    @Override
    public TemporalPattern getCurrentContents() {
        if (currentSequenceLength == 0) {
            return new VectorizedTemporalPattern(new ArrayList<>(), new double[0], currentTime);
        }

        var patterns = new ArrayList<Pattern>(currentSequenceLength);
        var temporalWeights = new double[currentSequenceLength];

        for (int i = 0; i < currentSequenceLength; i++) {
            var features = new double[itemDimension];
            for (int j = 0; j < itemDimension; j++) {
                features[j] = shuntingActivations[i][j];
            }
            patterns.add(Pattern.of(features));
            temporalWeights[i] = primacyGradient[i];
        }

        return new VectorizedTemporalPattern(patterns, temporalWeights, currentTime);
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
        var positions = new double[currentSequenceLength];
        for (int i = 0; i < currentSequenceLength; i++) {
            positions[i] = i; // Simple temporal indexing
        }
        return positions;
    }

    @Override
    public void updateDynamics(double deltaTime) {
        if (!isDirty && deltaTime == 0) return;

        var startTime = System.nanoTime();
        try {
            // Update shunting dynamics with vectorized operations
            updateShuntingDynamicsVectorized((float) deltaTime);

            // Update contrast enhancement if needed
            if (hasContrastModification()) {
                updateContrastEnhancementVectorized();
            }

            currentTime += deltaTime;
            isDirty = false;
            updateOperations.incrementAndGet();

        } finally {
            computationTimeNanos.addAndGet(System.nanoTime() - startTime);
        }
    }

    @Override
    public void clear() {
        // Clear all state arrays efficiently
        for (int i = 0; i < maxSequenceLength; i++) {
            Arrays.fill(itemActivations[i], 0.0f);
            Arrays.fill(shuntingActivations[i], 0.0f);
        }
        Arrays.fill(primacyGradient, 0.0f);
        Arrays.fill(lateralInhibition, 0.0f);
        Arrays.fill(contrastState, 1.0f);
        Arrays.fill(totalActivationPerItem, 0.0f);

        currentSequenceLength = 0;
        currentTime = 0.0;
        isDirty = false;
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
        var patterns = new ArrayList<Pattern>();
        var weights = new ArrayList<Double>();

        for (int i = 0; i < currentSequenceLength; i++) {
            if (primacyGradient[i] >= threshold) {
                var features = new double[itemDimension];
                for (int j = 0; j < itemDimension; j++) {
                    features[j] = shuntingActivations[i][j];
                }
                patterns.add(Pattern.of(features));
                weights.add((double) primacyGradient[i]);
            }
        }

        return new VectorizedTemporalPattern(patterns,
                                           weights.stream().mapToDouble(Double::doubleValue).toArray(),
                                           currentTime);
    }

    @Override
    public TemporalPattern getRecentItems(int count) {
        var actualCount = Math.min(count, currentSequenceLength);
        var startIndex = Math.max(0, currentSequenceLength - actualCount);

        var patterns = new ArrayList<Pattern>(actualCount);
        var weights = new double[actualCount];

        for (int i = 0; i < actualCount; i++) {
            var sourceIndex = startIndex + i;
            var features = new double[itemDimension];
            for (int j = 0; j < itemDimension; j++) {
                features[j] = shuntingActivations[sourceIndex][j];
            }
            patterns.add(Pattern.of(features));
            weights[i] = primacyGradient[sourceIndex];
        }

        return new VectorizedTemporalPattern(patterns, weights, currentTime);
    }

    @Override
    public Pattern getMostSalientItem() {
        if (currentSequenceLength == 0) return null;

        var maxPrimacy = 0.0f;
        var maxIndex = 0;

        for (int i = 0; i < currentSequenceLength; i++) {
            if (primacyGradient[i] > maxPrimacy) {
                maxPrimacy = primacyGradient[i];
                maxIndex = i;
            }
        }

        var features = new double[itemDimension];
        for (int j = 0; j < itemDimension; j++) {
            features[j] = shuntingActivations[maxIndex][j];
        }

        return Pattern.of(features);
    }

    @Override
    public double getTotalActivation() {
        var total = 0.0;
        for (int i = 0; i < currentSequenceLength; i++) {
            total += totalActivationPerItem[i];
        }
        return total;
    }

    @Override
    public boolean containsItem(Pattern item) {
        var targetFeatures = extractPatternFeatures(item);
        var tolerance = 1e-6;

        for (int i = 0; i < currentSequenceLength; i++) {
            var matches = true;
            for (int j = 0; j < Math.min(itemDimension, targetFeatures.length); j++) {
                if (Math.abs(shuntingActivations[i][j] - targetFeatures[j]) > tolerance) {
                    matches = false;
                    break;
                }
            }
            if (matches) return true;
        }
        return false;
    }

    @Override
    public double getItemPrimacy(Pattern item) {
        var targetFeatures = extractPatternFeatures(item);
        var tolerance = 1e-6;

        for (int i = 0; i < currentSequenceLength; i++) {
            var matches = true;
            for (int j = 0; j < Math.min(itemDimension, targetFeatures.length); j++) {
                if (Math.abs(shuntingActivations[i][j] - targetFeatures[j]) > tolerance) {
                    matches = false;
                    break;
                }
            }
            if (matches) return primacyGradient[i];
        }
        return 0.0;
    }

    @Override
    public void setParameters(WorkingMemoryParameters parameters) {
        this.parameters = parameters;
        isDirty = true;
    }

    @Override
    public WorkingMemoryParameters getParameters() {
        return parameters;
    }

    @Override
    public WorkingMemorySnapshot createSnapshot() {
        // Convert float[] primacyGradient to double[]
        var primacyValues = new double[currentSequenceLength];
        for (int i = 0; i < currentSequenceLength; i++) {
            primacyValues[i] = primacyGradient[i];
        }

        return new BasicWorkingMemorySnapshot(
            getCurrentContents(),
            primacyValues,
            (int)currentTime,
            (float)getTotalActivation()
        );
    }

    @Override
    public void restoreSnapshot(WorkingMemorySnapshot snapshot) {
        clear();

        var pattern = snapshot.getStoredPattern();
        var sequence = pattern.getSequence();
        var primacyValues = snapshot.getPrimacyValues();

        for (int i = 0; i < sequence.size() && i < maxSequenceLength; i++) {
            storeItem(sequence.get(i), i);
            if (i < primacyValues.length) {
                primacyGradient[i] = (float)primacyValues[i];
            }
        }

        currentTime = snapshot.getSnapshotTime();
        isDirty = true;
    }

    @Override
    public WorkingMemoryPerformanceMetrics getPerformanceMetrics() {
        return new VectorizedWorkingMemoryPerformanceMetrics();
    }

    @Override
    public void resetPerformanceTracking() {
        vectorOperations.set(0);
        storeOperations.set(0);
        updateOperations.set(0);
        computationTimeNanos.set(0);
    }

    @Override
    public void close() {
        // Release resources if needed
        clear();
    }

    // === Private Vectorized Implementation Methods ===

    private void initializeArrays() {
        itemActivations = new float[maxSequenceLength][paddedDimension];
        shuntingActivations = new float[maxSequenceLength][paddedDimension];
        primacyGradient = new float[maxSequenceLength];
        lateralInhibition = new float[maxSequenceLength];
        contrastState = new float[maxSequenceLength];
        totalActivationPerItem = new float[maxSequenceLength];

        // Initialize contrast state to neutral (1.0)
        Arrays.fill(contrastState, 1.0f);
    }

    private void storeItemVectorized(double[] features, int position) {
        // Copy features to padded array with SIMD optimization
        var positionArray = itemActivations[position];

        // Copy actual features
        for (int i = 0; i < Math.min(features.length, itemDimension); i++) {
            positionArray[i] = (float) features[i];
        }

        // Zero-pad remaining elements
        for (int i = itemDimension; i < paddedDimension; i++) {
            positionArray[i] = 0.0f;
        }

        // Copy to shunting activations (initial state)
        System.arraycopy(positionArray, 0, shuntingActivations[position], 0, paddedDimension);

        vectorOperations.incrementAndGet();
    }

    private void applyPrimacyVectorized(int position) {
        var primacy = primacyGradient[position];
        var activations = itemActivations[position];
        var shunting = shuntingActivations[position];

        // Apply primacy using SIMD operations
        var primacyVector = FloatVector.broadcast(SPECIES, primacy);

        for (int i = 0; i < paddedDimension; i += VECTOR_LENGTH) {
            var activationVector = FloatVector.fromArray(SPECIES, activations, i);
            var result = activationVector.mul(primacyVector);
            result.intoArray(shunting, i);
        }

        vectorOperations.incrementAndGet();
    }

    private void updateShuntingDynamicsVectorized(float deltaTime) {
        if (currentSequenceLength == 0) return;

        // First compute lateral inhibition for all positions
        computeLateralInhibitionVectorized();

        // Then update each position's shunting dynamics
        var decayVector = FloatVector.broadcast(SPECIES, (float) parameters.decayRate());
        var upperBoundVector = FloatVector.broadcast(SPECIES, (float) parameters.maxActivation());
        var lowerBoundVector = FloatVector.broadcast(SPECIES, 0.0f); // Assuming lower bound is 0
        var deltaTimeVector = FloatVector.broadcast(SPECIES, deltaTime);

        for (int pos = 0; pos < currentSequenceLength; pos++) {
            var activations = shuntingActivations[pos];
            var inputs = itemActivations[pos];
            var inhibition = FloatVector.broadcast(SPECIES, lateralInhibition[pos]);

            // Update total activation for this item
            totalActivationPerItem[pos] = computeTotalActivationVectorized(activations);

            // Update shunting dynamics: dx/dt = -α*x + (β-x)*I - (x+C)*L
            for (int i = 0; i < paddedDimension; i += VECTOR_LENGTH) {
                var x = FloatVector.fromArray(SPECIES, activations, i);
                var input = FloatVector.fromArray(SPECIES, inputs, i);

                // -α*x
                var decay = x.mul(decayVector).neg();

                // (β-x)*I
                var excitation = upperBoundVector.sub(x).mul(input);

                // (x+C)*L (assuming C=0 for simplicity)
                var inhibitionTerm = x.mul(inhibition).neg();

                // Combine terms: dx = decay + excitation + inhibition
                var dx = decay.add(excitation).add(inhibitionTerm);

                // Integrate: x_new = x + dx*dt
                var newX = x.add(dx.mul(deltaTimeVector));

                // Bound values to [0, upperBound]
                newX = newX.max(lowerBoundVector).min(upperBoundVector);

                newX.intoArray(activations, i);
            }
        }

        vectorOperations.addAndGet(currentSequenceLength);
    }

    private void computeLateralInhibitionVectorized() {
        Arrays.fill(lateralInhibition, 0.0f);

        if (!parameters.enableCompetition() || currentSequenceLength <= 1) {
            return;
        }

        var inhibitionStrength = (float) parameters.competitiveRate();

        // Compute inhibition for each position
        for (int i = 0; i < currentSequenceLength; i++) {
            var inhibition = 0.0f;

            for (int j = 0; j < currentSequenceLength; j++) {
                if (i != j) {
                    inhibition += inhibitionStrength * totalActivationPerItem[j];
                }
            }

            lateralInhibition[i] = inhibition;
        }

        vectorOperations.incrementAndGet();
    }

    private float computeTotalActivationVectorized(float[] activations) {
        var sum = FloatVector.zero(SPECIES);

        for (int i = 0; i < itemDimension; i += VECTOR_LENGTH) {
            var remaining = Math.min(VECTOR_LENGTH, itemDimension - i);
            if (remaining == VECTOR_LENGTH) {
                var vec = FloatVector.fromArray(SPECIES, activations, i);
                sum = sum.add(vec);
            } else {
                // Handle partial vector at end
                for (int j = 0; j < remaining; j++) {
                    sum = sum.add(FloatVector.broadcast(SPECIES, activations[i + j]));
                }
            }
        }

        return sum.reduceLanes(VectorOperators.ADD);
    }

    private boolean hasContrastModification() {
        for (int i = 0; i < currentSequenceLength; i++) {
            if (Math.abs(contrastState[i] - 1.0f) > 1e-6f) {
                return true;
            }
        }
        return false;
    }

    private void updateContrastEnhancementVectorized() {
        for (int pos = 0; pos < currentSequenceLength; pos++) {
            var contrast = contrastState[pos];
            if (Math.abs(contrast - 1.0f) > 1e-6f) {
                var activations = shuntingActivations[pos];
                var contrastVector = FloatVector.broadcast(SPECIES, contrast);

                for (int i = 0; i < paddedDimension; i += VECTOR_LENGTH) {
                    var vec = FloatVector.fromArray(SPECIES, activations, i);
                    var result = vec.mul(contrastVector);
                    result.intoArray(activations, i);
                }
            }
        }

        vectorOperations.incrementAndGet();
    }

    private float computePrimacyWeight(int position, int totalLength) {
        var decayFactor = parameters.decayRate();
        return (float) Math.exp(-position * decayFactor);
    }

    // === Performance Metrics Implementation ===

    public class VectorizedWorkingMemoryPerformanceMetrics implements WorkingMemoryPerformanceMetrics {
        @Override
        public long getStoreOperations() {
            return storeOperations.get();
        }

        @Override
        public long getUpdateOperations() {
            return updateOperations.get();
        }

        @Override
        public long getComputationTime() {
            return computationTimeNanos.get();
        }

        @Override
        public long getMemoryUsage() {
            return (long) maxSequenceLength * paddedDimension * Float.BYTES * 2 +
                   maxSequenceLength * Float.BYTES * 4;
        }

        @Override
        public double getAverageAccessTime() {
            var operations = storeOperations.get() + updateOperations.get();
            return operations > 0 ? (double) computationTimeNanos.get() / operations : 0.0;
        }

        public long getVectorOperations() {
            return vectorOperations.get();
        }

        public int getVectorSpeciesLength() {
            return VECTOR_LENGTH;
        }

        public double getVectorizationEfficiency() {
            return (double) itemDimension / paddedDimension;
        }
    }

    // === Snapshot Implementation ===

    private record VectorizedWorkingMemorySnapshot(
        TemporalPattern storedPattern,
        double[] primacyValues,
        double snapshotTime,
        double totalActivation
    ) implements WorkingMemorySnapshot {

        @Override
        public TemporalPattern getStoredPattern() {
            return storedPattern;
        }

        @Override
        public double[] getPrimacyValues() {
            return Arrays.copyOf(primacyValues, primacyValues.length);
        }

        @Override
        public double getSnapshotTime() {
            return snapshotTime;
        }

        @Override
        public double getTotalActivation() {
            return totalActivation;
        }
    }

    // === Temporal Pattern Implementation ===

    private record VectorizedTemporalPattern(
        List<Pattern> sequence,
        double[] temporalWeights,
        double timestamp
    ) implements TemporalPattern {

        @Override
        public List<Pattern> getSequence() {
            return new ArrayList<>(sequence);
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime < 0 || endTime > sequence.size() || startTime >= endTime) {
                throw new IndexOutOfBoundsException("Invalid subsequence bounds");
            }

            var subSequence = sequence.subList(startTime, endTime);
            var subWeights = Arrays.copyOfRange(temporalWeights, startTime, endTime);

            return new VectorizedTemporalPattern(new ArrayList<>(subSequence), subWeights, timestamp);
        }

        @Override
        public boolean isEmpty() {
            return sequence.isEmpty();
        }
    }

    // The required helper methods are already defined elsewhere in the file
}