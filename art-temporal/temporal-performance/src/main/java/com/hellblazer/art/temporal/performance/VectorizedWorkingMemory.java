package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.temporal.memory.WorkingMemoryParameters;
import com.hellblazer.art.temporal.memory.TemporalPattern;
import jdk.incubator.vector.*;
import org.jctools.queues.SpscArrayQueue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * High-performance vectorized working memory implementation.
 * Uses SIMD operations and lock-free data structures.
 */
public class VectorizedWorkingMemory {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final WorkingMemoryParameters parameters;
    private final int capacity;
    private final int vectorLength;
    private final int loopBound;

    // Lock-free queue for efficient storage
    private final SpscArrayQueue<MemoryItem> itemQueue;
    private final List<MemoryItem> storedSequence;

    // Lock for thread-safe operations
    private final ReentrantLock lock = new ReentrantLock();

    // Vectorized state arrays
    private double[] activations;
    private double[] transmitters;
    private double[] primacyWeights;

    // Pre-computed values
    private double[] decayRates;
    private double[] distanceMatrix;

    private double currentTime;

    public VectorizedWorkingMemory(WorkingMemoryParameters parameters) {
        this.parameters = parameters;
        this.capacity = parameters.getCapacity();
        this.vectorLength = SPECIES.length();
        this.loopBound = SPECIES.loopBound(capacity);

        this.itemQueue = new SpscArrayQueue<>(capacity * 2);
        this.storedSequence = new ArrayList<>(capacity);

        this.activations = new double[capacity];
        this.transmitters = new double[capacity];
        this.primacyWeights = new double[capacity];
        this.decayRates = new double[capacity];

        // Initialize transmitters
        var vBaseline = DoubleVector.broadcast(SPECIES, parameters.getTransmitterBaseline());
        for (int i = 0; i < loopBound; i += vectorLength) {
            vBaseline.intoArray(transmitters, i);
        }
        for (int i = loopBound; i < capacity; i++) {
            transmitters[i] = parameters.getTransmitterBaseline();
        }

        precomputeDecayRates();
        this.currentTime = 0.0;
    }

    /**
     * Store item with vectorized primacy computation.
     */
    public void storeItem(double[] pattern, double timestamp) {
        lock.lock();
        try {
            // Check capacity
            if (storedSequence.size() >= capacity) {
                removeOldestItem();
            }

            var item = new MemoryItem(pattern, timestamp);
            storedSequence.add(item);
            itemQueue.offer(item);

            // Update primacy weights (vectorized)
            updatePrimacyVectorized();

            // Initialize activation
            int index = storedSequence.size() - 1;
            if (index >= 0 && index < activations.length) {
                activations[index] = parameters.getInitialActivation();
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * Vectorized dynamics evolution.
     */
    public void evolveDynamics(double deltaT) {
        lock.lock();
        try {
            int size = storedSequence.size();
            if (size == 0) return;

            // Prepare bounds for vectorization
            int bound = SPECIES.loopBound(size);

            // Compute primacy-weighted activations (vectorized)
            computeActivationsVectorized(bound, size, deltaT);

            // Update transmitters (vectorized)
            updateTransmittersVectorized(bound, size, deltaT);

            currentTime += deltaT;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Vectorized activation computation.
     */
    private void computeActivationsVectorized(int bound, int size, double deltaT) {
        // Temporary arrays for computation
        double[] newActivations = new double[size];

        int i = 0;
        // Main vectorized loop - ensure we don't exceed array bounds
        // Must check against actual array length, not just logical size!
        int vectorBound = Math.min(size, activations.length - vectorLength + 1);
        for (; i < vectorBound && i + vectorLength <= size; i += vectorLength) {
            var vAct = DoubleVector.fromArray(SPECIES, activations, i);
            var vPrimacy = DoubleVector.fromArray(SPECIES, primacyWeights, i);
            var vTrans = DoubleVector.fromArray(SPECIES, transmitters, i);
            var vDecay = DoubleVector.fromArray(SPECIES, decayRates, i);

            // Compute lateral competition (simplified for vectorization)
            var vCompetition = computeCompetitionVectorized(i, size);

            // Shunting equation: dx/dt = -Ax + (B-x)P*T - xC
            var vDerivative = vAct.mul(vDecay).neg()
                .add(vPrimacy.mul(vTrans).mul(1.0 - vAct.reduceLanes(VectorOperators.ADD) / vectorLength))
                .sub(vAct.mul(vCompetition));

            // Euler integration
            var vNew = vAct.add(vDerivative.mul(deltaT));

            // Apply bounds
            vNew = vNew.max(0.0).min(1.0);
            vNew.intoArray(newActivations, i);
        }

        // Scalar tail
        for (; i < size; i++) {
            double competition = computeCompetition(i, size);
            double derivative = -decayRates[i] * activations[i] +
                              (1.0 - activations[i]) * primacyWeights[i] * transmitters[i] -
                              activations[i] * competition;
            newActivations[i] = Math.max(0, Math.min(1, activations[i] + deltaT * derivative));
        }

        // Copy back
        System.arraycopy(newActivations, 0, activations, 0, size);
    }

    /**
     * Vectorized transmitter update.
     */
    private void updateTransmittersVectorized(int bound, int size, double deltaT) {
        double recovery = parameters.getTransmitterRecovery();
        double depletion = parameters.getTransmitterDepletion();

        int i = 0;
        // Must check against actual array length, not just logical size!
        int vectorBound = Math.min(size, transmitters.length - vectorLength + 1);
        for (; i < vectorBound && i + vectorLength <= size; i += vectorLength) {
            var vTrans = DoubleVector.fromArray(SPECIES, transmitters, i);
            var vAct = DoubleVector.fromArray(SPECIES, activations, i);

            // dT/dt = ε(1-T) - T*A*(λ + μ*A)
            var vRecovery = DoubleVector.broadcast(SPECIES, recovery)
                .mul(DoubleVector.broadcast(SPECIES, 1.0).sub(vTrans));
            var vDepletion = vTrans.mul(vAct).mul(depletion);

            var vDerivative = vRecovery.sub(vDepletion);
            var vNew = vTrans.add(vDerivative.mul(deltaT));

            // Bounds
            vNew = vNew.max(0.0).min(1.0);
            vNew.intoArray(transmitters, i);
        }

        // Scalar tail
        for (; i < size; i++) {
            double derivative = recovery * (1.0 - transmitters[i]) -
                              transmitters[i] * activations[i] * depletion;
            transmitters[i] = Math.max(0, Math.min(1, transmitters[i] + deltaT * derivative));
        }
    }

    /**
     * Vectorized primacy weight update.
     */
    private void updatePrimacyVectorized() {
        int size = storedSequence.size();
        if (size == 0) return;

        double gamma = parameters.getPrimacyGradient();
        double recencyBoost = parameters.getRecencyBoost();

        // Compute primacy with exponential decay
        var maxIndex = Math.min(size, primacyWeights.length);
        for (int i = 0; i < maxIndex; i++) {
            primacyWeights[i] = Math.exp(-gamma * i);

            // Add recency boost for last few items
            if (i >= maxIndex - 2) {
                primacyWeights[i] *= (1.0 + recencyBoost);
            }
        }

        // Normalize weights (vectorized)
        double sum = 0.0;
        int i = 0;
        int bound = SPECIES.loopBound(maxIndex);

        for (; i < bound; i += vectorLength) {
            var vWeight = DoubleVector.fromArray(SPECIES, primacyWeights, i);
            sum += vWeight.reduceLanes(VectorOperators.ADD);
        }
        for (; i < maxIndex; i++) {
            sum += primacyWeights[i];
        }

        if (sum > 0) {
            var vNorm = DoubleVector.broadcast(SPECIES, 1.0 / sum);
            i = 0;
            for (; i < bound; i += vectorLength) {
                var vWeight = DoubleVector.fromArray(SPECIES, primacyWeights, i);
                vWeight = vWeight.mul(vNorm);
                vWeight.intoArray(primacyWeights, i);
            }
            for (; i < maxIndex; i++) {
                primacyWeights[i] /= sum;
            }
        }
    }

    /**
     * Vectorized competition computation.
     */
    private DoubleVector computeCompetitionVectorized(int start, int size) {
        var vSum = DoubleVector.zero(SPECIES);

        var maxIndex = Math.min(size, activations.length);
        for (int j = 0; j < maxIndex; j++) {
            if (j >= start && j < start + vectorLength) continue;  // Skip self

            double actJ = activations[j];
            if (actJ > 0.01) {
                // Simplified competition kernel
                var vDist = DoubleVector.broadcast(SPECIES, Math.abs(j - start - vectorLength/2));
                var vKernel = vDist.div(size).neg().add(1.0).max(0.0);
                vSum = vSum.add(vKernel.mul(actJ));
            }
        }

        return vSum.mul(parameters.getCompetitionStrength());
    }

    /**
     * Scalar competition computation for tail.
     */
    private double computeCompetition(int i, int size) {
        double sum = 0.0;
        for (int j = 0; j < size; j++) {
            if (i == j) continue;
            double distance = Math.abs(i - j) / (double) size;
            double kernel = Math.max(0, 1.0 - distance);
            sum += kernel * activations[j];
        }
        return sum * parameters.getCompetitionStrength();
    }

    /**
     * Get temporal pattern with vectorized computation.
     */
    public TemporalPattern getTemporalPattern() {
        lock.lock();
        try {
            // Create a snapshot to avoid concurrent modification issues
            List<MemoryItem> snapshot = new ArrayList<>(storedSequence);
            int size = snapshot.size();
            List<double[]> patterns = new ArrayList<>();
            List<Double> weights = new ArrayList<>();

            // Ensure we don't exceed bounds of any array/list
            var maxIndex = Math.min(size, Math.min(activations.length, transmitters.length));

            for (int i = 0; i < maxIndex; i++) {
                var item = snapshot.get(i);
                if (item != null) {  // Extra null check for safety
                    patterns.add(item.pattern());
                    weights.add(activations[i] * transmitters[i]);
                }
            }

            double primacyStrength = computePrimacyStrengthVectorized();
            return new TemporalPattern(patterns, weights, primacyStrength);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Vectorized primacy strength computation.
     */
    private double computePrimacyStrengthVectorized() {
        int size = storedSequence.size();
        if (size == 0) return 0.0;

        double strength = 0.0;
        var maxIndex = Math.min(size, Math.min(primacyWeights.length, activations.length));
        int bound = SPECIES.loopBound(maxIndex);

        int i = 0;
        // Conservative bound: stop if we can't fit a full vector
        for (; i + vectorLength <= maxIndex; i += vectorLength) {
            var vPrimacy = DoubleVector.fromArray(SPECIES, primacyWeights, i);
            var vAct = DoubleVector.fromArray(SPECIES, activations, i);
            var vProduct = vPrimacy.mul(vAct);
            strength += vProduct.reduceLanes(VectorOperators.ADD);
        }

        for (; i < maxIndex; i++) {
            strength += primacyWeights[i] * activations[i];
        }

        return strength;
    }

    private void precomputeDecayRates() {
        double baseDecay = parameters.getActivationDecay();
        for (int i = 0; i < capacity; i++) {
            decayRates[i] = baseDecay * (1.0 + 0.1 * Math.random());  // Slight variation
        }
    }

    private void removeOldestItem() {
        if (!storedSequence.isEmpty()) {
            storedSequence.remove(0);
            // Shift arrays
            System.arraycopy(activations, 1, activations, 0, capacity - 1);
            System.arraycopy(transmitters, 1, transmitters, 0, capacity - 1);
            activations[capacity - 1] = 0.0;
            transmitters[capacity - 1] = parameters.getTransmitterBaseline();
        }
    }

    public void reset() {
        storedSequence.clear();
        itemQueue.clear();

        // Reset arrays (vectorized)
        var vZero = DoubleVector.zero(SPECIES);
        var vBaseline = DoubleVector.broadcast(SPECIES, parameters.getTransmitterBaseline());

        for (int i = 0; i < loopBound; i += vectorLength) {
            vZero.intoArray(activations, i);
            vBaseline.intoArray(transmitters, i);
            vZero.intoArray(primacyWeights, i);
        }

        for (int i = loopBound; i < capacity; i++) {
            activations[i] = 0.0;
            transmitters[i] = parameters.getTransmitterBaseline();
            primacyWeights[i] = 0.0;
        }

        currentTime = 0.0;
    }

    /**
     * Get current state for compatibility.
     */
    public com.hellblazer.art.temporal.memory.WorkingMemoryState getState() {
        // Create a simplified state representation
        var items = new double[capacity][parameters.getItemDimension()];
        for (int i = 0; i < Math.min(storedSequence.size(), capacity); i++) {
            var item = storedSequence.get(i);
            System.arraycopy(item.pattern(), 0, items[i], 0,
                Math.min(item.pattern().length, parameters.getItemDimension()));
        }

        var primacyWeights = new double[capacity];
        var recencyWeights = new double[capacity];
        for (int i = 0; i < capacity; i++) {
            primacyWeights[i] = Math.exp(-parameters.getPrimacyGradient() * i);
            recencyWeights[i] = Math.exp(-parameters.getRecencyGradient() * (capacity - 1 - i));
        }

        return new com.hellblazer.art.temporal.memory.WorkingMemoryState(
            items, primacyWeights, recencyWeights, storedSequence.size(), storedSequence.size()
        );
    }

    // Getters
    public int getItemCount() {
        return storedSequence.size();
    }

    public double[] getActivations() {
        return activations.clone();
    }

    public double[] getTransmitters() {
        return transmitters.clone();
    }

    public WorkingMemoryParameters getParameters() {
        return parameters;
    }

    private record MemoryItem(double[] pattern, double timestamp) {}
}