package com.hellblazer.art.hybrid.pan.memory;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Experience replay buffer for continual learning.
 * Stores past experiences and samples them for replay training.
 */
public class ExperienceReplayBuffer implements AutoCloseable {

    private final int maxSize;
    private final int batchSize;
    private final List<Experience> buffer;
    private final Random random;
    private long totalExperiences = 0;
    private volatile boolean closed = false;

    public ExperienceReplayBuffer(int maxSize, int batchSize) {
        this.maxSize = maxSize;
        this.batchSize = batchSize;
        this.buffer = new ArrayList<>(maxSize);
        this.random = ThreadLocalRandom.current();
    }

    /**
     * Add experience to the buffer.
     */
    public synchronized void addExperience(Pattern features, Pattern target,
                                          BPARTWeight nodeState, double reward) {
        if (closed) return;

        totalExperiences++;

        var experience = new Experience(features, target, nodeState, reward, totalExperiences);

        if (buffer.size() >= maxSize) {
            // Reservoir sampling for uniform distribution
            int replaceIdx = random.nextInt(buffer.size());
            buffer.set(replaceIdx, experience);
        } else {
            buffer.add(experience);
        }
    }

    /**
     * Sample a batch of experiences for replay.
     */
    public synchronized Experience[] sampleBatch() {
        if (buffer.isEmpty()) {
            return new Experience[0];
        }

        int actualBatchSize = Math.min(batchSize, buffer.size());
        Experience[] batch = new Experience[actualBatchSize];

        // Sample with replacement
        for (int i = 0; i < actualBatchSize; i++) {
            batch[i] = buffer.get(random.nextInt(buffer.size()));
        }

        return batch;
    }

    /**
     * Get prioritized batch (higher reward experiences more likely).
     */
    public synchronized Experience[] samplePrioritizedBatch() {
        if (buffer.isEmpty()) {
            return new Experience[0];
        }

        // Calculate priorities based on reward
        double[] priorities = new double[buffer.size()];
        double sum = 0;

        for (int i = 0; i < buffer.size(); i++) {
            double priority = Math.abs(buffer.get(i).reward) + 0.01;  // Add small epsilon
            priorities[i] = priority;
            sum += priority;
        }

        // Normalize
        for (int i = 0; i < priorities.length; i++) {
            priorities[i] /= sum;
        }

        // Sample based on priorities
        int actualBatchSize = Math.min(batchSize, buffer.size());
        Experience[] batch = new Experience[actualBatchSize];

        for (int i = 0; i < actualBatchSize; i++) {
            batch[i] = buffer.get(sampleFromDistribution(priorities));
        }

        return batch;
    }

    /**
     * Sample index from probability distribution.
     */
    private int sampleFromDistribution(double[] probabilities) {
        double r = random.nextDouble();
        double cumulative = 0;

        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return i;
            }
        }

        return probabilities.length - 1;
    }

    /**
     * Get current buffer size.
     */
    public int size() {
        return buffer.size();
    }

    /**
     * Get total experiences seen.
     */
    public long getTotalExperiences() {
        return totalExperiences;
    }

    /**
     * Estimate memory usage.
     */
    public long estimateMemoryUsage() {
        // Rough estimate: each experience contains patterns and weight
        return buffer.size() * 4096L;  // ~4KB per experience
    }

    /**
     * Clear the buffer.
     */
    public synchronized void clear() {
        buffer.clear();
    }

    @Override
    public void close() {
        closed = true;
        buffer.clear();
    }

    /**
     * Single experience record.
     */
    public record Experience(
        Pattern features,
        Pattern target,
        BPARTWeight nodeState,
        double reward,
        long sequenceNumber
    ) {
        /**
         * Check if this is a supervised experience.
         */
        public boolean isSupervised() {
            return target != null;
        }

        /**
         * Get age of this experience.
         */
        public long getAge(long currentSequence) {
            return currentSequence - sequenceNumber;
        }
    }
}