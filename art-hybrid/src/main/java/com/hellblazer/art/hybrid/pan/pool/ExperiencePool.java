package com.hellblazer.art.hybrid.pan.pool;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Experience replay pool for continual learning in PAN.
 * Stores past experiences for periodic replay to prevent catastrophic forgetting.
 */
public class ExperiencePool {

    private final int maxSize;
    private final List<Experience> experiences;
    private long totalExperiences = 0;

    public ExperiencePool(int maxSize) {
        this.maxSize = maxSize;
        this.experiences = new ArrayList<>(maxSize);
    }

    /**
     * Add new experience to the pool
     */
    public synchronized void add(float[] pattern, int category) {
        totalExperiences++;

        var experience = new Experience(pattern.clone(), category, System.currentTimeMillis());

        if (experiences.size() >= maxSize) {
            // Replace random old experience (reservoir sampling)
            Random rand = ThreadLocalRandom.current();
            int replaceIdx = rand.nextInt(experiences.size());
            experiences.set(replaceIdx, experience);
        } else {
            experiences.add(experience);
        }
    }

    /**
     * Sample a batch of experiences for replay
     */
    public synchronized Experience[] sampleBatch(int batchSize) {
        if (experiences.isEmpty()) {
            return new Experience[0];
        }

        int actualBatchSize = Math.min(batchSize, experiences.size());
        Experience[] batch = new Experience[actualBatchSize];
        Random rand = ThreadLocalRandom.current();

        // Sample with replacement
        for (int i = 0; i < actualBatchSize; i++) {
            batch[i] = experiences.get(rand.nextInt(experiences.size()));
        }

        return batch;
    }

    /**
     * Get memory usage estimate
     */
    public long getMemoryUsage() {
        if (experiences.isEmpty()) {
            return 0;
        }

        // Estimate based on first experience
        var first = experiences.get(0);
        long singleSize = first.pattern.length * 4 + 8 + 4;  // pattern + timestamp + category
        return singleSize * experiences.size();
    }

    /**
     * Clear the pool
     */
    public synchronized void clear() {
        experiences.clear();
    }

    /**
     * Get pool statistics
     */
    public PoolStats getStats() {
        return new PoolStats(
            experiences.size(),
            maxSize,
            totalExperiences,
            getMemoryUsage()
        );
    }

    /**
     * Single experience entry
     */
    public static class Experience {
        public final float[] pattern;
        public final int category;
        public final long timestamp;

        public Experience(float[] pattern, int category, long timestamp) {
            this.pattern = pattern;
            this.category = category;
            this.timestamp = timestamp;
        }
    }

    /**
     * Pool statistics
     */
    public static class PoolStats {
        public final int currentSize;
        public final int maxSize;
        public final long totalExperiences;
        public final long memoryUsage;

        public PoolStats(int currentSize, int maxSize, long totalExperiences, long memoryUsage) {
            this.currentSize = currentSize;
            this.maxSize = maxSize;
            this.totalExperiences = totalExperiences;
            this.memoryUsage = memoryUsage;
        }
    }
}