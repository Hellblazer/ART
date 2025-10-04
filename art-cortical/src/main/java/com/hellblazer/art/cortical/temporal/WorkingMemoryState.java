package com.hellblazer.art.cortical.temporal;

/**
 * Immutable state snapshot of working memory system.
 *
 * <p>Captures complete state for:
 * <ul>
 *   <li>Stored item patterns</li>
 *   <li>Primacy-weighted activations</li>
 *   <li>Recency-weighted activations</li>
 *   <li>Current encoding position</li>
 *   <li>Number of items in memory</li>
 * </ul>
 *
 * <p>This state representation supports:
 * <ul>
 *   <li>State inspection for analysis</li>
 *   <li>Debugging primacy/recency gradients</li>
 *   <li>Serialization/deserialization</li>
 *   <li>State comparison</li>
 * </ul>
 *
 * @param items 2D array of stored patterns [capacity][dimension]
 * @param primacyWeights Weight profile showing primacy gradient
 * @param recencyWeights Weight profile showing recency gradient
 * @param currentPosition Current encoding position (0 to capacity-1)
 * @param itemCount Number of items actually stored
 *
 * @author Migrated from art-temporal/temporal-memory to art-cortical (Phase 2)
 */
public record WorkingMemoryState(
    double[][] items,
    double[] primacyWeights,
    double[] recencyWeights,
    int currentPosition,
    int itemCount
) {
    /**
     * Compact constructor with validation.
     */
    public WorkingMemoryState {
        if (items == null || primacyWeights == null || recencyWeights == null) {
            throw new IllegalArgumentException("State arrays cannot be null");
        }
        if (items.length != primacyWeights.length || items.length != recencyWeights.length) {
            throw new IllegalArgumentException(
                "Inconsistent dimensions: items=" + items.length +
                ", primacy=" + primacyWeights.length +
                ", recency=" + recencyWeights.length
            );
        }
        if (currentPosition < 0 || currentPosition > items.length) {
            throw new IllegalArgumentException(
                "Current position out of range: " + currentPosition +
                " (capacity=" + items.length + ")"
            );
        }
        if (itemCount < 0 || itemCount > items.length) {
            throw new IllegalArgumentException(
                "Item count out of range: " + itemCount +
                " (capacity=" + items.length + ")"
            );
        }
    }

    /**
     * Get memory capacity.
     */
    public int getCapacity() {
        return items.length;
    }

    /**
     * Flatten state into a single vector for analysis.
     * Format: [items_flat, primacyWeights, recencyWeights, position, count]
     */
    public double[] toArray() {
        if (items.length == 0 || items[0].length == 0) {
            return new double[0];
        }

        int itemDimension = items[0].length;
        int totalSize = items.length * itemDimension + primacyWeights.length +
                       recencyWeights.length + 2;
        var vector = new double[totalSize];
        int idx = 0;

        // Flatten items
        for (var item : items) {
            System.arraycopy(item, 0, vector, idx, item.length);
            idx += item.length;
        }

        // Add weights
        System.arraycopy(primacyWeights, 0, vector, idx, primacyWeights.length);
        idx += primacyWeights.length;
        System.arraycopy(recencyWeights, 0, vector, idx, recencyWeights.length);
        idx += recencyWeights.length;

        // Add position and count
        vector[idx++] = currentPosition;
        vector[idx] = itemCount;

        return vector;
    }

    /**
     * Reconstruct state from flattened vector.
     */
    public static WorkingMemoryState fromArray(double[] vector, int capacity, int dimension) {
        var newItems = new double[capacity][dimension];
        var newPrimacy = new double[capacity];
        var newRecency = new double[capacity];

        int idx = 0;

        // Reconstruct items
        for (int i = 0; i < capacity; i++) {
            System.arraycopy(vector, idx, newItems[i], 0, dimension);
            idx += dimension;
        }

        // Reconstruct weights
        System.arraycopy(vector, idx, newPrimacy, 0, capacity);
        idx += capacity;
        System.arraycopy(vector, idx, newRecency, 0, capacity);
        idx += capacity;

        // Reconstruct position and count
        int newPosition = (int) vector[idx++];
        int newCount = (int) vector[idx];

        return new WorkingMemoryState(newItems, newPrimacy, newRecency, newPosition, newCount);
    }

    /**
     * Compute distance between two states (Euclidean).
     */
    public double distance(WorkingMemoryState other) {
        var thisArray = this.toArray();
        var otherArray = other.toArray();

        if (thisArray.length != otherArray.length) {
            throw new IllegalArgumentException("States have incompatible dimensions");
        }

        double sum = 0.0;
        for (int i = 0; i < thisArray.length; i++) {
            double diff = thisArray[i] - otherArray[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Create deep copy of state.
     */
    public WorkingMemoryState copy() {
        var itemsCopy = new double[items.length][];
        for (int i = 0; i < items.length; i++) {
            itemsCopy[i] = items[i].clone();
        }
        return new WorkingMemoryState(
            itemsCopy,
            primacyWeights.clone(),
            recencyWeights.clone(),
            currentPosition,
            itemCount
        );
    }

    /**
     * Get combined pattern from all stored items, weighted by primacy.
     * Returns the weighted average of all items in memory.
     */
    public double[] getCombinedPattern() {
        if (itemCount == 0 || items.length == 0 || items[0].length == 0) {
            return new double[0];
        }

        var dimension = items[0].length;
        var combined = new double[dimension];
        var totalWeight = 0.0;

        for (int i = 0; i < itemCount; i++) {
            var weight = primacyWeights[i];
            for (int j = 0; j < dimension; j++) {
                combined[j] += items[i][j] * weight;
            }
            totalWeight += weight;
        }

        // Normalize by total weight
        if (totalWeight > 0) {
            for (int j = 0; j < dimension; j++) {
                combined[j] /= totalWeight;
            }
        }

        return combined;
    }

    /**
     * Get the most recent item stored.
     */
    public double[] getMostRecentItem() {
        if (itemCount == 0) {
            return new double[0];
        }
        var recentIndex = (currentPosition - 1 + items.length) % items.length;
        return items[recentIndex].clone();
    }
}
