package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.temporal.core.State;

/**
 * State of the working memory system.
 */
public class WorkingMemoryState extends State {
    private final double[][] items;
    private final double[] primacyWeights;
    private final double[] recencyWeights;
    private final int currentPosition;
    private final int itemCount;

    public WorkingMemoryState(
        double[][] items,
        double[] primacyWeights,
        double[] recencyWeights,
        int currentPosition,
        int itemCount
    ) {
        this.items = items;
        this.primacyWeights = primacyWeights;
        this.recencyWeights = recencyWeights;
        this.currentPosition = currentPosition;
        this.itemCount = itemCount;
    }

    @Override
    public double[] toArray() {
        // Flatten the state into a vector
        int totalSize = items.length * items[0].length + primacyWeights.length + recencyWeights.length + 2;
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

    @Override
    public State fromArray(double[] vector) {
        // Reconstruct state from vector
        int capacity = items.length;
        int dimension = items[0].length;
        var newItems = new double[capacity][dimension];
        var newPrimacy = new double[primacyWeights.length];
        var newRecency = new double[recencyWeights.length];

        int idx = 0;

        // Reconstruct items
        for (int i = 0; i < capacity; i++) {
            System.arraycopy(vector, idx, newItems[i], 0, dimension);
            idx += dimension;
        }

        // Reconstruct weights
        System.arraycopy(vector, idx, newPrimacy, 0, newPrimacy.length);
        idx += newPrimacy.length;
        System.arraycopy(vector, idx, newRecency, 0, newRecency.length);
        idx += newRecency.length;

        // Reconstruct position and count
        int newPosition = (int) vector[idx++];
        int newCount = (int) vector[idx];

        return new WorkingMemoryState(newItems, newPrimacy, newRecency, newPosition, newCount);
    }

    @Override
    public int dimension() {
        return items.length * items[0].length + primacyWeights.length + recencyWeights.length + 2;
    }

    @Override
    public State add(State other) {
        if (!(other instanceof WorkingMemoryState otherState)) {
            throw new IllegalArgumentException("Can only add WorkingMemoryState to WorkingMemoryState");
        }
        var result = copy();
        var thisArray = this.toArray();
        var otherArray = otherState.toArray();
        var sumArray = new double[thisArray.length];
        for (int i = 0; i < thisArray.length; i++) {
            sumArray[i] = thisArray[i] + otherArray[i];
        }
        return result.fromArray(sumArray);
    }

    @Override
    public State scale(double scalar) {
        var thisArray = this.toArray();
        var scaledArray = new double[thisArray.length];
        for (int i = 0; i < thisArray.length; i++) {
            scaledArray[i] = thisArray[i] * scalar;
        }
        return fromArray(scaledArray);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof WorkingMemoryState otherState)) {
            throw new IllegalArgumentException("Can only compute distance to WorkingMemoryState");
        }
        var thisArray = this.toArray();
        var otherArray = otherState.toArray();
        double sum = 0.0;
        for (int i = 0; i < thisArray.length; i++) {
            double diff = thisArray[i] - otherArray[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    @Override
    public State copy() {
        return new WorkingMemoryState(
            items.clone(),
            primacyWeights.clone(),
            recencyWeights.clone(),
            currentPosition,
            itemCount
        );
    }

    // Getters
    public double[][] getItems() {
        return items;
    }

    public double[] getPrimacyWeights() {
        return primacyWeights;
    }

    public double[] getRecencyWeights() {
        return recencyWeights;
    }

    public int getCurrentPosition() {
        return currentPosition;
    }

    public int getItemCount() {
        return itemCount;
    }

    /**
     * Get capacity (maximum number of items).
     */
    public int getCapacity() {
        return items.length;
    }
}