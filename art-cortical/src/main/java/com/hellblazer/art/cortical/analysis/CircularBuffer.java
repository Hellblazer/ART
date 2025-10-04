package com.hellblazer.art.cortical.analysis;

import java.util.Arrays;

/**
 * Fixed-size circular buffer for time-series data.
 *
 * <p>Efficiently stores the most recent N samples by overwriting oldest data.
 * Used for maintaining activation history windows for oscillation analysis.
 *
 * <h2>Design</h2>
 * <ul>
 *   <li>Fixed capacity - no dynamic resizing</li>
 *   <li>Constant-time add operation: O(1)</li>
 *   <li>Automatic oldest-data eviction</li>
 *   <li>Sequential data retrieval for FFT processing</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create buffer for 256 activation snapshots
 * var buffer = new CircularBuffer<double[]>(256);
 *
 * // Add activation patterns each timestep
 * for (int t = 0; t < 1000; t++) {
 *     double[] activation = layer.getActivation();
 *     buffer.add(activation);
 * }
 *
 * // When full, analyze oscillations
 * if (buffer.isFull()) {
 *     var history = buffer.toArray();
 *     var metrics = analyzer.analyze(history);
 * }
 * }</pre>
 *
 * @param <T> Type of elements stored
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class CircularBuffer<T> {

    private final Object[] buffer;
    private final int capacity;
    private int writeIndex;
    private int size;

    /**
     * Create circular buffer with specified capacity.
     *
     * @param capacity Maximum number of elements (must be > 0)
     * @throws IllegalArgumentException if capacity <= 0
     */
    public CircularBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException(
                "capacity must be positive, got: " + capacity
            );
        }
        this.capacity = capacity;
        this.buffer = new Object[capacity];
        this.writeIndex = 0;
        this.size = 0;
    }

    /**
     * Add element to buffer.
     *
     * <p>If buffer is full, overwrites oldest element.
     *
     * @param element Element to add (null allowed)
     */
    public void add(T element) {
        buffer[writeIndex] = element;
        writeIndex = (writeIndex + 1) % capacity;

        if (size < capacity) {
            size++;
        }
    }

    /**
     * Check if buffer is full (contains capacity elements).
     *
     * @return true if buffer has reached capacity
     */
    public boolean isFull() {
        return size == capacity;
    }

    /**
     * Get current number of elements in buffer.
     *
     * @return Number of elements [0, capacity]
     */
    public int size() {
        return size;
    }

    /**
     * Get buffer capacity.
     *
     * @return Maximum number of elements
     */
    public int capacity() {
        return capacity;
    }

    /**
     * Check if buffer is empty.
     *
     * @return true if buffer contains no elements
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Convert buffer to array in chronological order (oldest to newest).
     *
     * <p>For oscillation analysis, this provides time-series data ready for FFT.
     *
     * <p><b>Note</b>: Due to Java's type erasure, the returned array is actually Object[].
     * For double[][], cast using toDoubleArray2D() instead.
     *
     * @return Array of elements in insertion order
     */
    @SuppressWarnings("unchecked")
    public Object[] toArray() {
        var result = new Object[size];

        if (size < capacity) {
            // Buffer not yet full: elements are [0, size)
            System.arraycopy(buffer, 0, result, 0, size);
        } else {
            // Buffer full: oldest element is at writeIndex
            int oldestIndex = writeIndex;

            // Copy [oldestIndex, capacity) to result[0, ...]
            int firstPart = capacity - oldestIndex;
            System.arraycopy(buffer, oldestIndex, result, 0, firstPart);

            // Copy [0, oldestIndex) to result[firstPart, ...]
            if (oldestIndex > 0) {
                System.arraycopy(buffer, 0, result, firstPart, oldestIndex);
            }
        }

        return result;
    }

    /**
     * Convert buffer of double[] to double[][] in chronological order.
     *
     * <p>This is a type-safe method for the common case of double[] activation history.
     *
     * @return 2D array [timestep][neuron]
     * @throws ClassCastException if buffer doesn't contain double[]
     */
    public double[][] toDoubleArray2D() {
        var objArray = toArray();
        var result = new double[objArray.length][];

        for (int i = 0; i < objArray.length; i++) {
            result[i] = (double[]) objArray[i];
        }

        return result;
    }

    /**
     * Get element at specified index (0 = oldest, size-1 = newest).
     *
     * @param index Index in chronological order [0, size)
     * @return Element at index
     * @throws IndexOutOfBoundsException if index out of range
     */
    @SuppressWarnings("unchecked")
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                "Index %d out of range [0, %d)".formatted(index, size)
            );
        }

        int actualIndex;
        if (size < capacity) {
            // Not yet full: direct indexing
            actualIndex = index;
        } else {
            // Full: adjust for circular wrap
            actualIndex = (writeIndex + index) % capacity;
        }

        return (T) buffer[actualIndex];
    }

    /**
     * Get most recent element.
     *
     * @return Newest element
     * @throws IllegalStateException if buffer is empty
     */
    @SuppressWarnings("unchecked")
    public T getLast() {
        if (isEmpty()) {
            throw new IllegalStateException("Buffer is empty");
        }

        int lastIndex = (writeIndex - 1 + capacity) % capacity;
        return (T) buffer[lastIndex];
    }

    /**
     * Clear all elements from buffer.
     */
    public void clear() {
        Arrays.fill(buffer, null);
        writeIndex = 0;
        size = 0;
    }

    @Override
    public String toString() {
        return "CircularBuffer[size=%d, capacity=%d, full=%b]"
            .formatted(size, capacity, isFull());
    }
}
