package com.art.textgen.core;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.stream.Collectors;

/**
 * Item-Order-Rank Working Memory based on Grossberg's neural dynamics
 * Maintains primacy gradient where X₁ > X₂ > X₃ > X₄ represents temporal order
 */
public class WorkingMemory<T> {
    private static final int DEFAULT_CAPACITY = 7;
    private static final double DECAY_RATE = 0.95;
    private static final double LATERAL_INHIBITION_STRENGTH = 0.3;
    
    private final int capacity;
    private final Deque<MemoryItem<T>> items;
    private final double tau; // Time constant
    private double currentTime;
    
    public static class MemoryItem<T> {
        public final T content;
        public double activation;
        public final double timestamp;
        public double resonanceStrength;
        
        public MemoryItem(T content, double activation, double timestamp) {
            this.content = content;
            this.activation = activation;
            this.timestamp = timestamp;
            this.resonanceStrength = 1.0;
        }
        
        public void decay(double rate) {
            activation *= rate;
        }
        
        public void updateActivation(double dt, double input) {
            // Grossberg's shunting equation: dx/dt = -Ax + (B-x)I
            double A = 1.0;
            double B = 1.0;
            activation += dt * (-A * activation + (B - activation) * input);
            activation = Math.max(0, Math.min(1, activation)); // Bound [0,1]
        }
    }
    
    public WorkingMemory(int capacity, double tau) {
        this.capacity = capacity;
        this.tau = tau;
        this.items = new ConcurrentLinkedDeque<>();
        this.currentTime = 0.0;
    }
    
    public WorkingMemory() {
        this(DEFAULT_CAPACITY, 1.0);
    }
    
    public void addItem(T item, double activationStrength) {
        // Apply decay to existing items
        items.forEach(memItem -> memItem.decay(DECAY_RATE));
        
        // Add new item with primacy
        MemoryItem<T> newItem = new MemoryItem<>(item, activationStrength, currentTime);
        items.addFirst(newItem);        
        // Apply lateral inhibition to maintain distinctness
        applyLateralInhibition();
        
        // Remove oldest if over capacity
        while (items.size() > capacity) {
            items.removeLast();
        }
        
        currentTime += 1.0;
    }
    
    private void applyLateralInhibition() {
        List<MemoryItem<T>> itemList = new ArrayList<>(items);
        
        for (int i = 0; i < itemList.size(); i++) {
            double inhibition = 0.0;
            for (int j = 0; j < itemList.size(); j++) {
                if (i != j) {
                    double distance = Math.abs(i - j);
                    double weight = LATERAL_INHIBITION_STRENGTH / (1 + distance);
                    inhibition += weight * itemList.get(j).activation;
                }
            }
            
            // Apply inhibition but don't let activation go below a small threshold to maintain item
            double newActivation = itemList.get(i).activation - inhibition;
            itemList.get(i).activation = Math.max(0.01, newActivation); // Keep minimum activation
        }
    }    
    public List<T> getRecentItems(int n) {
        return items.stream()
            .limit(n)
            .map(item -> item.content)
            .collect(Collectors.toList());
    }
    
    public double[] getActivationGradient() {
        return items.stream()
            .mapToDouble(item -> item.activation)
            .toArray();
    }
    
    public boolean hasCapacity() {
        return items.size() < capacity;
    }
    
    public void clear() {
        items.clear();
    }
    
    public WorkingMemoryState<T> compress() {
        return new WorkingMemoryState<>(new ArrayList<>(items), currentTime);
    }
    
    public static class WorkingMemoryState<T> {
        public final List<MemoryItem<T>> items;
        public final double timestamp;
        
        public WorkingMemoryState(List<MemoryItem<T>> items, double timestamp) {
            this.items = new ArrayList<>(items);
            this.timestamp = timestamp;
        }
    }
}