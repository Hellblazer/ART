package com.art.textgen.memory;

import com.art.textgen.core.WorkingMemory;
import java.util.*;

/**
 * Multiple parallel working memories at different timescales
 * From 100ms phoneme level to 1-hour document level
 */
public class MultiTimescaleMemoryBank {
    
    public enum TimeScale {
        PHONEME("phoneme", 7, 10.0),      // ~10ms minimum for biological realism
        WORD("word", 7, 100.0),           // ~100ms (adjusted proportionally)
        PHRASE("phrase", 7, 1000.0),      // ~1s (adjusted proportionally)
        SENTENCE("sentence", 7, 10000.0), // ~10s (adjusted proportionally)
        PARAGRAPH("paragraph", 7, 60000.0), // ~1min (adjusted proportionally)
        DOCUMENT("document", 7, 3600000.0);  // ~1hour (adjusted proportionally)
        
        public final String name;
        public final int capacity;
        public final double tau;
        
        TimeScale(String name, int capacity, double tau) {
            this.name = name;
            this.capacity = capacity;
            this.tau = tau;
        }
    }
    
    private final Map<TimeScale, WorkingMemory<Object>> memories;    private final CompletionDetector completionDetector;
    
    public MultiTimescaleMemoryBank() {
        this.memories = new EnumMap<>(TimeScale.class);
        for (TimeScale scale : TimeScale.values()) {
            memories.put(scale, new WorkingMemory<>(scale.capacity, scale.tau));
        }
        this.completionDetector = new CompletionDetector();
    }
    
    public void update(Object token) {
        // Bottom-up activation
        memories.get(TimeScale.PHONEME).addItem(token, 1.0);
        
        // Check for completion and propagate up
        TimeScale[] scales = TimeScale.values();
        for (int i = 0; i < scales.length - 1; i++) {
            TimeScale current = scales[i];
            TimeScale next = scales[i + 1];
            
            if (completionDetector.detectCompletion(current, memories.get(current))) {
                Object completedUnit = extractUnit(memories.get(current));
                memories.get(next).addItem(completedUnit, 0.8);
            }
        }
    }
    
    private Object extractUnit(WorkingMemory<Object> memory) {
        List<Object> items = memory.getRecentItems(3);
        return new CompoundUnit(items);    }
    
    public Map<TimeScale, Prediction> generatePredictions() {
        Map<TimeScale, Prediction> predictions = new EnumMap<>(TimeScale.class);
        
        for (Map.Entry<TimeScale, WorkingMemory<Object>> entry : memories.entrySet()) {
            TimeScale scale = entry.getKey();
            WorkingMemory<Object> memory = entry.getValue();
            
            // Each level makes predictions at its timescale
            Object nextPrediction = predictNext(memory);
            double resonanceStrength = computeResonanceStrength(memory);
            
            predictions.put(scale, new Prediction(nextPrediction, resonanceStrength));
        }
        
        return predictions;
    }
    
    private Object predictNext(WorkingMemory<Object> memory) {
        List<Object> recent = memory.getRecentItems(3);
        if (recent.size() < 2) return null;
        return recent.get(0); // Simplified - predict repetition
    }
    
    private double computeResonanceStrength(WorkingMemory<Object> memory) {
        double[] gradient = memory.getActivationGradient();
        if (gradient.length == 0) return 0.0;
                
        double mean = Arrays.stream(gradient).average().orElse(0.0);
        double variance = Arrays.stream(gradient)
            .map(a -> Math.pow(a - mean, 2))
            .average().orElse(0.0);
        
        // Low variance = high resonance
        return Math.exp(-variance);
    }
    
    public Object combinePredictions(Map<TimeScale, Prediction> predictions) {
        // Shunting combination of predictions
        double V = 0.0; // Combined activation
        
        for (Map.Entry<TimeScale, Prediction> entry : predictions.entrySet()) {
            Prediction pred = entry.getValue();
            if (pred.content == null) continue;
            
            // Simplified combination
            V = V + (1 - V) * pred.weight * 0.5;
        }
        
        // Select prediction with highest weighted activation
        return predictions.entrySet().stream()
            .filter(e -> e.getValue().content != null)
            .max(Comparator.comparingDouble(e -> e.getValue().weight))
            .map(e -> e.getValue().content)
            .orElse(null);
    }    
    public Set<TimeScale> getActiveScales() {
        Set<TimeScale> active = EnumSet.noneOf(TimeScale.class);
        for (Map.Entry<TimeScale, WorkingMemory<Object>> entry : memories.entrySet()) {
            WorkingMemory<Object> memory = entry.getValue();
            // Consider scale active if it has any items in working memory
            // This reflects that the scale is participating in processing
            if (!memory.getRecentItems(entry.getKey().capacity).isEmpty()) {
                active.add(entry.getKey());
            }
        }
        return active;
    }
    
    // Helper classes
    private static class CompoundUnit {
        public final List<Object> items;
        
        public CompoundUnit(List<Object> items) {
            this.items = new ArrayList<>(items);
        }
    }
    
    public static class Prediction {
        public final Object content;
        public final double weight;
        
        public Prediction(Object content, double weight) {
            this.content = content;
            this.weight = weight;
        }
    }    
    private static class CompletionDetector {
        public boolean detectCompletion(TimeScale scale, WorkingMemory<Object> memory) {
            // Multiple completion criteria for robust hierarchical formation
            return detectCountBasedCompletion(scale, memory) ||
                   detectCapacityBasedCompletion(scale, memory) ||
                   detectPatternBasedCompletion(scale, memory);
        }
        
        private boolean detectCountBasedCompletion(TimeScale scale, WorkingMemory<Object> memory) {
            // Complete after processing a reasonable number of items for each scale
            int itemCount = memory.getRecentItems(scale.capacity).size();
            switch (scale) {
                case PHONEME:
                    return itemCount >= 2; // Form phoneme groups after 2+ items
                case WORD:
                    return itemCount >= 3; // Form word groups after 3+ phonemes/items
                case PHRASE:
                    return itemCount >= 3; // Form phrase groups after 3+ words
                case SENTENCE:
                    return itemCount >= 4; // Form sentence groups after 4+ phrases
                case PARAGRAPH:
                    return itemCount >= 3; // Form paragraph groups after 3+ sentences
                default:
                    return itemCount >= 2;
            }
        }
        
        private boolean detectCapacityBasedCompletion(TimeScale scale, WorkingMemory<Object> memory) {
            // Complete when working memory approaches capacity (Miller's 7±2 constraint)
            return memory.getRecentItems(scale.capacity).size() >= scale.capacity - 1;
        }
        
        private boolean detectPatternBasedCompletion(TimeScale scale, WorkingMemory<Object> memory) {
            // Original pattern-based detection as additional trigger
            switch (scale) {
                case PHONEME:
                    return memory.getRecentItems(1).stream()
                        .anyMatch(item -> item.toString().matches("\\s"));
                case WORD:
                    return memory.getRecentItems(1).stream()
                        .anyMatch(item -> item.toString().matches("[\\s.,;:!?]"));
                case PHRASE:
                    return memory.getRecentItems(1).stream()
                        .anyMatch(item -> item.toString().matches("[,;:]"));
                case SENTENCE:
                    return memory.getRecentItems(1).stream()
                        .anyMatch(item -> item.toString().matches("[.!?]"));
                case PARAGRAPH:
                    return memory.getRecentItems(1).stream()
                        .anyMatch(item -> item.toString().contains("\n\n"));
                default:
                    return false;
            }
        }
    }
}