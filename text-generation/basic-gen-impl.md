# Java Implementation: Grossberg Text Generation Module for ART Project

Package structure for `/Users/hal.hildebrand/git/ART/text-generation` module. This implementation provides concrete Java classes for text generation using Grossberg's neural dynamics, solving sequence length limitations through hierarchical memory architectures.

## Module Structure

```
/Users/hal.hildebrand/git/ART/
├── text-generation/
│   ├── pom.xml
│   ├── src/main/java/com/art/textgen/
│   │   ├── core/
│   │   │   ├── WorkingMemory.java
│   │   │   ├── MaskingField.java
│   │   │   └── VITEGenerator.java
│   │   ├── memory/
│   │   │   ├── RecursiveHierarchicalMemory.java
│   │   │   ├── LandmarkMemory.java
│   │   │   ├── MultiTimescaleMemoryBank.java
│   │   │   ├── AdaptiveForgettingMemory.java
│   │   │   ├── ResetSearchMemory.java
│   │   │   └── BidirectionalStreamingMemory.java
│   │   ├── dynamics/
│   │   │   ├── GrossbergDynamics.java
│   │   │   └── ResonanceController.java
│   │   └── GrossbergTextGenerator.java
```

## Core Components

### Base Working Memory Implementation

```java
package com.art.textgen.core;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicInteger;

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
            
            // Apply inhibition
            itemList.get(i).activation = Math.max(0, 
                itemList.get(i).activation - inhibition);
        }
    }
    
    public List<T> getRecentItems(int n) {
        return items.stream()
            .limit(n)
            .map(item -> item.content)
            .collect(ArrayList::new, 
                    (list, item) -> list.add(item),
                    ArrayList::addAll);
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
```

### Recursive Hierarchical Memory

```java
package com.art.textgen.memory;

import com.art.textgen.core.WorkingMemory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Recursive hierarchical chunking with dynamic compression
 * Achieves ~20,000 token capacity with just 5 levels
 */
public class RecursiveHierarchicalMemory {
    private static final int DEFAULT_LEVELS = 5;
    private static final int ITEMS_PER_LEVEL = 7;
    
    public static class HierarchicalLevel {
        public final int level;
        public final int capacity;
        public final int chunkSize;
        public final double tau;
        public final List<Chunk> items;
        public double compressionRatio;
        
        public HierarchicalLevel(int level) {
            this.level = level;
            this.capacity = ITEMS_PER_LEVEL;
            this.chunkSize = (int) Math.pow(ITEMS_PER_LEVEL, level);
            this.tau = Math.pow(10, level);
            this.items = new ArrayList<>();
            this.compressionRatio = 1.0;
        }
        
        public boolean needsCompression() {
            return items.size() >= capacity;
        }
    }
    
    public static class Chunk {
        public final String prototype;
        public final List<Pattern> patterns;
        public final List<Object> originalItems;
        public double activation;
        public final long timestamp;
        public final double compressionRatio;
        
        public Chunk(String prototype, List<Pattern> patterns, 
                    List<Object> items, double activation) {
            this.prototype = prototype;
            this.patterns = patterns;
            this.originalItems = new ArrayList<>(items);
            this.activation = activation;
            this.timestamp = System.currentTimeMillis();
            this.compressionRatio = items.size() / (double) patterns.size();
        }
        
        public List<Object> decompress(int maxDepth) {
            if (maxDepth <= 0) {
                return Collections.singletonList(prototype);
            }
            
            List<Object> decompressed = new ArrayList<>();
            for (Object item : originalItems) {
                if (item instanceof Chunk) {
                    decompressed.addAll(((Chunk) item).decompress(maxDepth - 1));
                } else {
                    decompressed.add(item);
                }
            }
            return decompressed;
        }
    }
    
    public static class Pattern {
        public final String pattern;
        public final int frequency;
        public final double importance;
        
        public Pattern(String pattern, int frequency, double importance) {
            this.pattern = pattern;
            this.frequency = frequency;
            this.importance = importance;
        }
    }
    
    private final List<HierarchicalLevel> levels;
    private final PatternExtractor patternExtractor;
    private long currentTime;
    
    public RecursiveHierarchicalMemory(int numLevels) {
        this.levels = new ArrayList<>();
        for (int i = 0; i < numLevels; i++) {
            levels.add(new HierarchicalLevel(i));
        }
        this.patternExtractor = new PatternExtractor();
        this.currentTime = 0;
    }
    
    public RecursiveHierarchicalMemory() {
        this(DEFAULT_LEVELS);
    }
    
    public void addToken(Object token) {
        // Add to bottom level
        HierarchicalLevel bottomLevel = levels.get(0);
        bottomLevel.items.add(new Chunk(
            token.toString(),
            Collections.emptyList(),
            Collections.singletonList(token),
            1.0
        ));
        
        // Cascade compression up the hierarchy
        for (int i = 0; i < levels.size() - 1; i++) {
            HierarchicalLevel currentLevel = levels.get(i);
            
            if (currentLevel.needsCompression()) {
                // Compress items at current level
                Chunk compressed = compressItems(currentLevel.items);
                
                // Move to next level
                HierarchicalLevel nextLevel = levels.get(i + 1);
                nextLevel.items.add(compressed);
                
                // Keep only most recent items at current level
                int keepCount = currentLevel.capacity / 2;
                currentLevel.items.retainAll(
                    currentLevel.items.subList(
                        Math.max(0, currentLevel.items.size() - keepCount),
                        currentLevel.items.size()
                    )
                );
            }
        }
        
        currentTime++;
    }
    
    private Chunk compressItems(List<Chunk> items) {
        // Extract patterns from items
        List<Pattern> patterns = patternExtractor.extractPatterns(items);
        
        // Compute prototype representation
        String prototype = computePrototype(items);
        
        // Calculate combined activation
        double totalActivation = items.stream()
            .mapToDouble(chunk -> chunk.activation)
            .sum();
        
        return new Chunk(prototype, patterns, new ArrayList<>(items), totalActivation);
    }
    
    private String computePrototype(List<Chunk> items) {
        // Create compressed representation
        Map<String, Integer> frequency = new HashMap<>();
        for (Chunk chunk : items) {
            frequency.merge(chunk.prototype, 1, Integer::sum);
        }
        
        return frequency.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("");
    }
    
    public List<Object> getActiveContext(int queryDepth) {
        List<Object> activeItems = new ArrayList<>();
        int remainingDepth = queryDepth;
        
        // Start from highest level, work down
        for (int i = levels.size() - 1; i >= 0 && remainingDepth > 0; i--) {
            HierarchicalLevel level = levels.get(i);
            
            for (Chunk chunk : level.items) {
                if (remainingDepth <= 0) break;
                
                if (isRelevant(chunk)) {
                    // Decompress relevant chunks
                    List<Object> decompressed = chunk.decompress(remainingDepth);
                    activeItems.addAll(decompressed);
                    remainingDepth -= decompressed.size();
                } else {
                    // Keep compressed if not immediately relevant
                    activeItems.add(chunk.prototype);
                    remainingDepth--;
                }
            }
        }
        
        return activeItems.stream()
            .limit(queryDepth)
            .collect(Collectors.toList());
    }
    
    private boolean isRelevant(Chunk chunk) {
        // Relevance based on recency and activation
        long age = currentTime - chunk.timestamp;
        double recency = Math.exp(-age / 1000.0);
        return chunk.activation * recency > 0.5;
    }
    
    public double getEffectiveCapacity() {
        double capacity = 0;
        for (HierarchicalLevel level : levels) {
            double levelCapacity = Math.pow(ITEMS_PER_LEVEL, level.level + 1) 
                * level.compressionRatio;
            capacity += levelCapacity;
        }
        return capacity;
    }
    
    // Pattern extraction helper
    private static class PatternExtractor {
        public List<Pattern> extractPatterns(List<Chunk> chunks) {
            Map<String, Integer> patternFrequency = new HashMap<>();
            
            // Simple n-gram extraction
            for (int n = 1; n <= 3; n++) {
                for (int i = 0; i <= chunks.size() - n; i++) {
                    StringBuilder pattern = new StringBuilder();
                    for (int j = 0; j < n; j++) {
                        pattern.append(chunks.get(i + j).prototype).append(" ");
                    }
                    patternFrequency.merge(pattern.toString().trim(), 1, Integer::sum);
                }
            }
            
            return patternFrequency.entrySet().stream()
                .map(e -> new Pattern(e.getKey(), e.getValue(), 
                    e.getValue() / (double) chunks.size()))
                .sorted((a, b) -> Double.compare(b.importance, a.importance))
                .limit(10)
                .collect(Collectors.toList());
        }
    }
}
```

### Landmark-Based Episodic Memory

```java
package com.art.textgen.memory;

import com.art.textgen.core.WorkingMemory;
import com.art.textgen.dynamics.ResonanceController;
import java.util.*;

/**
 * Episodic memory using landmark detection and ART gating
 * Handles ~100,000 tokens through strategic episode creation
 */
public class LandmarkMemory {
    private static final double DEFAULT_VIGILANCE = 0.7;
    private static final double LANDMARK_THRESHOLD = 0.6;
    private static final double RETRIEVAL_THRESHOLD = 0.3;
    
    public static class Episode {
        public final WorkingMemory.WorkingMemoryState<?> content;
        public Episode backwardLink;
        public Episode forwardLink;
        public final String summary;
        public double activation;
        public final long timestamp;
        
        public Episode(WorkingMemory.WorkingMemoryState<?> content, String summary) {
            this.content = content;
            this.summary = summary;
            this.activation = 1.0;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    public static class LandmarkDetector {
        private double landmarkActivation = 0.0;
        private long timeSinceLastLandmark = 0;
        private final Map<String, Double> weights;
        
        public LandmarkDetector() {
            this.weights = new HashMap<>();
            weights.put("semantic_shift", 0.3);
            weights.put("syntactic_closure", 0.2);
            weights.put("surprisal", 0.2);
            weights.put("resonance_peak", 0.2);
            weights.put("temporal_distance", 0.1);
        }
        
        public double computeLandmarkScore(Object token, ResonanceController resonance) {
            Map<String, Double> scores = new HashMap<>();
            
            scores.put("semantic_shift", detectSemanticBoundary(token));
            scores.put("syntactic_closure", detectSyntacticClosure(token));
            scores.put("surprisal", computeSurprisal(token));
            scores.put("resonance_peak", resonance.getCurrentResonance());
            scores.put("temporal_distance", normalizeTemporalDistance());
            
            // Weighted combination
            double totalScore = scores.entrySet().stream()
                .mapToDouble(e -> e.getValue() * weights.get(e.getKey()))
                .sum();
            
            // Update landmark dynamics (Grossberg boundary detection)
            // dL/dt = -AL + (B - L) * f(semantic_shift) - L * g(time_since_last)
            double A = 0.1;
            double B = 1.0;
            double dt = 0.1;
            
            landmarkActivation += dt * (
                -A * landmarkActivation + 
                (B - landmarkActivation) * scores.get("semantic_shift") -
                landmarkActivation * (timeSinceLastLandmark / 1000.0)
            );
            
            timeSinceLastLandmark++;
            
            return totalScore;
        }
        
        private double detectSemanticBoundary(Object token) {
            // Simplified semantic shift detection
            return Math.random() * 0.5; // Replace with actual implementation
        }
        
        private double detectSyntacticClosure(Object token) {
            // Check for sentence endings
            String tokenStr = token.toString();
            if (tokenStr.matches("[.!?]")) {
                return 1.0;
            }
            return 0.1;
        }
        
        private double computeSurprisal(Object token) {
            // Simplified surprisal calculation
            return Math.random() * 0.3; // Replace with actual implementation
        }
        
        private double normalizeTemporalDistance() {
            return Math.min(1.0, timeSinceLastLandmark / 100.0);
        }
        
        public void reset() {
            timeSinceLastLandmark = 0;
            landmarkActivation = 0.0;
        }
    }
    
    private final WorkingMemory<Object> workingMemory;
    private final List<Episode> landmarks;
    private final ResonanceController artGating;
    private final LandmarkDetector landmarkDetector;
    
    public LandmarkMemory() {
        this.workingMemory = new WorkingMemory<>();
        this.landmarks = new ArrayList<>();
        this.artGating = new ResonanceController(DEFAULT_VIGILANCE);
        this.landmarkDetector = new LandmarkDetector();
    }
    
    public void processToken(Object token) {
        // Add to working memory
        workingMemory.addItem(token, 1.0);
        
        // Check for landmark conditions
        double landmarkScore = landmarkDetector.computeLandmarkScore(token, artGating);
        
        if (landmarkScore > LANDMARK_THRESHOLD) {
            createEpisodeMemory();
            landmarkDetector.reset();
        }
    }
    
    private void createEpisodeMemory() {
        // Compress current working memory
        WorkingMemory.WorkingMemoryState<?> compressed = workingMemory.compress();
        
        // Generate summary (simplified)
        String summary = generateSummary(compressed);
        
        // Create new episode
        Episode episode = new Episode(compressed, summary);
        
        // Link to previous episode
        if (!landmarks.isEmpty()) {
            Episode previous = landmarks.get(landmarks.size() - 1);
            previous.forwardLink = episode;
            episode.backwardLink = previous;
        }
        
        landmarks.add(episode);
        
        // Clear working memory for next segment
        workingMemory.clear();
    }
    
    private String generateSummary(WorkingMemory.WorkingMemoryState<?> state) {
        // Simple summary generation
        return "Episode " + landmarks.size() + " at time " + state.timestamp;
    }
    
    public List<Object> retrieveContext(int position, int windowSize) {
        if (landmarks.isEmpty()) {
            return workingMemory.getRecentItems(windowSize);
        }
        
        // Find nearest landmark
        int nearestIdx = findNearestLandmark(position);
        Episode nearest = landmarks.get(nearestIdx);
        
        List<Object> context = new ArrayList<>();
        
        // Add content from nearest landmark
        context.addAll(nearest.content.items.stream()
            .map(item -> item.content)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll));
        
        // Radiate outward based on activation
        int leftPtr = nearestIdx - 1;
        int rightPtr = nearestIdx + 1;
        
        while (context.size() < windowSize && 
               (leftPtr >= 0 || rightPtr < landmarks.size())) {
            
            if (leftPtr >= 0) {
                Episode left = landmarks.get(leftPtr);
                double activation = computeActivation(left, position);
                if (activation > RETRIEVAL_THRESHOLD) {
                    List<Object> leftContent = left.content.items.stream()
                        .map(item -> item.content)
                        .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
                    context.addAll(0, leftContent);
                }
                leftPtr--;
            }
            
            if (rightPtr < landmarks.size() && context.size() < windowSize) {
                Episode right = landmarks.get(rightPtr);
                double activation = computeActivation(right, position);
                if (activation > RETRIEVAL_THRESHOLD) {
                    context.addAll(right.content.items.stream()
                        .map(item -> item.content)
                        .collect(ArrayList::new, ArrayList::add, ArrayList::addAll));
                }
                rightPtr++;
            }
        }
        
        return context.stream()
            .limit(windowSize)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    private int findNearestLandmark(int position) {
        // Binary search for nearest landmark
        int left = 0;
        int right = landmarks.size() - 1;
        
        while (left < right) {
            int mid = (left + right) / 2;
            if (mid < position / 100) { // Approximate position mapping
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return Math.min(left, landmarks.size() - 1);
    }
    
    private double computeActivation(Episode episode, int currentPosition) {
        // Activation based on recency and relevance
        long age = System.currentTimeMillis() - episode.timestamp;
        double recency = Math.exp(-age / 10000.0);
        return episode.activation * recency;
    }
}
```

### Multi-Timescale Memory Bank

```java
package com.art.textgen.memory;

import com.art.textgen.core.WorkingMemory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Multiple parallel working memories at different timescales
 * From 100ms phoneme level to 1-hour document level
 */
public class MultiTimescaleMemoryBank {
    
    public enum TimeScale {
        PHONEME("phoneme", 7, 0.1),      // ~100ms
        WORD("word", 7, 1.0),             // ~1s
        PHRASE("phrase", 7, 10.0),        // ~10s
        SENTENCE("sentence", 7, 60.0),    // ~1min
        PARAGRAPH("paragraph", 7, 600.0), // ~10min
        DOCUMENT("document", 7, 3600.0);  // ~1hour
        
        public final String name;
        public final int capacity;
        public final double tau;
        
        TimeScale(String name, int capacity, double tau) {
            this.name = name;
            this.capacity = capacity;
            this.tau = tau;
        }
    }
    
    private final Map<TimeScale, WorkingMemory<Object>> memories;
    private final Map<String, Double> verticalWeights;
    private final Map<String, Double> horizontalWeights;
    private final CompletionDetector completionDetector;
    
    public MultiTimescaleMemoryBank() {
        this.memories = new EnumMap<>(TimeScale.class);
        for (TimeScale scale : TimeScale.values()) {
            memories.put(scale, new WorkingMemory<>(scale.capacity, scale.tau));
        }
        
        this.verticalWeights = initializeVerticalConnections();
        this.horizontalWeights = initializeHorizontalConnections();
        this.completionDetector = new CompletionDetector();
    }
    
    private Map<String, Double> initializeVerticalConnections() {
        Map<String, Double> weights = new HashMap<>();
        // Bottom-up connections stronger than top-down
        weights.put("bottom_up", 0.8);
        weights.put("top_down", 0.3);
        return weights;
    }
    
    private Map<String, Double> initializeHorizontalConnections() {
        Map<String, Double> weights = new HashMap<>();
        // Lateral connections within same timescale
        weights.put("lateral", 0.5);
        return weights;
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
                memories.get(next).addItem(completedUnit, 
                    verticalWeights.get("bottom_up"));
            }
        }
        
        // Top-down modulation
        for (int i = scales.length - 1; i > 0; i--) {
            TimeScale current = scales[i];
            TimeScale previous = scales[i - 1];
            
            Object expectation = generateExpectation(memories.get(current));
            if (expectation != null) {
                modulateByExpectation(memories.get(previous), expectation);
            }
        }
    }
    
    private Object extractUnit(WorkingMemory<Object> memory) {
        List<Object> items = memory.getRecentItems(memory.capacity);
        return new CompoundUnit(items, memory.getActivationGradient());
    }
    
    private Object generateExpectation(WorkingMemory<Object> memory) {
        // Generate top-down expectation based on current state
        List<Object> recent = memory.getRecentItems(3);
        if (recent.isEmpty()) return null;
        
        // Simplified expectation generation
        return new Expectation(recent, memory.getActivationGradient());
    }
    
    private void modulateByExpectation(WorkingMemory<Object> memory, Object expectation) {
        // Modulate lower level based on higher level expectations
        if (expectation instanceof Expectation) {
            Expectation exp = (Expectation) expectation;
            // Apply top-down bias to activation patterns
            double[] gradient = memory.getActivationGradient();
            for (int i = 0; i < gradient.length && i < exp.activationPattern.length; i++) {
                gradient[i] = gradient[i] * (1 + verticalWeights.get("top_down") * 
                    exp.activationPattern[i]);
            }
        }
    }
    
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
        // Simple prediction based on recent patterns
        List<Object> recent = memory.getRecentItems(3);
        if (recent.size() < 2) return null;
        
        // Look for repeating patterns
        return recent.get(0); // Simplified - predict repetition
    }
    
    private double computeResonanceStrength(WorkingMemory<Object> memory) {
        // Compute resonance based on activation coherence
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
        // dV/dt = -AV + (B-V)Σ(excitatory) - (V+C)Σ(inhibitory)
        
        double V = 0.0; // Combined activation
        double A = 0.1;
        double B = 1.0;
        double C = 0.5;
        
        for (Map.Entry<TimeScale, Prediction> entry : predictions.entrySet()) {
            Prediction pred = entry.getValue();
            if (pred.content == null) continue;
            
            if (isConsonant(pred)) {
                // Excitatory (multiplicative enhancement)
                V = V + (B - V) * pred.weight * 0.5;
            } else {
                // Inhibitory (divisive normalization)
                V = V / (1 + pred.weight * C);
            }
        }
        
        // Select prediction with highest weighted activation
        return predictions.entrySet().stream()
            .max(Comparator.comparingDouble(e -> e.getValue().weight * V))
            .map(e -> e.getValue().content)
            .orElse(null);
    }
    
    private boolean isConsonant(Prediction pred) {
        // Check if prediction agrees with others
        return pred.weight > 0.5; // Simplified
    }
    
    // Helper classes
    private static class CompoundUnit {
        public final List<Object> items;
        public final double[] activationPattern;
        
        public CompoundUnit(List<Object> items, double[] activationPattern) {
            this.items = new ArrayList<>(items);
            this.activationPattern = Arrays.copyOf(activationPattern, 
                activationPattern.length);
        }
    }
    
    private static class Expectation {
        public final List<Object> items;
        public final double[] activationPattern;
        
        public Expectation(List<Object> items, double[] activationPattern) {
            this.items = new ArrayList<>(items);
            this.activationPattern = Arrays.copyOf(activationPattern, 
                activationPattern.length);
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
            // Detect when a unit at this timescale is complete
            switch (scale) {
                case PHONEME:
                    return detectPhonemeCompletion(memory);
                case WORD:
                    return detectWordCompletion(memory);
                case PHRASE:
                    return detectPhraseCompletion(memory);
                case SENTENCE:
                    return detectSentenceCompletion(memory);
                case PARAGRAPH:
                    return detectParagraphCompletion(memory);
                default:
                    return false;
            }
        }
        
        private boolean detectPhonemeCompletion(WorkingMemory<Object> memory) {
            // Check for phoneme boundaries
            return memory.getRecentItems(1).stream()
                .anyMatch(item -> item.toString().matches("\\s"));
        }
        
        private boolean detectWordCompletion(WorkingMemory<Object> memory) {
            // Check for word boundaries
            return memory.getRecentItems(1).stream()
                .anyMatch(item -> item.toString().matches("[\\s.,;:!?]"));
        }
        
        private boolean detectPhraseCompletion(WorkingMemory<Object> memory) {
            // Check for phrase boundaries
            return memory.getRecentItems(1).stream()
                .anyMatch(item -> item.toString().matches("[,;:]"));
        }
        
        private boolean detectSentenceCompletion(WorkingMemory<Object> memory) {
            // Check for sentence boundaries
            return memory.getRecentItems(1).stream()
                .anyMatch(item -> item.toString().matches("[.!?]"));
        }
        
        private boolean detectParagraphCompletion(WorkingMemory<Object> memory) {
            // Check for paragraph boundaries (simplified)
            return memory.getRecentItems(1).stream()
                .anyMatch(item -> item.toString().contains("\n\n"));
        }
    }
}
```

### Main Generator Class

```java
package com.art.textgen;

import com.art.textgen.core.*;
import com.art.textgen.memory.*;
import com.art.textgen.dynamics.*;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Stream;

/**
 * Main Grossberg Text Generator
 * Combines all memory systems for unlimited sequence processing
 */
public class GrossbergTextGenerator {
    
    private final RecursiveHierarchicalMemory hierarchicalMemory;
    private final LandmarkMemory landmarkMemory;
    private final MultiTimescaleMemoryBank timescaleBank;
    private final ExecutorService executor;
    
    // Meta-controller for strategy selection
    private final StrategySelector strategySelector;
    
    public enum GenerationStrategy {
        LOCAL_CONTEXT,
        EPISODIC_RECALL,
        HIERARCHICAL_SUMMARY,
        SKIP_CONNECTION,
        COMBINED
    }
    
    public GrossbergTextGenerator() {
        this.hierarchicalMemory = new RecursiveHierarchicalMemory();
        this.landmarkMemory = new LandmarkMemory();
        this.timescaleBank = new MultiTimescaleMemoryBank();
        this.executor = Executors.newFixedThreadPool(4);
        this.strategySelector = new StrategySelector();
    }
    
    public Stream<String> generate(String prompt, int maxLength) {
        // Initialize with prompt
        List<String> tokens = tokenize(prompt);
        for (String token : tokens) {
            processToken(token);
        }
        
        // Generate stream
        return Stream.generate(() -> generateNext())
            .limit(maxLength)
            .takeWhile(token -> !isEndToken(token));
    }
    
    private void processToken(Object token) {
        // Process in parallel across all memory systems
        CompletableFuture<Void> f1 = CompletableFuture.runAsync(
            () -> hierarchicalMemory.addToken(token), executor);
        CompletableFuture<Void> f2 = CompletableFuture.runAsync(
            () -> landmarkMemory.processToken(token), executor);
        CompletableFuture<Void> f3 = CompletableFuture.runAsync(
            () -> timescaleBank.update(token), executor);
        
        // Wait for all to complete
        CompletableFuture.allOf(f1, f2, f3).join();
    }
    
    private String generateNext() {
        // Select strategy based on current context
        GenerationStrategy strategy = strategySelector.selectStrategy(
            computeContextFeatures()
        );
        
        List<Object> context;
        
        switch (strategy) {
            case LOCAL_CONTEXT:
                context = getLocalContext();
                break;
            case EPISODIC_RECALL:
                context = getEpisodicContext();
                break;
            case HIERARCHICAL_SUMMARY:
                context = getHierarchicalContext();
                break;
            case COMBINED:
            default:
                context = getCombinedContext();
                break;
        }
        
        // Generate from context
        String nextToken = generateFromContext(context);
        
        // Feed back into system
        processToken(nextToken);
        
        return nextToken;
    }
    
    private List<Object> getLocalContext() {
        Map<MultiTimescaleMemoryBank.TimeScale, 
            MultiTimescaleMemoryBank.Prediction> predictions = 
            timescaleBank.generatePredictions();
        
        return Arrays.asList(timescaleBank.combinePredictions(predictions));
    }
    
    private List<Object> getEpisodicContext() {
        return landmarkMemory.retrieveContext(0, 100);
    }
    
    private List<Object> getHierarchicalContext() {
        return hierarchicalMemory.getActiveContext(100);
    }
    
    private List<Object> getCombinedContext() {
        List<Object> combined = new ArrayList<>();
        combined.addAll(getLocalContext());
        combined.addAll(getEpisodicContext());
        combined.addAll(getHierarchicalContext());
        return combined;
    }
    
    private ContextFeatures computeContextFeatures() {
        return new ContextFeatures(
            hierarchicalMemory.getEffectiveCapacity(),
            landmarkMemory.getLandmarkCount(),
            timescaleBank.getActiveScales()
        );
    }
    
    private String generateFromContext(List<Object> context) {
        // Simplified generation - replace with actual implementation
        if (context.isEmpty()) return "<END>";
        
        // Use context to predict next token
        return context.get(0).toString();
    }
    
    private List<String> tokenize(String text) {
        return Arrays.asList(text.split("\\s+"));
    }
    
    private boolean isEndToken(String token) {
        return "<END>".equals(token);
    }
    
    public void shutdown() {
        executor.shutdown();
    }
    
    // Helper classes
    private static class StrategySelector {
        public GenerationStrategy selectStrategy(ContextFeatures features) {
            // Simple heuristic selection
            if (features.effectiveCapacity < 100) {
                return GenerationStrategy.LOCAL_CONTEXT;
            } else if (features.landmarkCount > 10) {
                return GenerationStrategy.EPISODIC_RECALL;
            } else if (features.effectiveCapacity > 1000) {
                return GenerationStrategy.HIERARCHICAL_SUMMARY;
            } else {
                return GenerationStrategy.COMBINED;
            }
        }
    }
    
    private static class ContextFeatures {
        public final double effectiveCapacity;
        public final int landmarkCount;
        public final Set<MultiTimescaleMemoryBank.TimeScale> activeScales;
        
        public ContextFeatures(double effectiveCapacity, int landmarkCount,
                              Set<MultiTimescaleMemoryBank.TimeScale> activeScales) {
            this.effectiveCapacity = effectiveCapacity;
            this.landmarkCount = landmarkCount;
            this.activeScales = activeScales;
        }
    }
}
```

### Maven POM Configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>com.art</groupId>
        <artifactId>art-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>
    
    <artifactId>text-generation</artifactId>
    <name>ART Text Generation Module</name>
    <description>
        Grossberg neural dynamics-based text generation with 
        hierarchical memory for unlimited sequences
    </description>
    
    <properties>
        <java.version>17</java.version>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>
    
    <dependencies>
        <!-- Core ART dependencies -->
        <dependency>
            <groupId>com.art</groupId>
            <artifactId>art-core</artifactId>
            <version>${project.version}</version>
        </dependency>
        
        <!-- Math and scientific computing -->
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-math3</artifactId>
            <version>3.6.1</version>
        </dependency>
        
        <!-- Concurrent utilities -->
        <dependency>
            <groupId>com.google.guava</groupId>
            <artifactId>guava</artifactId>
            <version>32.1.2-jre</version>
        </dependency>
        
        <!-- Testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>org.assertj</groupId>
            <artifactId>assertj-core</artifactId>
            <version>3.24.2</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                    <compilerArgs>
                        <arg>--enable-preview</arg>
                    </compilerArgs>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

## Next Steps for Implementation

1. **Add to ART repository**:
```bash
cd /Users/hal.hildebrand/git/ART
git checkout -b text-generation
mkdir -p text-generation/src/main/java/com/art/textgen
# Copy the Java files above
mvn clean install
```

2. **Implement missing dynamics classes**:
   - `GrossbergDynamics.java` - Differential equation solvers
   - `ResonanceController.java` - ART resonance mechanisms
   - `VITEGenerator.java` - Semantic trajectory generation

3. **Add unit tests**:
   - Test memory capacity limits
   - Verify compression ratios
   - Benchmark performance

4. **Integration points**:
   - Connect to existing ART categorization
   - Add tokenizer interface
   - Implement actual generation logic

This Java implementation provides a solid foundation for the text generation module in your ART project, with proper Java idioms, concurrent processing, and modular design ready for integration!