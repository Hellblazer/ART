package com.art.textgen.memory;

import com.art.textgen.core.WorkingMemory;
import java.util.*;
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
            this.compressionRatio = items.size() / (double) Math.max(1, patterns.size());
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
                if (currentLevel.items.size() > keepCount) {
                    currentLevel.items.subList(0, currentLevel.items.size() - keepCount).clear();
                }
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