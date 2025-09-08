package com.art.textgen;

import com.art.textgen.core.*;
import com.art.textgen.memory.*;
import com.art.textgen.generation.*;
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
    private final MultiTimescaleMemoryBank timescaleBank;
    private final ExecutorService executor;
    private final StrategySelector strategySelector;
    private final Vocabulary vocabulary;
    private final PatternGenerator patternGenerator;
    private final IntegratedDynamics dynamics;
    private final List<String> generatedTokens;
    
    // Anti-runaway pattern tracking
    private final Map<String, Integer> recentTokenCounts = new HashMap<>();
    private final List<String> lastTokens = new ArrayList<>();
    private static final int ANTI_RUNAWAY_WINDOW = 10;
    
    public enum GenerationStrategy {
        LOCAL_CONTEXT,
        EPISODIC_RECALL,
        HIERARCHICAL_SUMMARY,
        SKIP_CONNECTION,
        COMBINED
    }
    
    public GrossbergTextGenerator() {
        this.hierarchicalMemory = new RecursiveHierarchicalMemory();
        this.timescaleBank = new MultiTimescaleMemoryBank();
        this.executor = Executors.newFixedThreadPool(4);
        this.strategySelector = new StrategySelector();
        this.vocabulary = new Vocabulary(64); // 64-dimensional embeddings
        this.patternGenerator = new PatternGenerator(vocabulary, 1.0); // Temperature 1.0
        this.dynamics = new IntegratedDynamics();
        this.generatedTokens = new ArrayList<>();
    }
    
    public Stream<String> generate(String prompt, int maxLength) {
        // Initialize with prompt and clear anti-runaway tracking
        List<String> tokens = vocabulary.tokenize(prompt);
        generatedTokens.clear();
        generatedTokens.addAll(tokens);
        lastTokens.clear();
        recentTokenCounts.clear();
        
        for (String token : tokens) {
            processToken(token);
            updateAntiRunawayTracking(token);
        }
        
        // Generate stream
        return Stream.generate(() -> generateNext())
            .limit(maxLength)
            .takeWhile(token -> !token.equals(Vocabulary.END_TOKEN));
    }
    
    private void processToken(Object token) {
        // Process in parallel across all memory systems
        CompletableFuture<Void> f1 = CompletableFuture.runAsync(
            () -> hierarchicalMemory.addToken(token), executor);
        CompletableFuture<Void> f2 = CompletableFuture.runAsync(
            () -> timescaleBank.update(token), executor);
        
        // Wait for all to complete
        CompletableFuture.allOf(f1, f2).join();
    }    
    private String generateNext() {
        // Get recent context
        List<String> context = generatedTokens.subList(
            Math.max(0, generatedTokens.size() - 10), 
            generatedTokens.size()
        );
        
        // Convert context to neural activation
        double[] contextVector = convertToVector(context);
        
        // Process through dynamics
        IntegratedDynamics.DynamicsState state = dynamics.process(contextVector);
        
        // Select strategy based on current context
        GenerationStrategy strategy = strategySelector.selectStrategy(
            computeContextFeatures()
        );
        
        String nextToken;
        
        switch (strategy) {
            case LOCAL_CONTEXT:
                // Use pattern generator with local context
                nextToken = patternGenerator.generateNext(context);
                break;
                
            case HIERARCHICAL_SUMMARY:
                // Use hierarchical memory
                List<Object> hierarchicalContext = hierarchicalMemory.getActiveContext(50);
                nextToken = selectFromHierarchical(hierarchicalContext);
                break;
                
            case COMBINED:
            default:
                // Combine multiple approaches
                nextToken = combineApproaches(context, state);
                break;
        }
        
        // Check for runaway patterns and apply correction if needed
        String finalToken = applyAntiRunawayCorrection(nextToken);
        
        // Process token through system
        processToken(finalToken);
        generatedTokens.add(finalToken);
        
        // Update anti-runaway tracking
        updateAntiRunawayTracking(finalToken);
        
        // Learn from the generated pattern
        if (generatedTokens.size() > 3) {
            List<String> recentPattern = generatedTokens.subList(
                generatedTokens.size() - 4, generatedTokens.size()
            );
            patternGenerator.learnPattern(recentPattern);
        }
        
        return finalToken;
    }    
    
    private double[] convertToVector(List<String> tokens) {
        if (tokens.isEmpty()) {
            return new double[64]; // Return zero vector
        }
        
        // Average embeddings of tokens
        double[] vector = new double[64];
        for (String token : tokens) {
            double[] embedding = vocabulary.getEmbedding(token);
            for (int i = 0; i < embedding.length && i < vector.length; i++) {
                vector[i] += embedding[i];
            }
        }
        
        // Normalize
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= tokens.size();
        }
        
        return vector;
    }
    
    private String selectFromHierarchical(List<Object> context) {
        if (context.isEmpty()) {
            return vocabulary.START_TOKEN;
        }
        
        // Convert objects to strings and find best continuation
        List<String> stringContext = new ArrayList<>();
        for (Object obj : context) {
            stringContext.add(obj.toString());
        }
        
        // Use pattern generator with hierarchical context
        return patternGenerator.generateNext(stringContext);
    }
    
    private String combineApproaches(List<String> context, IntegratedDynamics.DynamicsState state) {
        // Get predictions from multiple sources
        String patternPrediction = patternGenerator.generateNext(context);
        
        // Get semantic neighbors based on dynamics
        List<String> candidates = new ArrayList<>();
        candidates.add(patternPrediction);
        
        // Add semantic variations
        if (!context.isEmpty()) {
            String lastToken = context.get(context.size() - 1);
            candidates.addAll(vocabulary.getSemanticNeighbors(lastToken, 3));
        }
        
        // Select based on resonance strength
        if (state.resonanceState.isResonant && state.coherence > 0.8) {
            return patternPrediction; // High confidence - use pattern
        } else {
            // Lower confidence - add variation
            Random rand = new Random();
            return candidates.get(rand.nextInt(candidates.size()));
        }
    }
    
    private List<Object> getLocalContext() {
        Map<MultiTimescaleMemoryBank.TimeScale, 
            MultiTimescaleMemoryBank.Prediction> predictions = 
            timescaleBank.generatePredictions();
        
        return Arrays.asList(timescaleBank.combinePredictions(predictions));
    }
    
    private List<Object> getHierarchicalContext() {
        return hierarchicalMemory.getActiveContext(100);
    }
    
    private List<Object> getCombinedContext() {
        List<Object> combined = new ArrayList<>();
        combined.addAll(getLocalContext());
        combined.addAll(getHierarchicalContext());
        return combined;
    }
    
    private ContextFeatures computeContextFeatures() {
        return new ContextFeatures(
            hierarchicalMemory.getEffectiveCapacity(),
            timescaleBank.getActiveScales()
        );
    }
    
    /**
     * Apply anti-runaway correction to prevent repetitive patterns
     */
    private String applyAntiRunawayCorrection(String proposedToken) {
        // Check if token would cause excessive repetition
        if (lastTokens.size() >= 3) {
            // Check for immediate repetition (same token 3+ times in a row)
            boolean immediateRepetition = true;
            for (int i = Math.max(0, lastTokens.size() - 3); i < lastTokens.size(); i++) {
                if (!lastTokens.get(i).equals(proposedToken)) {
                    immediateRepetition = false;
                    break;
                }
            }
            
            if (immediateRepetition) {
                return selectAlternativeToken(proposedToken);
            }
        }
        
        // Check for excessive frequency in recent window
        Integer currentCount = recentTokenCounts.getOrDefault(proposedToken, 0);
        if (currentCount >= 5) { // More than 5 occurrences in window
            return selectAlternativeToken(proposedToken);
        }
        
        return proposedToken;
    }
    
    /**
     * Update anti-runaway tracking data structures
     */
    private void updateAntiRunawayTracking(String token) {
        // Update last tokens window
        lastTokens.add(token);
        if (lastTokens.size() > ANTI_RUNAWAY_WINDOW) {
            String removed = lastTokens.remove(0);
            // Update counts
            recentTokenCounts.merge(removed, -1, (old, val) -> Math.max(0, old + val));
            if (recentTokenCounts.get(removed) == 0) {
                recentTokenCounts.remove(removed);
            }
        }
        
        // Update counts for new token
        recentTokenCounts.merge(token, 1, Integer::sum);
    }
    
    /**
     * Select an alternative token to avoid runaway patterns
     */
    private String selectAlternativeToken(String avoidToken) {
        String[] alternatives = {
            "the", "and", "of", "to", "in", "that", "with", "for", 
            "on", "as", "by", "this", "it", "from", "or", "an"
        };
        
        // Find alternative that's not overused
        for (String alt : alternatives) {
            if (!alt.equals(avoidToken) && 
                recentTokenCounts.getOrDefault(alt, 0) < 3) {
                return alt;
            }
        }
        
        // If all alternatives are overused, use random selection with damping
        Random rand = new Random();
        return alternatives[rand.nextInt(alternatives.length)];
    }
    
    public void shutdown() {
        executor.shutdown();
        dynamics.shutdown();
    }
    
    // Helper classes
    private static class StrategySelector {
        public GenerationStrategy selectStrategy(ContextFeatures features) {
            // Simple heuristic selection
            if (features.effectiveCapacity < 100) {
                return GenerationStrategy.LOCAL_CONTEXT;
            } else if (features.effectiveCapacity > 1000) {
                return GenerationStrategy.HIERARCHICAL_SUMMARY;
            } else {
                return GenerationStrategy.COMBINED;
            }
        }    }
    
    private static class ContextFeatures {
        public final double effectiveCapacity;
        public final Set<MultiTimescaleMemoryBank.TimeScale> activeScales;
        
        public ContextFeatures(double effectiveCapacity,
                              Set<MultiTimescaleMemoryBank.TimeScale> activeScales) {
            this.effectiveCapacity = effectiveCapacity;
            this.activeScales = activeScales;
        }
    }
}