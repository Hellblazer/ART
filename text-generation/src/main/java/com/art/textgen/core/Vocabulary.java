package com.art.textgen.core;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Vocabulary management with semantic embeddings
 * Maps between tokens, indices, and semantic vectors
 */
public class Vocabulary {
    
    private final Map<String, Integer> tokenToIndex;
    private final Map<Integer, String> indexToToken;
    private final Map<String, double[]> tokenEmbeddings;
    private final Map<String, Set<String>> semanticAssociations;
    private int vocabSize;
    private final int embeddingDim;
    private final Random random;
    
    // Special tokens
    public static final String UNK_TOKEN = "<UNK>";
    public static final String START_TOKEN = "<START>";
    public static final String END_TOKEN = "<END>";
    public static final String PAD_TOKEN = "<PAD>";
    
    public Vocabulary(int embeddingDim) {
        this.tokenToIndex = new ConcurrentHashMap<>();
        this.indexToToken = new ConcurrentHashMap<>();
        this.tokenEmbeddings = new ConcurrentHashMap<>();
        this.semanticAssociations = new ConcurrentHashMap<>();
        this.embeddingDim = embeddingDim;
        this.vocabSize = 0;
        this.random = new Random();
        
        // Initialize special tokens
        addToken(UNK_TOKEN);
        addToken(START_TOKEN);
        addToken(END_TOKEN);
        addToken(PAD_TOKEN);
        
        // Initialize common semantic associations
        initializeSemanticAssociations();
    }
    
    /**
     * Add token to vocabulary
     */
    public int addToken(String token) {
        if (tokenToIndex.containsKey(token)) {
            return tokenToIndex.get(token);
        }
        
        int index = vocabSize++;
        tokenToIndex.put(token, index);
        indexToToken.put(index, token);
        
        // Initialize random embedding
        double[] embedding = generateEmbedding(token);
        tokenEmbeddings.put(token, embedding);
        
        return index;
    }
    
    /**
     * Generate semantic embedding for token
     */
    private double[] generateEmbedding(String token) {
        double[] embedding = new double[embeddingDim];
        
        // Use hash-based initialization for consistency
        int hash = token.hashCode();
        Random seededRandom = new Random(hash);
        
        for (int i = 0; i < embeddingDim; i++) {
            embedding[i] = seededRandom.nextGaussian() * 0.1;
        }
        
        // Add semantic bias based on token characteristics
        if (token.endsWith("ing")) {
            embedding[0] += 0.2; // Action bias
        }
        if (token.endsWith("ness") || token.endsWith("tion")) {
            embedding[1] += 0.2; // Abstract concept bias
        }
        if (Character.isUpperCase(token.charAt(0))) {
            embedding[2] += 0.2; // Proper noun bias
        }
        
        return normalize(embedding);
    }
    
    /**
     * Initialize semantic associations
     */
    private void initializeSemanticAssociations() {
        // Conceptual associations
        addAssociation("artificial", "intelligence", "machine", "learning", "neural", "network");
        addAssociation("intelligence", "artificial", "human", "cognitive", "understanding", "consciousness");
        addAssociation("consciousness", "awareness", "mind", "thought", "perception", "experience");
        addAssociation("future", "tomorrow", "technology", "progress", "evolution", "time");
        addAssociation("understanding", "knowledge", "comprehension", "insight", "learning", "wisdom");
        
        // Syntactic associations
        addAssociation("the", "a", "an", "this", "that", "these");
        addAssociation("is", "are", "was", "were", "be", "being");
        addAssociation("can", "could", "will", "would", "should", "might");
        
        // Temporal associations
        addAssociation("once", "upon", "time", "day", "moment", "era");
        addAssociation("beginning", "start", "first", "initial", "origin", "genesis");
        
        // Spatial associations
        addAssociation("in", "on", "at", "within", "inside", "through");
        addAssociation("over", "under", "above", "below", "beside", "between");
    }
    
    /**
     * Add semantic association between tokens
     */
    private void addAssociation(String token, String... associated) {
        Set<String> associations = semanticAssociations.computeIfAbsent(token, k -> new HashSet<>());
        associations.addAll(Arrays.asList(associated));
        
        // Make associations bidirectional
        for (String assoc : associated) {
            semanticAssociations.computeIfAbsent(assoc, k -> new HashSet<>()).add(token);
        }
    }
    
    /**
     * Get semantic neighbors of a token
     */
    public List<String> getSemanticNeighbors(String token, int k) {
        Set<String> directAssociations = semanticAssociations.getOrDefault(token, new HashSet<>());
        
        // If we have direct associations, use them
        if (!directAssociations.isEmpty()) {
            return directAssociations.stream()
                .limit(k)
                .collect(Collectors.toList());
        }
        
        // Otherwise, find nearest by embedding similarity
        double[] embedding = tokenEmbeddings.get(token);
        if (embedding == null) {
            return new ArrayList<>();
        }
        
        return tokenEmbeddings.entrySet().stream()
            .filter(e -> !e.getKey().equals(token))
            .sorted((a, b) -> {
                double simA = cosineSimilarity(embedding, a.getValue());
                double simB = cosineSimilarity(embedding, b.getValue());
                return Double.compare(simB, simA); // Descending order
            })
            .limit(k)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
    
    /**
     * Compute cosine similarity between embeddings
     */
    private double cosineSimilarity(double[] a, double[] b) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0) return 0;
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    /**
     * Get embedding for token
     */
    public double[] getEmbedding(String token) {
        return tokenEmbeddings.getOrDefault(token, tokenEmbeddings.get(UNK_TOKEN));
    }
    
    /**
     * Get token index
     */
    public int getIndex(String token) {
        return tokenToIndex.getOrDefault(token, tokenToIndex.get(UNK_TOKEN));
    }
    
    /**
     * Get token from index
     */
    public String getToken(int index) {
        return indexToToken.getOrDefault(index, UNK_TOKEN);
    }
    
    /**
     * Tokenize text
     */
    public List<String> tokenize(String text) {
        // Simple tokenization - split on whitespace and punctuation
        List<String> tokens = new ArrayList<>();
        
        // Split on whitespace but keep punctuation
        String[] words = text.toLowerCase().split("\\s+");
        
        for (String word : words) {
            // Handle punctuation
            List<String> subTokens = tokenizeWord(word);
            tokens.addAll(subTokens);
        }
        
        return tokens;
    }
    
    /**
     * Tokenize individual word with punctuation
     */
    private List<String> tokenizeWord(String word) {
        List<String> tokens = new ArrayList<>();
        
        if (word.isEmpty()) return tokens;
        
        // Check for leading punctuation
        int start = 0;
        while (start < word.length() && !Character.isLetterOrDigit(word.charAt(start))) {
            tokens.add(String.valueOf(word.charAt(start)));
            start++;
        }
        
        // Find trailing punctuation
        int end = word.length();
        List<String> trailingPunct = new ArrayList<>();
        while (end > start && !Character.isLetterOrDigit(word.charAt(end - 1))) {
            trailingPunct.add(0, String.valueOf(word.charAt(end - 1)));
            end--;
        }
        
        // Add main word
        if (start < end) {
            String mainWord = word.substring(start, end);
            tokens.add(mainWord);
            addToken(mainWord); // Add to vocabulary if new
        }
        
        // Add trailing punctuation
        tokens.addAll(trailingPunct);
        
        return tokens;
    }
    
    /**
     * Convert tokens to indices
     */
    public int[] tokensToIndices(List<String> tokens) {
        return tokens.stream()
            .mapToInt(this::getIndex)
            .toArray();
    }
    
    /**
     * Convert indices to tokens
     */
    public List<String> indicesToTokens(int[] indices) {
        return Arrays.stream(indices)
            .mapToObj(this::getToken)
            .collect(Collectors.toList());
    }
    
    /**
     * Normalize vector
     */
    private double[] normalize(double[] vector) {
        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        
        if (norm == 0) return vector;
        
        norm = Math.sqrt(norm);
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        return normalized;
    }
    
    /**
     * Get vocabulary size
     */
    public int size() {
        return vocabSize;
    }
    
    /**
     * Get embedding dimension
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    /**
     * Check if token exists
     */
    public boolean contains(String token) {
        return tokenToIndex.containsKey(token);
    }
    
    /**
     * Get all tokens
     */
    public Set<String> getAllTokens() {
        return new HashSet<>(tokenToIndex.keySet());
    }
    
    /**
     * Compute semantic field activation
     */
    public double[] computeSemanticField(List<String> context, String target) {
        double[] field = new double[embeddingDim];
        
        // Get target embedding
        double[] targetEmb = getEmbedding(target);
        
        // Compute weighted sum of context embeddings
        for (String contextToken : context) {
            double[] contextEmb = getEmbedding(contextToken);
            double similarity = cosineSimilarity(targetEmb, contextEmb);
            
            for (int i = 0; i < embeddingDim; i++) {
                field[i] += similarity * contextEmb[i];
            }
        }
        
        return normalize(field);
    }
}
