package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.core.SlidingWindow;

import java.util.HashMap;
import java.util.Map;

/**
 * Word embedding channel.
 * CRITICAL: For COMPREHENSION_ONLY - never used for generation to prevent hallucination.
 * In a full implementation, this would use Word2Vec or similar embeddings.
 */
public class WordChannel implements Channel {
    
    private static final int EMBEDDING_DIM = 128;
    private static final boolean COMPREHENSION_ONLY = true; // CRITICAL FLAG
    
    private final Map<String, float[]> vocabulary;
    private final float[] unknownWordVector;
    
    public WordChannel() {
        this.vocabulary = new HashMap<>();
        this.unknownWordVector = createRandomVector(EMBEDDING_DIM);
        initializeVocabulary();
    }
    
    /**
     * Initializes a basic vocabulary with random vectors.
     * In production, this would load pre-trained Word2Vec embeddings.
     */
    private void initializeVocabulary() {
        // Common words get initialized vectors
        String[] commonWords = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        };
        
        for (String word : commonWords) {
            vocabulary.put(word.toLowerCase(), createRandomVector(EMBEDDING_DIM));
        }
    }
    
    private float[] createRandomVector(int dim) {
        var vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = (float) (Math.random() * 2 - 1) * 0.1f; // Small random values
        }
        return vector;
    }
    
    @Override
    public float[] processWindow(Token[] tokens) {
        if (!COMPREHENSION_ONLY) {
            throw new IllegalStateException("WordChannel is for COMPREHENSION ONLY - never for generation!");
        }
        
        var output = new float[EMBEDDING_DIM];
        int wordCount = 0;
        
        // Average word embeddings in the window
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null && tokens[i].getType() == Token.TokenType.WORD) {
                var text = tokens[i].getText();
                if (text == null || text.isEmpty()) {
                    continue;
                }
                var word = text.toLowerCase();
                var embedding = vocabulary.getOrDefault(word, unknownWordVector);
                
                for (int j = 0; j < EMBEDDING_DIM; j++) {
                    output[j] += embedding[j];
                }
                wordCount++;
            }
        }
        
        // Average the embeddings
        if (wordCount > 0) {
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                output[i] /= wordCount;
            }
        }
        
        return output;
    }
    
    @Override
    public int getOutputDimension() {
        return EMBEDDING_DIM;
    }
    
    @Override
    public String getName() {
        return "WordChannel";
    }
    
    @Override
    public void reset() {
        // No state to reset
    }
    
    @Override
    public boolean isDeterministic() {
        return true; // Same words produce same embeddings
    }
    
    /**
     * Adds a word to the vocabulary with a specific embedding.
     * Used for extending the vocabulary during learning.
     */
    public void addWord(String word, float[] embedding) {
        if (embedding.length != EMBEDDING_DIM) {
            throw new IllegalArgumentException("Embedding dimension must be " + EMBEDDING_DIM);
        }
        vocabulary.put(word.toLowerCase(), embedding.clone());
    }
    
    public boolean isComprehensionOnly() {
        return COMPREHENSION_ONLY;
    }

    @Override
    public ChannelType getChannelType() {
        return ChannelType.WORD;
    }

    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var startTime = System.nanoTime();

        // Convert ProcessingWindow to Token array for legacy compatibility
        var tokens = window.getTokens().toArray(new Token[0]);
        var features = processWindow(tokens);

        var processingTime = System.nanoTime() - startTime;
        return ChannelOutput.valid(features, getChannelType(), processingTime);
    }
}