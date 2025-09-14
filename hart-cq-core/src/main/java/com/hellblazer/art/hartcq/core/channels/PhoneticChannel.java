package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.core.SlidingWindow;

/**
 * Phonetic pattern channel that captures sound patterns and rhythm.
 */
public class PhoneticChannel implements Channel {
    
    private static final int PHONETIC_DIM = 24;
    
    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[PHONETIC_DIM];
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null || tokens[i].getType() != Token.TokenType.WORD) {
                continue;
            }
            
            var text = tokens[i].getText();
            if (text == null || text.isEmpty()) continue;
            var word = text.toLowerCase();
            
            // Analyze phonetic features
            analyzePhoneticFeatures(word, output);
            
            // Analyze syllable patterns
            int syllables = estimateSyllables(word);
            output[10 + Math.min(syllables, 4)] += 1.0f;
            
            // Check for rhyme potential (ending patterns)
            if (word.length() >= 2) {
                var ending = word.substring(word.length() - 2);
                output[15 + (ending.hashCode() & 0x7) % 8] += 1.0f;
            }
        }
        
        // Normalize
        normalize(output);
        
        return output;
    }
    
    private void analyzePhoneticFeatures(String word, float[] output) {
        // Count vowels and consonants
        int vowels = 0;
        int consonants = 0;
        boolean hasDoubleConsonant = false;
        boolean hasDoubleVowel = false;
        
        char prev = '\0';
        for (char c : word.toCharArray()) {
            if (isVowel(c)) {
                vowels++;
                if (prev != '\0' && isVowel(prev)) {
                    hasDoubleVowel = true;
                }
            } else if (Character.isLetter(c)) {
                consonants++;
                if (prev != '\0' && !isVowel(prev) && Character.isLetter(prev)) {
                    hasDoubleConsonant = true;
                }
            }
            prev = c;
        }
        
        // Store phonetic features
        output[0] = (float) vowels / Math.max(word.length(), 1);
        output[1] = (float) consonants / Math.max(word.length(), 1);
        output[2] = hasDoubleConsonant ? 1.0f : 0.0f;
        output[3] = hasDoubleVowel ? 1.0f : 0.0f;
        
        // Check for specific sound patterns
        if (word.startsWith("th")) output[4] += 1.0f;
        if (word.startsWith("sh")) output[5] += 1.0f;
        if (word.startsWith("ch")) output[6] += 1.0f;
        if (word.endsWith("ing")) output[7] += 1.0f;
        if (word.endsWith("ed")) output[8] += 1.0f;
        if (word.endsWith("ly")) output[9] += 1.0f;
    }
    
    private int estimateSyllables(String word) {
        if (word.isEmpty()) return 0;
        
        int count = 0;
        boolean previousWasVowel = false;
        
        for (char c : word.toCharArray()) {
            boolean isVowel = isVowel(c);
            if (isVowel && !previousWasVowel) {
                count++;
            }
            previousWasVowel = isVowel;
        }
        
        // Adjust for silent 'e' at end
        if (word.endsWith("e") && count > 1) {
            count--;
        }
        
        return Math.max(count, 1);
    }
    
    private boolean isVowel(char c) {
        return "aeiouAEIOU".indexOf(c) >= 0;
    }
    
    private void normalize(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        
        if (sum > 0) {
            float norm = (float) Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
    
    @Override
    public int getOutputDimension() {
        return PHONETIC_DIM;
    }
    
    @Override
    public String getName() {
        return "PhoneticChannel";
    }
    
    @Override
    public void reset() {
        // No state to reset
    }
    
    @Override
    public boolean isDeterministic() {
        return true;
    }

    @Override
    public ChannelType getChannelType() {
        return ChannelType.WORD; // Phonetic relates to word characteristics
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