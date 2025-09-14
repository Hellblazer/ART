package com.hellblazer.art.hartcq;

import java.util.Objects;

/**
 * Features extracted from a sliding window in the HART-CQ system.
 * Contains statistical and linguistic features used for pattern recognition
 * and quality assessment of text processing windows.
 */
public class WindowFeatures {
    private final int wordCount;
    private final int punctuationCount;
    private final double avgWordLength;
    private final int uniqueWordCount;
    private final double qualityScore;
    
    /**
     * Constructs WindowFeatures with the specified metrics.
     * 
     * @param wordCount number of words in the window
     * @param punctuationCount number of punctuation marks
     * @param avgWordLength average length of words
     * @param uniqueWordCount number of unique words
     * @param qualityScore computed quality score [0.0, 1.0]
     */
    public WindowFeatures(int wordCount, int punctuationCount, double avgWordLength, 
                         int uniqueWordCount, double qualityScore) {
        this.wordCount = Math.max(0, wordCount);
        this.punctuationCount = Math.max(0, punctuationCount);
        this.avgWordLength = Math.max(0.0, avgWordLength);
        this.uniqueWordCount = Math.max(0, uniqueWordCount);
        this.qualityScore = Math.max(0.0, Math.min(1.0, qualityScore)); // Clamp to [0,1]
    }
    
    /**
     * Creates WindowFeatures by analyzing a text window.
     * This is a convenience factory method that computes all features.
     * 
     * @param text the text to analyze
     * @return computed WindowFeatures
     */
    public static WindowFeatures fromText(String text) {
        if (text == null || text.isEmpty()) {
            return new WindowFeatures(0, 0, 0.0, 0, 0.0);
        }
        
        // Basic tokenization for feature extraction
        var words = text.split("\\s+");
        var wordCount = 0;
        var totalWordLength = 0;
        var uniqueWords = new java.util.HashSet<String>();
        var punctuationCount = 0;
        
        for (String word : words) {
            if (word.trim().isEmpty()) continue;
            
            // Clean word and count punctuation
            var cleanWord = new StringBuilder();
            for (char c : word.toCharArray()) {
                if (Character.isLetter(c)) {
                    cleanWord.append(Character.toLowerCase(c));
                } else if (Character.getType(c) == Character.OTHER_PUNCTUATION ||
                          Character.getType(c) == Character.START_PUNCTUATION ||
                          Character.getType(c) == Character.END_PUNCTUATION) {
                    punctuationCount++;
                }
            }
            
            var cleanedWord = cleanWord.toString();
            if (!cleanedWord.isEmpty()) {
                wordCount++;
                totalWordLength += cleanedWord.length();
                uniqueWords.add(cleanedWord);
            }
        }
        
        double avgWordLength = wordCount > 0 ? (double) totalWordLength / wordCount : 0.0;
        int uniqueWordCount = uniqueWords.size();
        
        // Compute quality score based on various factors
        double qualityScore = computeQualityScore(wordCount, punctuationCount, 
                                                avgWordLength, uniqueWordCount);
        
        return new WindowFeatures(wordCount, punctuationCount, avgWordLength, 
                                uniqueWordCount, qualityScore);
    }
    
    /**
     * Computes a quality score for the window features.
     * Higher scores indicate better quality/coherence.
     * 
     * @param wordCount number of words
     * @param punctuationCount number of punctuation marks
     * @param avgWordLength average word length
     * @param uniqueWordCount number of unique words
     * @return quality score [0.0, 1.0]
     */
    private static double computeQualityScore(int wordCount, int punctuationCount, 
                                            double avgWordLength, int uniqueWordCount) {
        if (wordCount == 0) return 0.0;
        
        // Ideal ranges for quality assessment
        double wordCountScore = Math.min(1.0, wordCount / 20.0); // Up to 20 words is good
        double punctuationScore = punctuationCount > 0 ? Math.min(1.0, punctuationCount / 5.0) : 0.5;
        double avgLengthScore = avgWordLength >= 3.0 && avgWordLength <= 8.0 ? 1.0 : 
                               Math.max(0.2, 1.0 - Math.abs(avgWordLength - 5.5) / 5.5);
        double diversityScore = wordCount > 0 ? Math.min(1.0, (double) uniqueWordCount / wordCount) : 0.0;
        
        // Weighted combination
        return (wordCountScore * 0.3 + punctuationScore * 0.2 + 
                avgLengthScore * 0.3 + diversityScore * 0.2);
    }
    
    /**
     * Gets the number of words in the window.
     * @return word count
     */
    public int getWordCount() {
        return wordCount;
    }
    
    /**
     * Gets the number of punctuation marks in the window.
     * @return punctuation count
     */
    public int getPunctuationCount() {
        return punctuationCount;
    }
    
    /**
     * Gets the average word length in the window.
     * @return average word length
     */
    public double getAvgWordLength() {
        return avgWordLength;
    }
    
    /**
     * Gets the number of unique words in the window.
     * @return unique word count
     */
    public int getUniqueWordCount() {
        return uniqueWordCount;
    }
    
    /**
     * Gets the computed quality score for this window.
     * @return quality score [0.0, 1.0]
     */
    public double getQualityScore() {
        return qualityScore;
    }
    
    /**
     * Calculates lexical diversity (unique words / total words).
     * @return lexical diversity ratio [0.0, 1.0]
     */
    public double getLexicalDiversity() {
        return wordCount > 0 ? (double) uniqueWordCount / wordCount : 0.0;
    }
    
    /**
     * Gets the punctuation density (punctuation marks per word).
     * @return punctuation density
     */
    public double getPunctuationDensity() {
        return wordCount > 0 ? (double) punctuationCount / wordCount : 0.0;
    }
    
    /**
     * Checks if this window has sufficient content for processing.
     * @return true if the window meets minimum quality thresholds
     */
    public boolean isProcessable() {
        return wordCount >= 3 && qualityScore >= 0.3;
    }
    
    /**
     * Combines features from two windows for comparative analysis.
     * @param other the other WindowFeatures to combine with
     * @return combined WindowFeatures
     */
    public WindowFeatures combineWith(WindowFeatures other) {
        int combinedWordCount = this.wordCount + other.wordCount;
        int combinedPunctuation = this.punctuationCount + other.punctuationCount;
        double combinedAvgLength = combinedWordCount > 0 ? 
            ((this.avgWordLength * this.wordCount) + (other.avgWordLength * other.wordCount)) / combinedWordCount : 0.0;
        int combinedUnique = Math.max(this.uniqueWordCount, other.uniqueWordCount); // Conservative estimate
        double combinedQuality = (this.qualityScore + other.qualityScore) / 2.0;
        
        return new WindowFeatures(combinedWordCount, combinedPunctuation, 
                                combinedAvgLength, combinedUnique, combinedQuality);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var that = (WindowFeatures) o;
        return wordCount == that.wordCount &&
               punctuationCount == that.punctuationCount &&
               Double.compare(that.avgWordLength, avgWordLength) == 0 &&
               uniqueWordCount == that.uniqueWordCount &&
               Double.compare(that.qualityScore, qualityScore) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(wordCount, punctuationCount, avgWordLength, uniqueWordCount, qualityScore);
    }
    
    @Override
    public String toString() {
        return String.format("WindowFeatures[words=%d, punct=%d, avgLen=%.2f, unique=%d, quality=%.3f, diversity=%.3f]",
                           wordCount, punctuationCount, avgWordLength, uniqueWordCount, 
                           qualityScore, getLexicalDiversity());
    }
}