package com.hellblazer.art.nlp.core;

import java.util.Objects;

/**
 * Represents sentiment analysis results.
 * Thread-safe immutable class with compound emotions.
 */
public final class SentimentScore {
    
    public enum Sentiment {
        POSITIVE, NEGATIVE, NEUTRAL
    }
    
    private final Sentiment sentiment;
    private final double positive;
    private final double negative;
    private final double neutral;
    private final double confidence;
    
    // Emotion dimensions from NRC lexicon
    private final double joy;
    private final double sadness;
    private final double anger;
    private final double fear;
    private final double trust;
    private final double disgust;
    private final double surprise;
    private final double anticipation;

    public SentimentScore(Sentiment sentiment, double positive, double negative, double neutral, double confidence) {
        this(sentiment, positive, negative, neutral, confidence, 
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    public SentimentScore(Sentiment sentiment, double positive, double negative, double neutral, double confidence,
                         double joy, double sadness, double anger, double fear, 
                         double trust, double disgust, double surprise, double anticipation) {
        this.sentiment = Objects.requireNonNull(sentiment, "sentiment cannot be null");
        this.positive = validateScore(positive, "positive");
        this.negative = validateScore(negative, "negative");
        this.neutral = validateScore(neutral, "neutral");
        this.confidence = validateScore(confidence, "confidence");
        this.joy = validateScore(joy, "joy");
        this.sadness = validateScore(sadness, "sadness");
        this.anger = validateScore(anger, "anger");
        this.fear = validateScore(fear, "fear");
        this.trust = validateScore(trust, "trust");
        this.disgust = validateScore(disgust, "disgust");
        this.surprise = validateScore(surprise, "surprise");
        this.anticipation = validateScore(anticipation, "anticipation");
    }

    private static double validateScore(double score, String name) {
        if (score < 0.0 || score > 1.0) {
            throw new IllegalArgumentException(name + " score must be in [0.0, 1.0]: " + score);
        }
        return score;
    }

    public Sentiment getSentiment() {
        return sentiment;
    }

    public double getPositive() {
        return positive;
    }

    public double getNegative() {
        return negative;
    }

    public double getNeutral() {
        return neutral;
    }

    public double getConfidence() {
        return confidence;
    }

    public double getJoy() {
        return joy;
    }

    public double getSadness() {
        return sadness;
    }

    public double getAnger() {
        return anger;
    }

    public double getFear() {
        return fear;
    }

    public double getTrust() {
        return trust;
    }

    public double getDisgust() {
        return disgust;
    }

    public double getSurprise() {
        return surprise;
    }

    public double getAnticipation() {
        return anticipation;
    }

    /**
     * Get the dominant emotion (highest scoring emotion dimension).
     */
    public String getDominantEmotion() {
        double maxScore = Math.max(joy, Math.max(sadness, Math.max(anger, Math.max(fear, 
                         Math.max(trust, Math.max(disgust, Math.max(surprise, anticipation)))))));
        
        if (maxScore == 0.0) return "none";
        
        if (joy == maxScore) return "joy";
        if (sadness == maxScore) return "sadness";
        if (anger == maxScore) return "anger";
        if (fear == maxScore) return "fear";
        if (trust == maxScore) return "trust";
        if (disgust == maxScore) return "disgust";
        if (surprise == maxScore) return "surprise";
        if (anticipation == maxScore) return "anticipation";
        
        return "mixed";
    }

    /**
     * Get emotional intensity (sum of all emotion dimensions).
     */
    public double getEmotionalIntensity() {
        return joy + sadness + anger + fear + trust + disgust + surprise + anticipation;
    }

    /**
     * Create a neutral sentiment score.
     */
    public static SentimentScore neutral() {
        return new SentimentScore(Sentiment.NEUTRAL, 0.0, 0.0, 1.0, 1.0);
    }

    /**
     * Create a simple positive sentiment score.
     */
    public static SentimentScore positive(double confidence) {
        return new SentimentScore(Sentiment.POSITIVE, 1.0, 0.0, 0.0, confidence);
    }

    /**
     * Create a simple negative sentiment score.
     */
    public static SentimentScore negative(double confidence) {
        return new SentimentScore(Sentiment.NEGATIVE, 0.0, 1.0, 0.0, confidence);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        SentimentScore that = (SentimentScore) obj;
        return Double.compare(that.positive, positive) == 0 &&
               Double.compare(that.negative, negative) == 0 &&
               Double.compare(that.neutral, neutral) == 0 &&
               Double.compare(that.confidence, confidence) == 0 &&
               Double.compare(that.joy, joy) == 0 &&
               Double.compare(that.sadness, sadness) == 0 &&
               Double.compare(that.anger, anger) == 0 &&
               Double.compare(that.fear, fear) == 0 &&
               Double.compare(that.trust, trust) == 0 &&
               Double.compare(that.disgust, disgust) == 0 &&
               Double.compare(that.surprise, surprise) == 0 &&
               Double.compare(that.anticipation, anticipation) == 0 &&
               sentiment == that.sentiment;
    }

    @Override
    public int hashCode() {
        return Objects.hash(sentiment, positive, negative, neutral, confidence, 
                          joy, sadness, anger, fear, trust, disgust, surprise, anticipation);
    }

    @Override
    public String toString() {
        return String.format("SentimentScore{sentiment=%s, pos=%.3f, neg=%.3f, neu=%.3f, conf=%.3f, emotion=%s(%.3f)}",
                           sentiment, positive, negative, neutral, confidence, 
                           getDominantEmotion(), getEmotionalIntensity());
    }
}