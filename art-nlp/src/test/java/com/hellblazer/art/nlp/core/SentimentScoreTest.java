package com.hellblazer.art.nlp.core;

import com.hellblazer.art.nlp.core.SentimentScore.Sentiment;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive test suite for SentimentScore class.
 * Tests sentiment analysis results with emotion dimensions.
 */
@DisplayName("SentimentScore Tests")
class SentimentScoreTest {

    @Test
    @DisplayName("Should create basic sentiment score")
    void shouldCreateBasicSentimentScore() {
        var score = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.95);
        
        assertThat(score.getSentiment()).isEqualTo(Sentiment.POSITIVE);
        assertThat(score.getPositive()).isEqualTo(0.8);
        assertThat(score.getNegative()).isEqualTo(0.1);
        assertThat(score.getNeutral()).isEqualTo(0.1);
        assertThat(score.getConfidence()).isEqualTo(0.95);
        
        // Emotions should be zero by default
        assertThat(score.getJoy()).isZero();
        assertThat(score.getSadness()).isZero();
        assertThat(score.getAnger()).isZero();
        assertThat(score.getFear()).isZero();
        assertThat(score.getTrust()).isZero();
        assertThat(score.getDisgust()).isZero();
        assertThat(score.getSurprise()).isZero();
        assertThat(score.getAnticipation()).isZero();
    }

    @Test
    @DisplayName("Should create full sentiment score with emotions")
    void shouldCreateFullSentimentScoreWithEmotions() {
        var score = new SentimentScore(Sentiment.NEGATIVE, 0.1, 0.7, 0.2, 0.85,
                                     0.1, 0.6, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0);
        
        assertThat(score.getSentiment()).isEqualTo(Sentiment.NEGATIVE);
        assertThat(score.getPositive()).isEqualTo(0.1);
        assertThat(score.getNegative()).isEqualTo(0.7);
        assertThat(score.getNeutral()).isEqualTo(0.2);
        assertThat(score.getConfidence()).isEqualTo(0.85);
        
        assertThat(score.getJoy()).isEqualTo(0.1);
        assertThat(score.getSadness()).isEqualTo(0.6);
        assertThat(score.getAnger()).isEqualTo(0.3);
        assertThat(score.getFear()).isEqualTo(0.2);
        assertThat(score.getTrust()).isEqualTo(0.1);
        assertThat(score.getDisgust()).isEqualTo(0.1);
        assertThat(score.getSurprise()).isZero();
        assertThat(score.getAnticipation()).isZero();
    }

    @Test
    @DisplayName("Should validate null sentiment")
    void shouldValidateNullSentiment() {
        assertThatThrownBy(() -> new SentimentScore(null, 0.5, 0.5, 0.0, 1.0))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("sentiment cannot be null");
    }

    @Test
    @DisplayName("Should validate score ranges for basic scores")
    void shouldValidateScoreRangesForBasicScores() {
        // Test negative values
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, -0.1, 0.5, 0.5, 1.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("positive score must be in [0.0, 1.0]: -0.1");
        
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 0.5, -0.1, 0.5, 1.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("negative score must be in [0.0, 1.0]: -0.1");
        
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 0.5, 0.5, -0.1, 1.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("neutral score must be in [0.0, 1.0]: -0.1");
        
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 0.5, 0.5, 0.5, -0.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("confidence score must be in [0.0, 1.0]: -0.1");
        
        // Test values above 1.0
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 1.1, 0.0, 0.0, 1.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("positive score must be in [0.0, 1.0]: 1.1");
    }

    @Test
    @DisplayName("Should validate emotion score ranges")
    void shouldValidateEmotionScoreRanges() {
        // Test joy validation
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 0.5, 0.5, 0.0, 1.0,
                                                   -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("joy score must be in [0.0, 1.0]: -0.1");
        
        // Test anger validation
        assertThatThrownBy(() -> new SentimentScore(Sentiment.NEUTRAL, 0.5, 0.5, 0.0, 1.0,
                                                   0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("anger score must be in [0.0, 1.0]: 1.5");
    }

    @Test
    @DisplayName("Should accept boundary values")
    void shouldAcceptBoundaryValues() {
        // All zeros
        var allZeros = new SentimentScore(Sentiment.NEUTRAL, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertThat(allZeros.getPositive()).isZero();
        assertThat(allZeros.getJoy()).isZero();
        
        // All ones
        var allOnes = new SentimentScore(Sentiment.POSITIVE, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assertThat(allOnes.getPositive()).isOne();
        assertThat(allOnes.getJoy()).isOne();
    }

    @Test
    @DisplayName("Should find dominant emotion correctly")
    void shouldFindDominantEmotionCorrectly() {
        // Joy dominant
        var joyful = new SentimentScore(Sentiment.POSITIVE, 1.0, 0.0, 0.0, 1.0,
                                      0.8, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0);
        assertThat(joyful.getDominantEmotion()).isEqualTo("joy");
        
        // Sadness dominant
        var sad = new SentimentScore(Sentiment.NEGATIVE, 0.0, 1.0, 0.0, 1.0,
                                   0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertThat(sad.getDominantEmotion()).isEqualTo("sadness");
        
        // Anger dominant
        var angry = new SentimentScore(Sentiment.NEGATIVE, 0.0, 1.0, 0.0, 1.0,
                                     0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertThat(angry.getDominantEmotion()).isEqualTo("anger");
        
        // Fear dominant
        var fearful = new SentimentScore(Sentiment.NEGATIVE, 0.0, 1.0, 0.0, 1.0,
                                       0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0);
        assertThat(fearful.getDominantEmotion()).isEqualTo("fear");
        
        // Trust dominant
        var trusting = new SentimentScore(Sentiment.POSITIVE, 1.0, 0.0, 0.0, 1.0,
                                        0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0);
        assertThat(trusting.getDominantEmotion()).isEqualTo("trust");
        
        // Disgust dominant
        var disgusted = new SentimentScore(Sentiment.NEGATIVE, 0.0, 1.0, 0.0, 1.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0);
        assertThat(disgusted.getDominantEmotion()).isEqualTo("disgust");
        
        // Surprise dominant
        var surprised = new SentimentScore(Sentiment.NEUTRAL, 0.0, 0.0, 1.0, 1.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0);
        assertThat(surprised.getDominantEmotion()).isEqualTo("surprise");
        
        // Anticipation dominant
        var anticipating = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.0, 0.2, 1.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6);
        assertThat(anticipating.getDominantEmotion()).isEqualTo("anticipation");
        
        // No emotions
        var noEmotions = new SentimentScore(Sentiment.NEUTRAL, 0.0, 0.0, 1.0, 1.0);
        assertThat(noEmotions.getDominantEmotion()).isEqualTo("none");
        
        // Equal emotions (should return first found)
        var mixed = new SentimentScore(Sentiment.NEUTRAL, 0.5, 0.5, 0.0, 1.0,
                                     0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertThat(mixed.getDominantEmotion()).isIn("joy", "sadness");
    }

    @Test
    @DisplayName("Should calculate emotional intensity correctly")
    void shouldCalculateEmotionalIntensityCorrectly() {
        var noEmotions = new SentimentScore(Sentiment.NEUTRAL, 1.0, 0.0, 0.0, 1.0);
        assertThat(noEmotions.getEmotionalIntensity()).isZero();
        
        var someEmotions = new SentimentScore(Sentiment.POSITIVE, 1.0, 0.0, 0.0, 1.0,
                                            0.2, 0.1, 0.3, 0.0, 0.4, 0.0, 0.0, 0.1);
        assertThat(someEmotions.getEmotionalIntensity()).isEqualTo(1.1);
        
        var maxEmotions = new SentimentScore(Sentiment.POSITIVE, 1.0, 0.0, 0.0, 1.0,
                                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assertThat(maxEmotions.getEmotionalIntensity()).isEqualTo(8.0);
    }

    @Test
    @DisplayName("Should create static factory methods correctly")
    void shouldCreateStaticFactoryMethodsCorrectly() {
        var neutral = SentimentScore.neutral();
        assertThat(neutral.getSentiment()).isEqualTo(Sentiment.NEUTRAL);
        assertThat(neutral.getPositive()).isZero();
        assertThat(neutral.getNegative()).isZero();
        assertThat(neutral.getNeutral()).isOne();
        assertThat(neutral.getConfidence()).isOne();
        assertThat(neutral.getEmotionalIntensity()).isZero();
        
        var positive = SentimentScore.positive(0.8);
        assertThat(positive.getSentiment()).isEqualTo(Sentiment.POSITIVE);
        assertThat(positive.getPositive()).isOne();
        assertThat(positive.getNegative()).isZero();
        assertThat(positive.getNeutral()).isZero();
        assertThat(positive.getConfidence()).isEqualTo(0.8);
        
        var negative = SentimentScore.negative(0.9);
        assertThat(negative.getSentiment()).isEqualTo(Sentiment.NEGATIVE);
        assertThat(negative.getPositive()).isZero();
        assertThat(negative.getNegative()).isOne();
        assertThat(negative.getNeutral()).isZero();
        assertThat(negative.getConfidence()).isEqualTo(0.9);
    }

    @Test
    @DisplayName("Should implement equals correctly")
    void shouldImplementEqualsCorrectly() {
        var score1 = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score2 = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score3 = new SentimentScore(Sentiment.NEGATIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score4 = new SentimentScore(Sentiment.POSITIVE, 0.7, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score5 = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.8, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        
        // Same reference
        assertThat(score1).isEqualTo(score1);
        
        // Same values
        assertThat(score1).isEqualTo(score2);
        assertThat(score2).isEqualTo(score1);
        
        // Different values
        assertThat(score1).isNotEqualTo(score3); // Different sentiment
        assertThat(score1).isNotEqualTo(score4); // Different positive score
        assertThat(score1).isNotEqualTo(score5); // Different emotion
        
        // Null and different class
        assertThat(score1).isNotEqualTo(null);
        assertThat(score1).isNotEqualTo("not a sentiment");
    }

    @Test
    @DisplayName("Should implement hashCode correctly")
    void shouldImplementHashCodeCorrectly() {
        var score1 = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score2 = new SentimentScore(Sentiment.POSITIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        var score3 = new SentimentScore(Sentiment.NEGATIVE, 0.8, 0.1, 0.1, 0.9,
                                      0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        
        // Equal objects should have equal hash codes
        assertThat(score1.hashCode()).isEqualTo(score2.hashCode());
        
        // Different objects should preferably have different hash codes
        assertThat(score1.hashCode()).isNotEqualTo(score3.hashCode());
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        var score = new SentimentScore(Sentiment.POSITIVE, 0.856, 0.100, 0.044, 0.923,
                                     0.7, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0);
        var str = score.toString();
        
        assertThat(str).contains("POSITIVE");
        assertThat(str).contains("0.856");
        assertThat(str).contains("0.100");
        assertThat(str).contains("0.044");
        assertThat(str).contains("0.923");
        assertThat(str).contains("joy");
        assertThat(str).startsWith("SentimentScore{");
    }

    @Test
    @DisplayName("Should handle precision edge cases")
    void shouldHandlePrecisionEdgeCases() {
        // Very small values
        var precise = new SentimentScore(Sentiment.NEUTRAL, 0.001, 0.001, 0.998, 0.999,
                                       0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008);
        
        assertThat(precise.getPositive()).isEqualTo(0.001);
        assertThat(precise.getJoy()).isEqualTo(0.0001);
        assertThat(precise.getAnticipation()).isEqualTo(0.0008);
        assertThat(precise.getDominantEmotion()).isEqualTo("anticipation");
        assertThat(precise.getEmotionalIntensity()).isCloseTo(0.0036, within(0.0001));
    }
}