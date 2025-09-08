package com.hellblazer.art.nlp.channel;

import com.hellblazer.art.nlp.channels.SentimentChannel;
import com.hellblazer.art.nlp.config.ChannelConfig;
import com.hellblazer.art.nlp.core.SentimentScore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SentimentChannel Tests")
class SentimentChannelTest {
    
    private SentimentChannel sentimentChannel;
    
    @BeforeEach
    void setUp() {
        sentimentChannel = new SentimentChannel();
    }
    
    @Test
    @DisplayName("Should initialize with default configuration")
    void testDefaultInitialization() {
        assertNotNull(sentimentChannel);
        assertEquals("sentiment", sentimentChannel.getChannelName());
        assertEquals(0.5, sentimentChannel.getVigilance(), 0.001);
        // Note: Learning rate is not exposed in BaseChannel API
    }
    
    @Test
    @DisplayName("Should initialize with custom configuration")
    void testCustomInitialization() {
        var config = ChannelConfig.builder()
            .channelName("custom_sentiment")
            .vigilance(0.8)
            .learningRate(0.6)
            .maxTokensPerInput(120)
            .build();
        
        var customChannel = new SentimentChannel(config);
        
        assertEquals("custom_sentiment", customChannel.getChannelName());
        assertEquals(0.8, customChannel.getVigilance(), 0.001);
        // Note: Learning rate is not exposed in BaseChannel API
    }
    
    @Test
    @DisplayName("Should detect positive sentiment")
    void testPositiveSentiment() {
        var text = "This product is amazing! I love it. The quality is excellent and outstanding.";
        
        // Calculate sentiment score
        var sentimentScore = sentimentChannel.calculateSentimentScore(text);
        assertEquals(SentimentScore.Sentiment.POSITIVE, sentimentScore.getSentiment());
        assertTrue(sentimentScore.getPositive() > sentimentScore.getNegative());
        assertTrue(sentimentScore.getConfidence() > 0.5);
    }
    
    @Test
    @DisplayName("Should detect negative sentiment")
    void testNegativeSentiment() {
        var text = "This is terrible! I hate it. The quality is awful and disappointing.";
        
        // Calculate sentiment score
        var sentimentScore = sentimentChannel.calculateSentimentScore(text);
        assertEquals(SentimentScore.Sentiment.NEGATIVE, sentimentScore.getSentiment());
        assertTrue(sentimentScore.getNegative() > sentimentScore.getPositive());
        assertTrue(sentimentScore.getConfidence() > 0.5);
    }
    
    @Test
    @DisplayName("Should detect neutral sentiment")
    void testNeutralSentiment() {
        var text = "The weather is clear today. It is 75 degrees. The meeting is at 2 PM.";
        
        // Calculate sentiment score
        var sentimentScore = sentimentChannel.calculateSentimentScore(text);
        assertEquals(SentimentScore.Sentiment.NEUTRAL, sentimentScore.getSentiment());
        assertTrue(Math.abs(sentimentScore.getPositive() - sentimentScore.getNegative()) < 0.3);
    }
    
    @Test
    @DisplayName("Should handle negation properly")
    void testNegationHandling() {
        var positiveText = "This is good.";
        var negatedText = "This is not good.";
        
        var positiveScore = sentimentChannel.calculateSentimentScore(positiveText);
        var negatedScore = sentimentChannel.calculateSentimentScore(negatedText);
        
        // Negated sentiment should have different polarity
        assertTrue((positiveScore.getPositive() - positiveScore.getNegative()) > (negatedScore.getPositive() - negatedScore.getNegative()));
    }
    
    @Test
    @DisplayName("Should handle empty input gracefully")
    void testEmptyInput() {
        var sentimentScore = sentimentChannel.calculateSentimentScore("");
        assertEquals(SentimentScore.Sentiment.NEUTRAL, sentimentScore.getSentiment());
        assertEquals(0.0, sentimentScore.getPositive() - sentimentScore.getNegative(), 0.001);
        
        var nullScore = sentimentChannel.calculateSentimentScore(null);
        assertEquals(SentimentScore.Sentiment.NEUTRAL, nullScore.getSentiment());
    }
    
    @Test
    @DisplayName("Should manage sentiment lexicon")
    void testSentimentLexiconManagement() {
        // Test adding custom sentiment words
        sentimentChannel.addSentimentWord("superb", 0.9);
        sentimentChannel.addSentimentWord("dreadful", -0.8);
        
        var lexicon = sentimentChannel.getSentimentLexicon();
        assertTrue(lexicon.containsKey("superb"));
        assertTrue(lexicon.containsKey("dreadful"));
        assertEquals(0.9, lexicon.get("superb"), 0.001);
        assertEquals(-0.8, lexicon.get("dreadful"), 0.001);
        
        // Test removing sentiment words
        sentimentChannel.removeSentimentWord("superb");
        lexicon = sentimentChannel.getSentimentLexicon();
        assertFalse(lexicon.containsKey("superb"));
    }
    
    @Test
    @DisplayName("Should reset channel state properly")
    void testReset() {
        // Note: Individual channels don't maintain categories - that's done at the processor level
        // Test reset doesn't throw exceptions
        assertDoesNotThrow(() -> {
            sentimentChannel.reset();
        });
        
        // After reset, category count should be 0
        assertEquals(0, sentimentChannel.getCategoryCount());
    }
    
    @Test
    @DisplayName("Should maintain consistent sentiment classification")
    void testConsistentClassification() {
        var text = "This product is excellent and I highly recommend it!";
        
        var score1 = sentimentChannel.calculateSentimentScore(text);
        var score2 = sentimentChannel.calculateSentimentScore(text);
        
        assertEquals(score1.getSentiment(), score2.getSentiment());
        assertEquals(score1.getPositive() - score1.getNegative(), score2.getPositive() - score2.getNegative(), 0.001);
        assertEquals(score1.getConfidence(), score2.getConfidence(), 0.001);
    }
    
    @Test
    @DisplayName("Should handle mixed sentiment text")
    void testMixedSentiment() {
        var text = "The product has some good features, but it also has several bad issues.";
        
        var sentimentScore = sentimentChannel.calculateSentimentScore(text);
        
        assertNotNull(sentimentScore);
        
        // Mixed sentiment should result in neutral or low confidence
        assertTrue(Math.abs(sentimentScore.getPositive() - sentimentScore.getNegative()) < 0.7);
    }
    
    @Test
    @DisplayName("Should provide channel information")
    void testChannelInformation() {
        assertEquals("sentiment", sentimentChannel.getChannelName());
        assertEquals(0.5, sentimentChannel.getVigilance(), 0.001);
        assertTrue(sentimentChannel.getCategoryCount() >= 0);
    }
}