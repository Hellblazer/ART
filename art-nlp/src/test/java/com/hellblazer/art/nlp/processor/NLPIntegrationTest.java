package com.hellblazer.art.nlp.processor;

import com.hellblazer.art.nlp.core.NLPProcessor;
import com.hellblazer.art.nlp.core.Document;
import com.hellblazer.art.nlp.core.ProcessingResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.condition.EnabledIf;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("NLP System Integration Tests")
class NLPIntegrationTest {
    
    private NLPProcessor nlpProcessor;
    
    @BeforeEach
    void setUp() {
        try {
            nlpProcessor = MultiChannelProcessor.createWithDefaults();
        } catch (Exception e) {
            // Skip tests if required dependencies are not available
            nlpProcessor = null;
        }
    }
    
    // Helper method to check if NLP processor is available
    static boolean isNLPProcessorAvailable() {
        try {
            MultiChannelProcessor.createWithDefaults();
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    @Test
    @DisplayName("Should process simple text through all channels")
    @EnabledIf("isNLPProcessorAvailable")
    void testSimpleTextProcessing() {
        var text = "Natural language processing is a fascinating field of artificial intelligence.";
        var result = nlpProcessor.process(text);
        
        assertNotNull(result);
        assertNotNull(result.getAllCategories());
        assertNotNull(result.getEntities());
        assertTrue(result.getProcessingTimeMs() >= 0);
        
        // Should have categories from multiple channels
        var categories = result.getAllCategories();
        assertFalse(categories.isEmpty());
        
        // Should have some entities detected
        var entities = result.getEntities();
        // Entities may be empty if models are not available, that's ok for this test
    }
    
    @Test
    @DisplayName("Should process document with metadata")
    @EnabledIf("isNLPProcessorAvailable")
    void testDocumentProcessing() {
        var document = Document.builder()
            .withContent("Machine learning algorithms can classify text and analyze sentiment. " +
                    "However, they require proper training data.")
            .withTitle("ML Text Analysis")
            .withMetadata("author", "Test Author")
            .withMetadata("category", "technology")
            .build();
        
        var analysis = nlpProcessor.processDocument(document);
        
        assertNotNull(analysis);
        assertNotNull(analysis.getDocument());
        assertEquals(document, analysis.getDocument());
        
        // Should have processing results
        assertNotNull(analysis.getSemanticCategories());
        assertNotNull(analysis.getEntities());
        
        // Should have document-level analysis
        assertFalse(analysis.getSentences().isEmpty());
        var lengthStats = analysis.getLengthStatistics();
        assertTrue(lengthStats.sentenceCount() >= 2);
        assertTrue(lengthStats.tokenCount() > 10);
    }
    
    @Test
    @DisplayName("Should handle multi-channel semantic processing")
    @EnabledIf("isNLPProcessorAvailable")
    void testMultiChannelSemanticProcessing() {
        var text = "The innovative startup developed an amazing product. " +
                  "John Smith, the CEO, announced the launch in New York. " +
                  "However, some customers were disappointed with the quality.";
        
        var result = nlpProcessor.process(text);
        
        assertNotNull(result);
        var categories = result.getAllCategories();
        
        // Should have categories from different channels
        assertFalse(categories.isEmpty());
        
        // Check for channel-specific patterns
        var categoryKeys = categories.keySet();
        
        // May have semantic categories
        var hasSemanticCategories = categoryKeys.stream()
            .anyMatch(key -> key.startsWith("semantic_"));
        
        // May have syntactic categories
        var hasSyntacticCategories = categoryKeys.stream()
            .anyMatch(key -> key.startsWith("syntactic_"));
        
        // May have context categories
        var hasContextCategories = categoryKeys.stream()
            .anyMatch(key -> key.startsWith("context_"));
        
        // May have sentiment categories
        var hasSentimentCategories = categoryKeys.stream()
            .anyMatch(key -> key.startsWith("sentiment_"));
        
        // At least some categories should be present
        assertTrue(hasSemanticCategories || hasSyntacticCategories || 
                  hasContextCategories || hasSentimentCategories);
    }
    
    @Test
    @DisplayName("Should collect processing statistics")
    @EnabledIf("isNLPProcessorAvailable")
    void testProcessingStatistics() {
        var texts = new String[]{
            "This is the first test sentence.",
            "Here comes the second test sentence.",
            "Finally, we have the third test sentence."
        };
        
        // Process multiple texts
        for (var text : texts) {
            nlpProcessor.process(text);
        }
        
        var stats = nlpProcessor.getStatistics();
        
        assertNotNull(stats);
        assertTrue(stats.getTotalProcessed() >= texts.length);
        assertTrue(stats.getAverageProcessingTimeMs() >= 0);
        assertTrue(stats.getAverageProcessingTimeMs() >= 0);
        
        // Should have channel-specific statistics
        var channelStats = stats.getChannelStatistics();
        assertNotNull(channelStats);
        // Channel stats may be empty if channels are not configured
    }
    
    @Test
    @DisplayName("Should handle sentiment analysis integration")
    @EnabledIf("isNLPProcessorAvailable")
    void testSentimentAnalysisIntegration() {
        var positiveText = "I absolutely love this amazing product! It's fantastic and wonderful.";
        var negativeText = "This terrible product is awful and disappointing. I hate it completely.";
        var neutralText = "The meeting is scheduled for tomorrow at 3 PM in conference room B.";
        
        var positiveResult = nlpProcessor.process(positiveText);
        var negativeResult = nlpProcessor.process(negativeText);
        var neutralResult = nlpProcessor.process(neutralText);
        
        assertNotNull(positiveResult);
        assertNotNull(negativeResult);
        assertNotNull(neutralResult);
        
        // Check if sentiment categories are detected
        var positiveCategories = positiveResult.getAllCategories().keySet();
        var negativeCategories = negativeResult.getAllCategories().keySet();
        var neutralCategories = neutralResult.getAllCategories().keySet();
        
        // May have sentiment-related categories
        var hasPositiveSentiment = positiveCategories.stream()
            .anyMatch(key -> key.contains("POSITIVE") || key.contains("sentiment"));
        var hasNegativeSentiment = negativeCategories.stream()
            .anyMatch(key -> key.contains("NEGATIVE") || key.contains("sentiment"));
        
        // At least some sentiment processing should occur
        assertTrue(hasPositiveSentiment || hasNegativeSentiment || 
                  !neutralCategories.isEmpty());
    }
    
    @Test
    @DisplayName("Should handle context analysis integration")
    @EnabledIf("isNLPProcessorAvailable")
    void testContextAnalysisIntegration() {
        var text = "Machine learning is related to artificial intelligence. " +
                  "However, deep learning is different from traditional ML. " +
                  "Therefore, we need specialized algorithms for each approach.";
        
        var result = nlpProcessor.process(text);
        
        assertNotNull(result);
        var categories = result.getAllCategories();
        var entities = result.getEntities();
        
        // Should detect contextual relationships
        var hasContextCategories = categories.keySet().stream()
            .anyMatch(key -> key.startsWith("context_"));
        
        var hasDiscourseEntities = entities.stream()
            .anyMatch(entity -> entity.getType().startsWith("DISCOURSE_"));
        
        var hasRelationshipEntities = entities.stream()
            .anyMatch(entity -> entity.getType().equals("RELATIONSHIP"));
        
        // Some contextual processing should occur
        assertTrue(hasContextCategories || hasDiscourseEntities || hasRelationshipEntities);
    }
    
    @Test
    @DisplayName("Should handle empty and edge cases")
    @EnabledIf("isNLPProcessorAvailable")
    void testEdgeCases() {
        // Empty text
        var emptyResult = nlpProcessor.process("");
        assertNotNull(emptyResult);
        assertNotNull(emptyResult.getAllCategories());
        assertNotNull(emptyResult.getEntities());
        
        // Null text
        var nullResult = nlpProcessor.process(null);
        assertNotNull(nullResult);
        assertNotNull(nullResult.getAllCategories());
        assertNotNull(nullResult.getEntities());
        
        // Very short text
        var shortResult = nlpProcessor.process("Hi!");
        assertNotNull(shortResult);
        assertNotNull(shortResult.getAllCategories());
        assertNotNull(shortResult.getEntities());
        
        // Text with special characters
        var specialResult = nlpProcessor.process("Test @#$%^&*() text with symbols!!!");
        assertNotNull(specialResult);
        assertNotNull(specialResult.getAllCategories());
        assertNotNull(specialResult.getEntities());
    }
    
    @Test
    @DisplayName("Should maintain thread safety")
    @EnabledIf("isNLPProcessorAvailable")
    void testThreadSafety() {
        var texts = new String[]{
            "First concurrent text processing test.",
            "Second concurrent text processing test.",
            "Third concurrent text processing test."
        };
        
        // Process texts concurrently
        var results = java.util.Arrays.stream(texts)
            .parallel()
            .map(nlpProcessor::process)
            .toList();
        
        assertEquals(texts.length, results.size());
        
        for (var result : results) {
            assertNotNull(result);
            assertNotNull(result.getAllCategories());
            assertNotNull(result.getEntities());
            assertTrue(result.getProcessingTimeMs() >= 0);
        }
    }
    
    @Test
    @DisplayName("Should reset processor state")
    @EnabledIf("isNLPProcessorAvailable")
    void testProcessorReset() {
        // Process some text to build up state
        var text = "This text will create some categories and state.";
        nlpProcessor.process(text);
        
        var statsBefore = nlpProcessor.getStatistics();
        assertTrue(statsBefore.getTotalProcessed() > 0);
        
        // Reset the processor
        nlpProcessor.reset();
        
        var statsAfter = nlpProcessor.getStatistics();
        assertEquals(0, statsAfter.getTotalProcessed());
        assertEquals(0.0, statsAfter.getAverageProcessingTimeMs(), 0.001);
        assertEquals(0.0, statsAfter.getAverageProcessingTimeMs(), 0.001);
    }
    
    @Test
    @DisplayName("Should handle resource cleanup")
    @EnabledIf("isNLPProcessorAvailable")
    void testResourceCleanup() {
        assertDoesNotThrow(() -> {
            nlpProcessor.close();
        });
    }
    
    @Test
    @DisplayName("Should validate processing result structure")
    @EnabledIf("isNLPProcessorAvailable")
    void testProcessingResultStructure() {
        var text = "Natural language processing enables computers to understand human language.";
        var result = nlpProcessor.process(text);
        
        assertNotNull(result);
        
        // Validate basic structure
        assertNotNull(result.getAllCategories());
        assertNotNull(result.getEntities());
        assertTrue(result.getProcessingTimeMs() >= 0);
        
        // Validate categories structure (Map<String, Integer>)
        var categories = result.getAllCategories();
        for (var entry : categories.entrySet()) {
            assertNotNull(entry.getKey());
            assertNotNull(entry.getValue());
            assertTrue(entry.getValue() >= 0);
        }
        
        // Validate entities structure
        var entities = result.getEntities();
        for (var entity : entities) {
            assertNotNull(entity.getText());
            assertNotNull(entity.getType());
            assertTrue(entity.getStartToken() >= 0);
            assertTrue(entity.getEndToken() >= entity.getStartToken());
            assertTrue(entity.getConfidence() >= 0.0 && entity.getConfidence() <= 1.0);
        }
    }
    
    @Test
    @DisplayName("Should handle long text processing")
    @EnabledIf("isNLPProcessorAvailable")
    void testLongTextProcessing() {
        var longText = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence " +
                      "concerned with the interactions between computers and human language, in particular how to program computers " +
                      "to process and analyze large amounts of natural language data. The goal is a computer capable of understanding " +
                      "the contents of documents, including the contextual nuances of the language within them. The technology can " +
                      "then accurately extract information and insights contained in the documents as well as categorize and organize " +
                      "the documents themselves. Challenges in natural language processing frequently involve speech recognition, " +
                      "natural language understanding, and natural language generation. However, modern approaches often use machine " +
                      "learning and deep learning algorithms to achieve better performance.";
        
        var result = nlpProcessor.process(longText);
        
        assertNotNull(result);
        assertFalse(result.getAllCategories().isEmpty());
        assertTrue(result.getProcessingTimeMs() >= 0);
        
        // Should handle long text without errors
        var categories = result.getAllCategories();
        assertTrue(categories.size() > 0);
        
        // Processing time should be reasonable (less than 30 seconds for this text)
        assertTrue(result.getProcessingTimeMs() < 30000);
    }
}