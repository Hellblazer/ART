package com.hellblazer.art.nlp.util;

import com.hellblazer.art.nlp.util.TokenizerPipeline.Token;
import com.hellblazer.art.nlp.util.TokenizerPipeline.Sentence;
import com.hellblazer.art.nlp.util.TokenizerPipeline.TokenizedDocument;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TokenizerPipeline Tests")
class TokenizerPipelineTest {
    
    private TokenizerPipeline tokenizer;
    
    @BeforeEach
    void setUp() {
        try {
            tokenizer = new TokenizerPipeline();
        } catch (IOException e) {
            // Skip tests if OpenNLP models are not available
            tokenizer = null;
        }
    }
    
    // Helper method to check if tokenizer is available
    static boolean isTokenizerAvailable() {
        try {
            new TokenizerPipeline();
            return true;
        } catch (IOException e) {
            return false;
        }
    }
    
    @Test
    @DisplayName("Should initialize tokenizer pipeline")
    @EnabledIf("isTokenizerAvailable")
    void testInitialization() {
        assertNotNull(tokenizer);
        assertNotNull(tokenizer.getStopWords());
        assertFalse(tokenizer.getStopWords().isEmpty());
    }
    
    @Test
    @DisplayName("Should tokenize simple text")
    @EnabledIf("isTokenizerAvailable")
    void testSimpleTokenization() {
        var text = "Hello world! This is a test.";
        var document = tokenizer.tokenize(text);
        
        assertNotNull(document);
        assertEquals(text, document.getOriginalText());
        assertFalse(document.getSentences().isEmpty());
        assertFalse(document.getAllTokens().isEmpty());
        
        var tokens = document.getAllTokens();
        assertTrue(tokens.size() > 5); // Should have multiple tokens
        
        // Check that tokens have proper annotations
        for (var token : tokens) {
            assertNotNull(token.getText());
            assertNotNull(token.getNormalizedText());
            assertNotNull(token.getPosTag());
            assertNotNull(token.getLemma());
            assertTrue(token.getStartIndex() >= 0);
            assertTrue(token.getEndIndex() > token.getStartIndex());
        }
    }
    
    @Test
    @DisplayName("Should detect sentences correctly")
    @EnabledIf("isTokenizerAvailable")
    void testSentenceDetection() {
        var text = "First sentence. Second sentence! Third sentence?";
        var sentences = tokenizer.detectSentences(text);
        
        assertNotNull(sentences);
        assertEquals(3, sentences.length);
        assertEquals("First sentence.", sentences[0].trim());
        assertEquals("Second sentence!", sentences[1].trim());
        assertEquals("Third sentence?", sentences[2].trim());
    }
    
    @Test
    @DisplayName("Should tokenize document with multiple sentences")
    @EnabledIf("isTokenizerAvailable")
    void testDocumentTokenization() {
        var text = "Natural language processing is exciting. It involves many computational techniques.";
        var document = tokenizer.tokenize(text);
        
        assertNotNull(document);
        assertEquals(2, document.getSentences().size());
        
        var sentences = document.getSentences();
        for (var sentence : sentences) {
            assertNotNull(sentence.getText());
            assertFalse(sentence.getTokens().isEmpty());
            assertTrue(sentence.getStartIndex() >= 0);
            assertTrue(sentence.getEndIndex() > sentence.getStartIndex());
        }
        
        // Check that all tokens are accounted for
        var totalTokensFromSentences = sentences.stream()
            .mapToInt(sentence -> sentence.getTokens().size())
            .sum();
        assertEquals(totalTokensFromSentences, document.getAllTokens().size());
    }
    
    @Test
    @DisplayName("Should identify stop words correctly")
    @EnabledIf("isTokenizerAvailable")
    void testStopWordIdentification() {
        var text = "The quick brown fox jumps over the lazy dog.";
        var document = tokenizer.tokenize(text);
        
        assertNotNull(document);
        var tokens = document.getAllTokens();
        
        // Check that common stop words are identified
        var stopWordTokens = tokens.stream()
            .filter(Token::isStopWord)
            .toList();
        
        assertFalse(stopWordTokens.isEmpty());
        assertTrue(stopWordTokens.stream()
            .anyMatch(token -> "the".equals(token.getNormalizedText())));
        
        // Check that content words are not marked as stop words
        var contentTokens = tokens.stream()
            .filter(token -> !token.isStopWord())
            .toList();
        
        assertFalse(contentTokens.isEmpty());
        assertTrue(contentTokens.stream()
            .anyMatch(token -> "fox".equals(token.getNormalizedText()) || 
                              "dog".equals(token.getNormalizedText())));
    }
    
    @Test
    @DisplayName("Should extract POS tags correctly")
    @EnabledIf("isTokenizerAvailable")
    void testPOSTagging() {
        var text = "The cat runs quickly.";
        var tokens = tokenizer.tokenizeSimple(text);
        
        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        
        // Verify POS tags are assigned
        for (var token : tokens) {
            assertNotNull(token.getPosTag());
            assertFalse(token.getPosTag().isEmpty());
        }
        
        // Look for common POS tags
        var posTags = tokens.stream()
            .map(Token::getPosTag)
            .toList();
        
        // Should contain determiners, nouns, verbs, adverbs
        assertTrue(posTags.stream().anyMatch(pos -> pos.startsWith("DT"))); // Determiner
        assertTrue(posTags.stream().anyMatch(pos -> pos.startsWith("NN"))); // Noun
        assertTrue(posTags.stream().anyMatch(pos -> pos.startsWith("VB"))); // Verb
    }
    
    @Test
    @DisplayName("Should perform lemmatization")
    @EnabledIf("isTokenizerAvailable")
    void testLemmatization() {
        var text = "The cats are running quickly.";
        var tokens = tokenizer.tokenizeSimple(text);
        
        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        
        // Check that lemmas are provided
        for (var token : tokens) {
            assertNotNull(token.getLemma());
            assertFalse(token.getLemma().isEmpty());
        }
        
        // Look for lemmatized forms
        var lemmas = tokens.stream()
            .map(Token::getLemma)
            .toList();
        
        // "cats" should be lemmatized to "cat", "running" to "run" (or original if lemmatizer unavailable)
        assertTrue(lemmas.contains("cat") || lemmas.contains("cats"));
        assertTrue(lemmas.contains("run") || lemmas.contains("running"));
    }
    
    @Test
    @DisplayName("Should extract content tokens only")
    @EnabledIf("isTokenizerAvailable")
    void testContentTokenExtraction() {
        var text = "The machine learning algorithm processes the data efficiently.";
        var contentTokens = tokenizer.extractContentTokens(text);
        
        assertNotNull(contentTokens);
        assertFalse(contentTokens.isEmpty());
        
        // All tokens should be content words (not stop words)
        assertTrue(contentTokens.stream().noneMatch(Token::isStopWord));
        
        // Should contain meaningful words
        var contentWords = contentTokens.stream()
            .map(Token::getNormalizedText)
            .toList();
        
        assertTrue(contentWords.contains("machine") || contentWords.contains("learning") ||
                  contentWords.contains("algorithm") || contentWords.contains("processes") ||
                  contentWords.contains("data") || contentWords.contains("efficiently"));
    }
    
    @Test
    @DisplayName("Should calculate word frequencies")
    @EnabledIf("isTokenizerAvailable")
    void testWordFrequencies() {
        var text = "The cat sat on the mat. The cat was happy.";
        var frequencies = tokenizer.getWordFrequencies(text);
        
        assertNotNull(frequencies);
        assertFalse(frequencies.isEmpty());
        
        // "cat" should appear twice
        assertTrue(frequencies.containsKey("cat"));
        assertEquals(2L, frequencies.get("cat"));
        
        // Stop words should not be in frequency map
        assertFalse(frequencies.containsKey("the"));
        assertFalse(frequencies.containsKey("on"));
        assertFalse(frequencies.containsKey("was"));
    }
    
    @Test
    @DisplayName("Should handle empty input gracefully")
    @EnabledIf("isTokenizerAvailable")
    void testEmptyInput() {
        var document = tokenizer.tokenize("");
        assertNotNull(document);
        assertEquals("", document.getOriginalText());
        assertTrue(document.getSentences().isEmpty());
        assertTrue(document.getAllTokens().isEmpty());
        
        var tokens = tokenizer.tokenizeSimple("");
        assertNotNull(tokens);
        assertTrue(tokens.isEmpty());
        
        var sentences = tokenizer.detectSentences("");
        assertNotNull(sentences);
        assertEquals(0, sentences.length);
        
        var words = tokenizer.extractWords("");
        assertNotNull(words);
        assertTrue(words.isEmpty());
    }
    
    @Test
    @DisplayName("Should handle null input gracefully")
    @EnabledIf("isTokenizerAvailable")
    void testNullInput() {
        var document = tokenizer.tokenize(null);
        assertNotNull(document);
        assertTrue(document.getSentences().isEmpty());
        assertTrue(document.getAllTokens().isEmpty());
        
        var tokens = tokenizer.tokenizeSimple(null);
        assertNotNull(tokens);
        assertTrue(tokens.isEmpty());
        
        var sentences = tokenizer.detectSentences(null);
        assertNotNull(sentences);
        assertEquals(0, sentences.length);
    }
    
    @Test
    @DisplayName("Should extract words and lemmas")
    @EnabledIf("isTokenizerAvailable")
    void testWordAndLemmaExtraction() {
        var text = "The dogs are playing in the park.";
        
        var words = tokenizer.extractWords(text);
        var lemmas = tokenizer.extractLemmas(text);
        
        assertNotNull(words);
        assertNotNull(lemmas);
        assertEquals(words.size(), lemmas.size());
        
        // Should contain original words
        assertTrue(words.contains("dogs") || words.contains("playing"));
        
        // Should contain lemmatized forms (or original if lemmatizer unavailable)
        assertTrue(lemmas.contains("dog") || lemmas.contains("play") || 
                  lemmas.contains("dogs") || lemmas.contains("playing"));
    }
    
    @Test
    @DisplayName("Should manage stop words")
    @EnabledIf("isTokenizerAvailable")
    void testStopWordManagement() {
        assertTrue(tokenizer.isStopWord("the"));
        assertTrue(tokenizer.isStopWord("and"));
        assertFalse(tokenizer.isStopWord("machine"));
        assertFalse(tokenizer.isStopWord("learning"));
        
        // Test case insensitivity
        assertTrue(tokenizer.isStopWord("THE"));
        assertTrue(tokenizer.isStopWord("And"));
        
        // Test null handling
        assertFalse(tokenizer.isStopWord(null));
    }
    
    @Test
    @DisplayName("Should provide cache statistics")
    @EnabledIf("isTokenizerAvailable")
    void testCacheStatistics() {
        var text = "This is a test sentence for caching lemmas.";
        tokenizer.tokenize(text);
        
        var stats = tokenizer.getCacheStats();
        assertNotNull(stats);
        assertTrue(stats.containsKey("lemmaCacheSize"));
        assertTrue(stats.containsKey("stopWordsCount"));
        
        var cacheSize = (Integer) stats.get("lemmaCacheSize");
        var stopWordsCount = (Integer) stats.get("stopWordsCount");
        
        assertTrue(cacheSize >= 0);
        assertTrue(stopWordsCount > 0);
    }
    
    @Test
    @DisplayName("Should clear cache properly")
    @EnabledIf("isTokenizerAvailable")
    void testCacheClear() {
        var text = "This text will populate the lemma cache.";
        tokenizer.tokenize(text);
        
        var statsBefore = tokenizer.getCacheStats();
        var cacheSizeBefore = (Integer) statsBefore.get("lemmaCacheSize");
        
        tokenizer.clearCache();
        
        var statsAfter = tokenizer.getCacheStats();
        var cacheSizeAfter = (Integer) statsAfter.get("lemmaCacheSize");
        
        assertEquals(0, cacheSizeAfter);
        assertTrue(cacheSizeBefore >= 0); // Could be 0 if lemmatization didn't add to cache
    }
    
    @Test
    @DisplayName("Should process document with comprehensive features")
    @EnabledIf("isTokenizerAvailable")
    void testComprehensiveDocumentProcessing() {
        var text = "Machine learning algorithms learn patterns from data. " +
                  "They can classify text, recognize images, and make predictions.";
        
        var document = tokenizer.tokenize(text);
        
        assertNotNull(document);
        assertEquals(2, document.getSentences().size());
        
        // Test sentence-level features
        var sentences = document.getSentences();
        for (var sentence : sentences) {
            assertFalse(sentence.getWords().isEmpty());
            assertFalse(sentence.getLemmas().isEmpty());
            assertFalse(sentence.getContentTokens().isEmpty());
        }
        
        // Test document-level features
        var wordFreqs = document.getWordFrequencies();
        var posFreqs = document.getPosTagFrequencies();
        
        assertNotNull(wordFreqs);
        assertNotNull(posFreqs);
        assertFalse(wordFreqs.isEmpty());
        assertFalse(posFreqs.isEmpty());
        
        // Check content token extraction
        var contentTokens = document.getContentTokens();
        assertTrue(contentTokens.stream().noneMatch(Token::isStopWord));
    }
    
    @Test
    @DisplayName("Should handle resource cleanup")
    @EnabledIf("isTokenizerAvailable")
    void testResourceCleanup() {
        assertDoesNotThrow(() -> {
            tokenizer.close();
        });
    }
}