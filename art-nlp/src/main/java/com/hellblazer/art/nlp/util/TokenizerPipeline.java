package com.hellblazer.art.nlp.util;

import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.postag.POSModel;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Comprehensive tokenization pipeline using OpenNLP models.
 * Provides sentence detection, tokenization, POS tagging, and lemmatization.
 */
public class TokenizerPipeline implements AutoCloseable {
    
    private final SentenceDetector sentenceDetector;
    private final Tokenizer tokenizer;
    private final POSTagger posTagger;
    private final Lemmatizer lemmatizer;
    private final Map<String, String> lemmaCache;
    private final Set<String> stopWords;
    
    /**
     * Token representation with linguistic annotations.
     */
    public static class Token {
        private final String text;
        private final String normalizedText;
        private final String posTag;
        private final String lemma;
        private final int startIndex;
        private final int endIndex;
        private final boolean isStopWord;
        
        public Token(String text, String normalizedText, String posTag, String lemma, 
                    int startIndex, int endIndex, boolean isStopWord) {
            this.text = text;
            this.normalizedText = normalizedText;
            this.posTag = posTag;
            this.lemma = lemma;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
            this.isStopWord = isStopWord;
        }
        
        public String getText() { return text; }
        public String getNormalizedText() { return normalizedText; }
        public String getPosTag() { return posTag; }
        public String getLemma() { return lemma; }
        public int getStartIndex() { return startIndex; }
        public int getEndIndex() { return endIndex; }
        public boolean isStopWord() { return isStopWord; }
        
        @Override
        public String toString() {
            return String.format("Token{text='%s', pos='%s', lemma='%s'}", text, posTag, lemma);
        }
    }
    
    /**
     * Sentence representation with tokens.
     */
    public static class Sentence {
        private final String text;
        private final List<Token> tokens;
        private final int startIndex;
        private final int endIndex;
        
        public Sentence(String text, List<Token> tokens, int startIndex, int endIndex) {
            this.text = text;
            this.tokens = Collections.unmodifiableList(new ArrayList<>(tokens));
            this.startIndex = startIndex;
            this.endIndex = endIndex;
        }
        
        public String getText() { return text; }
        public List<Token> getTokens() { return tokens; }
        public int getStartIndex() { return startIndex; }
        public int getEndIndex() { return endIndex; }
        
        public List<Token> getContentTokens() {
            return tokens.stream()
                    .filter(token -> !token.isStopWord())
                    .collect(Collectors.toList());
        }
        
        public List<String> getWords() {
            return tokens.stream()
                    .map(Token::getText)
                    .collect(Collectors.toList());
        }
        
        public List<String> getLemmas() {
            return tokens.stream()
                    .map(Token::getLemma)
                    .collect(Collectors.toList());
        }
        
        @Override
        public String toString() {
            return String.format("Sentence{text='%s', tokenCount=%d}", 
                    text.length() > 50 ? text.substring(0, 50) + "..." : text, 
                    tokens.size());
        }
    }
    
    /**
     * Document representation with sentences and tokens.
     */
    public static class TokenizedDocument {
        private final String originalText;
        private final List<Sentence> sentences;
        private final List<Token> allTokens;
        
        public TokenizedDocument(String originalText, List<Sentence> sentences) {
            this.originalText = originalText;
            this.sentences = Collections.unmodifiableList(new ArrayList<>(sentences));
            this.allTokens = sentences.stream()
                    .flatMap(sentence -> sentence.getTokens().stream())
                    .collect(Collectors.toUnmodifiableList());
        }
        
        public String getOriginalText() { return originalText; }
        public List<Sentence> getSentences() { return sentences; }
        public List<Token> getAllTokens() { return allTokens; }
        
        public List<Token> getContentTokens() {
            return allTokens.stream()
                    .filter(token -> !token.isStopWord())
                    .collect(Collectors.toList());
        }
        
        public List<String> getAllWords() {
            return allTokens.stream()
                    .map(Token::getText)
                    .collect(Collectors.toList());
        }
        
        public List<String> getAllLemmas() {
            return allTokens.stream()
                    .map(Token::getLemma)
                    .collect(Collectors.toList());
        }
        
        public Map<String, Long> getWordFrequencies() {
            return allTokens.stream()
                    .filter(token -> !token.isStopWord())
                    .map(Token::getNormalizedText)
                    .collect(Collectors.groupingBy(
                            word -> word,
                            Collectors.counting()
                    ));
        }
        
        public Map<String, Long> getPosTagFrequencies() {
            return allTokens.stream()
                    .map(Token::getPosTag)
                    .collect(Collectors.groupingBy(
                            pos -> pos,
                            Collectors.counting()
                    ));
        }
        
        @Override
        public String toString() {
            return String.format("TokenizedDocument{sentenceCount=%d, tokenCount=%d}", 
                    sentences.size(), allTokens.size());
        }
    }
    
    /**
     * Create TokenizerPipeline with default OpenNLP models.
     * Models are loaded from classpath resources.
     */
    public TokenizerPipeline() throws IOException {
        this.lemmaCache = new ConcurrentHashMap<>();
        this.stopWords = initializeStopWords();
        
        // Load OpenNLP models from resources
        this.sentenceDetector = loadSentenceDetector();
        this.tokenizer = loadTokenizer();
        this.posTagger = loadPosTagger();
        this.lemmatizer = loadLemmatizer();
    }
    
    private SentenceDetector loadSentenceDetector() throws IOException {
        try (var modelStream = getResourceStream("opennlp/en-sent.bin")) {
            var sentenceModel = new SentenceModel(modelStream);
            return new SentenceDetectorME(sentenceModel);
        }
    }
    
    private Tokenizer loadTokenizer() throws IOException {
        try (var modelStream = getResourceStream("opennlp/en-token.bin")) {
            var tokenizerModel = new TokenizerModel(modelStream);
            return new TokenizerME(tokenizerModel);
        }
    }
    
    private POSTagger loadPosTagger() throws IOException {
        try (var modelStream = getResourceStream("opennlp/en-pos-maxent.bin")) {
            var posModel = new POSModel(modelStream);
            return new POSTaggerME(posModel);
        }
    }
    
    private Lemmatizer loadLemmatizer() throws IOException {
        try (var modelStream = getResourceStream("opennlp/en-lemmatizer.bin")) {
            var lemmatizerModel = new LemmatizerModel(modelStream);
            return new LemmatizerME(lemmatizerModel);
        } catch (IOException e) {
            // Fallback: return null if lemmatizer model is not available
            System.out.println("Warning: Lemmatizer model not found, using fallback implementation");
            return null;
        }
    }
    
    private InputStream getResourceStream(String resourcePath) throws IOException {
        var stream = getClass().getClassLoader().getResourceAsStream("models/" + resourcePath);
        if (stream == null) {
            throw new IOException("Could not find OpenNLP model resource: " + resourcePath);
        }
        return stream;
    }
    
    private Set<String> initializeStopWords() {
        return Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their",
            "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more",
            "go", "no", "way", "could", "my", "than", "first", "been", "call",
            "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
            "come", "made", "may", "part"
        );
    }
    
    /**
     * Tokenize input text into a structured document.
     * 
     * @param text Input text to tokenize
     * @return TokenizedDocument with sentences and tokens
     */
    public TokenizedDocument tokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new TokenizedDocument("", Collections.emptyList());
        }
        
        var sentences = new ArrayList<Sentence>();
        var sentenceSpans = sentenceDetector.sentPosDetect(text);
        
        for (var span : sentenceSpans) {
            var sentenceText = span.getCoveredText(text).toString();
            var sentenceTokens = tokenizeSentence(sentenceText, span.getStart());
            
            sentences.add(new Sentence(
                sentenceText,
                sentenceTokens,
                span.getStart(),
                span.getEnd()
            ));
        }
        
        return new TokenizedDocument(text, sentences);
    }
    
    private List<Token> tokenizeSentence(String sentenceText, int sentenceOffset) {
        var tokens = new ArrayList<Token>();
        var tokenSpans = tokenizer.tokenizePos(sentenceText);
        var tokenTexts = new String[tokenSpans.length];
        
        // Extract token texts
        for (int i = 0; i < tokenSpans.length; i++) {
            tokenTexts[i] = tokenSpans[i].getCoveredText(sentenceText).toString();
        }
        
        // Get POS tags
        var posTags = posTagger.tag(tokenTexts);
        
        // Get lemmas (with fallback if lemmatizer is not available)
        var lemmas = (lemmatizer != null) ? lemmatizer.lemmatize(tokenTexts, posTags) : tokenTexts;
        
        // Create token objects
        for (int i = 0; i < tokenTexts.length; i++) {
            var tokenText = tokenTexts[i];
            var normalizedText = tokenText.toLowerCase();
            var posTag = posTags[i];
            var lemma = getLemmaWithCache(tokenText, posTag, lemmas[i]);
            var isStopWord = stopWords.contains(normalizedText);
            
            tokens.add(new Token(
                tokenText,
                normalizedText,
                posTag,
                lemma,
                sentenceOffset + tokenSpans[i].getStart(),
                sentenceOffset + tokenSpans[i].getEnd(),
                isStopWord
            ));
        }
        
        return tokens;
    }
    
    private String getLemmaWithCache(String token, String posTag, String lemma) {
        var cacheKey = token + "_" + posTag;
        return lemmaCache.computeIfAbsent(cacheKey, k -> 
            lemma != null && !lemma.equals("O") ? lemma : token.toLowerCase()
        );
    }
    
    /**
     * Simple tokenization without sentence detection.
     * 
     * @param text Input text
     * @return List of tokens
     */
    public List<Token> tokenizeSimple(String text) {
        if (text == null || text.trim().isEmpty()) {
            return Collections.emptyList();
        }
        
        return tokenizeSentence(text, 0);
    }
    
    /**
     * Extract sentences from text.
     * 
     * @param text Input text
     * @return Array of sentence strings
     */
    public String[] detectSentences(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new String[0];
        }
        
        return sentenceDetector.sentDetect(text);
    }
    
    /**
     * Extract words only (no linguistic annotations).
     * 
     * @param text Input text
     * @return List of word strings
     */
    public List<String> extractWords(String text) {
        var document = tokenize(text);
        return document.getAllWords();
    }
    
    /**
     * Extract lemmas only.
     * 
     * @param text Input text
     * @return List of lemma strings
     */
    public List<String> extractLemmas(String text) {
        var document = tokenize(text);
        return document.getAllLemmas();
    }
    
    /**
     * Extract content words (excluding stop words).
     * 
     * @param text Input text
     * @return List of content word tokens
     */
    public List<Token> extractContentTokens(String text) {
        var document = tokenize(text);
        return document.getContentTokens();
    }
    
    /**
     * Get word frequencies from text.
     * 
     * @param text Input text
     * @return Map of words to frequencies
     */
    public Map<String, Long> getWordFrequencies(String text) {
        var document = tokenize(text);
        return document.getWordFrequencies();
    }
    
    /**
     * Check if a word is a stop word.
     * 
     * @param word Word to check
     * @return True if the word is a stop word
     */
    public boolean isStopWord(String word) {
        return word != null && stopWords.contains(word.toLowerCase());
    }
    
    /**
     * Add custom stop words.
     * 
     * @param words Words to add as stop words
     */
    public void addStopWords(String... words) {
        if (words != null) {
            var mutableStopWords = new HashSet<>(stopWords);
            for (var word : words) {
                if (word != null) {
                    mutableStopWords.add(word.toLowerCase().trim());
                }
            }
        }
    }
    
    /**
     * Get all configured stop words.
     * 
     * @return Set of stop words
     */
    public Set<String> getStopWords() {
        return new HashSet<>(stopWords);
    }
    
    /**
     * Clear the lemma cache.
     */
    public void clearCache() {
        lemmaCache.clear();
    }
    
    /**
     * Get cache statistics.
     * 
     * @return Map with cache size and other metrics
     */
    public Map<String, Object> getCacheStats() {
        var stats = new HashMap<String, Object>();
        stats.put("lemmaCacheSize", lemmaCache.size());
        stats.put("stopWordsCount", stopWords.size());
        return stats;
    }
    
    @Override
    public void close() {
        // Clear caches
        lemmaCache.clear();
        
        // OpenNLP models don't need explicit closing
        // They will be garbage collected
    }
    
    @Override
    public String toString() {
        return String.format("TokenizerPipeline{cacheSize=%d, stopWords=%d}", 
                lemmaCache.size(), stopWords.size());
    }
}