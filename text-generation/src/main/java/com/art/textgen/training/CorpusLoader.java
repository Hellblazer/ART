package com.art.textgen.training;

import com.art.textgen.core.Vocabulary;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Corpus loader for training the text generation system
 * Supports multiple formats and batch processing
 */
public class CorpusLoader {
    
    private final Vocabulary vocabulary;
    private final List<Document> documents;
    private final Map<String, Integer> globalWordFrequency;
    private final AtomicInteger totalTokens;
    private final Set<String> stopWords;
    
    public static class Document {
        public final String id;
        public final String title;
        public final String content;
        public final List<String> tokens;
        public final Map<String, Integer> wordFrequency;
        public final String source;
        
        public Document(String id, String title, String content, List<String> tokens, String source) {
            this.id = id;
            this.title = title;
            this.content = content;
            this.tokens = tokens;
            this.source = source;
            this.wordFrequency = computeFrequency(tokens);
        }
        
        private Map<String, Integer> computeFrequency(List<String> tokens) {
            Map<String, Integer> freq = new HashMap<>();
            for (String token : tokens) {
                freq.merge(token, 1, Integer::sum);
            }
            return freq;
        }
        
        public List<List<String>> getSentences() {
            List<List<String>> sentences = new ArrayList<>();
            List<String> current = new ArrayList<>();
            
            for (String token : tokens) {
                current.add(token);
                if (token.matches("[.!?]")) {
                    sentences.add(new ArrayList<>(current));
                    current.clear();
                }
            }
            
            if (!current.isEmpty()) {
                sentences.add(current);
            }
            
            return sentences;
        }
    }
    
    public CorpusLoader(Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.documents = new ArrayList<>();
        this.globalWordFrequency = new HashMap<>();
        this.totalTokens = new AtomicInteger(0);
        this.stopWords = initializeStopWords();
    }
    
    /**
     * Initialize common stop words
     */
    private Set<String> initializeStopWords() {
        return new HashSet<>(Arrays.asList(
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "shall"
        ));
    }
    
    /**
     * Load corpus from directory
     */
    public void loadFromDirectory(String directoryPath) throws IOException {
        Path dir = Paths.get(directoryPath);
        
        if (!Files.exists(dir) || !Files.isDirectory(dir)) {
            throw new IOException("Invalid directory: " + directoryPath);
        }
        
        // Find all text files
        try (Stream<Path> paths = Files.walk(dir)) {
            List<Path> textFiles = paths
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().matches(".*\\.(txt|md|text)$"))
                .collect(Collectors.toList());
            
            System.out.println("Found " + textFiles.size() + " text files");
            
            // Load each file
            for (Path file : textFiles) {
                try {
                    loadFile(file);
                } catch (Exception e) {
                    System.err.println("Error loading " + file + ": " + e.getMessage());
                }
            }
        }
        
        System.out.println("Loaded " + documents.size() + " documents");
        System.out.println("Total tokens: " + totalTokens.get());
        System.out.println("Unique tokens: " + globalWordFrequency.size());
    }
    
    /**
     * Load single file
     */
    public void loadFile(Path file) throws IOException {
        String content = Files.readString(file);
        String filename = file.getFileName().toString();
        String id = "doc_" + documents.size();
        
        // Tokenize
        List<String> tokens = vocabulary.tokenize(content);
        
        // Add to vocabulary
        for (String token : tokens) {
            vocabulary.addToken(token);
            globalWordFrequency.merge(token, 1, Integer::sum);
        }
        
        totalTokens.addAndGet(tokens.size());
        
        // Create document
        Document doc = new Document(id, filename, content, tokens, file.toString());
        documents.add(doc);
        
        if (documents.size() % 10 == 0) {
            System.out.println("Loaded " + documents.size() + " documents...");
        }
    }
    
    /**
     * Load sample corpus for testing
     */
    public void loadSampleCorpus() {
        // Classic literature samples
        addSample("Alice in Wonderland", 
            "Alice was beginning to get very tired of sitting by her sister on the bank, " +
            "and of having nothing to do. Once or twice she had peeped into the book her sister " +
            "was reading, but it had no pictures or conversations in it. And what is the use " +
            "of a book, thought Alice, without pictures or conversations?");
        
        addSample("Pride and Prejudice",
            "It is a truth universally acknowledged, that a single man in possession of a good " +
            "fortune, must be in want of a wife. However little known the feelings or views of " +
            "such a man may be on his first entering a neighbourhood, this truth is so well " +
            "fixed in the minds of the surrounding families, that he is considered as the " +
            "rightful property of some one or other of their daughters.");
        
        addSample("1984",
            "It was a bright cold day in April, and the clocks were striking thirteen. " +
            "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, " +
            "slipped quickly through the glass doors of Victory Mansions, though not quickly enough " +
            "to prevent a swirl of gritty dust from entering along with him.");
        
        // Technical writing
        addSample("AI Introduction",
            "Artificial intelligence is the simulation of human intelligence processes by machines, " +
            "especially computer systems. These processes include learning, reasoning, and self-correction. " +
            "Particular applications of AI include expert systems, speech recognition, and machine vision. " +
            "Machine learning is a subset of artificial intelligence that provides systems the ability " +
            "to automatically learn and improve from experience without being explicitly programmed.");
        
        // Scientific writing
        addSample("Neuroscience",
            "The human brain contains approximately 86 billion neurons, each connected to thousands " +
            "of other neurons through synapses. These neural networks process information through " +
            "electrochemical signals. Neurons communicate via neurotransmitters released at synaptic " +
            "junctions. The patterns of neural activity give rise to cognition, emotion, and consciousness. " +
            "Understanding these mechanisms is fundamental to neuroscience and artificial intelligence.");
        
        // Philosophy
        addSample("Consciousness",
            "Consciousness refers to the state of being aware of and able to think about one's existence, " +
            "sensations, thoughts, and surroundings. The hard problem of consciousness is the question of " +
            "how and why we have qualitative, subjective experiences. This is distinct from the easy problems " +
            "of explaining cognitive functions. Some philosophers argue that consciousness is fundamental to " +
            "the universe, while others maintain it emerges from complex information processing.");
        
        System.out.println("Loaded sample corpus with " + documents.size() + " documents");
    }
    
    /**
     * Add sample document
     */
    private void addSample(String title, String content) {
        String id = "sample_" + documents.size();
        List<String> tokens = vocabulary.tokenize(content);
        
        for (String token : tokens) {
            vocabulary.addToken(token);
            globalWordFrequency.merge(token, 1, Integer::sum);
        }
        
        totalTokens.addAndGet(tokens.size());
        
        Document doc = new Document(id, title, content, tokens, "sample");
        documents.add(doc);
    }
    
    /**
     * Get all documents
     */
    public List<Document> getDocuments() {
        return new ArrayList<>(documents);
    }
    
    /**
     * Get sentences from all documents
     */
    public List<List<String>> getAllSentences() {
        List<List<String>> allSentences = new ArrayList<>();
        
        for (Document doc : documents) {
            allSentences.addAll(doc.getSentences());
        }
        
        return allSentences;
    }
    
    /**
     * Get word frequency statistics
     */
    public Map<String, Integer> getWordFrequency() {
        return new HashMap<>(globalWordFrequency);
    }
    
    /**
     * Get most frequent words
     */
    public List<String> getMostFrequentWords(int n) {
        return globalWordFrequency.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(n)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
    
    /**
     * Get content words (non-stop words)
     */
    public List<String> getContentWords() {
        return globalWordFrequency.keySet().stream()
            .filter(word -> !stopWords.contains(word.toLowerCase()))
            .filter(word -> word.length() > 2)
            .collect(Collectors.toList());
    }
    
    /**
     * Get statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("total_documents", documents.size());
        stats.put("total_tokens", totalTokens.get());
        stats.put("unique_tokens", globalWordFrequency.size());
        stats.put("vocabulary_size", vocabulary.size());
        
        // Average document length
        double avgLength = documents.stream()
            .mapToInt(d -> d.tokens.size())
            .average()
            .orElse(0.0);
        stats.put("avg_document_length", avgLength);
        
        // Most frequent words
        stats.put("top_10_words", getMostFrequentWords(10));
        
        // Content word ratio
        double contentRatio = getContentWords().size() / (double) globalWordFrequency.size();
        stats.put("content_word_ratio", contentRatio);
        
        return stats;
    }
    
    /**
     * Save corpus metadata
     */
    public void saveMetadata(String filepath) throws IOException {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("statistics", getStatistics());
        metadata.put("documents", documents.stream()
            .map(d -> Map.of("id", d.id, "title", d.title, "tokens", d.tokens.size()))
            .collect(Collectors.toList()));
        
        // Simple JSON-like format
        try (PrintWriter writer = new PrintWriter(filepath)) {
            writer.println("Corpus Metadata");
            writer.println("===============");
            for (Map.Entry<String, Object> entry : metadata.entrySet()) {
                writer.println(entry.getKey() + ": " + entry.getValue());
            }
        }
    }
}
