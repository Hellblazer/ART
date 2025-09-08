package com.hellblazer.art.nlp.core;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Collections;
import java.util.HashMap;
import java.util.ArrayList;

/**
 * Document analysis result containing comprehensive NLP analysis.
 * Extends ProcessingResult with document-specific analysis capabilities.
 */
public final class DocumentAnalysis {
    private final Document document;
    private final ProcessingResult processingResult;
    private final List<String> sentences;
    private final List<String> paragraphs;
    private final Map<String, Double> topicDistribution;
    private final DocumentSummary summary;
    private final Map<String, Object> analysisMetadata;
    
    private DocumentAnalysis(Builder builder) {
        this.document = Objects.requireNonNull(builder.document, "document cannot be null");
        this.processingResult = Objects.requireNonNull(builder.processingResult, "processingResult cannot be null");
        this.sentences = Collections.unmodifiableList(new ArrayList<>(builder.sentences));
        this.paragraphs = Collections.unmodifiableList(new ArrayList<>(builder.paragraphs));
        this.topicDistribution = Collections.unmodifiableMap(new HashMap<>(builder.topicDistribution));
        this.summary = builder.summary;
        this.analysisMetadata = Collections.unmodifiableMap(new HashMap<>(builder.analysisMetadata));
    }
    
    /**
     * Get the original document.
     */
    public Document getDocument() {
        return document;
    }
    
    /**
     * Get the core processing result.
     */
    public ProcessingResult getProcessingResult() {
        return processingResult;
    }
    
    /**
     * Get detected sentences.
     */
    public List<String> getSentences() {
        return sentences;
    }
    
    /**
     * Get detected paragraphs.
     */
    public List<String> getParagraphs() {
        return paragraphs;
    }
    
    /**
     * Get topic distribution.
     */
    public Map<String, Double> getTopicDistribution() {
        return topicDistribution;
    }
    
    /**
     * Get document summary (may be null).
     */
    public DocumentSummary getSummary() {
        return summary;
    }
    
    /**
     * Get analysis metadata.
     */
    public Map<String, Object> getAnalysisMetadata() {
        return analysisMetadata;
    }
    
    /**
     * Get analysis metadata value.
     */
    public Object getAnalysisMetadata(String key) {
        return analysisMetadata.get(key);
    }
    
    // Delegate methods to ProcessingResult for convenience
    
    /**
     * Get semantic categories.
     */
    public Map<String, Integer> getSemanticCategories() {
        return processingResult.getAllCategories();
    }
    
    /**
     * Get extracted entities.
     */
    public List<Entity> getEntities() {
        return processingResult.getEntities();
    }
    
    /**
     * Get sentiment scores.
     */
    public SentimentScore getSentiment() {
        return processingResult.getSentiment();
    }
    
    /**
     * Get processing time.
     */
    public long getProcessingTimeMs() {
        return processingResult.getProcessingTimeMs();
    }
    
    /**
     * Check if analysis was successful.
     */
    public boolean isSuccess() {
        return processingResult.isSuccess();
    }
    
    /**
     * Get document length statistics.
     */
    public LengthStatistics getLengthStatistics() {
        return new LengthStatistics(
            document.getContentLength(),
            sentences.size(),
            paragraphs.size(),
            processingResult.getTokenCount()
        );
    }
    
    @Override
    public String toString() {
        return String.format("DocumentAnalysis{title='%s', sentences=%d, entities=%d, sentiment=%s, time=%dms}",
                           document.getTitle() != null ? document.getTitle() : "Untitled",
                           sentences.size(),
                           processingResult.getEntities().size(),
                           processingResult.getSentiment() != null ? processingResult.getSentiment().getSentiment() : "none",
                           processingResult.getProcessingTimeMs());
    }
    
    /**
     * Document summary information.
     */
    public record DocumentSummary(
        String abstractText,
        List<String> keyPhrases,
        double relevanceScore,
        String primaryTopic
    ) {}
    
    /**
     * Document length statistics.
     */
    public record LengthStatistics(
        int characterCount,
        int sentenceCount,
        int paragraphCount,
        int tokenCount
    ) {
        public double averageWordsPerSentence() {
            return sentenceCount > 0 ? (double) tokenCount / sentenceCount : 0.0;
        }
        
        public double averageSentencesPerParagraph() {
            return paragraphCount > 0 ? (double) sentenceCount / paragraphCount : 0.0;
        }
        
        public double averageCharactersPerSentence() {
            return sentenceCount > 0 ? (double) characterCount / sentenceCount : 0.0;
        }
    }
    
    /**
     * Builder for DocumentAnalysis.
     */
    public static class Builder {
        private Document document;
        private ProcessingResult processingResult;
        private final List<String> sentences = new ArrayList<>();
        private final List<String> paragraphs = new ArrayList<>();
        private final Map<String, Double> topicDistribution = new HashMap<>();
        private DocumentSummary summary;
        private final Map<String, Object> analysisMetadata = new HashMap<>();
        
        /**
         * Set the document being analyzed.
         */
        public Builder withDocument(Document document) {
            this.document = document;
            return this;
        }
        
        /**
         * Set the core processing result.
         */
        public Builder withProcessingResult(ProcessingResult processingResult) {
            this.processingResult = processingResult;
            return this;
        }
        
        /**
         * Add detected sentence.
         */
        public Builder withSentence(String sentence) {
            this.sentences.add(Objects.requireNonNull(sentence));
            return this;
        }
        
        /**
         * Add detected sentences.
         */
        public Builder withSentences(List<String> sentences) {
            if (sentences != null) {
                this.sentences.addAll(sentences);
            }
            return this;
        }
        
        /**
         * Add detected paragraph.
         */
        public Builder withParagraph(String paragraph) {
            this.paragraphs.add(Objects.requireNonNull(paragraph));
            return this;
        }
        
        /**
         * Add detected paragraphs.
         */
        public Builder withParagraphs(List<String> paragraphs) {
            if (paragraphs != null) {
                this.paragraphs.addAll(paragraphs);
            }
            return this;
        }
        
        /**
         * Add topic probability.
         */
        public Builder withTopic(String topic, double probability) {
            this.topicDistribution.put(Objects.requireNonNull(topic), probability);
            return this;
        }
        
        /**
         * Add topic distribution.
         */
        public Builder withTopicDistribution(Map<String, Double> distribution) {
            if (distribution != null) {
                this.topicDistribution.putAll(distribution);
            }
            return this;
        }
        
        /**
         * Set document summary.
         */
        public Builder withSummary(DocumentSummary summary) {
            this.summary = summary;
            return this;
        }
        
        /**
         * Add analysis metadata.
         */
        public Builder withAnalysisMetadata(String key, Object value) {
            this.analysisMetadata.put(Objects.requireNonNull(key), value);
            return this;
        }
        
        /**
         * Add analysis metadata.
         */
        public Builder withAnalysisMetadata(Map<String, Object> metadata) {
            if (metadata != null) {
                this.analysisMetadata.putAll(metadata);
            }
            return this;
        }
        
        /**
         * Build immutable DocumentAnalysis.
         */
        public DocumentAnalysis build() {
            return new DocumentAnalysis(this);
        }
    }
    
    /**
     * Create new builder.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Create simple document analysis.
     */
    public static DocumentAnalysis of(Document document, ProcessingResult processingResult) {
        return new Builder()
            .withDocument(document)
            .withProcessingResult(processingResult)
            .build();
    }
}