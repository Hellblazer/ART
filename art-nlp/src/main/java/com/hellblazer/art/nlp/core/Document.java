package com.hellblazer.art.nlp.core;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Document representation for NLP processing with metadata support.
 * Immutable document container with builder pattern for construction.
 */
public final class Document {
    private final String content;
    private final String title;
    private final String source;
    private final Instant timestamp;
    private final Map<String, Object> metadata;
    private final String language;
    private final DocumentType type;
    
    /**
     * Document types for processing optimization.
     */
    public enum DocumentType {
        PLAIN_TEXT,
        MARKDOWN,
        HTML,
        PDF,
        EMAIL,
        SOCIAL_MEDIA,
        ACADEMIC_PAPER,
        NEWS_ARTICLE,
        LEGAL_DOCUMENT,
        TECHNICAL_DOCUMENTATION,
        UNKNOWN
    }
    
    private Document(Builder builder) {
        this.content = Objects.requireNonNull(builder.content, "content cannot be null");
        this.title = builder.title;
        this.source = builder.source;
        this.timestamp = builder.timestamp != null ? builder.timestamp : Instant.now();
        this.metadata = Collections.unmodifiableMap(new HashMap<>(builder.metadata));
        this.language = builder.language != null ? builder.language : "en";
        this.type = builder.type != null ? builder.type : DocumentType.PLAIN_TEXT;
        
        if (content.isBlank()) {
            throw new IllegalArgumentException("Document content cannot be blank");
        }
    }
    
    /**
     * Get document content.
     */
    public String getContent() {
        return content;
    }
    
    /**
     * Get document title (may be null).
     */
    public String getTitle() {
        return title;
    }
    
    /**
     * Get document source (may be null).
     */
    public String getSource() {
        return source;
    }
    
    /**
     * Get document timestamp.
     */
    public Instant getTimestamp() {
        return timestamp;
    }
    
    /**
     * Get document metadata.
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    /**
     * Get metadata value.
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }
    
    /**
     * Get document language.
     */
    public String getLanguage() {
        return language;
    }
    
    /**
     * Get document type.
     */
    public DocumentType getType() {
        return type;
    }
    
    /**
     * Get content length in characters.
     */
    public int getContentLength() {
        return content.length();
    }
    
    /**
     * Check if document has metadata key.
     */
    public boolean hasMetadata(String key) {
        return metadata.containsKey(key);
    }
    
    /**
     * Get document summary information.
     */
    public String getSummary() {
        var titleStr = title != null ? title : "Untitled";
        var sourceStr = source != null ? " from " + source : "";
        return String.format("%s%s (%d chars, %s)", 
                           titleStr, sourceStr, content.length(), type);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Document document = (Document) obj;
        return Objects.equals(content, document.content) &&
               Objects.equals(title, document.title) &&
               Objects.equals(source, document.source) &&
               Objects.equals(timestamp, document.timestamp) &&
               Objects.equals(metadata, document.metadata) &&
               Objects.equals(language, document.language) &&
               type == document.type;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(content, title, source, timestamp, metadata, language, type);
    }
    
    @Override
    public String toString() {
        return String.format("Document{title='%s', source='%s', type=%s, length=%d, metadata=%d}",
                           title, source, type, content.length(), metadata.size());
    }
    
    /**
     * Builder for Document construction.
     */
    public static class Builder {
        private String content;
        private String title;
        private String source;
        private Instant timestamp;
        private final Map<String, Object> metadata = new HashMap<>();
        private String language;
        private DocumentType type;
        
        /**
         * Set document content.
         */
        public Builder withContent(String content) {
            this.content = content;
            return this;
        }
        
        /**
         * Set document title.
         */
        public Builder withTitle(String title) {
            this.title = title;
            return this;
        }
        
        /**
         * Set document source.
         */
        public Builder withSource(String source) {
            this.source = source;
            return this;
        }
        
        /**
         * Set document timestamp.
         */
        public Builder withTimestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }
        
        /**
         * Add metadata entry.
         */
        public Builder withMetadata(String key, Object value) {
            this.metadata.put(Objects.requireNonNull(key), value);
            return this;
        }
        
        /**
         * Add multiple metadata entries.
         */
        public Builder withMetadata(Map<String, Object> metadata) {
            if (metadata != null) {
                this.metadata.putAll(metadata);
            }
            return this;
        }
        
        /**
         * Set document language.
         */
        public Builder withLanguage(String language) {
            this.language = language;
            return this;
        }
        
        /**
         * Set document type.
         */
        public Builder withType(DocumentType type) {
            this.type = type;
            return this;
        }
        
        /**
         * Build immutable Document.
         */
        public Document build() {
            return new Document(this);
        }
    }
    
    /**
     * Create new builder.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Create simple text document.
     */
    public static Document of(String content) {
        return new Builder().withContent(content).build();
    }
    
    /**
     * Create document with title and content.
     */
    public static Document of(String title, String content) {
        return new Builder()
            .withTitle(title)
            .withContent(content)
            .build();
    }
}