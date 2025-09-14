package com.hellblazer.art.hartcq.templates;

import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Immutable template definition that prevents hallucination by enforcing
 * strict output boundaries through predefined patterns and variable placeholders.
 * 
 * Templates support variable placeholders in the format [VARIABLE_NAME] and
 * include pattern matching rules for input classification and confidence scoring.
 * 
 * @author Claude Code
 */
public record Template(
    String id,
    String category,
    String pattern,
    List<String> variables,
    List<Pattern> matchingPatterns,
    double baseConfidence,
    Map<String, Object> metadata
) {
    
    /**
     * Template categories for classification
     */
    public enum Category {
        GREETING("greeting"),
        QUESTION("question"), 
        STATEMENT("statement"),
        RESPONSE("response"),
        TRANSITION("transition");
        
        private final String value;
        
        Category(String value) {
            this.value = value;
        }
        
        public String getValue() {
            return value;
        }
    }
    
    /**
     * Builder for creating templates with validation
     */
    public static class Builder {
        private String id;
        private String category;
        private String pattern;
        private final List<String> variables = new ArrayList<>();
        private final List<Pattern> matchingPatterns = new ArrayList<>();
        private double baseConfidence = 0.5;
        private final Map<String, Object> metadata = new HashMap<>();
        
        public Builder id(String id) {
            this.id = Objects.requireNonNull(id, "Template ID cannot be null");
            return this;
        }
        
        public Builder category(String category) {
            this.category = Objects.requireNonNull(category, "Category cannot be null");
            return this;
        }
        
        public Builder category(Category category) {
            this.category = category.getValue();
            return this;
        }
        
        public Builder pattern(String pattern) {
            this.pattern = Objects.requireNonNull(pattern, "Pattern cannot be null");
            // Extract variables from pattern
            this.variables.clear();
            var variablePattern = Pattern.compile("\\[([A-Z_]+)\\]");
            var matcher = variablePattern.matcher(pattern);
            while (matcher.find()) {
                var variable = matcher.group(1);
                if (!variables.contains(variable)) {
                    variables.add(variable);
                }
            }
            return this;
        }
        
        public Builder addMatchingPattern(String regex) {
            this.matchingPatterns.add(Pattern.compile(regex, Pattern.CASE_INSENSITIVE));
            return this;
        }
        
        public Builder addMatchingPattern(Pattern pattern) {
            this.matchingPatterns.add(Objects.requireNonNull(pattern));
            return this;
        }
        
        public Builder baseConfidence(double confidence) {
            if (confidence < 0.0 || confidence > 1.0) {
                throw new IllegalArgumentException("Confidence must be between 0.0 and 1.0");
            }
            this.baseConfidence = confidence;
            return this;
        }
        
        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }
        
        public Template build() {
            if (id == null || id.trim().isEmpty()) {
                throw new IllegalStateException("Template ID is required");
            }
            if (category == null || category.trim().isEmpty()) {
                throw new IllegalStateException("Template category is required");
            }
            if (pattern == null || pattern.trim().isEmpty()) {
                throw new IllegalStateException("Template pattern is required");
            }
            
            return new Template(
                id.trim(),
                category.trim(),
                pattern.trim(),
                List.copyOf(variables),
                List.copyOf(matchingPatterns),
                baseConfidence,
                Map.copyOf(metadata)
            );
        }
    }
    
    /**
     * Create a new template builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Compact constructor with validation
     */
    public Template {
        Objects.requireNonNull(id, "Template ID cannot be null");
        Objects.requireNonNull(category, "Category cannot be null");
        Objects.requireNonNull(pattern, "Pattern cannot be null");
        Objects.requireNonNull(variables, "Variables list cannot be null");
        Objects.requireNonNull(matchingPatterns, "Matching patterns list cannot be null");
        Objects.requireNonNull(metadata, "Metadata map cannot be null");
        
        if (id.trim().isEmpty()) {
            throw new IllegalArgumentException("Template ID cannot be empty");
        }
        if (category.trim().isEmpty()) {
            throw new IllegalArgumentException("Category cannot be empty");
        }
        if (pattern.trim().isEmpty()) {
            throw new IllegalArgumentException("Pattern cannot be empty");
        }
        if (baseConfidence < 0.0 || baseConfidence > 1.0) {
            throw new IllegalArgumentException("Base confidence must be between 0.0 and 1.0");
        }
        
        // Make defensive copies
        variables = List.copyOf(variables);
        matchingPatterns = List.copyOf(matchingPatterns);
        metadata = Map.copyOf(metadata);
    }
    
    /**
     * Calculate confidence score for given input text
     */
    public double calculateConfidence(String input) {
        if (input == null || input.trim().isEmpty()) {
            return 0.0;
        }
        
        if (matchingPatterns.isEmpty()) {
            return baseConfidence;
        }
        
        var maxScore = 0.0;
        for (var matchPattern : matchingPatterns) {
            var matcher = matchPattern.matcher(input);
            if (matcher.find()) {
                // Calculate score based on match quality
                var matchLength = matcher.end() - matcher.start();
                var inputLength = input.length();
                var matchRatio = (double) matchLength / inputLength;
                var score = baseConfidence + (matchRatio * (1.0 - baseConfidence));
                maxScore = Math.max(maxScore, score);
            }
        }
        
        return Math.min(maxScore, 1.0);
    }
    
    /**
     * Check if this template matches the given input
     */
    public boolean matches(String input) {
        return calculateConfidence(input) > 0.0;
    }
    
    /**
     * Get all placeholder variables in this template
     */
    public Set<String> getRequiredVariables() {
        return Set.copyOf(variables);
    }
    
    /**
     * Check if template has the specified variable
     */
    public boolean hasVariable(String variableName) {
        return variables.contains(variableName);
    }
    
    /**
     * Get metadata value by key
     */
    public Optional<Object> getMetadata(String key) {
        return Optional.ofNullable(metadata.get(key));
    }
    
    /**
     * Check if template is valid for rendering with given variables
     */
    public boolean canRender(Map<String, String> variableValues) {
        if (variableValues == null) {
            return variables.isEmpty();
        }
        
        for (var requiredVar : variables) {
            if (!variableValues.containsKey(requiredVar) || 
                variableValues.get(requiredVar) == null ||
                variableValues.get(requiredVar).trim().isEmpty()) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public String toString() {
        return "Template{id='%s', category='%s', pattern='%s', variables=%s}"
            .formatted(id, category, pattern, variables);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Template other)) return false;
        return Objects.equals(id, other.id) &&
               Objects.equals(category, other.category) &&
               Objects.equals(pattern, other.pattern);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id, category, pattern);
    }
}