package com.hellblazer.art.hartcq.templates;

import java.util.*;
import java.util.stream.Collectors;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;

/**
 * Intelligent template matcher that uses pattern-based matching and confidence scoring
 * to select the most appropriate template for given input. Provides deterministic
 * selection (same input always produces same template) to prevent hallucination.
 * 
 * @author Claude Code
 */
public class TemplateMatcher {
    
    private final TemplateRepository repository;
    private final double minimumConfidenceThreshold;
    private final MessageDigest digest;
    
    /**
     * Matching result containing the selected template and confidence score
     */
    public record MatchResult(
        Template template,
        double confidence,
        String matchReason
    ) {
        public boolean isSuccessful() {
            return template != null;
        }
    }

    /**
     * Multi-match result containing all matching templates with confidence scores
     */
    public static class MultiMatchResult {
        private final Map<Template, Double> templateConfidences;

        public MultiMatchResult(Map<Template, Double> templateConfidences) {
            this.templateConfidences = new HashMap<>(templateConfidences);
        }

        public List<Template> getTemplates() {
            return new ArrayList<>(templateConfidences.keySet());
        }

        public double getConfidence(Template template) {
            return templateConfidences.getOrDefault(template, 0.0);
        }

        public boolean isEmpty() {
            return templateConfidences.isEmpty();
        }
    }
    
    /**
     * Create matcher with default confidence threshold of 0.3
     */
    public TemplateMatcher(TemplateRepository repository) {
        this(repository, 0.3);
    }
    
    /**
     * Create matcher with custom confidence threshold
     */
    public TemplateMatcher(TemplateRepository repository, double minimumConfidenceThreshold) {
        this.repository = Objects.requireNonNull(repository, "Repository cannot be null");
        this.minimumConfidenceThreshold = validateConfidenceThreshold(minimumConfidenceThreshold);
        
        try {
            this.digest = MessageDigest.getInstance("SHA-256");
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize message digest", e);
        }
    }
    
    /**
     * Find all matching templates for the given input
     * Used for providing multiple valid outputs for ambiguous input
     */
    public MultiMatchResult findAllMatches(String input) {
        if (input == null || input.trim().isEmpty()) {
            return new MultiMatchResult(Map.of());
        }

        var candidateTemplates = new HashMap<Template, Double>();
        var normalizedInput = input.trim().toLowerCase();

        // Find all templates that match with sufficient confidence
        for (var template : repository.getAllTemplates()) {
            double confidence = template.calculateConfidence(normalizedInput);
            if (confidence >= minimumConfidenceThreshold) {
                candidateTemplates.put(template, confidence);
            }
        }

        return new MultiMatchResult(candidateTemplates);
    }

    /**
     * Find the best matching template for the given input
     * DETERMINISTIC: Same input will always return the same template
     */
    public MatchResult findBestMatch(String input) {
        if (input == null || input.trim().isEmpty()) {
            return new MatchResult(null, 0.0, "Empty or null input");
        }
        
        var normalizedInput = normalizeInput(input);
        var candidates = findCandidateTemplates(normalizedInput);
        
        if (candidates.isEmpty()) {
            return new MatchResult(null, 0.0, "No matching templates found");
        }
        
        // For deterministic selection, sort by confidence then by template ID
        var bestCandidate = candidates.stream()
            .sorted((a, b) -> {
                var confidenceCompare = Double.compare(b.confidence, a.confidence);
                if (confidenceCompare != 0) {
                    return confidenceCompare;
                }
                // Secondary sort by template ID for determinism
                return a.template.id().compareTo(b.template.id());
            })
            .findFirst()
            .orElse(null);
            
        if (bestCandidate == null || bestCandidate.confidence < minimumConfidenceThreshold) {
            return new MatchResult(null, 0.0, "No template meets confidence threshold");
        }
        
        return bestCandidate;
    }
    
    /**
     * Find best matching template in a specific category
     */
    public MatchResult findBestMatchInCategory(String input, String category) {
        if (input == null || input.trim().isEmpty()) {
            return new MatchResult(null, 0.0, "Empty or null input");
        }
        
        var categoryTemplates = repository.getByCategory(category);
        if (categoryTemplates.isEmpty()) {
            return new MatchResult(null, 0.0, "Category not found: " + category);
        }
        
        var normalizedInput = normalizeInput(input);
        var candidates = categoryTemplates.stream()
            .map(template -> new MatchResult(
                template,
                template.calculateConfidence(normalizedInput),
                "Category match: " + category
            ))
            .filter(result -> result.confidence >= minimumConfidenceThreshold)
            .sorted((a, b) -> {
                var confidenceCompare = Double.compare(b.confidence, a.confidence);
                if (confidenceCompare != 0) {
                    return confidenceCompare;
                }
                return a.template.id().compareTo(b.template.id());
            })
            .collect(Collectors.toList());
            
        if (candidates.isEmpty()) {
            return new MatchResult(null, 0.0, "No templates in category meet threshold");
        }
        
        return candidates.get(0);
    }
    
    /**
     * Find multiple matching templates (up to maxResults)
     */
    public List<MatchResult> findMultipleMatches(String input, int maxResults) {
        if (input == null || input.trim().isEmpty()) {
            return List.of();
        }
        
        var normalizedInput = normalizeInput(input);
        var candidates = findCandidateTemplates(normalizedInput);
        
        return candidates.stream()
            .filter(result -> result.confidence >= minimumConfidenceThreshold)
            .sorted((a, b) -> {
                var confidenceCompare = Double.compare(b.confidence, a.confidence);
                if (confidenceCompare != 0) {
                    return confidenceCompare;
                }
                return a.template.id().compareTo(b.template.id());
            })
            .limit(maxResults)
            .collect(Collectors.toList());
    }
    
    /**
     * Get deterministic template selection based on input hash
     * This ensures the same input always selects the same template from candidates
     */
    public MatchResult getDeterministicMatch(String input) {
        var candidates = findMultipleMatches(input, Integer.MAX_VALUE);
        if (candidates.isEmpty()) {
            return new MatchResult(null, 0.0, "No candidates found");
        }
        
        // Use hash to deterministically select from top candidates with same confidence
        var topConfidence = candidates.get(0).confidence;
        var topCandidates = candidates.stream()
            .filter(c -> Math.abs(c.confidence - topConfidence) < 0.001) // Same confidence within epsilon
            .collect(Collectors.toList());
            
        if (topCandidates.size() == 1) {
            return topCandidates.get(0);
        }
        
        // Use input hash for deterministic selection among equals
        var hash = computeInputHash(input);
        var selectedIndex = Math.abs(hash) % topCandidates.size();
        var selected = topCandidates.get(selectedIndex);
        
        return new MatchResult(
            selected.template,
            selected.confidence,
            "Deterministic selection from " + topCandidates.size() + " candidates"
        );
    }
    
    /**
     * Find candidate templates with their confidence scores
     */
    private List<MatchResult> findCandidateTemplates(String input) {
        return repository.getAllTemplates().stream()
            .map(template -> new MatchResult(
                template,
                template.calculateConfidence(input),
                "Pattern match"
            ))
            .filter(result -> result.confidence > 0.0)
            .collect(Collectors.toList());
    }
    
    /**
     * Normalize input for consistent matching
     */
    private String normalizeInput(String input) {
        return input.trim().toLowerCase();
    }
    
    /**
     * Compute deterministic hash of input string
     */
    private int computeInputHash(String input) {
        synchronized (digest) {
            digest.reset();
            var hashBytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            
            // Convert first 4 bytes to int
            var hash = 0;
            for (var i = 0; i < Math.min(4, hashBytes.length); i++) {
                hash = (hash << 8) | (hashBytes[i] & 0xFF);
            }
            return hash;
        }
    }
    
    /**
     * Validate confidence threshold
     */
    private double validateConfidenceThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be between 0.0 and 1.0");
        }
        return threshold;
    }
    
    /**
     * Check if input would match any template
     */
    public boolean hasMatch(String input) {
        return findBestMatch(input).isSuccessful();
    }
    
    /**
     * Get all templates that match the input above threshold
     */
    public List<Template> getAllMatching(String input) {
        return findMultipleMatches(input, Integer.MAX_VALUE).stream()
            .map(MatchResult::template)
            .collect(Collectors.toList());
    }
    
    /**
     * Get confidence threshold
     */
    public double getMinimumConfidenceThreshold() {
        return minimumConfidenceThreshold;
    }
    
    /**
     * Create a matcher with different confidence threshold
     */
    public TemplateMatcher withThreshold(double newThreshold) {
        return new TemplateMatcher(repository, newThreshold);
    }
    
    @Override
    public String toString() {
        return "TemplateMatcher{threshold=%.2f, templateCount=%d}"
            .formatted(minimumConfidenceThreshold, repository.getTemplateCount());
    }
}