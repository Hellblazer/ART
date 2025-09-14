package com.hellblazer.art.hartcq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Processes individual windows and produces results.
 * Core logic for window-level processing.
 */
public class WindowProcessor {
    private static final Logger log = LoggerFactory.getLogger(WindowProcessor.class);

    private final FeatureExtractor featureExtractor;
    private final PatternMatcher patternMatcher;

    public WindowProcessor() {
        this.featureExtractor = new FeatureExtractor();
        this.patternMatcher = new PatternMatcher();
    }

    /**
     * Process a window and produce a result.
     */
    public WindowResult process(ProcessingWindow window) {
        log.debug("Processing window {}", window.getWindowId());

        // Extract features from the window
        var features = featureExtractor.extract(window);

        // Match patterns
        var patterns = patternMatcher.match(window, features);

        // Calculate confidence scores
        var confidence = calculateConfidence(features, patterns);

        // Create result
        return new WindowResult(
            window.getWindowId(),
            features,
            patterns,
            confidence,
            System.nanoTime() - window.getCreationTime()
        );
    }

    /**
     * Calculate confidence score for the window processing.
     */
    private double calculateConfidence(WindowFeatures features, List<WindowResult.Pattern> patterns) {
        double score = 0.0;

        // Base confidence from feature quality
        score += features.getQualityScore() * 0.5;

        // Pattern match contribution
        if (!patterns.isEmpty()) {
            double patternScore = patterns.stream()
                .mapToDouble(WindowResult.Pattern::getStrength)
                .average()
                .orElse(0.0);
            score += patternScore * 0.5;
        }

        return Math.min(1.0, score);
    }

    /**
     * Feature extraction from windows.
     */
    static class FeatureExtractor {
        public WindowFeatures extract(ProcessingWindow window) {
            var tokens = window.getTokens();

            // Extract various features
            int wordCount = 0;
            int punctuationCount = 0;
            double avgWordLength = 0;
            var uniqueWords = new HashSet<String>();

            for (var token : tokens) {
                switch (token.getType()) {
                    case WORD -> {
                        wordCount++;
                        uniqueWords.add(token.getText().toLowerCase());
                        avgWordLength += token.getText().length();
                    }
                    case PUNCTUATION -> punctuationCount++;
                }
            }

            if (wordCount > 0) {
                avgWordLength /= wordCount;
            }

            // Calculate quality score
            double qualityScore = calculateQualityScore(wordCount, uniqueWords.size(), avgWordLength);

            return new WindowFeatures(
                wordCount,
                punctuationCount,
                avgWordLength,
                uniqueWords.size(),
                qualityScore
            );
        }

        private double calculateQualityScore(int wordCount, int uniqueWords, double avgLength) {
            double diversity = uniqueWords > 0 ? (double) uniqueWords / wordCount : 0;
            double lengthScore = Math.min(1.0, avgLength / 7.0); // 7 chars is average word
            return (diversity + lengthScore) / 2.0;
        }
    }

    /**
     * Pattern matching within windows.
     */
    static class PatternMatcher {
        private final List<PatternTemplate> templates;

        public PatternMatcher() {
            this.templates = initializeTemplates();
        }

        public List<WindowResult.Pattern> match(ProcessingWindow window, WindowFeatures features) {
            var matches = new ArrayList<WindowResult.Pattern>();
            var text = window.getText().toLowerCase();

            // Debug output
            log.debug("Matching text: '{}' against {} templates", text, templates.size());

            for (var template : templates) {
                boolean isMatch = template.matches(text, features);
                log.debug("Template {} ({}): match={}", template.name, template.type, isMatch);
                if (isMatch) {
                    matches.add(new WindowResult.Pattern(
                        template.type.name(),  // type should be the pattern type (INTERROGATIVE, etc.)
                        template.name,          // value should be the template name (question, etc.)
                        template.calculateConfidence(text, features),
                        0  // position - could be enhanced to find actual position
                    ));
                }
            }

            return matches;
        }

        private List<PatternTemplate> initializeTemplates() {
            // Initialize basic pattern templates
            return List.of(
                new PatternTemplate("question", PatternType.INTERROGATIVE,
                    text -> text.contains("?") || text.contains("what") || text.contains("how")),
                new PatternTemplate("statement", PatternType.DECLARATIVE,
                    text -> text.contains(".") && !text.contains("?")),
                new PatternTemplate("exclamation", PatternType.EXCLAMATORY,
                    text -> text.contains("!")),
                new PatternTemplate("command", PatternType.IMPERATIVE,
                    text -> text.startsWith("please") || text.startsWith("could"))
            );
        }
    }

    /**
     * Pattern template for matching.
     */
    record PatternTemplate(String name, PatternType type, java.util.function.Predicate<String> matcher) {
        boolean matches(String text, WindowFeatures features) {
            return matcher.test(text);
        }

        double calculateConfidence(String text, WindowFeatures features) {
            return matches(text, features) ? 0.8 + (features.getQualityScore() * 0.2) : 0.0;
        }
    }

    /**
     * Pattern types.
     */
    enum PatternType {
        DECLARATIVE,
        INTERROGATIVE,
        EXCLAMATORY,
        IMPERATIVE,
        NARRATIVE,
        DESCRIPTIVE
    }

}