package com.hellblazer.art.hartcq.feedback;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.templates.Template;
import com.hellblazer.art.hartcq.templates.TemplateManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Manages top-down expectations in the HART-CQ feedback system.
 * Generates template-based predictions and context-aware expectations
 * to guide bottom-up processing and prevent hallucination.
 *
 * The expectation manager maintains a repository of learned patterns
 * and generates predictions based on current context and historical data.
 *
 * @author Claude Code
 */
public class ExpectationManager implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(ExpectationManager.class);

    private final HARTCQConfig config;
    private final TemplateManager templateManager;

    // Context management
    private final Map<String, ContextState> contextHistory;
    private final Queue<String> recentContexts;
    private final int maxContextHistory;

    // Expectation prediction
    private final Map<String, ExpectationPattern> learnedPatterns;
    private final Map<Template, Double> templateConfidenceCache;

    // Statistics
    private final AtomicInteger predictionsGenerated;
    private final AtomicInteger correctPredictions;
    private final AtomicLong totalPredictionTime;

    /**
     * Represents the current context state for expectation generation.
     */
    public static class ContextState {
        private final String context;
        private final List<Token> recentTokens;
        private final Template currentTemplate;
        private final double confidence;
        private final long timestamp;

        public ContextState(String context, List<Token> recentTokens, Template currentTemplate,
                           double confidence) {
            this.context = Objects.requireNonNull(context);
            this.recentTokens = List.copyOf(recentTokens != null ? recentTokens : Collections.emptyList());
            this.currentTemplate = currentTemplate;
            this.confidence = Math.max(0.0, Math.min(1.0, confidence));
            this.timestamp = System.currentTimeMillis();
        }

        public String getContext() { return context; }
        public List<Token> getRecentTokens() { return recentTokens; }
        public Template getCurrentTemplate() { return currentTemplate; }
        public double getConfidence() { return confidence; }
        public long getTimestamp() { return timestamp; }

        public boolean isExpired(long maxAgeMs) {
            return (System.currentTimeMillis() - timestamp) > maxAgeMs;
        }

        @Override
        public String toString() {
            return "ContextState{context='%s', tokens=%d, template=%s, confidence=%.3f}"
                .formatted(context, recentTokens.size(),
                         currentTemplate != null ? currentTemplate.id() : "null", confidence);
        }
    }

    /**
     * Learned expectation pattern for prediction.
     */
    public static class ExpectationPattern {
        private final String contextSignature;
        private final List<String> expectedSequences;
        private final Map<String, Double> transitionProbabilities;
        private final double baseConfidence;
        private int observationCount;
        private long lastUpdated;

        public ExpectationPattern(String contextSignature, double baseConfidence) {
            this.contextSignature = Objects.requireNonNull(contextSignature);
            this.expectedSequences = new ArrayList<>();
            this.transitionProbabilities = new ConcurrentHashMap<>();
            this.baseConfidence = Math.max(0.0, Math.min(1.0, baseConfidence));
            this.observationCount = 0;
            this.lastUpdated = System.currentTimeMillis();
        }

        public void addObservation(String sequence, double weight) {
            if (!expectedSequences.contains(sequence)) {
                expectedSequences.add(sequence);
            }

            var currentProb = transitionProbabilities.getOrDefault(sequence, 0.0);
            var newProb = currentProb + (weight / (observationCount + 1));
            transitionProbabilities.put(sequence, Math.min(1.0, newProb));

            observationCount++;
            lastUpdated = System.currentTimeMillis();
        }

        public double getProbability(String sequence) {
            return transitionProbabilities.getOrDefault(sequence, baseConfidence);
        }

        public List<String> getTopPredictions(int maxPredictions) {
            return transitionProbabilities.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(maxPredictions)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        }

        public String getContextSignature() { return contextSignature; }
        public double getBaseConfidence() { return baseConfidence; }
        public int getObservationCount() { return observationCount; }
        public long getLastUpdated() { return lastUpdated; }
    }

    /**
     * Result of expectation generation.
     */
    public static class ExpectationResult {
        private final List<String> predictions;
        private final Map<String, Double> predictionConfidences;
        private final Template suggestedTemplate;
        private final String contextSignature;
        private final double overallConfidence;

        public ExpectationResult(List<String> predictions, Map<String, Double> predictionConfidences,
                               Template suggestedTemplate, String contextSignature, double overallConfidence) {
            this.predictions = List.copyOf(predictions != null ? predictions : Collections.emptyList());
            this.predictionConfidences = Map.copyOf(predictionConfidences != null ? predictionConfidences : Collections.emptyMap());
            this.suggestedTemplate = suggestedTemplate;
            this.contextSignature = contextSignature;
            this.overallConfidence = Math.max(0.0, Math.min(1.0, overallConfidence));
        }

        public List<String> getPredictions() { return predictions; }
        public Map<String, Double> getPredictionConfidences() { return predictionConfidences; }
        public Template getSuggestedTemplate() { return suggestedTemplate; }
        public String getContextSignature() { return contextSignature; }
        public double getOverallConfidence() { return overallConfidence; }

        public boolean hasPredictions() { return !predictions.isEmpty(); }

        public String getTopPrediction() {
            return predictions.isEmpty() ? null : predictions.get(0);
        }

        public double getTopPredictionConfidence() {
            var topPrediction = getTopPrediction();
            return topPrediction != null ? predictionConfidences.getOrDefault(topPrediction, 0.0) : 0.0;
        }

        @Override
        public String toString() {
            return "ExpectationResult{predictions=%d, topPrediction='%s', confidence=%.3f}"
                .formatted(predictions.size(), getTopPrediction(), getTopPredictionConfidence());
        }
    }

    /**
     * Creates an expectation manager with the given configuration.
     *
     * @param config HART-CQ configuration
     */
    public ExpectationManager(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.templateManager = new TemplateManager();

        this.maxContextHistory = Math.max(10, config.getTemplateConfig().getMaxTemplates() / 10);
        this.contextHistory = new ConcurrentHashMap<>();
        this.recentContexts = new LinkedList<>();
        this.learnedPatterns = new ConcurrentHashMap<>();
        this.templateConfidenceCache = new ConcurrentHashMap<>();

        // Initialize statistics
        this.predictionsGenerated = new AtomicInteger(0);
        this.correctPredictions = new AtomicInteger(0);
        this.totalPredictionTime = new AtomicLong(0);

        logger.info("ExpectationManager initialized with maxContextHistory={}", maxContextHistory);
    }

    /**
     * Generates expectations based on current input tokens and context.
     *
     * @param inputTokens Current input tokens
     * @return Expectation result with predictions and confidence
     */
    public ExpectationResult generateExpectations(Token[] inputTokens) {
        if (inputTokens == null || inputTokens.length == 0) {
            return createEmptyResult("No input tokens provided");
        }

        var startTime = System.currentTimeMillis();
        try {
            // Extract context from input tokens
            var contextSignature = extractContextSignature(inputTokens);

            // Find matching templates
            var candidateResults = templateManager.getAllPossibleOutputs(
                tokenArrayToString(inputTokens), Map.of());
            var candidateTemplates = candidateResults.stream()
                .filter(result -> result.template() != null)
                .map(result -> result.template())
                .toList();

            // Generate predictions based on learned patterns
            var predictions = generatePredictionsFromPatterns(contextSignature);

            // Enhance predictions with template-based expectations
            var enhancedPredictions = enhanceWithTemplates(predictions, candidateTemplates);

            // Calculate overall confidence
            var overallConfidence = calculateOverallConfidence(enhancedPredictions, candidateTemplates);

            // Select best template
            var suggestedTemplate = selectBestTemplate(candidateTemplates, contextSignature);

            var result = new ExpectationResult(
                enhancedPredictions.keySet().stream().toList(),
                enhancedPredictions,
                suggestedTemplate,
                contextSignature,
                overallConfidence
            );

            // Update statistics
            predictionsGenerated.incrementAndGet();

            logger.debug("Generated expectations: {}", result);
            return result;

        } finally {
            totalPredictionTime.addAndGet(System.currentTimeMillis() - startTime);
        }
    }

    /**
     * Updates the expectation manager with observed outcome.
     *
     * @param inputTokens Original input tokens
     * @param actualOutcome What actually occurred
     * @param template Template that was used (if any)
     */
    public void updateWithObservation(Token[] inputTokens, String actualOutcome, Template template) {
        if (inputTokens == null || actualOutcome == null) {
            return;
        }

        var contextSignature = extractContextSignature(inputTokens);

        // Update context history
        var contextState = new ContextState(contextSignature, Arrays.asList(inputTokens),
                                          template, template != null ? 0.8 : 0.5);
        updateContextHistory(contextSignature, contextState);

        // Update learned patterns
        updateLearnedPatterns(contextSignature, actualOutcome);

        // Update template confidence cache
        if (template != null) {
            updateTemplateConfidence(template, true);
        }

        logger.debug("Updated expectation manager with observation: context='{}', outcome='{}'",
                    contextSignature, actualOutcome);
    }

    /**
     * Validates an expectation against actual outcome.
     *
     * @param expectation Generated expectation
     * @param actualOutcome Actual outcome
     * @return true if expectation was correct
     */
    public boolean validateExpectation(ExpectationResult expectation, String actualOutcome) {
        if (expectation == null || actualOutcome == null) {
            return false;
        }

        var isCorrect = expectation.getPredictions().contains(actualOutcome) ||
                       (expectation.getTopPrediction() != null &&
                        expectation.getTopPrediction().equals(actualOutcome));

        if (isCorrect) {
            correctPredictions.incrementAndGet();
        }

        return isCorrect;
    }

    /**
     * Gets expectation statistics.
     *
     * @return expectation statistics
     */
    public ExpectationStats getExpectationStats() {
        var totalPredictions = predictionsGenerated.get();
        var accuracy = totalPredictions > 0 ?
            (double) correctPredictions.get() / totalPredictions : 0.0;
        var avgPredictionTime = totalPredictions > 0 ?
            (double) totalPredictionTime.get() / totalPredictions : 0.0;

        return new ExpectationStats(
            totalPredictions,
            correctPredictions.get(),
            accuracy,
            avgPredictionTime,
            learnedPatterns.size(),
            contextHistory.size()
        );
    }

    /**
     * Resets the expectation manager.
     */
    public void reset() {
        logger.info("Resetting expectation manager");

        contextHistory.clear();
        recentContexts.clear();
        learnedPatterns.clear();
        templateConfidenceCache.clear();

        predictionsGenerated.set(0);
        correctPredictions.set(0);
        totalPredictionTime.set(0);

        templateManager.resetStats();
        templateManager.clearCaches();

        logger.info("Expectation manager reset completed");
    }

    /**
     * Extracts context signature from input tokens.
     */
    private String extractContextSignature(Token[] tokens) {
        if (tokens.length == 0) {
            return "EMPTY";
        }

        // Create context signature from first and last tokens plus length
        var firstToken = tokens[0].getText();
        var lastToken = tokens[tokens.length - 1].getText();
        var length = tokens.length;

        return "%s|%s|%d".formatted(firstToken, lastToken, length);
    }

    /**
     * Converts token array to string for template matching.
     */
    private String tokenArrayToString(Token[] tokens) {
        return Arrays.stream(tokens)
            .map(Token::getText)
            .collect(Collectors.joining(" "));
    }

    /**
     * Generates predictions from learned patterns.
     */
    private Map<String, Double> generatePredictionsFromPatterns(String contextSignature) {
        var predictions = new HashMap<String, Double>();

        var pattern = learnedPatterns.get(contextSignature);
        if (pattern != null) {
            var topPredictions = pattern.getTopPredictions(5);
            for (var prediction : topPredictions) {
                predictions.put(prediction, pattern.getProbability(prediction));
            }
        }

        return predictions;
    }

    /**
     * Enhances predictions with template-based expectations.
     */
    private Map<String, Double> enhanceWithTemplates(Map<String, Double> predictions,
                                                    List<Template> templates) {
        var enhanced = new HashMap<>(predictions);

        for (var template : templates) {
            var templatePrediction = "TEMPLATE_" + template.category();
            var confidence = templateConfidenceCache.getOrDefault(template, template.baseConfidence());
            enhanced.put(templatePrediction, confidence);
        }

        return enhanced;
    }

    /**
     * Calculates overall confidence for expectation result.
     */
    private double calculateOverallConfidence(Map<String, Double> predictions, List<Template> templates) {
        if (predictions.isEmpty() && templates.isEmpty()) {
            return 0.0;
        }

        var maxPredictionConfidence = predictions.values().stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElse(0.0);

        var maxTemplateConfidence = templates.stream()
            .mapToDouble(t -> templateConfidenceCache.getOrDefault(t, t.baseConfidence()))
            .max()
            .orElse(0.0);

        return Math.max(maxPredictionConfidence, maxTemplateConfidence);
    }

    /**
     * Selects the best template for the given context.
     */
    private Template selectBestTemplate(List<Template> templates, String contextSignature) {
        return templates.stream()
            .max(Comparator.comparingDouble(t ->
                templateConfidenceCache.getOrDefault(t, t.baseConfidence())))
            .orElse(null);
    }

    /**
     * Updates context history with new state.
     */
    private void updateContextHistory(String contextSignature, ContextState state) {
        contextHistory.put(contextSignature, state);

        recentContexts.offer(contextSignature);
        if (recentContexts.size() > maxContextHistory) {
            var oldContext = recentContexts.poll();
            // Keep the old context for some time before removing
            // This is a simple implementation - could be more sophisticated
        }
    }

    /**
     * Updates learned patterns with new observation.
     */
    private void updateLearnedPatterns(String contextSignature, String outcome) {
        var pattern = learnedPatterns.computeIfAbsent(contextSignature,
            ctx -> new ExpectationPattern(ctx, config.getTemplateConfig().getMatchThreshold()));

        pattern.addObservation(outcome, 1.0);
    }

    /**
     * Updates template confidence based on success.
     */
    private void updateTemplateConfidence(Template template, boolean success) {
        var currentConfidence = templateConfidenceCache.getOrDefault(template, template.baseConfidence());
        var learningRate = config.getTemplateConfig().getLearningRate();
        var newConfidence = success ?
            currentConfidence + (learningRate * (1.0 - currentConfidence)) :
            currentConfidence * (1.0 - learningRate);

        templateConfidenceCache.put(template, Math.max(0.0, Math.min(1.0, newConfidence)));
    }

    /**
     * Creates an empty expectation result.
     */
    private ExpectationResult createEmptyResult(String reason) {
        logger.debug("Creating empty expectation result: {}", reason);
        return new ExpectationResult(Collections.emptyList(), Collections.emptyMap(),
                                   null, "EMPTY", 0.0);
    }

    /**
     * Closes the expectation manager and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing expectation manager");
        reset();
        logger.info("Expectation manager closed");
    }

    /**
     * Statistics about expectation generation and validation.
     */
    public static class ExpectationStats {
        private final int totalPredictions;
        private final int correctPredictions;
        private final double accuracy;
        private final double averagePredictionTimeMs;
        private final int learnedPatternsCount;
        private final int contextHistorySize;

        public ExpectationStats(int totalPredictions, int correctPredictions, double accuracy,
                              double averagePredictionTimeMs, int learnedPatternsCount,
                              int contextHistorySize) {
            this.totalPredictions = totalPredictions;
            this.correctPredictions = correctPredictions;
            this.accuracy = accuracy;
            this.averagePredictionTimeMs = averagePredictionTimeMs;
            this.learnedPatternsCount = learnedPatternsCount;
            this.contextHistorySize = contextHistorySize;
        }

        public int getTotalPredictions() { return totalPredictions; }
        public int getCorrectPredictions() { return correctPredictions; }
        public double getAccuracy() { return accuracy; }
        public double getAveragePredictionTimeMs() { return averagePredictionTimeMs; }
        public int getLearnedPatternsCount() { return learnedPatternsCount; }
        public int getContextHistorySize() { return contextHistorySize; }

        @Override
        public String toString() {
            return "ExpectationStats{predictions=%d, accuracy=%.2f%%, avgTime=%.2fms, patterns=%d}"
                .formatted(totalPredictions, accuracy * 100, averagePredictionTimeMs, learnedPatternsCount);
        }
    }
}